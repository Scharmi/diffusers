import os
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
from loguru import logger
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import diffuse
from src.distributed import RANK, is_distributed
from src.model import Predictor, PredictorMetadata
from src.schedule import ScheduleGroup
from src.timestep import Timestep


class DatasetProtocol(Protocol):
    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]: ...


@dataclass
class TrainingConfig:
    epochs: int = 1000
    lr: float = 1e-4
    checkpoint_dir: str = "models/"
    checkpoint_interval_steps: int = 5
    continuous_time: bool = True


class Trainer:
    raw_model: Predictor
    model: nn.Module

    def __init__(
        self,
        model: Predictor,
        schedules: ScheduleGroup,
        config: TrainingConfig = TrainingConfig(),
    ):
        self.raw_model = model
        self.model = model

        if is_distributed():
            self.model = DDP(self.raw_model, device_ids=[torch.cuda.current_device()])

        self.schedules = schedules
        self.config = config

        self.optimizer = optim.AdamW(self.raw_model.parameters(), lr=self.config.lr)
        self.criterion = nn.MSELoss()

        self.current_epoch = 0
        self.total_steps_executed = 0

        if RANK == 0:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _get_trainer_state_path(self):
        return os.path.join(
            self.config.checkpoint_dir, f"trainer_{self.raw_model.file_name}"
        )

    def save_checkpoint(self):
        metadata = {
            PredictorMetadata.AlphaSchedule: self.schedules.alpha.__class__.__name__,
            PredictorMetadata.SigmaSchedule: self.schedules.sigma.__class__.__name__,
            PredictorMetadata.EtaSchedule: self.schedules.eta.__class__.__name__,
        }

        self.raw_model.save(**metadata)

        trainer_state = {
            "epoch": self.current_epoch,
            "total_steps": self.total_steps_executed,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(trainer_state, self._get_trainer_state_path())

    def load_checkpoint(self):
        path = self._get_trainer_state_path()
        if os.path.exists(path):
            state = torch.load(path, weights_only=False)
            self.current_epoch = state["epoch"]
            self.total_steps_executed = state["total_steps"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

            logger.info(
                f"Resuming training from epoch {self.current_epoch}, step {self.total_steps_executed}"
            )
        else:
            logger.info("No trainer checkpoint found, starting fresh training")

    def train(self, dataloader: DataLoader):
        device = next(self.model.parameters()).device
        max_t = self.raw_model.timestep_config.T

        total_training_steps = self.config.epochs * len(dataloader)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_training_steps,
            eta_min=1e-6,
            last_epoch=self.total_steps_executed - 1,
        )

        self.model.train()
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            if is_distributed():
                dataloader.sampler.set_epoch(epoch)  # ty: ignore

            total_loss = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                disable=RANK != 0,
            )

            for batch_idx, (X, _) in enumerate(pbar):
                self.optimizer.zero_grad()

                X = X.to(device)

                if self.config.continuous_time:
                    t = torch.rand(X.size(0), device=X.device) * max_t
                else:
                    t = torch.randint(1, int(max_t) + 1, (X.size(0),), device=X.device)

                t = Timestep(self.raw_model.timestep_config, t)

                X_noisy, noise = diffuse(X, t, self.schedules)
                noise_pred = self.model(X_noisy, timestep=t)

                loss = self.criterion(
                    noise_pred, noise
                )  # TODO: Add loss weighting (EDM)
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

                if (
                    self.total_steps_executed % self.config.checkpoint_interval_steps
                    == 0
                    and RANK == 0
                ):
                    self.save_checkpoint()

                self.total_steps_executed += 1

        if RANK == 0:
            self.save_checkpoint()
