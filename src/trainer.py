import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Protocol

import torch
import wandb
from loguru import logger
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion import diffuse
from src.distributed import get_rank, is_distributed
from src.model import PredictionTarget, Predictor, PredictorMetadata
from src.schedule import ScheduleGroup
from src.timestep import Timestep


class DatasetProtocol(Protocol):
    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]: ...


class TimeSampler(StrEnum):
    UNIFORM_DISCRETE = "uniform_discrete"
    UNIFORM_CONTINUOUS = "uniform_continuous"
    EDM = "edm"


@dataclass
class TrainingConfig:
    epochs: int = 1000
    lr: float = 1e-4
    checkpoint_dir: str = "models/"
    checkpoint_interval_steps: int = 100
    time_sampler: TimeSampler = TimeSampler.UNIFORM_CONTINUOUS


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
        self.criterion = nn.MSELoss(reduction="none")

        self.current_epoch = 0
        self.total_steps_executed = 0

        if get_rank() == 0:
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
                disable=get_rank() != 0,
            )

            for batch_idx, (X, _) in enumerate(pbar):
                self.optimizer.zero_grad()

                X = X.to(device)

                match self.config.time_sampler:
                    case TimeSampler.UNIFORM_CONTINUOUS:
                        t = torch.rand(X.size(0), device=X.device) * max_t
                    case TimeSampler.UNIFORM_DISCRETE:
                        t = torch.randint(
                            1, int(max_t) + 1, (X.size(0),), device=X.device
                        )
                    case TimeSampler.EDM:
                        P_mean = -1.2
                        P_std = 1.2
                        t = torch.randn(X.size(0), device=X.device) * P_std + P_mean
                        t = torch.exp(t)
                        t = torch.clamp(t, max=max_t)

                t = Timestep(self.raw_model.timestep_config, t)

                X_noisy, noise = diffuse(X, t, self.schedules)
                pred = self.model(X_noisy, timestep=t, schedules=self.schedules)

                target = noise if self.raw_model.target == PredictionTarget.Noise else X

                weight = self.raw_model.loss_weight(t, self.schedules).view(-1, 1, 1, 1)
                loss = self.criterion(pred, target) * weight
                loss = loss.mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), max_norm=1.0
                )

                self.optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": avg_loss})

                if get_rank() == 0 and is_distributed():
                    self._log_metrics(
                        loss=loss.item(),
                        avg_loss=avg_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                        epoch=epoch,
                        global_step=self.total_steps_executed,
                    )

                if (
                    self.total_steps_executed % self.config.checkpoint_interval_steps
                    == 0
                    and get_rank() == 0
                ):
                    self.save_checkpoint()

                self.total_steps_executed += 1

        if get_rank() == 0:
            self.save_checkpoint()

    def _log_metrics(
        self,
        loss: float,
        avg_loss: float,
        lr: float,
        epoch: int,
        global_step: int,
    ):
        device = torch.cuda.current_device()

        vram_total_bytes = torch.cuda.get_device_properties(device).total_memory
        vram_allocated_bytes = torch.cuda.memory_allocated(device)
        vram_reserved_bytes = torch.cuda.memory_reserved(device)
        gb_divider = 1024**3

        wandb.log(
            {
                "train/loss_step": loss,
                "train/loss_avg": avg_loss,
                "train/learning_rate": lr,
                "train/epoch": epoch,
                "train/vram_allocated_gb": vram_allocated_bytes / gb_divider,
                "train/vram_reserved_gb": vram_reserved_bytes / gb_divider,
                "train/vram_total_gb": vram_total_bytes / gb_divider,
                "train/vram_utilization_%": (vram_allocated_bytes / vram_total_bytes)
                * 100,
                "global_step": self.total_steps_executed,
            },
            step=self.total_steps_executed,
        )
