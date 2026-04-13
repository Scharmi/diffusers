import os
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Protocol

import psutil
import torch
import wandb
from loguru import logger
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion import diffuse
from src.distributed import get_rank, get_world_size, is_distributed
from src.model import VAE, PredictionTarget, Predictor, PredictorMetadata
from src.schedule import ScheduleGroup
from src.timestep import Timestep
from src.train import ExpMovingAverageWrapper, WarmupCosineLR


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
    use_ema: bool = True  # Exponential Moving Average
    ema_update_every_n_steps: int = 10
    use_amp: bool = True  # Automatic Mixed Precision


class Trainer:
    raw_model: Predictor
    model: nn.Module
    vae: VAE | None

    def __init__(
        self,
        model: Predictor,
        schedules: ScheduleGroup,
        config: TrainingConfig = TrainingConfig(),
        vae: VAE | None = None,
    ):
        self.raw_model = model
        self.model = model
        self.vae = vae

        if is_distributed():
            self.model = DDP(self.raw_model, device_ids=[torch.cuda.current_device()])

        self.schedules = schedules
        self.config = config

        self.optimizer = optim.AdamW(self.raw_model.parameters(), lr=self.config.lr)
        self.criterion = nn.MSELoss(reduction="none")

        # self.amp_dtype = (
        #     torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # )

        # For some reason the code above is lying as V100 does not support bf16
        # and it slows down training significantly.
        self.amp_dtype = torch.float16

        needs_scaler = self.config.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(device="cuda", enabled=needs_scaler)

        self.current_epoch = 0
        self.total_steps_executed = 0

        if get_rank() == 0:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _get_trainer_state_path(self):
        return os.path.join(
            self.config.checkpoint_dir, f"trainer_{self.raw_model.file_name}"
        )

    def save_checkpoint(self, model: nn.Module):
        metadata = {
            PredictorMetadata.AlphaSchedule: self.schedules.alpha.__class__.__name__,
            PredictorMetadata.SigmaSchedule: self.schedules.sigma.__class__.__name__,
            PredictorMetadata.EtaSchedule: self.schedules.eta.__class__.__name__,
            PredictorMetadata.VAE: self.vae.model_id if self.vae is not None else None,
        }

        model.save(**metadata)

        trainer_state = {
            "epoch": self.current_epoch,
            "total_steps": self.total_steps_executed,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(trainer_state, self._get_trainer_state_path())

    def load_checkpoint(self):
        path = self._get_trainer_state_path()
        if os.path.exists(path):
            state = torch.load(path, weights_only=False)
            self.current_epoch = state["epoch"]
            self.total_steps_executed = state["total_steps"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])

            logger.info(
                f"Resuming training from epoch {self.current_epoch}, step {self.total_steps_executed}"
            )
        else:
            logger.info("No trainer checkpoint found, starting fresh training")

    def train(self, dataloader: DataLoader):
        device = next(self.model.parameters()).device
        solver_T = self.raw_model.timestep_config.T

        total_training_steps = self.config.epochs * len(dataloader)
        lr_scheduler = WarmupCosineLR(
            self.optimizer,
            total_steps=total_training_steps,
            last_epoch=self.total_steps_executed - 1,
        )

        ema_wrapper = None
        if self.config.use_ema:
            ema_wrapper = ExpMovingAverageWrapper(self.raw_model)
            logger.info("Using Exponential Moving Average (EMA)")

        if self.config.use_amp:
            logger.info(
                f"Using Automatic Mixed Precision (AMP) with dtype {self.amp_dtype}"
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

            end_time = time.time()
            for batch_idx, (X, _) in enumerate(pbar):
                data_time = time.time() - end_time

                self.optimizer.zero_grad()

                X = X.to(device, non_blocking=True)
                if self.vae is not None:
                    X = self.vae.encode(X.half()).float()

                vae_compute_time = time.time() - end_time - data_time

                match self.config.time_sampler:
                    case TimeSampler.UNIFORM_CONTINUOUS:
                        t = torch.rand(X.size(0), device=X.device) * solver_T
                    case TimeSampler.UNIFORM_DISCRETE:
                        t = torch.randint(
                            1, int(solver_T) + 1, (X.size(0),), device=X.device
                        )
                    case TimeSampler.EDM:
                        P_mean = -1.2
                        P_std = 1.2
                        t = torch.randn(X.size(0), device=X.device) * P_std + P_mean
                        t = torch.exp(t)
                        t = torch.clamp(t, max=solver_T)

                t = Timestep(self.raw_model.timestep_config, t)

                with autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=self.config.use_amp,
                ):
                    X_noisy, noise = diffuse(X, t, self.schedules)
                    pred = self.model(X_noisy, timestep=t, schedules=self.schedules)

                    target = (
                        noise if self.raw_model.target == PredictionTarget.Noise else X
                    )

                    weight = self.raw_model.loss_weight(t, self.schedules).view(
                        -1, 1, 1, 1
                    )
                    loss = self.criterion(pred, target) * weight
                    loss = loss.mean()

                self.scaler.scale(loss).backward()

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), max_norm=1.0
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                lr_scheduler.step()

                unet_compute_time = (
                    time.time() - end_time - data_time - vae_compute_time
                )

                if (
                    ema_wrapper is not None
                    and self.total_steps_executed % self.config.ema_update_every_n_steps
                    == 0
                ):
                    ema_wrapper.update(self.raw_model)

                with torch.no_grad():
                    loss_to_log = loss.detach().clone()
                    if is_distributed():
                        torch.distributed.all_reduce(
                            loss_to_log, op=torch.distributed.ReduceOp.SUM
                        )
                        loss_to_log /= get_world_size()

                total_loss += loss_to_log.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": avg_loss})

                if get_rank() == 0 and is_distributed():
                    self._log_metrics(
                        loss=loss.item(),
                        avg_loss=avg_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                        epoch=epoch,
                        global_step=self.total_steps_executed,
                        data_time=data_time,
                        vae_compute_time=vae_compute_time,
                        unet_compute_time=unet_compute_time,
                    )

                if (
                    self.total_steps_executed % self.config.checkpoint_interval_steps
                    == 0
                    and get_rank() == 0
                ):
                    self.save_checkpoint(
                        ema_wrapper.ema_model
                        if ema_wrapper is not None
                        else self.raw_model
                    )

                self.total_steps_executed += 1
                end_time = time.time()

        if get_rank() == 0:
            self.save_checkpoint(
                ema_wrapper.ema_model if ema_wrapper is not None else self.raw_model
            )

    def _log_metrics(
        self,
        loss: float,
        avg_loss: float,
        lr: float,
        epoch: int,
        global_step: int,
        data_time: float,
        vae_compute_time: float,
        unet_compute_time: float,
    ):
        device = torch.cuda.current_device()

        vram_total_bytes = torch.cuda.get_device_properties(device).total_memory
        vram_allocated_bytes = torch.cuda.memory_allocated(device)
        vram_reserved_bytes = torch.cuda.memory_reserved(device)
        gb_divider = 1024**3

        sys_mem = psutil.virtual_memory()

        wandb.log(
            {
                "train/loss_step": loss,
                "train/loss_avg": avg_loss,
                "train/learning_rate": lr,
                "train/epoch": epoch,
                "stats/vram_allocated_gb": vram_allocated_bytes / gb_divider,
                "stats/vram_reserved_gb": vram_reserved_bytes / gb_divider,
                "stats/vram_total_gb": vram_total_bytes / gb_divider,
                "stats/vram_utilization_%": (vram_allocated_bytes / vram_total_bytes)
                * 100,
                "stats/ram_used_gb": sys_mem.used / gb_divider,
                "stats/ram_total_gb": sys_mem.total / gb_divider,
                "stats/ram_utilization_%": sys_mem.percent,
                "perf/data_time_sec": data_time,
                "perf/vae_compute_time_sec": vae_compute_time,
                "perf/unet_compute_time_sec": unet_compute_time,
                "global_step": self.total_steps_executed,
            },
            step=self.total_steps_executed,
        )
