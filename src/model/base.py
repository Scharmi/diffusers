import os
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Self

import torch
from loguru import logger
from torch import nn

from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig


class PredictorMetadata(StrEnum):
    AlphaSchedule = "alpha_schedule"
    SigmaSchedule = "sigma_schedule"
    EtaSchedule = "eta_schedule"
    VAE: str = "vae"


class PersistableModule(nn.Module):
    file_name: str
    metadata: dict = {}

    def save(self, **extra_metadata):
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {"type": self.__class__.__name__, **extra_metadata},
        }
        torch.save(checkpoint, f"models/{self.file_name}")

    def load(self):
        checkpoint = torch.load(f"models/{self.file_name}", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("config", {})

        type_name = self.metadata.get("type", None)
        if type_name is not None and type_name != self.__class__.__name__:
            logger.warning(
                f"Loaded model type '{type_name}' does not match current class '{self.__class__.__name__}'"
            )

    def try_load(self) -> bool:
        if os.path.exists(f"models/{self.file_name}"):
            self.load()
            return True
        return False

    @classmethod
    def load_from_file(cls, file_path: str) -> Self:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        checkpoint = torch.load(file_path, weights_only=False)
        config = checkpoint.get("config", {})
        type_name = config.get("type")

        if not type_name:
            raise ValueError(f"Model {file_path} does not contain metadata 'type'.")

        target_class = cls._get_subclass_by_name(type_name)
        if not target_class:
            raise ValueError(f"Subclass '{type_name}' not found.")

        model = target_class(**config)

        model.load_state_dict(checkpoint["state_dict"])
        model.metadata = config
        model.file_name = os.path.basename(file_path)

        return model

    @classmethod
    def _get_subclass_by_name(cls, name: str):
        if cls.__name__ == name:
            return cls

        for subclass in cls.__subclasses__():
            found = subclass._get_subclass_by_name(name)
            if found:
                return found

        return None


class PredictionTarget(StrEnum):
    Noise = "noise"
    x0 = "x0"
    Score = "score"
    Vecolcity = "velocity"

    @staticmethod
    def from_value(value: str) -> "PredictionTarget":
        for target in PredictionTarget:
            if target.value == value:
                return target
        raise ValueError(f"Unsupported PredictionTarget value: {value}")

    def to_hf(self) -> str:
        match self:
            case PredictionTarget.Noise:
                return "epsilon"
            case PredictionTarget.x0:
                return "v_prediction"
            case PredictionTarget.Vecolcity:
                return "v_prediction"
            case PredictionTarget.Score:
                raise NotImplementedError("Not supported")
            case _:
                raise ValueError(f"Unsupported PredictionTarget: {self}")


class Predictor(PersistableModule, ABC):
    timestep_config: TimestepConfig
    n_channels: int
    img_width: int
    img_height: int
    target: PredictionTarget

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
    ) -> torch.Tensor:
        pass

    def loss_weight(
        self, t: Timestep, schedules: ScheduleGroup | None = None
    ) -> torch.Tensor:
        return torch.tensor(1.0, device=t.device)

    def save(self, **extra_metadata):
        super().save(
            T=self.timestep_config.T,
            n_channels=self.n_channels,
            img_width=self.img_width,
            img_height=self.img_height,
            target=self.target.value,
            **extra_metadata,
        )

    def load(self):
        super().load()
        meta_T = self.metadata.get("T", None)
        meta_n_channels = self.metadata.get("n_channels", None)
        meta_img_width = self.metadata.get("img_width", None)
        meta_img_height = self.metadata.get("img_height", None)
        meta_target = self.metadata.get("target", None)

        if meta_T is not None and meta_T != self.timestep_config.T:
            logger.warning(
                f"Loaded model T '{meta_T}' does not match current timestep_config.T '{self.timestep_config.T}'"
            )

        for attr_name, meta_value, current_value in [
            ("n_channels", meta_n_channels, self.n_channels),
            ("img_width", meta_img_width, self.img_width),
            ("img_height", meta_img_height, self.img_height),
            ("target", meta_target, self.target.value if self.target else None),
        ]:
            if meta_value is not None and meta_value != current_value:
                logger.warning(
                    f"Loaded model {attr_name} '{meta_value}' does not match current value '{current_value}'"
                )

    @classmethod
    def load_from_file(cls, file_path: str) -> "Predictor":
        instance = super().load_from_file(file_path)

        if not issubclass(instance.__class__, Predictor):
            raise ValueError(f"Loaded model from {file_path} is not a Predictor")

        return instance
