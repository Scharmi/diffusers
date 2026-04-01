from abc import ABC

import torch

from src.common import assert_type, load_scheduler_config
from src.timestep import Timestep, TimestepConfig


class HuggingFaceDDPMBaseSchedule(ABC):
    def __init__(self, model_id: str):
        config = load_scheduler_config(model_id)

        self.T = assert_type(config.get("num_train_timesteps"), int)
        beta_start = assert_type(config.get("beta_start"), float)
        beta_end = assert_type(config.get("beta_end"), float)
        schedule_type = assert_type(config.get("beta_schedule"), str)

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, self.T)
        elif schedule_type == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.T) ** 2
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_alpha_at_t(self, timestep: Timestep) -> torch.Tensor:
        t_adapted = timestep.adapt(TimestepConfig(kind="discrete", T=self.T))
        t_indices = t_adapted.steps.long()
        self.alphas_cumprod = self.alphas_cumprod.to(t_indices.device)
        return torch.sqrt(self.alphas_cumprod[t_indices])

    def get_sigma_at_t(self, timestep: Timestep) -> torch.Tensor:
        t_adapted = timestep.adapt(TimestepConfig(kind="discrete", T=self.T))
        t_indices = t_adapted.steps.long()
        self.alphas_cumprod = self.alphas_cumprod.to(t_indices.device)
        return torch.sqrt(1.0 - self.alphas_cumprod[t_indices])
