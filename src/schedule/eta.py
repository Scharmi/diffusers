from abc import ABC, abstractmethod

import torch

from src.schedule.alpha import AlphaSchedule
from src.schedule.sigma import SigmaSchedule
from src.timestep import Timestep, TimestepConfig


class EtaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", T=1.0)
    alpha: AlphaSchedule
    sigma: SigmaSchedule

    def __init__(self, alpha_schedule: AlphaSchedule, sigma_schedule: SigmaSchedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule

    @abstractmethod
    def __call__(self, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        pass


class ConstantEtaSchedule(EtaSchedule):
    def __init__(self, eta_value: float):
        self.eta_value = eta_value

    def __call__(self, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        return torch.tensor(self.eta_value, device=t.steps.device)


class DDPMEtaSchedule(EtaSchedule):
    def __call__(self, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        adapted_t = t.adapt(self.timestep_config)
        adapted_t_prev = t_prev.adapt(self.timestep_config)

        alpha_t = self.alpha(adapted_t)
        alpha_t_prev = self.alpha(adapted_t_prev)
        sigma_t = self.sigma(adapted_t)

        return torch.sqrt(alpha_t_prev**2 - alpha_t**2) / (alpha_t_prev * sigma_t)
