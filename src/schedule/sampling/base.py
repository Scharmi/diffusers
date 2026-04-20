from abc import ABC, abstractmethod

import torch

from src.timestep import Timestep, TimestepConfig

EPSILON = 1e-9


class SamplingSchedule(ABC):
    max_t: float
    T: float

    # If values are close to 1.0 it generates noisy images.
    def __init__(self, *, max_t: float = 0.95, T: float = 1.0):
        self.max_t = max_t
        self.T = T

    @abstractmethod
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        pass


class LinearSamplingSchedule(SamplingSchedule):
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        steps = torch.linspace(self.max_t, 0.0, n_steps + 1)
        return Timestep(TimestepConfig(kind="continuous", T=self.T), steps)
