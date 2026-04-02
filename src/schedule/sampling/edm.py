import torch

from src.schedule.sampling import SamplingSchedule
from src.timestep import Timestep, TimestepConfig


class EDMSamplingSchedule(SamplingSchedule):
    sigma_min: float
    sigma_max: float
    rho: float

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    # Note: This assumes sigma_edm(t) = t. Otherwise t_i = sigma_edm^{-1}(sigma_i).
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        indices = torch.arange(n_steps, dtype=torch.float32)

        inv_rho = 1.0 / self.rho
        sigmas = self.sigma_max**inv_rho + indices / (n_steps - 1) * (
            self.sigma_min**inv_rho - self.sigma_max**inv_rho
        )
        sigmas = sigmas**self.rho

        sigmas = torch.cat([sigmas, torch.tensor([0.0])])

        return Timestep(TimestepConfig(kind="continuous", T=self.T), sigmas)
