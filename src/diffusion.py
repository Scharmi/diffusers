import torch
from torch import Tensor

from src.schedule import ScheduleGroup
from src.timestep import Timestep


def diffuse(x0: Tensor, t: Timestep, schedules: ScheduleGroup) -> tuple[Tensor, Tensor]:
    noise = torch.randn_like(x0)
    alpha = schedules.alpha(t).view(-1, 1, 1, 1)
    sigma = schedules.sigma(t).view(-1, 1, 1, 1)

    return alpha * x0 + sigma * noise, noise


# Samples from p(x_t | x__{s}) where t > s
def diffuse_from(
    x_0: Tensor,
    x_s: Tensor,
    t_s: Timestep,
    t: Timestep,
    schedules: ScheduleGroup,
) -> Tensor:
    noise = torch.rand_like(x_0)
    alpha_t = schedules.alpha(t).view(-1, 1, 1, 1)
    sigma_t = schedules.sigma(t).view(-1, 1, 1, 1)
    alpha_s = schedules.alpha(t_s).view(-1, 1, 1, 1)
    sigma_s = schedules.sigma(t_s).view(-1, 1, 1, 1)

    return x_s + (alpha_t - alpha_s) * x_0 + torch.sqrt(sigma_t**2 - sigma_s**2) * noise
