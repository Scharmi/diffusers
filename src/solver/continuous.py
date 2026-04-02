import math

import torch

from src.common import assert_type
from src.equation import DifferentialEquation
from src.solver import Solver
from src.timestep import Timestep


class ContinuousSolver(Solver):
    equation: DifferentialEquation

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        T = assert_type(self.T, float)
        return self._step(x_t, t.as_continuous(T), s.as_continuous(T))


class EulerODESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        drift = self.equation.drift(x_t, t)
        x_s = x_t + h * drift

        return x_s


class HeunODESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        k1 = self.equation.drift(x_t, t)
        x_s_euler = x_t + h * k1

        if s.steps.max() > 0.0:
            k2 = self.equation.drift(x_s_euler, s)
            x_s = x_t + h / 2 * (k1 + k2)

        return x_s


class EulerMaruyamaSDESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        drift = self.equation.drift(x_t, t)

        diff_coeff = self.equation.diffusion_coeff(x_t, t)
        diffusion = diff_coeff * math.sqrt(-h) * torch.randn_like(x_t)

        x_s = x_t + h * drift + diffusion

        return x_s
