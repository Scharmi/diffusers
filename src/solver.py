import math
from abc import ABC, abstractmethod

import torch
from loguru import logger

from src.model import ModuleMetadata, Predictor
from src.schedule import ScheduleGroup
from src.timestep import Timestep


class Solver(ABC):
    model: Predictor
    schedules: ScheduleGroup

    def __init__(
        self,
        *,
        model: Predictor,
        schedules: ScheduleGroup,
    ):
        self.model = model
        self.schedules = schedules

        schedule_meta_checks = (
            (ModuleMetadata.AlphaSchedule, "alpha"),
            (ModuleMetadata.SigmaSchedule, "sigma"),
        )

        for meta_key, schedule_attr in schedule_meta_checks:
            trained_schedule_meta = model.metadata.get(meta_key, None)
            current_schedule = getattr(schedules, schedule_attr, None)

            if trained_schedule_meta is None or current_schedule is None:
                continue

            current_schedule_name = current_schedule.__class__.__name__
            if current_schedule_name != trained_schedule_meta:
                logger.warning(
                    f"Denoiser model was trained with {schedule_attr} schedule '{trained_schedule_meta}', "
                    f"but current schedule is '{current_schedule_name}'"
                )

    # Assumes t_prev < t
    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        return self._denoise(x_t, t, t_prev)

    @abstractmethod
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        pass


class DiscreteSolver(Solver):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        T = int(self.model.timestep_config.T)
        t = t.as_discrete(T)
        t_prev = t_prev.as_discrete(T)

        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        alpha_t_prev = self.schedules.alpha(t_prev).view(-1, 1, 1, 1)
        sigma_t_prev = self.schedules.sigma(t_prev).view(-1, 1, 1, 1)
        eta_t_prev = self.schedules.eta(t, t_prev).view(-1, 1, 1, 1)

        noise_pred = self.model(x_t, timestep=t)
        mean = (alpha_t_prev / alpha_t) * x_t + (
            sigma_t_prev * torch.sqrt(1 - eta_t_prev**2)
            - sigma_t * alpha_t_prev / alpha_t
        ) * noise_pred

        noise = torch.randn_like(x_t) * sigma_t_prev * eta_t_prev

        return mean + noise


class ContinuousSolver(Solver):
    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        return self._denoise(x_t, t.as_continuous(1.0), t_prev.as_continuous(1.0))


class EulerODESolver(ContinuousSolver):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        noise_pred = self.model(x_t, timestep=t)
        f = d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * noise_pred
        x_t_prev = x_t + h * f

        return x_t_prev


class HeunODESolver(ContinuousSolver):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        def f(x_t: torch.Tensor, t: Timestep):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            noise_pred = self.model(x_t, timestep=t)
            return d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * noise_pred

        k1 = f(x_t, t)
        k2 = f(x_t + h * k1, t_prev)
        x_t_prev = x_t + h / 2 * (k1 + k2)

        return x_t_prev


class EulerMaruyamaSDESolver(ContinuousSolver):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)
        eta_t_inf = 1.0

        noise_pred = self.model(x_t, timestep=t)
        drift = (
            d_alpha_t / alpha_t * x_t
            + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * noise_pred
        )
        diffusion = sigma_t * eta_t_inf * math.sqrt(-h) * torch.randn_like(x_t)

        x_t_prev = x_t + h * drift + diffusion

        return x_t_prev


class HeunSDESolver(ContinuousSolver):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)
        eta_t_inf = 1.0

        def drift(x_t: torch.Tensor, t: Timestep):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            noise_pred = self.model(x_t, timestep=t)
            return (
                d_alpha_t / alpha_t * x_t
                + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * noise_pred
            )

        def diffusion(t: Timestep):
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            return sigma_t * eta_t_inf * math.sqrt(-h)

        k1 = drift(x_t, t)
        k2 = drift(x_t + h * k1, t_prev)
        dr = (k1 + k2) / 2

        dif = (diffusion(t) + diffusion(t_prev)) / 2 * torch.randn_like(x_t)
        # dif = diffusion(t) * torch.randn_like(x_t)

        x_t_prev = x_t + h * dr + dif

        return x_t_prev
