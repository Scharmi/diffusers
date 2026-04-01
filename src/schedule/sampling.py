import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_device
from src.diffusion import DiffusionMixin
from src.solver import Solver
from src.timestep import Timestep, TimestepConfig

EPSILON = 1e-5


class SamplingSchedule(ABC):
    max_t: float

    # If values are close to 1.0 it generates noisy images.
    def __init__(self, *, max_t: float = 0.95):
        self.max_t = max_t

    @abstractmethod
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        pass


class LinearSamplingSchedule(SamplingSchedule):
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        steps = torch.linspace(self.max_t, 0.0, n_steps + 1)
        return Timestep(TimestepConfig(kind="continuous", T=1.0), steps)


@dataclass
class AYSConfig:
    max_iter: int = 300
    max_finetune_iter: int = 10
    device: torch.device = get_device()
    n_candidates: int = 11
    n_monte_carlo_iter: int = 1000
    save_interval_iter: int = 1
    save_file: str = "generated/ays_timesteps.pt"


class AYSSamplingSchedule(SamplingSchedule, DiffusionMixin):
    denoiser: Solver
    dataloader: DataLoader
    timestep_config: TimestepConfig

    def __init__(
        self,
        *,
        max_t: float = 0.95,
        denoiser: Solver,
        dataloader: DataLoader,
        config: AYSConfig = AYSConfig(),
    ):
        super().__init__(max_t=max_t)
        self.denoiser = denoiser
        self.dataloader = dataloader
        self.config = config

        self.timestep_config = TimestepConfig(kind="continuous", T=1.0)

    def get_timesteps(
        self, n_steps: int = 10, *, initial_t: Timestep | None = None, **kwargs
    ) -> Timestep:
        if initial_t is not None:
            t = initial_t.as_continuous(1.0).steps
        else:
            t = torch.linspace(0.0, self.max_t, n_steps + 1)

        t = self._get_10_timesteps(t)

        if n_steps <= 10:
            return self.interpolate_timesteps(t, n_steps)

        t = self.get_20_timesteps(t)

        if n_steps <= 20:
            return self.interpolate_timesteps(t, n_steps)

        t = self.get_40_timesteps(t)

        return self.interpolate_timesteps(t, n_steps)

    def _get_10_timesteps(self, initial_steps: torch.Tensor) -> Timestep:
        steps = self._optimize(
            initial_steps,
            max_iter=self.config.max_iter,
            desc="AYS 10-step",
            suffix="_10",
        )

        return Timestep(self.timestep_config, steps)

    def get_20_timesteps(self, steps_10: Timestep) -> Timestep:  # ty: ignore
        assert len(steps_10) == 11
        steps = steps_10.as_continuous(1.0).steps
        steps = self._subdivide(steps)

        steps = self._optimize(
            steps,
            max_iter=self.config.max_finetune_iter,
            desc="AYS 20-step",
            skip_even=True,
            suffix="_20",
        )

        return Timestep(self.timestep_config, steps)

    def get_40_timesteps(self, steps_20: Timestep) -> Timestep:  # ty: ignore
        assert len(steps_20) == 21
        steps = steps_20.as_continuous(1.0).steps
        steps = self._subdivide(steps)

        steps = self._optimize(
            steps,
            max_iter=self.config.max_finetune_iter,
            desc="AYS 40-step",
            skip_even=True,
            suffix="_40",
        )

        return Timestep(self.timestep_config, steps)

    def _optimize(
        self,
        steps: torch.Tensor,
        max_iter: int,
        desc: str,
        *,
        skip_even: bool = False,
        suffix: str = "",
    ) -> torch.Tensor:
        pbar_outer = tqdm(range(max_iter), desc=desc)

        no_change = False
        current_iter = 0

        while not no_change and current_iter < max_iter:
            no_change = True
            current_iter += 1

            t_indices = range(1, len(steps) - 1)
            pbar_steps = tqdm(
                t_indices, desc=f"Iter {current_iter}: Steps", leave=False
            )

            for i in pbar_steps:
                if skip_even and i % 2 == 0:
                    continue

                t_prev = steps[i - 1].item()
                t = steps[i].item()
                t_next = steps[i + 1].item()

                candidates = self._get_candidates(t_prev, t, t_next)
                klub_per_candidate = []

                pbar_steps.set_description(f"Iter {current_iter}: Optimizing t_{i}")

                pbar_candidates = tqdm(candidates, desc="Candidates", leave=False)
                for cand in pbar_candidates:
                    pbar_candidates.set_description(
                        f"t_{i} candidate: {cand.item():.4f}"
                    )

                    klub = self._estimate_klub(t_prev, cand.item())
                    klub += self._estimate_klub(cand.item(), t_next)
                    klub_per_candidate.append(klub)

                argmin = int(torch.argmin(torch.tensor(klub_per_candidate)).item())

                if candidates[argmin].item() != t:
                    steps[i] = candidates[argmin]
                    no_change = False

            pbar_outer.update(1)

            if current_iter % self.config.save_interval_iter == 0:
                torch.save(
                    Timestep(self.timestep_config, steps),
                    f"{self.config.save_file}{suffix}",
                )

        pbar_outer.close()

        return steps

    def interpolate_timesteps(self, timesteps: Timestep, n_steps: int) -> Timestep:
        steps = timesteps.as_continuous(1.0).steps

        if len(steps) == n_steps + 1:
            return Timestep(self.timestep_config, steps)

        xs = torch.linspace(0, 1, len(steps)).cpu().detach().numpy()
        ys = torch.log(steps).cpu().detach().numpy()

        new_xs = torch.linspace(0, 1, n_steps + 1).cpu().detach().numpy()
        new_ys = np.interp(new_xs, xs, ys)
        new_steps = torch.exp(torch.tensor(new_ys, device=steps.device))

        return Timestep(self.timestep_config, new_steps)

    def _subdivide(self, steps: torch.Tensor) -> torch.Tensor:
        new_steps = []

        # log t_{2n+1} = 0.5 * (log t_{n} + log t_{n+1})

        for i in range(len(steps) - 1):
            t_start = steps[i].item()
            t_end = steps[i + 1].item()

            new_steps.append(t_start)

            if t_start < EPSILON:
                t_mid = (t_start + t_end) / 2.0
            else:
                # t_mid = math.exp(0.5 * (math.log(t_start) + math.log(t_end)))
                t_mid = math.sqrt(t_start * t_end)

            new_steps.append(t_mid)

        new_steps.append(steps[-1].item())

        return torch.tensor(new_steps, device=steps.device)

    def _get_candidates(self, t_prev: float, t: float, t_next: float) -> torch.Tensor:
        candidates = torch.linspace(
            t_prev + EPSILON, t_next - EPSILON, self.config.n_candidates - 1
        )
        candidates = torch.cat([torch.tensor([t]), candidates])

        return candidates.sort().values

    def _estimate_klub(self, t_start: float, t_end: float) -> float:
        klub_sum = 0.0
        sample_count = 0

        for X in self._get_data_samples():
            # TODO: Add importance sampling
            t_samples = (
                torch.rand(X.size(0), device=X.device) * (t_end - t_start) + t_start
            )
            timestep_samples = Timestep(self.timestep_config, t_samples)
            timestep_end = Timestep(
                self.timestep_config, torch.tensor([t_end], device=X.device)
            )

            # (batch, channels, height, width)
            X_t, _ = self.diffuse(
                X,
                timestep_samples,
                self.denoiser.schedules,
            )

            X_t_end = self.diffuse_from(
                X,
                X_t,
                timestep_samples,
                timestep_end,
                self.denoiser.schedules,
            )

            noise_t = self.denoiser.model(X_t, timestep=timestep_samples)
            noise_t_end = self.denoiser.model(
                X_t_end,
                timestep=timestep_end,
            )

            noise_diff_norm = (noise_t_end - noise_t).view(X.size(0), -1).norm(
                dim=1
            ) ** 2

            likelihood = 1 / (t_end - t_start)
            integral = (
                -0.5
                * (
                    self.denoiser.schedules.lambda_.derivative(timestep_samples)
                    * noise_diff_norm
                )
                / likelihood
            )

            klub_sum += integral.sum().item()
            sample_count += X.size(0)

        return klub_sum / sample_count

    def _get_data_samples(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        while count < self.config.n_monte_carlo_iter:
            for batch_idx, (X, _) in enumerate(self.dataloader):
                X = X.to(self.config.device)
                yield X
                count += X.size(0)
                if count >= self.config.n_monte_carlo_iter:
                    break
