import math
import os
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_device
from src.config import EquationType
from src.diffusion import diffuse, diffuse_from
from src.model import VAE, PredictionTarget, Predictor
from src.schedule import ScheduleGroup
from src.schedule.sampling import EPSILON, SamplingSchedule
from src.timestep import Timestep, TimestepConfig

STAGE_10 = 10
STAGE_20 = 20
STAGE_40 = 40

MIN_T = 1e-4
MIN_GAP = 1e-5


@dataclass
class AYSConfig:
    max_iter: int = 300
    max_finetune_iter: int = 10
    device: torch.device = get_device()
    n_candidates: int = 11
    n_monte_carlo_iter: int = 1000
    save_interval_iter: int = 10
    importance_sampling: bool = True
    inverse_transform_sampling_grid_size: int = 1000
    save_file: str = "generated/ays_timesteps.pth"


class AYSSamplingSchedule(SamplingSchedule):
    model: Predictor
    schedules: ScheduleGroup
    dataloader: DataLoader
    timestep_config: TimestepConfig
    equation_type: EquationType
    vae: VAE | None = None

    def __init__(
        self,
        *,
        max_t: float = 0.95,
        model: Predictor,
        schedules: ScheduleGroup,
        dataloader: DataLoader,
        solver_T: float,
        equation_type: EquationType,
        vae: VAE | None = None,
        config: AYSConfig = AYSConfig(),
    ):
        super().__init__(max_t=max_t, T=solver_T)
        self.model = model
        self.schedules = schedules
        self.equation_type = equation_type
        self.dataloader = dataloader
        self.vae = vae
        self.config = config

        self.timestep_config = TimestepConfig(kind="continuous", T=solver_T)

    def _save_state(self, steps: torch.Tensor, stage: int, current_iter: int):
        state = {
            "steps": torch.flip(steps.cpu(), dims=[0]),
            "stage": stage,
            "current_iter": current_iter,
        }
        os.makedirs(os.path.dirname(self.config.save_file), exist_ok=True)
        torch.save(state, self.config.save_file)

    def get_timesteps(
        self, n_steps: int = 40, *, initial_t: Timestep | None = None, **kwargs
    ) -> Timestep:
        assert n_steps >= 10

        stage = STAGE_10
        start_iter = 0
        steps = None

        if os.path.exists(self.config.save_file):
            logger.info(
                f"Resuming AYS optimization from checkpoint: {self.config.save_file}"
            )
            state = torch.load(
                self.config.save_file,
                weights_only=False,
            )
            steps = torch.flip(state["steps"].to(self.config.device), dims=[0])
            stage = state["stage"]
            start_iter = state["current_iter"]
            logger.info(f"Resuming Stage: {stage}-step, Iteration: {start_iter}")
        else:
            logger.info("No checkpoint found. Starting fresh AYS optimization.")
            if initial_t is not None:
                steps = initial_t.adapt(self.timestep_config).reverse().steps
                assert len(steps) == 11
            else:
                logger.warning(
                    f"No initial_t provided. Falling back to linear schedule. Remember to set max_t: {self.max_t}"
                )
                steps = torch.linspace(0.0, self.max_t, 11, device=self.config.device)

        assert steps[0] <= steps[-1]

        logger.info(f"Starting with steps: {steps}")

        while True:
            is_first_stage = stage == STAGE_10
            max_iterations = (
                self.config.max_iter
                if is_first_stage
                else self.config.max_finetune_iter
            )

            steps = self._optimize(
                steps,
                max_iter=max_iterations,
                desc=f"AYS {stage}-step",
                stage=stage,
                start_iter=start_iter,
                skip_even=not is_first_stage,
            )

            if (stage >= n_steps and not is_first_stage) or stage == STAGE_40:
                break

            stage *= 2
            start_iter = 0
            steps = self._subdivide(steps)

        return self._interpolate_timesteps(
            Timestep(self.timestep_config, steps), n_steps
        ).reverse()

    def _optimize(
        self,
        steps: torch.Tensor,
        max_iter: int,
        desc: str,
        stage: int,
        start_iter: int = 0,
        skip_even: bool = False,
    ) -> torch.Tensor:
        if start_iter >= max_iter:
            return steps

        pbar_outer = tqdm(total=max_iter, initial=start_iter, desc=desc)

        no_change = False
        current_iter = start_iter

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

                s = steps[i - 1].item()
                t = steps[i].item()
                t_next = steps[i + 1].item()

                candidates = self._get_candidates(s, t, t_next)
                klub_per_candidate = []

                pbar_steps.set_description(f"Iter {current_iter}: Optimizing t_{i}")

                pbar_candidates = tqdm(candidates, desc="Candidates", leave=False)
                for cand in pbar_candidates:
                    pbar_candidates.set_description(
                        f"t_{i} candidate: {cand.item():.4f}"
                    )

                    klub = self._estimate_klub(s, cand.item())
                    klub += self._estimate_klub(cand.item(), t_next)
                    klub_per_candidate.append(klub)

                argmin = int(torch.argmin(torch.tensor(klub_per_candidate)).item())

                if candidates[argmin].item() != t:
                    steps[i] = candidates[argmin]
                    no_change = False

            pbar_outer.update(1)

            assert torch.all(steps[1:] >= steps[:-1])

            if (
                current_iter % self.config.save_interval_iter == 0
                or current_iter == max_iter
                or no_change
            ):
                self._save_state(
                    steps, stage, current_iter if not no_change else max_iter
                )

        pbar_outer.close()
        return steps

    def _interpolate_timesteps(self, timesteps: Timestep, n_steps: int) -> Timestep:
        steps = timesteps.adapt(self.timestep_config).steps

        if len(steps) == n_steps + 1:
            return Timestep(self.timestep_config, steps)

        xs = torch.linspace(0, self.T, len(steps)).cpu().detach().numpy()
        ys = torch.log(steps).cpu().detach().numpy()

        new_xs = torch.linspace(0, self.T, n_steps + 1).cpu().detach().numpy()
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

            if t_start < MIN_T:
                t_mid = (t_start + t_end) / 2.0
            else:
                # t_mid = math.exp(0.5 * (math.log(t_start) + math.log(t_end)))
                t_mid = math.sqrt(t_start * t_end)

            new_steps.append(t_mid)

        new_steps.append(steps[-1].item())
        return torch.tensor(new_steps, device=steps.device)

    def _get_candidates(self, s: float, t: float, t_next: float) -> torch.Tensor:
        if t_next - s <= 2 * MIN_GAP:
            return torch.tensor([t], device=self.config.device)

        lower_bound = s + MIN_GAP
        upper_bound = t_next - MIN_GAP

        candidates = torch.linspace(
            lower_bound, upper_bound, self.config.n_candidates - 1
        )
        candidates = torch.cat([torch.tensor([t]), candidates])
        candidates = torch.clamp(candidates, min=lower_bound, max=upper_bound)

        return candidates.sort().values

    def _estimate_klub(self, t_start: float, t_end: float) -> float:
        klub_sum = 0.0
        sample_count = 0

        t_start = max(t_start, MIN_T)
        t_end = max(t_end, t_start + MIN_GAP)

        for X in self._get_data_samples():
            t_samples = (
                self._importance_sample(X.size(0), t_start, t_end)
                if self.config.importance_sampling
                else (
                    torch.rand(X.size(0), device=X.device) * (t_end - t_start) + t_start
                )
            )
            timestep_samples = Timestep(self.timestep_config, t_samples)
            timestep_end = Timestep(
                self.timestep_config, torch.tensor([t_end], device=X.device)
            )

            # (batch, channels, height, width)
            X_t, _ = diffuse(
                X,
                timestep_samples,
                self.schedules,
            )

            X_t_end = diffuse_from(
                X,
                X_t,
                timestep_samples,
                timestep_end,
                self.schedules,
            )

            pred_t = self.model(
                X_t, timestep=timestep_samples, schedules=self.schedules
            )
            pred_t_end = self.model(
                X_t_end,
                timestep=timestep_end,
                schedules=self.schedules,
            )

            pred_diff_norm = (pred_t_end - pred_t).view(X.size(0), -1).norm(dim=1) ** 2

            factor = (
                torch.tensor((t_end - t_start), device=self.config.device)
                if not self.config.importance_sampling
                else self.schedules.edm_sigma(timestep_samples) ** 3
                / (1 / (t_start**2 + 0.5**2) - 1 / (t_end**2 + 0.5**2))
            )

            assert torch.all(factor > 0)  # ty: ignore

            if self.equation_type in [
                EquationType.song_sde,
                EquationType.probability_flow,
            ]:
                assert self.model.target == PredictionTarget.x0

                integral = (
                    self.schedules.s(timestep_samples)
                    * self.schedules.edm_sigma.derivative(timestep_samples)
                    * (1 / self.schedules.edm_sigma(timestep_samples) ** 3)
                    * pred_diff_norm
                    * factor
                )
            elif self.equation_type == EquationType.generalized_differential:
                assert self.model.target == PredictionTarget.Noise

                integral = (
                    -0.5
                    * (
                        self.schedules.lambda_.derivative(timestep_samples)
                        * pred_diff_norm
                    )
                    * factor
                )
            else:
                raise ValueError(f"Unsupported equation type: {self.equation_type}")

            klub_sum += integral.sum().item()
            sample_count += X.size(0)

        return klub_sum / sample_count

    # Inverse Transform Sampling
    def _importance_sample(self, n: int, t_start: float, t_end: float) -> torch.Tensor:
        t_start = max(t_start, MIN_T)
        t_end = max(t_end, t_start + MIN_GAP)

        c = 0.5
        t_grid = torch.linspace(
            t_start,
            t_end,
            self.config.inverse_transform_sampling_grid_size,
            device=self.config.device,
        )

        pi_t = (1.0 / t_grid**3) * (1.0 / (t_grid**2 + c**2) - 1.0 / (t_end**2 + c**2))
        print(t_start, t_end, t_grid.min(), t_grid.max(), pi_t.min(), pi_t.max())
        assert torch.all(pi_t >= 0)

        pi_t = torch.clamp(pi_t, min=EPSILON)

        cdf = torch.cumsum(pi_t, dim=0)
        cdf = cdf / cdf[-1]

        u = torch.rand(n, device=self.config.device)

        indices = torch.searchsorted(cdf, u)
        indices = torch.clamp(indices, 0, len(t_grid) - 1)

        return t_grid[indices]

    def _get_data_samples(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        while count < self.config.n_monte_carlo_iter:
            for batch_idx, (X, _) in enumerate(self.dataloader):
                X = X.to(self.config.device)
                yield self.vae.encode(X.half()).float() if self.vae is not None else X
                count += X.size(0)
                if count >= self.config.n_monte_carlo_iter:
                    break
