from dataclasses import dataclass

import torch
from loguru import logger
from PIL.GifImagePlugin import TYPE_CHECKING

from src.solver import Solver
from src.timestep import Timestep, TimestepConfig


@dataclass
class Generator:
    solver: Solver

    n_channels: int
    img_width: int
    img_height: int

    # Assumes timesteps are provided in decreasing order.
    # In case of continuous timesteps, max_t should be set to values
    # close to T e.g. 0.99 when T=1 to deal with exploding values.
    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        *,
        max_t: float | int | None = None,
        n_steps: int | None = None,
        timesteps: Timestep | None = None,
        skip_last_step: bool = False,
    ) -> torch.Tensor:
        if n_steps is None and timesteps is None:
            n_steps = int(self.solver.model.timestep_config.T) - 1
            logger.info(
                f"Neither n_steps nor timesteps were provided. Using max number of steps according to model config: {n_steps}."
                " For continuous generation, provide max_t as well."
            )

        device = next(self.solver.model.parameters()).device
        x_t = torch.randn(
            n_samples, self.n_channels, self.img_width, self.img_height, device=device
        )

        if TYPE_CHECKING:
            assert n_steps is not None

        if timesteps is None:
            max_t = max_t or self.solver.model.timestep_config.T
            timesteps_tensor = torch.linspace(
                max_t,
                0.0,
                n_steps + 1 + (1 if skip_last_step else 0),
                device=device,
            )

            if skip_last_step:
                timesteps_tensor = timesteps_tensor[1:]

            timesteps = Timestep(
                TimestepConfig(
                    kind="continuous", T=self.solver.model.timestep_config.T
                ),
                timesteps_tensor,
            )

        logger.info(f"Using timesteps: {timesteps.steps.cpu().numpy()}")

        for i in range(len(timesteps) - 1):
            x_t = self.solver.denoise(x_t, timesteps[i], timesteps[i + 1])

        return x_t
