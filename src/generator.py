from dataclasses import dataclass

import torch
from loguru import logger
from PIL.GifImagePlugin import TYPE_CHECKING

from src.model import VAE
from src.solver import Solver
from src.timestep import Timestep, TimestepConfig


@dataclass
class Generator:
    solver: Solver
    vae: VAE | None = None

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
        variance_exploding: bool = False,
    ) -> torch.Tensor:
        if n_steps is None and timesteps is None:
            n_steps = int(self.solver.equation.model.timestep_config.T) - 1
            logger.info(
                f"Neither n_steps nor timesteps were provided. Using max number of steps according to model config: {n_steps}."
                " For continuous generation, provide max_t as well."
            )

        device = next(self.solver.equation.model.parameters()).device

        if TYPE_CHECKING:
            assert n_steps is not None

        if timesteps is None:
            max_t = max_t or self.solver.equation.model.timestep_config.T
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
                    kind="continuous", T=self.solver.equation.model.timestep_config.T
                ),
                timesteps_tensor,
            )

        logger.info(f"Using timesteps: {timesteps.steps.cpu().numpy()}")

        n_channels = self.solver.equation.model.n_channels
        img_width = self.solver.equation.model.img_width
        img_height = self.solver.equation.model.img_height

        x_t = torch.randn(n_samples, n_channels, img_width, img_height, device=device)

        if variance_exploding:
            logger.info("Using variance exploding noise for generation.")
            t_start = timesteps[0]
            sigma_start = self.solver.equation.schedules.edm_sigma(t_start).view(
                -1, 1, 1, 1
            )
            x_t = x_t * sigma_start

        for i in range(len(timesteps) - 1):
            x_t = self.solver.step(x_t, timesteps[i], timesteps[i + 1])

        if self.vae is not None:
            x_t = self.vae.decode(x_t.half()).float()

        return x_t
