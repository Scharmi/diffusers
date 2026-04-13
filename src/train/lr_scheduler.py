from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class WarmupCosineLR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps_percentage: float = 0.05,
        warmup_start_factor: float = 1e-5,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        warmup_steps = int(total_steps * warmup_steps_percentage)
        assert warmup_steps < total_steps, "Warmup steps must be less than total steps"

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=eta_min,
        )

        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            last_epoch=-1,
        )

        if last_epoch > 0:
            for _ in range(last_epoch):
                self.step()
