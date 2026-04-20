from typing import Optional

import typer
from loguru import logger

from src.common import get_dataloader
from src.config import (
    DatasetType,
    EquationType,
    EtaType,
    ModelType,
    SamplingScheduleType,
    ScheduleType,
)
from src.config.presets import (
    DATASET_CONFIGS,
    ETA_CONFIGS,
    SCHEDULE_CONFIGS,
    get_solver_T,
)
from src.model import VAE, Predictor, PredictorHuggingface, PredictorMetadata
from src.schedule import ScheduleGroup
from src.schedule.sampling import EDMSamplingSchedule, LinearSamplingSchedule
from src.schedule.sampling.ays import (
    AYSConfig,
    AYSSamplingSchedule,
)

app = typer.Typer(help="Find steps with AYS")


@app.callback(invoke_without_command=True)
def ays(
    model_name: ModelType = typer.Option(
        ModelType.edm, "--model", help="Network architecture"
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", help="Path to the local pre-trained weights"
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID (required if model=huggingface)",
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.flowers, "--dataset", help="Target dataset for training"
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm, "--schedule", help="Signal/Noise schedule"
    ),
    eta: EtaType = typer.Option(EtaType.ddpm, "--eta", help="Stochasticity"),
    equation_name: EquationType = typer.Option(
        EquationType.probability_flow, "--equation", help="Reverse diffusion equation"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    initial_sampling_schedule: SamplingScheduleType = typer.Option(
        SamplingScheduleType.edm,
        "--initial-schedule",
        help="Initial time sampling",
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None, "--checkpoint-path", help="Path to resume AYS tuning from a checkpoint"
    ),
    suffix: str | None = typer.Option(None, "--suffix", help="Suffix for checkpoint"),
    max_iter: int = typer.Option(
        300, "--max-iter", help="Max iterations for 10-step stage"
    ),
    max_finetune_iter: int = typer.Option(
        10, "--max-finetune-iter", help="Max fine tune iterations for 20/40-step stage"
    ),
    n_candidates: int = typer.Option(
        11, "--n-candidates", help="Number of timestep candidates"
    ),
    n_monte_carlo_iter: int = typer.Option(
        1000, "--n-mc-iter", help="Number of Monte Carlo steps"
    ),
    save_interval_iter: int = typer.Option(
        1, "--save-interval", help="Checkpoint frequency"
    ),
    importance_sampling: bool = typer.Option(
        True,
        "--importance-sampling/--no-importance-sampling",
        help="Use importance sampling for AYS",
    ),
    its_grid_size: int = typer.Option(
        1000, "--its-grid-size", help="Grid size for Inverse Transform Sampling"
    ),
    vae_low_memory: bool = typer.Option(
        False, "--vae-low-memory", help="Enable low memory mode for VAE"
    ),
    n_workers: int = typer.Option(
        1, "--n-workers", help="Number of dataloader workers"
    ),
):
    logger.info("Starting the AYS Generation")
    logger.info(f"Model Architecture: {model_name.value}")
    logger.info(f"Signal/Noise Schedule: {schedule.value}")
    logger.info(f"Equation: {equation_name.value}")
    logger.info(f"Initial Sampling Schedule: {initial_sampling_schedule.value}")

    dataset_config = DATASET_CONFIGS[dataset]

    if model_name == ModelType.huggingface:
        if not model_id:
            logger.error(
                "You must provide '--model-id' when using the huggingface model."
            )
            raise typer.Exit(code=1)
        logger.info(f"Loading HuggingFace model: {model_id}")
        model = PredictorHuggingface(model_id=model_id).cuda()
    else:
        if not model_path:
            logger.error(
                "You must provide '--model-path' when using a local model type."
            )
            raise typer.Exit(code=1)
        logger.info(f"Loading local model from: {model_path}")
        model = Predictor.load_from_file(model_path).cuda()
        model.load()

    model.eval()
    logger.info(
        f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    schedule_kwargs = {}
    if model_id:
        schedule_kwargs["model_id"] = model_id

    alpha_schedule = SCHEDULE_CONFIGS[schedule].alpha_schedule_factory(
        **schedule_kwargs
    )
    sigma_schedule = SCHEDULE_CONFIGS[schedule].sigma_schedule_factory(
        **schedule_kwargs
    )
    eta_schedule = ETA_CONFIGS[eta](alpha_schedule, sigma_schedule)

    schedules = ScheduleGroup(
        alpha_schedule=alpha_schedule,
        sigma_schedule=sigma_schedule,
        eta_schedule=eta_schedule,
    )

    predictor_T = model.timestep_config.T
    solver_T = get_solver_T(equation_name, schedule, predictor_T)

    vae_model_id = model.metadata.get(PredictorMetadata.VAE, None)
    vae = None
    if vae_model_id:
        logger.info(f"Using VAE: {vae_model_id}")
        vae = VAE(model_id=vae_model_id, low_memory=vae_low_memory).cuda()

    if initial_sampling_schedule == SamplingScheduleType.edm:
        initial_t = EDMSamplingSchedule(T=solver_T).get_timesteps(n_steps=10).cuda()
    elif initial_sampling_schedule == SamplingScheduleType.linear:
        initial_t = LinearSamplingSchedule(T=solver_T).get_timesteps(n_steps=10).cuda()
    else:
        raise ValueError(
            f"Unsupported initial schedule: {initial_sampling_schedule.value}"
        )

    save_file_path = (
        checkpoint_path
        if checkpoint_path
        else (
            f"generated/ays_timesteps_{suffix}.pth"
            if suffix
            else "generated/ays_timesteps.pth"
        )
    )
    ays_config = AYSConfig(
        max_iter=max_iter,
        max_finetune_iter=max_finetune_iter,
        n_candidates=n_candidates,
        n_monte_carlo_iter=n_monte_carlo_iter,
        save_interval_iter=save_interval_iter,
        importance_sampling=importance_sampling,
        inverse_transform_sampling_grid_size=its_grid_size,
        save_file=save_file_path,
    )

    ays_schedule = AYSSamplingSchedule(
        model=model,
        schedules=schedules,
        solver_T=solver_T,
        equation_type=equation_name,
        vae=vae,
        dataloader=get_dataloader(
            batch_size=batch_size,
            dataset_config=dataset_config,
            num_workers=n_workers,
        ),
        config=ays_config,
    )

    logger.info("Starting AYS tuning")

    timesteps = ays_schedule.get_timesteps(n_steps=40, initial_t=initial_t)

    logger.info(f"Final AYS schedule: {timesteps.steps}")
