import typer
import wandb
from loguru import logger

from src import distributed
from src.common import get_dataloader
from src.config import (
    DatasetType,
    EtaType,
    ModelType,
    ScheduleType,
)
from src.config.presets import (
    DATASET_CONFIGS,
    ETA_CONFIGS,
    MODEL_CONFIGS,
    SCHEDULE_CONFIGS,
    get_timesampler,
)
from src.model.presets import ModelSize
from src.schedule import ScheduleGroup
from src.train import Trainer, TrainingConfig

app = typer.Typer(help="Train model")


@app.callback(invoke_without_command=True)
def train(
    run_id: str = typer.Option(None, "--run-id", help="Run id suffix for Wandb"),
    model_name: ModelType = typer.Option(
        ModelType.edm, "--model", help="Network architecture"
    ),
    model_id: str = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID (required if schedule=ddpm)",
    ),
    model_size: ModelSize = typer.Option(
        ModelSize.SMALL, "--model-size", help="Model size (number of parameters)"
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.flowers, "--dataset", help="Target dataset for training"
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm, "--schedule", help="Signal/Noise schedule"
    ),
    eta: EtaType = typer.Option(
        EtaType.ddpm,
        "--eta",
        help="Stochasticity",
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    predictor_t: int = typer.Option(
        1000, "--predictor-t", help="Max time step predictor is trained on"
    ),
    n_epochs: int = typer.Option(1000, "--n-epochs", help="Number of epochs"),
    checkpoint_interval: int = typer.Option(
        100, "--checkpoint-interval", help="save a checkpoint every N steps"
    ),
):
    dataset_config = DATASET_CONFIGS[dataset]
    model_class = MODEL_CONFIGS[model_name]
    timesampler_config = get_timesampler(schedule)

    logger.info("Starting the training")
    logger.info(f"Model architecture: {model_name.value}")
    logger.info(f"Model size: {model_size.value}")
    logger.info(f"Dataset: {dataset.value}")
    logger.info(f"Signal/Noise schedule: {schedule.value}")
    logger.info(f"Stochasticity (eta): {eta.value}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Predictor T: {predictor_t}")
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Timesampler: {timesampler_config.value}")

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

    if distributed.get_rank() == 0 and distributed.is_distributed():
        run_identifier = f"{model_name.value}_{schedule.value}_{dataset.value}"
        if run_id:
            run_identifier += f"_{run_id}"

        wandb.init(
            project="prosem",
            name=run_identifier,
            id=run_identifier,
            resume="allow",
            config={
                "model": model_name.value,
                "batch_size": batch_size,
                "predictor_T": predictor_t,
                "dataset": dataset.value,
                "schedule": schedule.value,
                "eta": eta.value,
            },
        )

    model = model_class(
        T=predictor_t,
        model_size=model_size,
        n_channels=dataset_config.channels,
        img_width=dataset_config.img_size,
        img_height=dataset_config.img_size,
        suffix=f"_{dataset.value}",
    ).cuda()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.try_load()

    trainer = Trainer(
        model=model,
        schedules=schedules,
        config=TrainingConfig(
            time_sampler=timesampler_config,
            epochs=n_epochs,
            checkpoint_interval_steps=checkpoint_interval,
        ),
    )
    trainer.load_checkpoint()

    dataloader = get_dataloader(
        batch_size=batch_size,
        dataset_config=dataset_config,
    )

    trainer.train(dataloader)

    logger.success("Training completed successfully")

    if distributed.get_rank() == 0 and distributed.is_distributed():
        wandb.finish()
