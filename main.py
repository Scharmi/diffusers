import os
import sys
from datetime import datetime

import torch
import torchvision
import wandb
from loguru import logger

from src import distributed
from src.common import get_dataloader, unnormalize
from src.config import (
    BATCH_SIZE,
    DATASET_CONFIG_NAME,
    EQUATION_CONFIG_NAME,
    ETA_CONFIG_NAME,
    PREDICTOR_T,
    SCHEDULE_CONFIG_NAME,
    SOLVER_CONFIG_NAME,
    SOLVER_T,
    dataset_config,
    equation_config,
    eta_config,
    schedule_config,
    solver_config,
    timesampler_config,
)
from src.distributed import get_rank, is_distributed
from src.equation import Equation
from src.generator import Generator
from src.model import (  # noqa
    Predictor,
    PredictorEDM,
    PredictorEDM2,
    PredictorHuggingface,
    PredictorUNet,
)
from src.schedule import ScheduleGroup
from src.schedule.sampling import (  # noqa
    AYSConfig,
    AYSSamplingSchedule,
    EDMSamplingSchedule,
    LinearSamplingSchedule,
)
from src.solver import Solver
from src.trainer import Trainer, TrainingConfig

# model_id = "1aurent/ddpm-mnist"
# model_id = "google/ddpm-cifar10-32"
model_id = "google/ddpm-celebahq-256"

schedule_kwargs = {}
if SCHEDULE_CONFIG_NAME == "ddpm":
    schedule_kwargs["model_id"] = model_id

alpha_schedule = schedule_config["alpha_schedule"](**schedule_kwargs)  # ty: ignore
sigma_schedule = schedule_config["sigma_schedule"](**schedule_kwargs)  # ty: ignore

eta_schedule = eta_config(alpha_schedule, sigma_schedule)  # ty: ignore

schedules = ScheduleGroup(
    alpha_schedule=alpha_schedule,  # ty: ignore
    sigma_schedule=sigma_schedule,  # ty: ignore
    eta_schedule=eta_schedule,  # ty: ignore
)


def train():
    logger.info("Starting training")
    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    logger.info(f"Using predictor T: {PREDICTOR_T}")
    logger.info(f"Using timesampler config: {timesampler_config}")
    logger.info(f"Using dataset config: {DATASET_CONFIG_NAME}")

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if get_rank() == 0 and is_distributed():
        wandb.init(
            project="prosem",
            name=f"EDM_{DATASET_CONFIG_NAME}_{date}",
            config={
                "batch_size": BATCH_SIZE,
                "predictor_t": PREDICTOR_T,
                "dataset": DATASET_CONFIG_NAME,
                "time_sampler": timesampler_config.value,
            },
        )

    model = PredictorEDM(
        T=PREDICTOR_T,
        suffix=f"_{DATASET_CONFIG_NAME}",
        n_channels=dataset_config["channels"],  # ty: ignore
        img_width=dataset_config["img_width"],  # ty: ignore
        img_height=dataset_config["img_height"],  # ty: ignore
    ).cuda()
    model.try_load()

    trainer = Trainer(
        model=model,
        schedules=schedules,
        config=TrainingConfig(time_sampler=timesampler_config),
    )
    trainer.load_checkpoint()

    dataloader = get_dataloader(
        batch_size=BATCH_SIZE,
        dataset_class=dataset_config["class"],  # ty: ignore
        width=dataset_config["img_width"],  # ty: ignore
        height=dataset_config["img_height"],  # ty: ignore
    )
    trainer.train(dataloader)

    logger.success("Training complete")

    if get_rank() == 0 and is_distributed():
        wandb.finish()


def generate():
    if not os.path.exists(f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}"):
        os.makedirs(f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}")

    # model = NoisePredictorHuggingface(model_id=model_id).cuda()
    model = Predictor.load_from_file("./models/x0_predictor_edm_cifar10.pth").cuda()
    print(sum(p.numel() for p in model.parameters()))

    model.load()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    logger.info(f"Using eta config: {ETA_CONFIG_NAME}")
    logger.info(f"Using equation config: {EQUATION_CONFIG_NAME}")
    logger.info(f"Using solver config: {SOLVER_CONFIG_NAME}")
    logger.info(f"Using solver T: {SOLVER_T}")

    equation: Equation = equation_config(
        model=model,
        schedules=schedules,
    )

    solver: Solver = solver_config(
        equation=equation,
        T=SOLVER_T,
    )

    generator = Generator(
        solver=solver,
        n_channels=dataset_config["channels"],  # ty: ignore
        img_width=dataset_config["img_width"],  # ty: ignore
        img_height=dataset_config["img_height"],  # ty: ignore
    )

    # timesteps = cast(
    #     Timestep,
    #     torch.load(
    #         f"generated/ays_timesteps_{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}.pt",
    #         weights_only=False,
    #     ),
    # )
    # timesteps.steps = timesteps.steps.cuda()
    # timesteps = timesteps.reverse()

    timesteps = EDMSamplingSchedule(T=PREDICTOR_T).get_timesteps(n_steps=100).cuda()

    n_samples = 30
    generated = generator.generate(
        n_samples=n_samples,
        timesteps=timesteps,
        variance_exploding=(SCHEDULE_CONFIG_NAME == "edm"),
    )

    for i in range(n_samples):
        img = unnormalize(generated[i])
        torchvision.utils.save_image(
            img, f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}/{i + 1}.png"
        )


def ays():
    model = PredictorUNet(
        T=PREDICTOR_T,
        suffix=f"_{DATASET_CONFIG_NAME}",
        n_channels=dataset_config["channels"],  # ty: ignore
        img_width=dataset_config["img_width"],  # ty: ignore
        img_height=dataset_config["img_height"],  # ty: ignore
    ).cuda()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    logger.info(f"Using denoiser config: {SOLVER_CONFIG_NAME}")
    logger.info(f"Using equation config: {EQUATION_CONFIG_NAME}")

    equation = equation_config(
        model=model,
        schedules=schedules,
    )

    solver: Solver = solver_config(
        equation=equation,
        T=SOLVER_T,
    )

    ays_schedule = AYSSamplingSchedule(
        denoiser=solver,
        dataloader=get_dataloader(
            batch_size=BATCH_SIZE,
            dataset_class=dataset_config["class"],  # ty: ignore
            width=dataset_config["img_width"],  # ty: ignore
            height=dataset_config["img_height"],  # ty: ignore
        ),
        config=AYSConfig(
            max_iter=1,
            max_finetune_iter=1,
            save_file=f"generated/ays_timesteps_{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}.pt",
        ),
    )

    try:
        initial_t = torch.load(
            f"generated/ays_timesteps_{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}.pt_10",
            weights_only=False,
        )
    except FileNotFoundError:
        initial_t = None

    logger.info(f"Starting AYS tuning with initial_t: {initial_t}")
    timesteps = ays_schedule.get_20_timesteps(initial_t)  # ty: ignore
    logger.info(f"Final timesteps: {timesteps.steps}")


if __name__ == "__main__":
    mode = "train"

    if len(sys.argv) == 0:
        raise ValueError("Please provide a mode: train, test, ays")

    mode = sys.argv[1]

    if distributed.is_distributed():
        distributed.setup()

    match mode:
        case "train":
            train()
        case "generate":
            generate()
        case "ays":
            ays()
        case _:
            logger.error(f"Unknown mode: {mode}")

    if distributed.is_distributed():
        torch.distributed.barrier()
        distributed.cleanup()
