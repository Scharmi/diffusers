import os
import sys

import torch
import torch.distributed as dist
import torchvision
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from src import distributed
from src.distributed import RANK, WORLD_SIZE
from src.generator import Generator
from src.model import (  # noqa
    NoisePredictorHuggingface,
    Predictor,
    PredictorUNet,
)
from src.schedule import (
    ConstantEtaSchedule,
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    DDPMEtaSchedule,
    HuggingFaceDDPMAlphaSchedule,
    HuggingFaceDDPMSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
    ScheduleGroup,
)
from src.schedule.sampling import AYSConfig, AYSSamplingSchedule, LinearSamplingSchedule
from src.solver import (
    DiscreteSolver,
    EulerMaruyamaSDESolver,
    EulerODESolver,
    HeunODESolver,
    HeunSDESolver,
    Solver,
)
from src.trainer import Trainer

BATCH_SIZE = 512
T = 1000

SOLVER_CONFIGS = {
    "discrete": {
        "denoiser": DiscreteSolver,
    },
    "euler": {
        "denoiser": EulerODESolver,
    },
    "heun": {
        "denoiser": HeunODESolver,
    },
    "euler_maruyama": {
        "denoiser": EulerMaruyamaSDESolver,
    },
    "heun_sde": {
        "denoiser": HeunSDESolver,
    },
}
SCHEDULE_CONFIGS = {
    "linear": {
        "alpha_schedule": LinearAlphaSchedule,
        "sigma_schedule": LinearSigmaSchedule,
        "eta_schedule": lambda: ConstantEtaSchedule(eta_value=1.0),
    },
    "cosine": {
        "alpha_schedule": CosineAlphaSchedule,
        "sigma_schedule": CosineSigmaSchedule,
        "eta_schedule": lambda: ConstantEtaSchedule(eta_value=1.0),
    },
    "hf_ddpm": {
        "alpha_schedule": HuggingFaceDDPMAlphaSchedule,
        "sigma_schedule": HuggingFaceDDPMSigmaSchedule,
        "eta_schedule": DDPMEtaSchedule,
    },
}
DATASET_CONFIGS = {
    "mnist": {
        "class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "fashion": {
        "class": datasets.FashionMNIST,
        "mean": (0.5,),
        "std": (0.5,),
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "channels": 3,
        "img_width": 32,
        "img_height": 32,
    },
    "celeb": {
        "class": datasets.CelebA,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "channels": 3,
        "img_width": 256,
        "img_height": 256,
    },
}

SOLVER_CONFIG_NAME = "discrete"
solver_config = SOLVER_CONFIGS[SOLVER_CONFIG_NAME]

SCHEDULE_CONFIG_NAME = "hf_ddpm"
schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

DATASET_CONFIG_NAME = "celeb"
dataset_config = DATASET_CONFIGS[DATASET_CONFIG_NAME]


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    config = DATASET_CONFIGS[DATASET_CONFIG_NAME]
    transform = transforms.Normalize(
        (-torch.tensor(config["mean"]) / torch.tensor(config["std"])),
        (1.0 / torch.tensor(config["std"])),
    )
    return transform(img)


def get_dataloader(batch_size=BATCH_SIZE, train=True, shuffle=True, num_workers=2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config["mean"], dataset_config["std"]),
        ]
    )

    dataset = dataset_config["class"](root="./data", download=True, transform=transform)  # ty: ignore

    sampler = None
    if distributed.is_distributed():
        sampler = DistributedSampler(
            dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=shuffle
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
    )

    return loader


def train():
    logger.info("Starting training")

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    schedules = ScheduleGroup(
        alpha_schedule=schedule_config["alpha_schedule"](),  # ty: ignore
        sigma_schedule=schedule_config["sigma_schedule"](),  # ty: ignore
        eta_schedule=schedule_config["eta_schedule"](),  # ty: ignore
    )

    logger.info(f"Using T: {T}")
    model = PredictorUNet(
        T=T,
        suffix=f"_{DATASET_CONFIG_NAME}",
        n_channels=dataset_config["channels"],  # ty: ignore
        img_width=dataset_config["img_width"],  # ty: ignore
        img_height=dataset_config["img_height"],  # ty: ignore
    ).cuda()
    model.try_load()

    trainer = Trainer(
        model=model,
        schedules=schedules,
    )
    trainer.load_checkpoint()

    dataloader = get_dataloader()
    trainer.train(dataloader)

    logger.success("Training complete")


def generate():
    if not os.path.exists(f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}"):
        os.makedirs(f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}")

    # model_id = "1aurent/ddpm-mnist"
    # model_id = "google/ddpm-cifar10-32"
    model_id = "google/ddpm-celebahq-256"
    model = NoisePredictorHuggingface(model_id=model_id).cuda()
    # model = Predictor.load_from_file("./models/noise_predictor_unet_fashion.pth").cuda()
    # model.load()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")

    alpha_schedule = schedule_config["alpha_schedule"](model_id)  # ty: ignore
    sigma_schedule = schedule_config["sigma_schedule"](model_id)  # ty: ignore
    schedules = ScheduleGroup(
        alpha_schedule=alpha_schedule,  # ty: ignore
        sigma_schedule=sigma_schedule,  # ty: ignore
        eta_schedule=schedule_config["eta_schedule"](alpha_schedule, sigma_schedule),  # ty: ignore
    )

    # print(
    #     schedules.alpha(
    #         Timestep(
    #             TimestepConfig(kind="continuous", max_t=1.0),
    #             torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    #         )
    #     )
    # )

    logger.info(f"Using denoiser config: {SOLVER_CONFIG_NAME}")
    solver: Solver = solver_config["denoiser"](
        model=model,
        schedules=schedules,
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

    timesteps = (
        LinearSamplingSchedule(max_t=0.95).get_timesteps(n_steps=50).as_discrete(T)
    )
    timesteps.steps = timesteps.steps.cuda()

    n_samples = 1
    generated = generator.generate(n_samples=n_samples, timesteps=timesteps)

    for i in range(n_samples):
        img = generated[i]
        torchvision.utils.save_image(
            img, f"generated/{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}/{i + 1}.png"
        )


def ays():
    model = PredictorUNet(
        T=T,
        suffix=f"_{DATASET_CONFIG_NAME}",
        n_channels=dataset_config["channels"],  # ty: ignore
        img_width=dataset_config["img_width"],  # ty: ignore
        img_height=dataset_config["img_height"],  # ty: ignore
    ).cuda()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    schedules = ScheduleGroup(
        alpha_schedule=schedule_config["alpha_schedule"](),  # ty: ignore
        sigma_schedule=schedule_config["sigma_schedule"](),  # ty: ignore
        eta_schedule=schedule_config["eta_schedule"](),  # ty: ignore
    )

    logger.info(f"Using denoiser config: {SOLVER_CONFIG_NAME}")
    denoiser: Solver = solver_config["denoiser"](
        model=model,
        schedules=schedules,
    )

    ays_schedule = AYSSamplingSchedule(
        denoiser=denoiser,
        dataloader=get_dataloader(batch_size=128),
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
        dist.barrier()
        distributed.cleanup()
