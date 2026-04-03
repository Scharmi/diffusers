from typing import cast

from torchvision import datasets

from src.equation import GeneralizedDifferential, GeneralizedDiscrete, ProbabilityFlow
from src.schedule import (
    ConstantAlphaSchedule,
    ConstantEtaSchedule,
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    DDPMEtaSchedule,
    HuggingFaceDDPMAlphaSchedule,
    HuggingFaceDDPMSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
)
from src.solver import (
    DiscreteSolver,
    EulerMaruyamaSDESolver,
    EulerODESolver,
    HeunODESolver,
)
from src.trainer import TimeSampler

BATCH_SIZE = 128
PREDICTOR_T = 1000

SOLVER_CONFIGS = {
    "discrete": DiscreteSolver,
    "euler": EulerODESolver,
    "heun": HeunODESolver,
    "euler_maruyama": EulerMaruyamaSDESolver,
}

SCHEDULE_CONFIGS = {
    "linear": {
        "alpha_schedule": LinearAlphaSchedule,
        "sigma_schedule": LinearSigmaSchedule,
    },
    "cosine": {
        "alpha_schedule": CosineAlphaSchedule,
        "sigma_schedule": CosineSigmaSchedule,
    },
    "ddpm": {
        "alpha_schedule": HuggingFaceDDPMAlphaSchedule,
        "sigma_schedule": HuggingFaceDDPMSigmaSchedule,
    },
    "edm": {
        "alpha_schedule": lambda: ConstantAlphaSchedule(1.0),
        "sigma_schedule": lambda: LinearSigmaSchedule(
            exploding=True
        ),  # In this setting sigma_EDM(t) = sigma(t)
    },
}

ETA_CONFIGS = {
    "deterministic": lambda alpha, sigma: ConstantEtaSchedule(0.0),
    "stochastic": lambda alpha, sigma: ConstantEtaSchedule(1.0),
    "ddpm": lambda alpha, sigma: DDPMEtaSchedule(alpha, sigma),
}

EQUATION_CONFIGS = {
    "generalized_discrete": GeneralizedDiscrete,
    "generalized_differential": GeneralizedDifferential,
    "probability_flow": ProbabilityFlow,
}

DATASET_CONFIGS = {
    "mnist": {
        "class": datasets.MNIST,
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "fashion": {
        "class": datasets.FashionMNIST,
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "channels": 3,
        "img_width": 32,
        "img_height": 32,
    },
    "celeb": {
        "class": datasets.CelebA,
        "channels": 3,
        "img_width": 256,
        "img_height": 256,
    },
    "flowers": {
        "class": datasets.Flowers102,
        "channels": 3,
        "img_width": 224,
        "img_height": 224,
    },
}

SOLVER_CONFIG_NAME = "heun"
solver_config = SOLVER_CONFIGS[SOLVER_CONFIG_NAME]

SCHEDULE_CONFIG_NAME = "edm"
schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

ETA_CONFIG_NAME = "ddpm"
eta_config = ETA_CONFIGS[ETA_CONFIG_NAME]

EQUATION_CONFIG_NAME = "probability_flow"
equation_config = EQUATION_CONFIGS[EQUATION_CONFIG_NAME]

DATASET_CONFIG_NAME = "flowers"
dataset_config = DATASET_CONFIGS[DATASET_CONFIG_NAME]

timesampler_config = cast(TimeSampler, None)
match SCHEDULE_CONFIG_NAME:
    case "edm":
        timesampler_config = TimeSampler.EDM
    case "ddpm":
        timesampler_config = TimeSampler.UNIFORM_DISCRETE
    case _:
        timesampler_config = TimeSampler.UNIFORM_CONTINUOUS

SOLVER_T = cast(int | float, None)  # full time-space
match EQUATION_CONFIG_NAME:
    case "generalized_discrete":
        SOLVERT_T = PREDICTOR_T
    case "generalized_differential":
        SOLVERT_T = 1.0
    case "probability_flow":
        if SCHEDULE_CONFIG_NAME == "edm":
            # I assume that edm_sigma(t) = t, then choosing edm_sigma is equivalent to choosing time parametrization.
            # Theoretically in EDM sigma has no upper bound, but due to how this codebase handles timespace, here it is bound by large value.
            # This should be no problem as during training model will sample time close to 0 and during inference no more than around 80.
            SOLVER_T = float(PREDICTOR_T)
        else:
            SOLVER_T = 1.0
    case _:
        raise ValueError(f"Unknown equation config name: {EQUATION_CONFIG_NAME}")
