# from typing import cast

# from torchvision import datasets

# from src.equation import GeneralizedDifferential, GeneralizedDiscrete, ProbabilityFlow
# from src.schedule import (
#     ConstantAlphaSchedule,
#     ConstantEtaSchedule,
#     CosineAlphaSchedule,
#     CosineSigmaSchedule,
#     DDPMEtaSchedule,
#     HuggingFaceDDPMAlphaSchedule,
#     HuggingFaceDDPMSigmaSchedule,
#     LinearAlphaSchedule,
#     LinearSigmaSchedule,
# )
# from src.solver import (
#     DiscreteSolver,
#     EulerMaruyamaSDESolver,
#     EulerODESolver,
#     HeunODESolver,
# )
# from src.trainer import TimeSampler

# BATCH_SIZE = 128
# PREDICTOR_T = 1000

# SOLVER_CONFIGS = {
#     "discrete": DiscreteSolver,
#     "euler": EulerODESolver,
#     "heun": HeunODESolver,
#     "euler_maruyama": EulerMaruyamaSDESolver,
# }

# SCHEDULE_CONFIGS = {
#     "linear": {
#         "alpha_schedule": LinearAlphaSchedule,
#         "sigma_schedule": LinearSigmaSchedule,
#     },
#     "cosine": {
#         "alpha_schedule": CosineAlphaSchedule,
#         "sigma_schedule": CosineSigmaSchedule,
#     },
#     "ddpm": {
#         "alpha_schedule": HuggingFaceDDPMAlphaSchedule,
#         "sigma_schedule": HuggingFaceDDPMSigmaSchedule,
#     },
#     "edm": {
#         "alpha_schedule": lambda: ConstantAlphaSchedule(1.0),
#         "sigma_schedule": lambda: LinearSigmaSchedule(
#             exploding=True
#         ),  # In this setting sigma_EDM(t) = sigma(t)
#     },
# }

# ETA_CONFIGS = {
#     "deterministic": lambda alpha, sigma: ConstantEtaSchedule(0.0),
#     "stochastic": lambda alpha, sigma: ConstantEtaSchedule(1.0),
#     "ddpm": lambda alpha, sigma: DDPMEtaSchedule(alpha, sigma),
# }

# EQUATION_CONFIGS = {
#     "generalized_discrete": GeneralizedDiscrete,
#     "generalized_differential": GeneralizedDifferential,
#     "probability_flow": ProbabilityFlow,
# }

# DATASET_CONFIGS = {
#     "mnist": {
#         "class": datasets.MNIST,
#         "channels": 1,
#         "img_width": 28,
#         "img_height": 28,
#     },
#     "fashion": {
#         "class": datasets.FashionMNIST,
#         "channels": 1,
#         "img_width": 28,
#         "img_height": 28,
#     },
#     "cifar10": {
#         "class": datasets.CIFAR10,
#         "channels": 3,
#         "img_width": 32,
#         "img_height": 32,
#     },
#     "celeb": {
#         "class": datasets.CelebA,
#         "channels": 3,
#         "img_width": 256,
#         "img_height": 256,
#     },
#     "flowers": {
#         "class": datasets.Flowers102,
#         "channels": 3,
#         "img_width": 224,
#         "img_height": 224,
#     },
# }

# SOLVER_CONFIG_NAME = "heun"
# solver_config = SOLVER_CONFIGS[SOLVER_CONFIG_NAME]

# SCHEDULE_CONFIG_NAME = "edm"
# schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

# ETA_CONFIG_NAME = "ddpm"
# eta_config = ETA_CONFIGS[ETA_CONFIG_NAME]

# EQUATION_CONFIG_NAME = "probability_flow"
# equation_config = EQUATION_CONFIGS[EQUATION_CONFIG_NAME]

# DATASET_CONFIG_NAME = "flowers"
# dataset_config = DATASET_CONFIGS[DATASET_CONFIG_NAME]

# timesampler_config = cast(TimeSampler, None)
# match SCHEDULE_CONFIG_NAME:
#     case "edm":
#         timesampler_config = TimeSampler.EDM
#     case "ddpm":
#         timesampler_config = TimeSampler.UNIFORM_DISCRETE
#     case _:
#         timesampler_config = TimeSampler.UNIFORM_CONTINUOUS

# SOLVER_T = cast(int | float, None)  # full time-space
# match EQUATION_CONFIG_NAME:
#     case "generalized_discrete":
#         SOLVERT_T = PREDICTOR_T
#     case "generalized_differential":
#         SOLVERT_T = 1.0
#     case "probability_flow":
#         if SCHEDULE_CONFIG_NAME == "edm":
#             # I assume that edm_sigma(t) = t, then choosing edm_sigma is equivalent to choosing time parametrization.
#             # Theoretically in EDM sigma has no upper bound, but due to how this codebase handles timespace, here it is bound by large value.
#             # This should be no problem as during training model will sample time close to 0 and during inference no more than around 80.
#             SOLVER_T = float(PREDICTOR_T)
#         else:
#             SOLVER_T = 1.0
#     case _:
#         raise ValueError(f"Unknown equation config name: {EQUATION_CONFIG_NAME}")

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Type

from torchvision import datasets
from torchvision.datasets import VisionDataset

from src.equation import (
    Equation,
    GeneralizedDifferential,
    GeneralizedDiscrete,
    ProbabilityFlow,
)
from src.model import Predictor, PredictorEDM, PredictorEDM2, PredictorUNet
from src.schedule import (
    AlphaSchedule,
    ConstantAlphaSchedule,
    ConstantEtaSchedule,
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    DDPMEtaSchedule,
    EtaSchedule,
    HuggingFaceDDPMAlphaSchedule,
    HuggingFaceDDPMSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
    SigmaSchedule,
)
from src.solver import (
    DiscreteSolver,
    EulerMaruyamaSDESolver,
    EulerODESolver,
    HeunODESolver,
    Solver,
)
from src.trainer import TimeSampler


class SolverType(str, Enum):
    discrete = "discrete"
    euler = "euler"
    heun = "heun"
    euler_maruyama = "euler_maruyama"


class ScheduleType(str, Enum):
    linear = "linear"
    cosine = "cosine"
    ddpm = "ddpm"
    edm = "edm"


class EtaType(str, Enum):
    deterministic = "deterministic"
    stochastic = "stochastic"
    ddpm = "ddpm"


class EquationType(str, Enum):
    generalized_discrete = "generalized_discrete"
    generalized_differential = "generalized_differential"
    probability_flow = "probability_flow"


class DatasetType(str, Enum):
    mnist = "mnist"
    fashion = "fashion"
    cifar10 = "cifar10"
    celeb = "celeb"
    flowers = "flowers"


@dataclass
class DatasetConfig:
    dataset_class: Type[VisionDataset]
    channels: int
    img_width: int
    img_height: int


@dataclass
class ScheduleConfig:
    alpha_schedule_factory: Callable[..., AlphaSchedule]
    sigma_schedule_factory: Callable[..., SigmaSchedule]


class ModelType(str, Enum):
    edm = "edm"
    edm2 = "edm2"
    unet = "unet"
    huggingface = "huggingface"


SOLVER_CONFIGS: dict[SolverType, Type[Solver]] = {
    SolverType.discrete: DiscreteSolver,
    SolverType.euler: EulerODESolver,
    SolverType.heun: HeunODESolver,
    SolverType.euler_maruyama: EulerMaruyamaSDESolver,
}

SCHEDULE_CONFIGS: dict[ScheduleType, ScheduleConfig] = {
    ScheduleType.linear: ScheduleConfig(
        alpha_schedule_factory=LinearAlphaSchedule,
        sigma_schedule_factory=LinearSigmaSchedule,
    ),
    ScheduleType.cosine: ScheduleConfig(
        alpha_schedule_factory=CosineAlphaSchedule,
        sigma_schedule_factory=CosineSigmaSchedule,
    ),
    ScheduleType.ddpm: ScheduleConfig(
        alpha_schedule_factory=HuggingFaceDDPMAlphaSchedule,
        sigma_schedule_factory=HuggingFaceDDPMSigmaSchedule,
    ),
    ScheduleType.edm: ScheduleConfig(
        alpha_schedule_factory=lambda **kwargs: ConstantAlphaSchedule(1.0),
        sigma_schedule_factory=lambda **kwargs: LinearSigmaSchedule(exploding=True),
    ),
}

ETA_CONFIGS: dict[EtaType, Callable[[AlphaSchedule, SigmaSchedule], EtaSchedule]] = {
    EtaType.deterministic: lambda alpha, sigma: ConstantEtaSchedule(0.0),
    EtaType.stochastic: lambda alpha, sigma: ConstantEtaSchedule(1.0),
    EtaType.ddpm: lambda alpha, sigma: DDPMEtaSchedule(alpha, sigma),
}

EQUATION_CONFIGS: dict[EquationType, Type[Equation]] = {
    EquationType.generalized_discrete: GeneralizedDiscrete,
    EquationType.generalized_differential: GeneralizedDifferential,
    EquationType.probability_flow: ProbabilityFlow,
}

DATASET_CONFIGS: dict[DatasetType, DatasetConfig] = {
    DatasetType.mnist: DatasetConfig(datasets.MNIST, 1, 28, 28),
    DatasetType.fashion: DatasetConfig(datasets.FashionMNIST, 1, 28, 28),
    DatasetType.cifar10: DatasetConfig(datasets.CIFAR10, 3, 32, 32),
    DatasetType.celeb: DatasetConfig(datasets.CelebA, 3, 256, 256),
    DatasetType.flowers: DatasetConfig(datasets.Flowers102, 3, 128, 128),
}

MODEL_CONFIGS: dict[ModelType, Type[Predictor]] = {
    ModelType.edm: PredictorEDM,
    ModelType.edm2: PredictorEDM2,
    ModelType.unet: PredictorUNet,
}


def get_timesampler(schedule_name: ScheduleType) -> TimeSampler:
    if schedule_name == ScheduleType.edm:
        return TimeSampler.EDM
    if schedule_name == ScheduleType.ddpm:
        return TimeSampler.UNIFORM_DISCRETE
    return TimeSampler.UNIFORM_CONTINUOUS


def get_solver_T(
    equation_name: EquationType, schedule_name: ScheduleType, predictor_t: int
) -> float:
    """Returns maximum time step T solver can be run with."""

    if equation_name == EquationType.generalized_discrete:
        return predictor_t
    if equation_name == EquationType.generalized_differential:
        return 1.0
    if equation_name == EquationType.probability_flow:
        return float(predictor_t) if schedule_name == ScheduleType.edm else 1.0
    raise ValueError(f"Unknown equation config name: {equation_name}")
