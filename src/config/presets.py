from typing import Callable, Type

from torchvision import datasets

from src.config import (
    DatasetConfig,
    DatasetType,
    EquationType,
    EtaType,
    ModelType,
    ScheduleConfig,
    ScheduleType,
    SolverType,
)
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
from src.train import TimeSampler

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

# Image sizes should be divisible by 8.
DATASET_CONFIGS: dict[DatasetType, DatasetConfig] = {
    DatasetType.mnist: DatasetConfig(datasets.MNIST, 1, 32),
    DatasetType.fashion: DatasetConfig(datasets.FashionMNIST, 1, 32),
    DatasetType.cifar10: DatasetConfig(datasets.CIFAR10, 3, 32),
    DatasetType.celeb: DatasetConfig(datasets.CelebA, 3, 64, path="data/celebA"),
    DatasetType.flowers: DatasetConfig(datasets.Flowers102, 3, 128),
    DatasetType.stl10: DatasetConfig(datasets.STL10, 3, 64, split="unlabeled"),
    DatasetType.food101: DatasetConfig(datasets.Food101, 3, 512, split="train"),
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
