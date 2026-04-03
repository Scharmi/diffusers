import os
from typing import Optional

import torch
import torchvision
import typer
from loguru import logger

from src.common import unnormalize
from src.config import (
    EQUATION_CONFIGS,
    ETA_CONFIGS,
    SCHEDULE_CONFIGS,
    SOLVER_CONFIGS,
    EquationType,
    EtaType,
    ModelType,
    ScheduleType,
    SolverType,
    get_solver_T,
)
from src.generator import Generator
from src.model import Predictor, PredictorHuggingface
from src.schedule import ScheduleGroup
from src.schedule.sampling import EDMSamplingSchedule

app = typer.Typer(help="Generate images")


@app.callback(invoke_without_command=True)
def generate(
    model_name: ModelType = typer.Option(
        ModelType.edm,
        "--model",
        help="Network architecture",
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", help="Path to the local pre-trained weights"
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID (required if model=huggingface)",
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm, "--schedule", help="Signal/Noise schedule"
    ),
    eta: EtaType = typer.Option(EtaType.ddpm, "--eta", help="Stochasticity"),
    solver_name: SolverType = typer.Option(
        SolverType.heun, "--solver", help="Discrete/ODE/SDE Solver"
    ),
    equation_name: EquationType = typer.Option(
        EquationType.probability_flow,
        "--equation",
        help="Reverse diffusion equation",
    ),
    n_samples: int = typer.Option(
        30, "--n-samples", help="Number of images to generate"
    ),
):
    output_path = f"generated/{solver_name.value}_{schedule.value}"
    os.makedirs(output_path, exist_ok=True)

    logger.info("Starting the generation")
    logger.info(f"Model Architecture: {model_name.value}")
    logger.info(f"Signal/Noise Schedule: {schedule.value}")
    logger.info(f"Stochasticity (eta): {eta.value}")
    logger.info(f"Equation: {equation_name.value}")
    logger.info(f"Solver: {solver_name.value}")

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

    equation = EQUATION_CONFIGS[equation_name](
        model=model,
        schedules=schedules,
    )

    predictor_T = model.timestep_config.T
    solver_T = get_solver_T(equation_name, schedule, predictor_T)
    solver = SOLVER_CONFIGS[solver_name](
        equation=equation,
        T=solver_T,
    )

    generator = Generator(solver=solver)

    logger.info(f"Generating {n_samples} samples...")

    generator_kwargs = {}
    if schedule == ScheduleType.edm:
        generator_kwargs["timesteps"] = (
            EDMSamplingSchedule(T=predictor_T).get_timesteps(n_steps=100).cuda()
        )
        generator_kwargs["variance_exploding"] = True
    else:
        generator_kwargs["skip_last_step"] = True

    with torch.no_grad():
        generated = generator.generate(
            n_samples=n_samples,
            **generator_kwargs,
        )

    for i in range(n_samples):
        img = unnormalize(generated[i])
        torchvision.utils.save_image(
            img,
            f"{output_path}/{i + 1}.png",
        )
