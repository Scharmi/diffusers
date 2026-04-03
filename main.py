import torch
import typer
from loguru import logger

from src import distributed
from src.cli import generate, train

app = typer.Typer(no_args_is_help=True)

app.add_typer(train.app, name="train")
app.add_typer(generate.app, name="generate")
# app.add_typer(ays.app, name="ays")


def setup_env():
    if distributed.is_distributed():
        distributed.setup()


def cleanup_env():
    if distributed.is_distributed():
        torch.distributed.barrier()
        distributed.cleanup()


if __name__ == "__main__":
    setup_env()
    try:
        app()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        cleanup_env()
