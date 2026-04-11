from typing import Any, Type, cast

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from src.config import DatasetConfig
from src.distributed import get_rank, get_world_size, is_distributed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_type[T](val: Any, typ: Type[T]) -> T:
    if not isinstance(val, typ):
        raise TypeError(f"Expexted type {typ}, got {type(val)}")
    return val


def load_unet_config(model_id: str, unet_class: Type = UNet2DModel, **kwargs) -> dict:
    try:
        return cast(dict, unet_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(dict, unet_class.load_config(model_id, subfolder="unet", **kwargs))  # type: ignore


def load_unet_pretrained(
    model_id: str, unet_class: Type = UNet2DModel, **kwargs
) -> UNet2DModel:
    try:
        return unet_class.from_pretrained(model_id, **kwargs)  # type: ignore
    except Exception:
        return unet_class.from_pretrained(model_id, subfolder="unet", **kwargs)  # type: ignore


def load_scheduler_config(
    model_id: str, scheduler_class: Type = DDPMScheduler, **kwargs
) -> dict:
    try:
        return cast(dict, scheduler_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(
            dict,
            scheduler_class.load_config(model_id, subfolder="scheduler", **kwargs),  # type: ignore
        )


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    img = (img + 1.0) / 2.0

    return torch.clamp(img, 0.0, 1.0)


def get_dataloader(
    batch_size: int,
    dataset_config: DatasetConfig,
    shuffle=True,
    num_workers=1,
):
    transform = transforms.Compose(
        [
            transforms.Resize(dataset_config.img_size),
            transforms.CenterCrop(dataset_config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(dataset_config.channels, 1, 1)
                if x.shape[0] == 1 and dataset_config.channels > 1
                else x
            ),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    dataset_kwargs = {}
    if dataset_config.split is not None:
        dataset_kwargs["split"] = dataset_config.split

    if is_distributed() and get_rank() != 0:
        torch.distributed.barrier()

    should_download = True
    if is_distributed():
        should_download = get_rank() == 0

    if dataset_config.path is None:
        dataset = dataset_config.dataset_class(
            root="./data",
            download=should_download,  # ty: ignore
            transform=transform,
            **dataset_kwargs,
        )
    else:
        dataset = datasets.ImageFolder(
            root=dataset_config.path,
            transform=transform,
        )

    if is_distributed() and get_rank() == 0:
        torch.distributed.barrier()

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
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
