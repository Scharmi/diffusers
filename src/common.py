from typing import Any, Type, cast

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from src.distributed import _RANK, _WORLD_SIZE, is_distributed


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
    dataset_class: Type,
    width: int,
    height: int,
    train=True,
    shuffle=True,
    num_workers=2,
):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    dataset = dataset_class(root="./data", download=True, transform=transform)  # ty: ignore

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset, num_replicas=_WORLD_SIZE, rank=_RANK, shuffle=shuffle
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
