import torch
from diffusers import AutoencoderKL, AutoencoderTiny
from loguru import logger
from torch import nn


class VAE(nn.Module):
    def __init__(
        self,
        model_id: str = "madebyollin/taesd",
    ):
        super().__init__()
        self.model_id = model_id

        if "taesd" in model_id.lower() or "tiny" in model_id.lower():
            logger.info("Switching to Tiny AutoEncoder")
            self.autoencoder = AutoencoderTiny.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.is_tiny = True
        else:
            self.autoencoder = AutoencoderKL.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.is_tiny = False

        # self.autoencoder.enable_tiling()
        # self.autoencoder.enable_slicing()

        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        if hasattr(self.autoencoder.config, "block_out_channels"):
            num_blocks = len(self.autoencoder.config.block_out_channels)  # ty: ignore
            self.compression_factor = 2 ** (num_blocks - 1)
        elif hasattr(self.autoencoder.config, "encoder_block_out_channels"):
            num_blocks = len(self.autoencoder.config.encoder_block_out_channels)  # ty: ignore
            self.compression_factor = 2 ** (num_blocks - 1)
        else:
            logger.warning(
                "Could not determine compression factor from autoencoder config. Defaulting to 8."
            )
            self.compression_factor = 8

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tiny:
            z = self.autoencoder.encode(x).latents
        else:
            z = self.autoencoder.encode(x).latent_dist.sample()

        return z * self.autoencoder.config.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.autoencoder.config.scaling_factor
        x = self.autoencoder.decode(z).sample
        return x

    def get_latent_channels(self) -> int:
        return self.autoencoder.config.latent_channels

    def get_latent_width(self, img_width: int) -> int:
        return img_width // self.compression_factor

    def get_latent_height(self, img_height: int) -> int:
        return img_height // self.compression_factor
