from __future__ import annotations

import torch
from torch import nn


class InletSurrogateCNN(nn.Module):
    def __init__(self, input_dim: int = 4, height: int = 32, width: int = 64) -> None:
        super().__init__()
        self.height = height
        self.width = width
        latent_channels = 32
        seed_h = height // 8
        seed_w = width // 8
        self.seed_h = seed_h
        self.seed_w = seed_w

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_channels * seed_h * seed_w),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x).view(x.shape[0], 32, self.seed_h, self.seed_w)
        fields = self.decoder(latent)
        metrics = self.metric_head(x)
        return fields, metrics


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x + self.block(x))


class InletSurrogateResNet(nn.Module):
    def __init__(self, input_dim: int = 4, height: int = 32, width: int = 64) -> None:
        super().__init__()
        self.height = height
        self.width = width
        channels = 48
        seed_h = height // 8
        seed_w = width // 8
        self.seed_h = seed_h
        self.seed_w = seed_w
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.SiLU(),
            nn.Linear(192, channels * seed_h * seed_w),
            nn.SiLU(),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x).view(x.shape[0], 48, self.seed_h, self.seed_w)
        return self.upsample(latent), self.metric_head(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InletSurrogateUNet(nn.Module):
    def __init__(self, input_dim: int = 4, height: int = 32, width: int = 64) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.seed = nn.Sequential(nn.Linear(input_dim, 64 * height * width), nn.SiLU())
        self.down1 = ConvBlock(64, 32)
        self.down2 = ConvBlock(32, 64)
        self.down3 = ConvBlock(64, 96)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(96, 128)
        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(192, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 2, kernel_size=1)
        self.metric_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seeded = self.seed(x).view(x.shape[0], 64, self.height, self.width)
        enc1 = self.down1(seeded)
        enc2 = self.down2(self.pool(enc1))
        enc3 = self.down3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.out(dec1), self.metric_head(x)


class MetricOnlySurrogate(nn.Module):
    def __init__(self, input_dim: int = 4, height: int = 32, width: int = 64) -> None:
        super().__init__()
        self.field_bias = nn.Parameter(torch.zeros(1, 2, height, width))
        self.metric_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.field_bias.expand(x.shape[0], -1, -1, -1), self.metric_head(x)


def build_model(model_type: str, height: int, width: int) -> nn.Module:
    if model_type == "cnn":
        return InletSurrogateCNN(height=height, width=width)
    if model_type == "resnet":
        return InletSurrogateResNet(height=height, width=width)
    if model_type == "unet":
        return InletSurrogateUNet(height=height, width=width)
    if model_type == "metric_mlp":
        return MetricOnlySurrogate(height=height, width=width)
    raise ValueError(
        f"Unknown model_type {model_type!r}. Expected 'cnn', 'resnet', 'unet', or 'metric_mlp'."
    )
