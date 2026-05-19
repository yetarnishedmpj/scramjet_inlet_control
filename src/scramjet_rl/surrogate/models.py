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
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 8),
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_attention: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.attention = CBAM(channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.attention(residual)
        return torch.nn.functional.silu(x + residual)


class InletSurrogateResNet(nn.Module):
    def __init__(self, input_dim: int = 4, height: int = 32, width: int = 64) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.channels = 64
        seed_h = height // 8
        seed_w = width // 8
        self.seed_h = seed_h
        self.seed_w = seed_w
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.channels * seed_h * seed_w),
            nn.SiLU(),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.channels),
            nn.SiLU(),
            ResidualBlock(self.channels, use_attention=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualBlock(32, use_attention=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x).view(x.shape[0], self.channels, self.seed_h, self.seed_w)
        return self.upsample(latent), self.metric_head(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
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
        self.bot_cbam = CBAM(128)
        
        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2)
        self.cbam3 = CBAM(96)
        self.dec3 = ConvBlock(192, 96)
        
        self.up2 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.cbam2 = CBAM(64)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.cbam1 = CBAM(32)
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
        bottleneck = self.bot_cbam(bottleneck)
        
        skip3 = self.cbam3(enc3)
        dec3 = self.dec3(torch.cat([self.up3(bottleneck), skip3], dim=1))
        
        skip2 = self.cbam2(enc2)
        dec2 = self.dec2(torch.cat([self.up2(dec3), skip2], dim=1))
        
        skip1 = self.cbam1(enc1)
        dec1 = self.dec1(torch.cat([self.up1(dec2), skip1], dim=1))
        
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
