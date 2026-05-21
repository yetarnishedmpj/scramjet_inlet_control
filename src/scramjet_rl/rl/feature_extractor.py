from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class InletCNNExtractor(BaseFeaturesExtractor):
    """Custom CNN + MLP feature extractor for the ScramjetInletEnv Dict observation.

    Architecture:
        * ``"sensors"`` branch: small CNN that preserves spatial structure of
          the pressure/temperature field patches, producing ``cnn_out_dim`` features.
        * ``"state"`` branch: two-layer MLP that embeds the scalar flight
          condition vector, producing ``state_out_dim`` features.
        * Both branches are concatenated into a single feature vector of length
          ``cnn_out_dim + state_out_dim``.

    Parameters
    ----------
    observation_space:
        The ``spaces.Dict`` observation space from ``ScramjetInletEnv``.
    cnn_out_dim:
        Number of features produced by the CNN branch. Default 64.
    state_out_dim:
        Number of features produced by the state MLP branch. Default 32.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out_dim: int = 64,
        state_out_dim: int = 32,
    ) -> None:
        features_dim = cnn_out_dim + state_out_dim
        super().__init__(observation_space, features_dim=features_dim)

        sensor_space = observation_space["sensors"]
        in_channels = sensor_space.shape[0]   # 2 (pressure + temperature)
        h = sensor_space.shape[1]
        w = sensor_space.shape[2]
        state_dim = observation_space["state"].shape[0]

        # --- CNN branch: lightweight conv stack ---
        self.cnn = nn.Sequential(
            # (B, 2, H, W)
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # (B, 16, H, W)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            # (B, 32, H/2, W/2)
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 4)),   # fixed output regardless of H, W
            # (B, 32, 2, 4) -> (B, 256) after flatten
            nn.Flatten(),
        )

        # Compute CNN output size by doing a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            cnn_flat_dim = self.cnn(dummy).shape[1]

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_flat_dim, cnn_out_dim),
            nn.SiLU(),
        )

        # --- Scalar state branch ---
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, state_out_dim),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_feat = self.cnn_proj(self.cnn(observations["sensors"]))
        state_feat = self.state_mlp(observations["state"])
        return torch.cat([cnn_feat, state_feat], dim=1)
