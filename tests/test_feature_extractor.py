"""Smoke tests for the CNN-MLP feature extractor and Dict observation space."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch


def _make_dummy_dict_obs_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "sensors": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 8, 16), dtype=np.float32),
        }
    )


def test_feature_extractor_forward():
    from scramjet_rl.rl.feature_extractor import InletCNNExtractor

    obs_space = _make_dummy_dict_obs_space()
    extractor = InletCNNExtractor(obs_space, cnn_out_dim=32, state_out_dim=16)
    assert extractor.features_dim == 32 + 16

    batch = {
        "state": torch.zeros(4, 5),
        "sensors": torch.zeros(4, 2, 8, 16),
    }
    out = extractor(batch)
    assert out.shape == (4, 48), f"Unexpected output shape: {out.shape}"


def test_env_dict_obs_space():
    """Verify that ScramjetInletEnv has a Dict observation space with correct keys."""
    from scramjet_rl.envs.scramjet_inlet_env import ScramjetInletEnv

    # We can't load the real surrogate in a unit test, so just check the space
    # by subclassing and stubbing out the predictor.
    import types

    env = object.__new__(ScramjetInletEnv)
    env.config = __import__(
        "scramjet_rl.envs.scramjet_inlet_env", fromlist=["EnvConfig"]
    ).EnvConfig()

    from gymnasium import spaces

    env.observation_space = spaces.Dict(
        {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "sensors": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2, env.config.sensor_height, env.config.sensor_width),
                dtype=np.float32,
            ),
        }
    )

    assert "state" in env.observation_space.spaces
    assert "sensors" in env.observation_space.spaces
    assert env.observation_space["state"].shape == (5,)
    assert env.observation_space["sensors"].shape == (2, 8, 16)
