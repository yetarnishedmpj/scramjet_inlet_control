import pytest

from scramjet_rl.config import validate_config


def test_validate_surrogate_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="model_type"):
        validate_config(
            {
                "dataset_path": "data.h5",
                "model_path": "model.pt",
                "model_type": "unknown",
            },
            "surrogate",
        )


def test_validate_rl_accepts_ppo():
    validate_config(
        {
            "surrogate_path": "model.pt",
            "output_path": "policy.zip",
            "algorithm": "ppo",
        },
        "rl",
    )
