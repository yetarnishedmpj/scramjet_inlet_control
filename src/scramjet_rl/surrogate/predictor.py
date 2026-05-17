from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scramjet_rl.surrogate.models import build_model


class SurrogatePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.input_mean = _to_numpy(checkpoint["input_mean"])
        self.input_std = _to_numpy(checkpoint["input_std"])
        self.field_mean = _to_numpy(checkpoint["field_mean"])
        self.field_std = _to_numpy(checkpoint["field_std"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(
            checkpoint.get("model_type", "cnn"),
            height=checkpoint["height"],
            width=checkpoint["width"],
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        normalized = (inputs - self.input_mean) / self.input_std
        x = torch.from_numpy(normalized).float().to(self.device)
        fields, metrics = self.model(x)
        fields_np = fields.cpu().numpy() * self.field_std + self.field_mean
        return fields_np, metrics.cpu().numpy()


class EnsembleSurrogatePredictor:
    def __init__(self, checkpoint_paths: list[str | Path], device: str | None = None) -> None:
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty")
        self.predictors = [SurrogatePredictor(path, device=device) for path in checkpoint_paths]

    def predict(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        field_predictions = []
        metric_predictions = []
        for predictor in self.predictors:
            fields, metrics = predictor.predict(inputs)
            field_predictions.append(fields)
            metric_predictions.append(metrics)
        field_stack = np.stack(field_predictions, axis=0)
        metric_stack = np.stack(metric_predictions, axis=0)
        return {
            "field_mean": field_stack.mean(axis=0),
            "field_std": field_stack.std(axis=0),
            "metric_mean": metric_stack.mean(axis=0),
            "metric_std": metric_stack.std(axis=0),
        }


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)
