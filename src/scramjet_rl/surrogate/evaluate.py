from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scramjet_rl.surrogate.predictor import SurrogatePredictor


def evaluate_surrogate(dataset_path: str | Path, checkpoint_path: str | Path) -> dict[str, float]:
    predictor = SurrogatePredictor(checkpoint_path)
    with h5py.File(dataset_path, "r") as file:
        inputs = file["inputs"][:].astype(np.float32)
        pressure = file["pressure"][:].astype(np.float32)
        temperature = file["temperature"][:].astype(np.float32)
        metrics = file["metrics"][:].astype(np.float32)

    pred_fields, pred_metrics = predictor.predict(inputs)
    true_fields = np.stack([pressure, temperature], axis=1)
    field_mae = float(np.mean(np.abs(pred_fields - true_fields)))
    pressure_mae = float(np.mean(np.abs(pred_fields[:, 0] - pressure)))
    temperature_mae = float(np.mean(np.abs(pred_fields[:, 1] - temperature)))
    metric_mae = float(np.mean(np.abs(pred_metrics - metrics)))
    unstart_mae = float(np.mean(np.abs(pred_metrics[:, 2] - metrics[:, 2])))
    return {
        "field_mae": field_mae,
        "pressure_mae": pressure_mae,
        "temperature_mae": temperature_mae,
        "metric_mae": metric_mae,
        "unstart_probability_mae": unstart_mae,
    }


def plot_prediction_sample(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    index: int = 0,
) -> Path:
    predictor = SurrogatePredictor(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_path, "r") as file:
        inputs = file["inputs"][index].astype(np.float32)
        pressure = file["pressure"][index].astype(np.float32)
        temperature = file["temperature"][index].astype(np.float32)
    pred_fields, _ = predictor.predict(inputs)
    pred_pressure = pred_fields[0, 0]
    pred_temperature = pred_fields[0, 1]

    panels = [
        ("Pressure Truth", pressure),
        ("Pressure Prediction", pred_pressure),
        ("Pressure Error", np.abs(pred_pressure - pressure)),
        ("Temperature Truth", temperature),
        ("Temperature Prediction", pred_temperature),
        ("Temperature Error", np.abs(pred_temperature - temperature)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, (title, values) in zip(axes.reshape(-1), panels, strict=True):
        image = ax.imshow(values, origin="lower", aspect="auto")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
