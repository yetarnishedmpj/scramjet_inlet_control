from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scramjet_rl.logging import write_json
from scramjet_rl.physics.shock import shock_location_from_pressure, shock_on_lip_error
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
    true_shock_errors = np.asarray([shock_on_lip_error(field) for field in pressure], dtype=np.float32)
    pred_shock_errors = np.asarray(
        [shock_on_lip_error(field) for field in pred_fields[:, 0]],
        dtype=np.float32,
    )
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
        "shock_location_error_mae": float(np.mean(np.abs(pred_shock_errors - true_shock_errors))),
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


def generate_surrogate_report(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    sample_count: int = 3,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_surrogate(dataset_path, checkpoint_path)
    write_json(output_dir / "metrics.json", metrics)
    _plot_error_histograms(dataset_path, checkpoint_path, output_dir / "error_histograms.png")
    _plot_metric_parity(dataset_path, checkpoint_path, output_dir / "metric_parity.png")
    for index in range(sample_count):
        plot_prediction_sample(dataset_path, checkpoint_path, output_dir / f"sample_{index}.png", index)
    _write_markdown_summary(output_dir / "README.md", metrics, dataset_path, checkpoint_path)
    return output_dir


def _plot_error_histograms(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: Path,
) -> None:
    predictor = SurrogatePredictor(checkpoint_path)
    with h5py.File(dataset_path, "r") as file:
        inputs = file["inputs"][:].astype(np.float32)
        pressure = file["pressure"][:].astype(np.float32)
        temperature = file["temperature"][:].astype(np.float32)
    pred_fields, _ = predictor.predict(inputs)
    pressure_error = np.abs(pred_fields[:, 0] - pressure).reshape(-1)
    temperature_error = np.abs(pred_fields[:, 1] - temperature).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(pressure_error, bins=40)
    axes[0].set_title("Pressure Absolute Error")
    axes[1].hist(temperature_error, bins=40)
    axes[1].set_title("Temperature Absolute Error")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_metric_parity(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: Path,
) -> None:
    predictor = SurrogatePredictor(checkpoint_path)
    with h5py.File(dataset_path, "r") as file:
        inputs = file["inputs"][:].astype(np.float32)
        metrics = file["metrics"][:].astype(np.float32)
    _, pred_metrics = predictor.predict(inputs)
    names = ["shock_error", "pressure_recovery", "unstart_probability", "efficiency"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for ax, index, name in zip(axes.reshape(-1), range(4), names, strict=True):
        ax.scatter(metrics[:, index], pred_metrics[:, index], s=12, alpha=0.7)
        min_value = float(min(metrics[:, index].min(), pred_metrics[:, index].min()))
        max_value = float(max(metrics[:, index].max(), pred_metrics[:, index].max()))
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("truth")
        ax.set_ylabel("prediction")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_markdown_summary(
    path: Path,
    metrics: dict[str, Any],
    dataset_path: str | Path,
    checkpoint_path: str | Path,
) -> None:
    lines = [
        "# Surrogate Evaluation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{key}`: {value:.6g}" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `error_histograms.png`",
            "- `metric_parity.png`",
            "- `sample_*.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
