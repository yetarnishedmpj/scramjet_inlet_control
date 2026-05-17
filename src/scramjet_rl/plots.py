from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def plot_dataset_sample(dataset_path: str | Path, index: int, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_path, "r") as file:
        pressure = file["pressure"][index]
        temperature = file["temperature"][index]
        inputs = file["inputs"][index]
        metrics = file["metrics"][index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    pressure_plot = axes[0].imshow(pressure, origin="lower", aspect="auto")
    axes[0].set_title("Pressure")
    fig.colorbar(pressure_plot, ax=axes[0])
    temperature_plot = axes[1].imshow(temperature, origin="lower", aspect="auto")
    axes[1].set_title("Temperature")
    fig.colorbar(temperature_plot, ax=axes[1])
    fig.suptitle(
        "mach={:.2f}, altitude={:.0f}m, angle={:.2f}deg | shock_err={:.3f}, unstart={:.3f}".format(
            inputs[0], inputs[1], inputs[3], metrics[0], metrics[2]
        )
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
