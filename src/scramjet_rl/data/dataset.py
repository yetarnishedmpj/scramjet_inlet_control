from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class InletFieldDataset(Dataset):
    def __init__(self, path: str | Path, indices: np.ndarray | None = None) -> None:
        with h5py.File(path, "r") as file:
            self.inputs = file["inputs"][:].astype(np.float32)
            pressure = file["pressure"][:].astype(np.float32)
            temperature = file["temperature"][:].astype(np.float32)
            self.metrics = file["metrics"][:].astype(np.float32)

        self.fields = np.stack([pressure, temperature], axis=1)
        self.input_mean = self.inputs.mean(axis=0)
        self.input_std = self.inputs.std(axis=0) + 1e-6
        self.field_mean = self.fields.mean(axis=(0, 2, 3), keepdims=True)
        self.field_std = self.fields.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        if indices is None:
            self.indices = np.arange(len(self.inputs))
        else:
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[item]
        x = (self.inputs[idx] - self.input_mean) / self.input_std
        y = (self.fields[idx] - self.field_mean[0]) / self.field_std[0]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(self.metrics[idx]).float(),
        )
