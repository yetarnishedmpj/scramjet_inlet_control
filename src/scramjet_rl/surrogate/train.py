from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from scramjet_rl.config import ensure_parent
from scramjet_rl.data.dataset import InletFieldDataset
from scramjet_rl.data.splits import load_split_indices
from scramjet_rl.logging import append_csv, timestamp, write_json
from scramjet_rl.surrogate.models import build_model


def train_surrogate(config: dict) -> Path:
    torch.manual_seed(int(config.get("seed", 0)))
    dataset = InletFieldDataset(config["dataset_path"])
    num_items = len(dataset)
    if "split_dir" in config:
        splits = load_split_indices(config["split_dir"])
        train_indices = splits["train"]
        val_indices = splits["val"]
    else:
        rng = np.random.default_rng(int(config.get("seed", 0)))
        indices = rng.permutation(num_items)
        val_count = max(1, int(num_items * float(config.get("validation_fraction", 0.15))))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

    train_dataset = InletFieldDataset(config["dataset_path"], train_indices)
    val_dataset = InletFieldDataset(config["dataset_path"], val_indices)
    train_dataset.input_mean = dataset.input_mean
    train_dataset.input_std = dataset.input_std
    train_dataset.field_mean = dataset.field_mean
    train_dataset.field_std = dataset.field_std
    val_dataset.input_mean = dataset.input_mean
    val_dataset.input_std = dataset.input_std
    val_dataset.field_mean = dataset.field_mean
    val_dataset.field_std = dataset.field_std

    sample_fields = dataset.fields
    height, width = sample_fields.shape[-2:]
    model_type = str(config.get("model_type", "cnn"))
    model = build_model(model_type, height=height, width=width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    mse = nn.MSELoss()
    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    metric_weight = float(config.get("metric_loss_weight", 0.25))
    field_weight = float(config.get("field_loss_weight", 1.0))

    run_id = str(config.get("run_id", timestamp()))
    log_dir = Path(str(config.get("log_dir", "outputs/experiments"))) / run_id
    history_path = log_dir / "surrogate_history.csv"
    epochs = int(config.get("epochs", 5))
    best_val_loss = float("inf")
    best_state = None

    for epoch in trange(epochs, desc="training surrogate"):
        model.train()
        train_losses = []
        for inputs, fields, metrics in train_loader:
            inputs = inputs.to(device)
            fields = fields.to(device)
            metrics = metrics.to(device)
            pred_fields, pred_metrics = model(inputs)
            loss = field_weight * mse(pred_fields, fields) + metric_weight * mse(pred_metrics, metrics)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, fields, metrics in val_loader:
                inputs = inputs.to(device)
                fields = fields.to(device)
                metrics = metrics.to(device)
                pred_fields, pred_metrics = model(inputs)
                v_loss = field_weight * mse(pred_fields, fields) + metric_weight * mse(pred_metrics, metrics)
                val_losses.append(float(v_loss.detach().cpu()))

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        scheduler.step(mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        append_csv(
            history_path,
            {
                "epoch": epoch,
                "train_loss": mean_train,
                "val_loss": mean_val,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = ensure_parent(config["model_path"])
    checkpoint = {
        "model_state": best_state if best_state else model.cpu().state_dict(),
        "input_mean": torch.as_tensor(dataset.input_mean),
        "input_std": torch.as_tensor(dataset.input_std),
        "field_mean": torch.as_tensor(dataset.field_mean),
        "field_std": torch.as_tensor(dataset.field_std),
        "height": height,
        "width": width,
        "model_type": model_type,
        "validation_loss": best_val_loss if best_state else float(np.mean(val_losses)),
    }
    torch.save(checkpoint, model_path)
    write_json(
        log_dir / "surrogate_summary.json",
        {
            "run_id": run_id,
            "model_path": str(model_path),
            "dataset_path": str(config["dataset_path"]),
            "model_type": model_type,
            "epochs": epochs,
            "validation_loss": checkpoint["validation_loss"],
        },
    )
    return model_path
