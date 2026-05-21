"""Export a trained PyTorch surrogate model to ONNX format.

Usage (CLI):
    scramjet export-onnx --checkpoint models/surrogate_unet.pt \\
                         --output    models/surrogate_unet.onnx

The exported ONNX graph accepts a single float32 input tensor of shape
``[batch, 4]`` (mach, altitude_m, density_kg_m3, ramp_angle_deg) and
produces two outputs:
    - ``fields``  – shape ``[batch, 2, H, W]``  (pressure / temperature)
    - ``metrics`` – shape ``[batch, 4]``         (shock_error, pressure_recovery,
                                                   unstart_probability, efficiency)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def export_surrogate_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    input_dim: int = 4,
    opset_version: int = 17,
    validate: bool = True,
) -> Path:
    """Load a ``.pt`` surrogate checkpoint and export it to ONNX.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``torch.save``-d dict with keys ``model_state_dict``,
        ``model_type``, ``height``, and ``width``.
    output_path:
        Destination ``.onnx`` file path (parent dirs created automatically).
    input_dim:
        Dimensionality of the scalar input vector. Default 4.
    opset_version:
        ONNX opset to target. Default 17.
    validate:
        If True, run a forward pass through both the original PyTorch model and
        the exported ONNX model and compare outputs with ``np.allclose``.

    Returns
    -------
    Path
        Resolved path to the written ``.onnx`` file.
    """
    try:
        import onnx  # noqa: F401 – only needed at validation time
    except ModuleNotFoundError:
        validate = False  # silently skip validation if onnx is not installed

    from scramjet_rl.surrogate.models import build_model

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type: str = checkpoint.get("model_type", "cnn")
    height: int = int(checkpoint.get("height", 32))
    width: int = int(checkpoint.get("width", 64))

    model = build_model(model_type, height=height, width=width)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Dummy input ───────────────────────────────────────────────────────────
    dummy_input = torch.zeros(1, input_dim)

    # ── Export ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["inputs"],
        output_names=["fields", "metrics"],
        dynamic_axes={
            "inputs":  {0: "batch"},
            "fields":  {0: "batch"},
            "metrics": {0: "batch"},
        },
    )

    # ── Optional validation ───────────────────────────────────────────────────
    if validate:
        _validate_onnx(model, output_path, dummy_input)

    return output_path.resolve()


def _validate_onnx(
    pt_model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Compare PyTorch and ONNX Runtime outputs on the dummy input."""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("onnxruntime not installed – skipping ONNX output validation.")
        return

    # PyTorch reference outputs
    with torch.no_grad():
        pt_fields, pt_metrics = pt_model(dummy_input)
    pt_fields_np = pt_fields.numpy()
    pt_metrics_np = pt_metrics.numpy()

    # ONNX Runtime outputs
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
    ort_fields, ort_metrics = sess.run(None, ort_inputs)

    fields_ok = np.allclose(pt_fields_np, ort_fields, rtol=rtol, atol=atol)
    metrics_ok = np.allclose(pt_metrics_np, ort_metrics, rtol=rtol, atol=atol)

    if fields_ok and metrics_ok:
        print("ONNX validation passed – PyTorch and ONNX outputs match.")
    else:
        max_field_diff = float(np.abs(pt_fields_np - ort_fields).max())
        max_metric_diff = float(np.abs(pt_metrics_np - ort_metrics).max())
        print(
            f"ONNX validation WARNING – max field diff: {max_field_diff:.6g}, "
            f"max metric diff: {max_metric_diff:.6g}"
        )
