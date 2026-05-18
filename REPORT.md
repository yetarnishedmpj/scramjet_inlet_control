# Surrogate-Assisted RL for Scramjet Inlet Control

## Problem Statement

The project investigates adaptive control of a scramjet inlet ramp. The control objective is to keep the inlet near shock-on-lip operation while avoiding unstart and limiting actuator movement.

Training a reinforcement learning agent directly inside a CFD solver is computationally impractical. The project therefore uses a surrogate-assisted architecture.

## Architecture

```text
OpenFOAM / Imported Data
        ↓
HDF5 dataset
        ↓
PyTorch surrogate model
        ↓
Gymnasium environment
        ↓
Stable Baselines3 SAC/PPO agent
        ↓
CFD replay for validation
```

## Data Schema

All data sources are converted into a common HDF5 format:

```text
inputs:      [N, 4] -> mach, altitude_m, density_kg_m3, ramp_angle_deg
pressure:    [N, H, W]
temperature: [N, H, W]
metrics:     [N, 4] -> shock_error, pressure_recovery, unstart_probability, efficiency
```

The current repository includes a synthetic data generator for software validation. Real CFD or experimental data can be imported through `scramjet import-arrays` or `scramjet postprocess-cfd`.

## Surrogate Model

The surrogate maps operating condition and ramp angle to flow-field outputs and scalar performance metrics:

```text
(Mach, altitude, density, ramp angle) -> pressure field, temperature field, metrics
```

Implemented model options:

- `cnn`: compact baseline field predictor
- `resnet`: residual decoder for stronger spatial prediction
- `unet`: U-Net-style model for field reconstruction
- `metric_mlp`: scalar-metric-only predictor for fast reward estimation

The evaluation pipeline reports field errors, metric errors, shock-location error, parity plots, error histograms, and truth/prediction/error field plots.

## Reinforcement Learning Environment

The Gymnasium environment exposes:

- observations: Mach, altitude, density, ramp angle, ramp rate, downsampled pressure/temperature sensors
- action: continuous ramp movement command
- reward: efficiency and pressure recovery reward, actuator movement penalty, unstart penalty
- termination: unstart probability threshold

Actuator lag and rate limits are included to avoid unrealistic instant ramp movement.

## Baseline Controller

A non-RL baseline controller is included:

```text
target_ramp_angle = 6.0 + 1.6 * (Mach - 4.0)
```

This provides a comparison point for SAC and PPO. The RL policy should be evaluated against this baseline before claiming controller improvement.

## Uncertainty Guardrails

The project supports ensemble surrogate prediction. Multiple checkpoints can be loaded and the environment can penalize policy actions in regions where surrogate metric predictions disagree.

This matters because RL agents can exploit surrogate errors if uncertainty is ignored.

## Current Limitations

- The included OpenFOAM case is a structural template, not a validated scramjet simulation.
- The default dataset is synthetic and should not be used for aerospace conclusions.
- Real claims require mesh-validated CFD sweeps, held-out CFD testing, and replay of learned policies in OpenFOAM.
- Shock and unstart metrics must be validated against domain-specific definitions before being used as final reward terms.

## Validation Plan

1. Build and validate a 2D `rhoCentralFoam` inlet case.
2. Run Mach, altitude, and ramp-angle sweeps.
3. Export pressure and temperature fields to the HDF5 schema.
4. Train surrogate models with fixed train/validation/test splits.
5. Compare surrogate predictions against held-out CFD.
6. Train SAC/PPO agents against the surrogate.
7. Compare RL policies against the Mach-scheduled baseline controller.
8. Replay selected RL trajectories in OpenFOAM and compare surrogate predictions with CFD truth.
