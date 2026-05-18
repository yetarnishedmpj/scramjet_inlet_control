# AI Context Handoff

## Project

Repository path:

```text
C:\Users\mahar\OneDrive\Documents\scramjet-surrogate-rl
```

Git remote:

```text
https://github.com/yetarnishedmpj/scramjet_inlet_control.git
```

Purpose:

```text
Surrogate-assisted reinforcement learning scaffold for scramjet inlet ramp control.
```

Core flow:

```text
OpenFOAM / imported data
  -> HDF5 dataset
  -> PyTorch surrogate model
  -> Gymnasium environment
  -> Stable Baselines3 SAC/PPO agent
  -> CFD replay manifest
```

## Current State

Implemented:

- Synthetic CFD-like data generator.
- HDF5 dataset schema validation.
- Direct real-data import from `.npy`, `.npz`, `.csv`, `.txt`.
- Fixed train/validation/test split generation.
- PyTorch surrogate models:
  - `cnn`
  - `resnet`
  - `unet`
  - `metric_mlp`
- Surrogate evaluation:
  - scalar MAE metrics
  - prediction sample plots
  - error histograms
  - metric parity plots
  - report folder generation
- Gymnasium inlet-control environment.
- Stable Baselines3 SAC/PPO training.
- Baseline Mach-scheduled controller.
- Ensemble uncertainty support for surrogate guardrails.
- OpenFOAM case templating, manifest creation, postprocessing hooks, and solver check.
- Browser-based interactive inlet dashboard.
- Professor-facing `REPORT.md`.

Important limitation:

```text
The included OpenFOAM template is structural only. It is not a validated scramjet CFD case.
The default dataset is synthetic and should not be used for aerospace conclusions.
```

## Key Commands

Use CMD:

```bat
cd C:\Users\mahar\OneDrive\Documents\scramjet-surrogate-rl
.venv\Scripts\activate
set PYTHONPATH=src
```

Run tests:

```bat
python -m pytest
```

Generate synthetic data:

```bat
scramjet generate-data --config configs\data_synthetic.yaml
scramjet validate-dataset data\processed\synthetic_inlet.h5
```

Train surrogate:

```bat
scramjet train-surrogate --config configs\surrogate.yaml
```

Evaluate surrogate:

```bat
scramjet evaluate-surrogate --dataset data\processed\synthetic_inlet.h5 --checkpoint models\surrogate_cnn.pt --plot-output outputs\surrogate_eval.png
```

Generate surrogate report:

```bat
scramjet surrogate-report --dataset data\processed\synthetic_inlet.h5 --checkpoint models\surrogate_cnn.pt --output-dir outputs\reports\surrogate_cnn
```

Open interactive dashboard:

```bat
scramjet open-dashboard
```

Train RL agent:

```bat
scramjet train-agent --config configs\rl_sac.yaml
```

Evaluate baseline:

```bat
scramjet evaluate-baseline --config configs\rl_sac.yaml --episodes 5 --rollout-csv outputs\baseline_rollouts.csv
```

Evaluate RL policy:

```bat
scramjet evaluate-agent --config configs\rl_sac.yaml --policy-path models\sac_scramjet_env.zip --rollout-csv outputs\policy_rollouts.csv
```

## Important Files

```text
README.md
REPORT.md
apps/inlet_dashboard/index.html
apps/inlet_dashboard/styles.css
apps/inlet_dashboard/app.js
src/scramjet_rl/cli.py
src/scramjet_rl/data/synthetic.py
src/scramjet_rl/data/importers.py
src/scramjet_rl/data/schema.py
src/scramjet_rl/data/splits.py
src/scramjet_rl/surrogate/models.py
src/scramjet_rl/surrogate/train.py
src/scramjet_rl/surrogate/evaluate.py
src/scramjet_rl/envs/scramjet_inlet_env.py
src/scramjet_rl/rl/train.py
src/scramjet_rl/rl/evaluate.py
src/scramjet_rl/rl/baseline.py
src/scramjet_rl/cfd/
tests/
```

## Data Schema

All real or synthetic data must be converted to:

```text
inputs:      float32 [N, 4] -> mach, altitude_m, density_kg_m3, ramp_angle_deg
pressure:    float32 [N, H, W]
temperature: float32 [N, H, W]
metrics:     float32 [N, 4] -> shock_error, pressure_recovery, unstart_probability, efficiency
```

## Suggested Next Work

Highest-value next tasks:

1. Replace synthetic fields with real OpenFOAM or experimental data.
2. Build a validated `rhoCentralFoam` scramjet inlet case with a real 2D mesh.
3. Validate surrogate accuracy on held-out CFD cases.
4. Compare SAC/PPO policies against the baseline controller.
5. Replay learned trajectories in OpenFOAM and compare with surrogate predictions.
6. Improve the dashboard so it can load real rollout CSVs and trained-policy results.

## Current GitHub Blocker

Earlier push attempts failed with:

```text
remote: Repository not found.
fatal: repository 'https://github.com/yetarnishedmpj/scramjet_inlet_control.git/' not found
```

Likely causes:

- repository does not exist,
- repository is private and local credentials lack access,
- owner/name is wrong,
- GitHub CLI is not installed/authenticated.

Once access is fixed:

```bat
git push -u origin main
```
