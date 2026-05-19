# Scramjet Surrogate RL

This project scaffolds a practical pipeline for adaptive scramjet inlet ramp control:

1. Generate or import CFD fields.
2. Train a fast surrogate model for pressure and temperature fields.
3. Wrap the surrogate in a Gymnasium environment.
4. Train a continuous-control RL policy.

The initial implementation uses synthetic CFD-like fields so the software stack can be tested without OpenFOAM. Replace the synthetic generator with postprocessed `rhoCentralFoam` outputs once the CFD cases are available.

## Layout

```text
configs/                 YAML configs for data, surrogate, and RL runs
cfd/templates/           OpenFOAM case template placeholder
src/scramjet_rl/cfd/     CFD sweep and postprocessing interfaces
src/scramjet_rl/data/    Dataset generation and loading
src/scramjet_rl/surrogate/  PyTorch surrogate models (with CBAM/GroupNorm) and training
src/scramjet_rl/envs/    Gymnasium environment (configurable rewards)
src/scramjet_rl/rl/      Stable Baselines3 training/evaluation scripts
tests/                   Smoke tests
```

## Quick Start

Use Python 3.10-3.12 for the ML/RL stack. The local machine currently has Python 3.14 as default, which is fine for lightweight smoke tests but not the safest target for PyTorch and Stable Baselines3.

```powershell
cd C:\Users\mahar\OneDrive\Documents\scramjet-surrogate-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts\generate_synthetic_dataset.py --config configs\data_synthetic.yaml
python scripts\validate_dataset.py data\processed\synthetic_inlet.h5
python scripts\train_surrogate.py --config configs\surrogate.yaml
python scripts\evaluate_surrogate.py --dataset data\processed\synthetic_inlet.h5 --checkpoint models\surrogate_cnn.pt
python scripts\train_agent.py --config configs\rl_sac.yaml
python scripts\evaluate_agent.py --config configs\rl_sac.yaml --policy-path models\sac_scramjet_env.zip
pytest
```

The same workflow is available through the installed CLI:

```powershell
scramjet generate-data --config configs\data_synthetic.yaml
scramjet validate-dataset data\processed\synthetic_inlet.h5
scramjet train-surrogate --config configs\surrogate.yaml
scramjet train-agent --config configs\rl_sac.yaml
```

For a professor-facing technical summary, see [REPORT.md](REPORT.md).

## OpenFOAM Workflow

Create a CFD sweep manifest:

```powershell
python scripts\make_cfd_manifest.py --output cfd\sweep_manifest.csv --count 256
```

Place a validated OpenFOAM case template at:

```text
cfd/templates/openfoam_case/
```

Any text files in that template may use these placeholders:

```text
{{MACH}}
{{ALTITUDE_M}}
{{RAMP_ANGLE_DEG}}
```

Then materialize and run cases:

```powershell
python scripts\check_openfoam.py --solver rhoCentralFoam
python scripts\materialize_cfd_cases.py --manifest cfd\sweep_manifest.csv --template-dir cfd\templates\openfoam_case --output-dir data\raw_openfoam\cases
python scripts\run_cfd_cases.py --cases-dir data\raw_openfoam\cases --solver rhoCentralFoam
```

After sampling each OpenFOAM result to CSV files named `case_00000.csv`, `case_00001.csv`, etc. with columns `x,y,pressure,temperature`, build the ML dataset:

```powershell
python scripts\postprocess_openfoam_samples.py --manifest cfd\sweep_manifest.csv --sampled-dir data\raw_openfoam\sampled --output data\processed\openfoam_inlet.h5
python scripts\validate_dataset.py data\processed\openfoam_inlet.h5
```

The included `cfd/templates/openfoam_case` is a structural template, not a validated scramjet simulation. A real run still needs a checked mesh in `constant/polyMesh`, physically correct freestream values, and verified boundary patches.

To replay an RL policy trajectory through CFD case generation:

```powershell
python scripts\evaluate_agent.py --config configs\rl_sac.yaml --policy-path models\sac_scramjet_env.zip --rollout-csv outputs\policy_rollouts.csv
python scripts\replay_rollout_to_cfd_manifest.py --rollout-csv outputs\policy_rollouts.csv --output cfd\replay_manifest.csv
python scripts\materialize_cfd_cases.py --manifest cfd\replay_manifest.csv --template-dir cfd\templates\openfoam_case --output-dir data\raw_openfoam\replay_cases
```

## Visualization

```powershell
python scripts\plot_dataset_sample.py --dataset data\processed\synthetic_inlet.h5 --index 0 --output outputs\sample_fields.png
```

## Interactive Inlet Dashboard

The project includes a browser-based dashboard for explaining how inlet ramp angle, Mach number, altitude, shock position, pressure recovery, efficiency, and unstart risk interact. It features a modern Cyber-Aerospace aesthetic with dynamic, high-fidelity canvas rendering of oblique shocks and pressure fields.

Run:

```powershell
scramjet open-dashboard
```

or open the HTML file directly:

```text
apps/inlet_dashboard/index.html
```

Dashboard modes:

- `Manual`: the user directly controls ramp angle.
- `Baseline`: the ramp tracks a Mach-scheduled target angle.
- `RL Policy`: a demonstrator controller reacts to shock error and unstart risk.

This dashboard is an explanatory visualization. It uses the same simplified relationships as the synthetic environment, not a validated CFD solver.

## Importing Real Data

The synthetic dataset is only for pipeline testing. For actual research use, import CFD or experimental data into the same HDF5 schema:

```text
inputs:      float32 [N, 4] -> mach, altitude_m, density_kg_m3, ramp_angle_deg
pressure:    float32 [N, H, W]
temperature: float32 [N, H, W]
metrics:     float32 [N, 4] -> shock_error, pressure_recovery, unstart_probability, efficiency
```

After creating the HDF5 file, validate it:

```powershell
scramjet validate-dataset data\processed\openfoam_inlet.h5
```

Then train on the real dataset by creating a config such as `configs/surrogate_real.yaml`:

```yaml
dataset_path: data/processed/openfoam_inlet.h5
model_path: models/surrogate_real_unet.pt
model_type: unet
epochs: 20
batch_size: 16
learning_rate: 0.0005
validation_fraction: 0.15
seed: 7
field_loss_weight: 1.0
metric_loss_weight: 0.25
log_dir: outputs/experiments
```

Run:

```powershell
scramjet train-surrogate --config configs\surrogate_real.yaml
scramjet evaluate-surrogate --dataset data\processed\openfoam_inlet.h5 --checkpoint models\surrogate_real_unet.pt --plot-output outputs\real_surrogate_eval.png
scramjet surrogate-report --dataset data\processed\openfoam_inlet.h5 --checkpoint models\surrogate_real_unet.pt --output-dir outputs\reports\real_surrogate_unet
```

If OpenFOAM sampled outputs are available as CSV files named by case, use this input format:

```text
x,y,pressure,temperature
```

Create a manifest:

```csv
case_id,mach,altitude_m,ramp_angle_deg
case_00000,4.5,18000,8.0
case_00001,5.0,20000,9.5
```

Then convert the sampled CSV files into the project dataset format:

```powershell
scramjet postprocess-cfd --manifest cfd\sweep_manifest.csv --sampled-dir data\raw_openfoam\sampled --output data\processed\openfoam_inlet.h5
scramjet validate-dataset data\processed\openfoam_inlet.h5
```

For non-OpenFOAM sources, write the four required arrays directly to HDF5. The ML and RL stages do not depend on OpenFOAM specifically once the HDF5 schema is satisfied.

You can import `.npy`, `.npz`, or CSV arrays directly:

```powershell
scramjet import-arrays --inputs inputs.npy --pressure pressure.npy --temperature temperature.npy --metrics metrics.npy --output data\processed\real_data.h5
scramjet create-splits --dataset data\processed\real_data.h5 --output-dir data\splits\real_data --seed 7
```

## CFD Integration Contract

The surrogate expects HDF5 datasets with:

```text
inputs: float32 [N, 4] -> mach, altitude_m, density_kg_m3, ramp_angle_deg
pressure: float32 [N, H, W]
temperature: float32 [N, H, W]
metrics: float32 [N, 4] -> shock_error, pressure_recovery, unstart_probability, efficiency
```

For real OpenFOAM data, write postprocessed fields into `data/processed/*.h5` using the same schema.

Metric columns are:

```text
shock_error
pressure_recovery
unstart_probability
efficiency
```

## Model Options

The surrogate config supports:

```yaml
model_type: cnn
```

or:

```yaml
model_type: resnet
```

or:

```yaml
model_type: unet
```

or:

```yaml
model_type: metric_mlp
```

`unet` and `resnet` models are enhanced with Convolutional Block Attention Modules (CBAM) and Group Normalization layers to better capture high-frequency boundary layers and shockwaves from CFD data. `metric_mlp` is useful when the RL loop only needs reward metrics and not full fields. `EnsembleSurrogatePredictor` can combine multiple checkpoints and return mean/std estimates for uncertainty-aware analysis.

## RL Options

The agent config supports:

```yaml
algorithm: sac
```

or:

```yaml
algorithm: ppo
```

The environment includes actuator lag, rate limits, movement penalty, shock-efficiency reward, and unstart termination.

Baseline comparison:

```powershell
scramjet evaluate-baseline --config configs\rl_sac.yaml --episodes 5 --rollout-csv outputs\baseline_rollouts.csv
```

Use this before claiming that SAC/PPO improves inlet control.

## Scope

This is an engineering scaffold, not a validated aerospace model. Before using it for technical conclusions, validate:

- mesh independence and CFD solver settings,
- air model and boundary conditions,
- surrogate error near shocks and unstart boundaries,
- RL policy behavior by replaying selected trajectories in OpenFOAM.
