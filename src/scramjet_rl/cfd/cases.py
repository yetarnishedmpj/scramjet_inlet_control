from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def materialize_cases(
    manifest_path: str | Path,
    template_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    rows = read_manifest(manifest_path)
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_paths: list[Path] = []

    for row in rows:
        case_id = row["case_id"]
        case_dir = output_dir / case_id
        if case_dir.exists():
            if not overwrite:
                case_paths.append(case_dir)
                continue
            shutil.rmtree(case_dir)
        shutil.copytree(template_dir, case_dir)
        _render_placeholders(case_dir, row)
        case_paths.append(case_dir)
    return case_paths


def run_cases(case_dirs: list[str | Path], solver: str = "rhoCentralFoam", dry_run: bool = False) -> None:
    for case_dir in [Path(path) for path in case_dirs]:
        if dry_run:
            print(f"Would run {solver} in {case_dir}")
            continue
        subprocess.run([solver, "-case", str(case_dir)], check=True)


def _render_placeholders(case_dir: Path, values: dict[str, str]) -> None:
    replacements = {
        "{{MACH}}": values["mach"],
        "{{ALTITUDE_M}}": values["altitude_m"],
        "{{RAMP_ANGLE_DEG}}": values["ramp_angle_deg"],
    }
    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token, value in replacements.items():
            text = text.replace(token, value)
        path.write_text(text, encoding="utf-8")
