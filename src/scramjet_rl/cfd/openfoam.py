from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenFOAMCheck:
    ok: bool
    solver: str
    message: str


def check_openfoam(solver: str = "rhoCentralFoam") -> OpenFOAMCheck:
    solver_path = shutil.which(solver)
    if solver_path is None:
        return OpenFOAMCheck(
            ok=False,
            solver=solver,
            message=f"{solver} was not found on PATH. Install/source OpenFOAM before running CFD cases.",
        )
    try:
        result = subprocess.run(
            [solver, "-help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except OSError as exc:
        return OpenFOAMCheck(False, solver, f"{solver} exists at {solver_path}, but failed: {exc}")
    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0] if output else "solver responded"
    return OpenFOAMCheck(True, solver, f"{solver} found at {solver_path}: {detail}")
