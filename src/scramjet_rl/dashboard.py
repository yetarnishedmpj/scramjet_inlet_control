from __future__ import annotations

import os
import webbrowser
from pathlib import Path


def dashboard_path() -> Path:
    return Path(__file__).resolve().parents[2] / "apps" / "inlet_dashboard" / "index.html"


def open_dashboard() -> Path:
    path = dashboard_path()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except AttributeError:
        webbrowser.open(path.as_uri())
    return path
