"""Centralized path resolution and environment configuration for NucEvo."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = REPO_ROOT / "assets"
BANKS_DIR = ASSETS_DIR / "banks"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
LIBRARY_DIR = REPO_ROOT / "nucevo" / "benchmark" / "nuclear_edu"

DEFAULT_CACHE_ROOT = Path(
    os.environ.get("NUCEVO_CACHE_DIR", str(Path.home() / ".cache" / "nucevo"))
).expanduser()


def resolve_omc_binary() -> Path:
    """Locate the OpenModelica compiler (`omc`) via env override or PATH."""
    override = os.environ.get("NUCEVO_OMC")
    if override:
        return Path(override).expanduser()
    om_env = os.environ.get("NUCEVO_OM_ENV")
    if om_env:
        return Path(om_env).expanduser() / "bin" / "omc"
    found = shutil.which("omc")
    if found:
        return Path(found)
    return Path("omc")


def cache_subdir(name: str) -> Path:
    path = DEFAULT_CACHE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path
