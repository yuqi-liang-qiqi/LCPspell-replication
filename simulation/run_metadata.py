"""Collect reproducibility metadata for simulation runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_commit(repo_root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def build_run_metadata(
    *,
    simulation_version: str,
    bundle_dir: Path,
    extra: dict[str, Any],
) -> dict[str, Any]:
    repo_root = bundle_dir.parent
    return {
        "simulation_engine_version": simulation_version,
        "sequenzo_version": _package_version("sequenzo"),
        "numpy_version": _package_version("numpy"),
        "pandas_version": _package_version("pandas"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit_simulation_bundle": _git_commit(bundle_dir),
        "git_dirty_simulation_bundle": _git_dirty(bundle_dir),
        "git_commit_repo_root": _git_commit(repo_root),
        "git_dirty_repo_root": _git_dirty(repo_root),
        **extra,
    }
