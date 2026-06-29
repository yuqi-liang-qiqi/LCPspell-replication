"""Resolve group sample sizes for R²_disc affine maps in plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path


def _read_n_per_group(payload: dict, *, config_path: Path) -> int:
    raw = payload.get("n_sequences_per_group")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"Config {config_path} must contain integer n_sequences_per_group."
        )
    if raw < 1:
        raise ValueError(f"n_sequences_per_group must be >= 1, got {raw}.")
    return raw


def n_total_from_config(config_path: Path) -> int:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {config_path} must be a JSON object.")
    return 2 * _read_n_per_group(payload, config_path=config_path)


def resolve_n_total(
    *,
    n_per_group: int | None,
    config_path: Path | None,
) -> int:
    if n_per_group is not None:
        if n_per_group < 1:
            raise ValueError(f"--n-per-group must be >= 1, got {n_per_group}.")
        return 2 * n_per_group
    if config_path is not None and config_path.is_file():
        return n_total_from_config(config_path)
    raise ValueError(
        "Pass --n-per-group or provide --config-json with n_sequences_per_group."
    )


def ensure_matching_sample_size_source(
    *,
    input_path: Path,
    default_input_path: Path,
    config_path: Path,
    default_config_path: Path,
    n_per_group: int | None,
) -> None:
    """Reject custom CSV paired with the default production config."""
    if input_path.resolve() == default_input_path.resolve():
        return
    if n_per_group is not None:
        return
    if config_path.resolve() != default_config_path.resolve():
        return
    raise ValueError(
        "When --input-csv is overridden, also pass --config-json or --n-per-group "
        "so the R²_disc affine map uses the matching sample size."
    )
