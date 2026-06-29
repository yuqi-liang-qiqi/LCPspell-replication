"""
Load simulation results.json into a tidy DataFrame for figure pipeline.

Columns align with main.tex Study 1 (pseudo-R²) and Study 2 (paired contrast).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.config import SimulationConfig
from simulation.studies.registry import (
    EXPECTED_PANEL_LETTERS,
    EXPECTED_RESULTS_JSON_KEYS,
    METRIC_CHANCE_CORRECTED_PSEUDO_R2,
    METRIC_NORMALIZED_PAIRED_CONTRAST,
    PANEL_SPECS,
    SEQUENCING_PATTERN_KEYS,
    STUDY2_JSON_KEYS,
)

PANEL_TO_STRAND = {spec["panel"]: spec["strand_label"] for spec in PANEL_SPECS}
PANEL_TO_METRIC = {spec["panel"]: spec["metric"] for spec in PANEL_SPECS}


def convert_method_name(method_raw: str) -> tuple[str, str]:
    """Map JSON method key to display label and raw key."""
    if method_raw == "LCP":
        return "LCP", "LCP"
    if method_raw == "RLCP":
        return "RLCP", "RLCP"
    if method_raw == "LCPmst":
        return "LCPmst", "LCPmst"
    if method_raw == "RLCPmst":
        return "RLCPmst", "RLCPmst"
    if method_raw == "OM":
        return "OM", "OM"
    if method_raw == "HAM":
        return "Hamming", "HAM"

    for prefix, label_prefix in (
        ("LCPspell_expcost_", "LCPspell(expcost="),
        ("RLCPspell_expcost_", "RLCPspell(expcost="),
        ("OMspell_expcost_", "OMspell(expcost="),
        ("OMspellRS_expcost_", "OMspellRS(expcost="),
    ):
        if method_raw.startswith(prefix):
            param_value = method_raw.replace(prefix, "")
            return f"{label_prefix}{param_value})", method_raw

    return method_raw, method_raw


def _validate_results_schema(data: dict[str, Any]) -> None:
    actual_top_level = set(data)
    missing = EXPECTED_RESULTS_JSON_KEYS - actual_top_level
    extra = actual_top_level - EXPECTED_RESULTS_JSON_KEYS
    if missing or extra:
        raise ValueError(
            "Invalid results.json schema. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )

    sequencing = data["sequencing"]
    if not isinstance(sequencing, dict):
        raise ValueError("Invalid results.json: 'sequencing' must be an object.")
    actual_patterns = set(sequencing)
    if actual_patterns != SEQUENCING_PATTERN_KEYS:
        raise ValueError(
            "Invalid sequencing patterns. "
            f"Expected={sorted(SEQUENCING_PATTERN_KEYS)}, "
            f"got={sorted(actual_patterns)}"
        )

    expected_methods = set(SimulationConfig().get_distance_methods())
    study2_stat_keys = frozenset({"mean", "std", "win_mean", "win_std"})
    study1_stat_keys = frozenset({"mean", "std"})

    for strand_key in EXPECTED_RESULTS_JSON_KEYS:
        if strand_key == "sequencing":
            for pattern_key in SEQUENCING_PATTERN_KEYS:
                pattern_data = sequencing[pattern_key]
                if not isinstance(pattern_data, dict):
                    raise ValueError(
                        f"sequencing.{pattern_key} must be an object."
                    )
                _validate_method_block(
                    pattern_data,
                    expected_methods=expected_methods,
                    expected_stat_keys=study1_stat_keys,
                    context=f"sequencing.{pattern_key}",
                )
            continue

        strand_data = data[strand_key]
        if not isinstance(strand_data, dict):
            raise ValueError(f"{strand_key} must be an object.")
        stat_keys = (
            study2_stat_keys if strand_key in STUDY2_JSON_KEYS else study1_stat_keys
        )
        _validate_method_block(
            strand_data,
            expected_methods=expected_methods,
            expected_stat_keys=stat_keys,
            context=strand_key,
        )


def _validate_method_block(
    block: dict[str, Any],
    *,
    expected_methods: set[str],
    expected_stat_keys: frozenset[str],
    context: str,
) -> None:
    actual_methods = set(block)
    if actual_methods != expected_methods:
        missing = expected_methods - actual_methods
        extra = actual_methods - expected_methods
        raise ValueError(
            f"Incomplete method set in {context}. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for method, stats in block.items():
        if not isinstance(stats, dict):
            raise ValueError(f"{context}.{method} stats must be an object.")
        actual_keys = set(stats)
        if actual_keys != expected_stat_keys:
            raise ValueError(
                f"Invalid stats schema for {context}.{method}. "
                f"Expected={sorted(expected_stat_keys)}, got={sorted(actual_keys)}"
            )
        for key, value in stats.items():
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite stat {context}.{method}.{key}: {value!r}"
                )


def _append_row(
    rows: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    method_raw: str,
    values: dict[str, float],
) -> None:
    method_display, method_raw_out = convert_method_name(method_raw)
    rows.append(
        {
            "panel": spec["panel"],
            "study": spec["study"],
            "strand_number": spec["strand_number"],
            "strand": spec["strand_label"],
            "strand_key": spec["json_key"],
            "sequencing_pattern": spec["sequencing_pattern"],
            "metric": spec["metric"],
            "method": method_display,
            "method_raw": method_raw_out,
            "mean_score": float(values["mean"]),
            "sd_score": float(values["std"]),
            "win_mean": float(values.get("win_mean", float("nan"))),
            "win_sd": float(values.get("win_std", float("nan"))),
            # Legacy column names (Study 1 pseudo-R² scripts).
            "mean_r2": float(values["mean"]),
            "sd_r2": float(values["std"]),
        }
    )


def validate_figure_frame(df: pd.DataFrame) -> None:
    """Validate aggregated CSV or JSON-derived DataFrame for figure pipeline."""
    required_columns = {
        "panel",
        "strand",
        "metric",
        "method",
        "method_raw",
        "mean_score",
        "sd_score",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    if df.duplicated(["panel", "method_raw"]).any():
        raise ValueError("Duplicate panel-method rows found in input data.")

    mean_scores = df["mean_score"].astype(float)
    sd_scores = df["sd_score"].astype(float)
    if not np.isfinite(mean_scores).all():
        raise ValueError("mean_score contains non-finite values.")
    if not np.isfinite(sd_scores).all():
        raise ValueError("sd_score contains non-finite values.")

    actual_panels = df["panel"].drop_duplicates().tolist()
    if actual_panels != EXPECTED_PANEL_LETTERS:
        raise ValueError(
            f"Expected panels {EXPECTED_PANEL_LETTERS}, got {actual_panels}. "
            "Regenerate the CSV from a current results.json."
        )

    if df.groupby("panel")["strand"].nunique().max() != 1:
        raise ValueError("A panel maps to multiple strand labels.")
    if df.groupby("panel")["metric"].nunique().max() != 1:
        raise ValueError("A panel maps to multiple metrics.")

    actual_panel_to_strand = (
        df[["panel", "strand"]]
        .drop_duplicates()
        .set_index("panel")["strand"]
        .to_dict()
    )
    if actual_panel_to_strand != PANEL_TO_STRAND:
        raise ValueError(
            "Panel-to-strand mapping differs from registry. "
            f"Expected={PANEL_TO_STRAND}, got={actual_panel_to_strand}"
        )

    actual_panel_to_metric = (
        df[["panel", "metric"]]
        .drop_duplicates()
        .set_index("panel")["metric"]
        .to_dict()
    )
    if actual_panel_to_metric != PANEL_TO_METRIC:
        raise ValueError(
            "Panel-to-metric mapping differs from registry. "
            f"Expected={PANEL_TO_METRIC}, got={actual_panel_to_metric}"
        )

    expected_methods = set(SimulationConfig().get_distance_methods())
    for panel in EXPECTED_PANEL_LETTERS:
        panel_methods = set(df.loc[df["panel"] == panel, "method_raw"])
        if panel_methods != expected_methods:
            missing = expected_methods - panel_methods
            extra = panel_methods - expected_methods
            raise ValueError(
                f"Incomplete method set for panel {panel}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}"
            )

    study2_rows = df[df["metric"] == METRIC_NORMALIZED_PAIRED_CONTRAST]
    if not study2_rows.empty:
        for col in ("win_mean", "win_sd"):
            if col not in df.columns:
                raise ValueError(f"Study 2 rows require column {col!r}.")
            values = study2_rows[col].astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{col} contains non-finite values for Study 2 rows.")


def load_results_json(json_path: str | Path) -> pd.DataFrame:
    """Parse results.json into one row per (panel, method)."""
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    _validate_results_schema(data)

    rows: list[dict[str, Any]] = []

    for spec in PANEL_SPECS:
        if spec["json_key"] == "sequencing":
            if spec["sequencing_pattern"] is None:
                continue
            pattern_data = data["sequencing"][spec["sequencing_pattern"]]
            for method_raw, values in pattern_data.items():
                _append_row(rows, spec=spec, method_raw=method_raw, values=values)
            continue

        strand_data = data[spec["json_key"]]
        for method_raw, values in strand_data.items():
            _append_row(rows, spec=spec, method_raw=method_raw, values=values)

    if not rows:
        raise ValueError(f"No panel rows parsed from {json_path}")

    df = pd.DataFrame(rows)
    validate_figure_frame(df)
    return df


def study1_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Study 1 rows (chance-corrected pseudo-R² panels)."""
    return df[df["metric"] == METRIC_CHANCE_CORRECTED_PSEUDO_R2].copy()
