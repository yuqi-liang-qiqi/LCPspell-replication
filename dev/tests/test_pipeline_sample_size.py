"""Tests for R²_disc sample-size resolution in plotting scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUNDLE / "pipeline"))

from _sample_size import (
    ensure_matching_sample_size_source,
    n_total_from_config,
    resolve_n_total,
)


def test_resolve_n_total_from_cli(tmp_path):
    assert resolve_n_total(n_per_group=500, config_path=None) == 1000


def test_resolve_n_total_from_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"n_sequences_per_group": 123}), encoding="utf-8")
    assert resolve_n_total(n_per_group=None, config_path=cfg) == 246
    assert n_total_from_config(cfg) == 246


def test_resolve_n_total_requires_input(tmp_path):
    with pytest.raises(ValueError, match="Pass --n-per-group"):
        resolve_n_total(n_per_group=None, config_path=tmp_path / "missing.json")


def test_n_total_from_config_rejects_fractional_value(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"n_sequences_per_group": 123.7}), encoding="utf-8")
    with pytest.raises(ValueError, match="integer n_sequences_per_group"):
        n_total_from_config(cfg)


def test_custom_input_csv_requires_matching_sample_size_source(tmp_path):
    default_csv = tmp_path / "default.csv"
    custom_csv = tmp_path / "custom.csv"
    default_cfg = tmp_path / "default_config.json"
    default_csv.write_text("x", encoding="utf-8")
    custom_csv.write_text("y", encoding="utf-8")
    default_cfg.write_text(json.dumps({"n_sequences_per_group": 2000}), encoding="utf-8")

    ensure_matching_sample_size_source(
        input_path=default_csv,
        default_input_path=default_csv,
        config_path=default_cfg,
        default_config_path=default_cfg,
        n_per_group=None,
    )

    with pytest.raises(ValueError, match="--config-json or --n-per-group"):
        ensure_matching_sample_size_source(
            input_path=custom_csv,
            default_input_path=default_csv,
            config_path=default_cfg,
            default_config_path=default_cfg,
            n_per_group=None,
        )

    ensure_matching_sample_size_source(
        input_path=custom_csv,
        default_input_path=default_csv,
        config_path=default_cfg,
        default_config_path=default_cfg,
        n_per_group=100,
    )
