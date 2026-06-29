"""Integration tests for supplementary data-generating-check runner I/O."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUNDLE))

from simulation.supplementary_dg_checks.extension_experiments import (
    supplementary_dg_checks_metadata_extra,
)
from simulation.config import SimulationConfig


def _load_supplementary_checks_module():
    path = _BUNDLE / "pipeline" / "03_run_supplementary_data_generating_checks.py"
    spec = importlib.util.spec_from_file_location(
        "run_supplementary_data_generating_checks", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_supplementary_check_results(config: SimulationConfig) -> dict:
    methods = config.get_distance_methods()
    stat = {"mean": 0.05, "std": 0.01}
    return {
        strand: {method: {"mean": stat["mean"], "std": stat["std"]} for method in methods}
        for strand in (
            "event_order",
            "event_timing",
            "event_inter_duration",
            "small_perturbation_token",
            "small_perturbation_boundary",
        )
    }


def test_run_supplementary_checks_on_fresh_directory(tmp_path, monkeypatch):
    mod = _load_supplementary_checks_module()
    monkeypatch.setattr(mod, "_BUNDLE_DIR", tmp_path)

    config = SimulationConfig(
        total_length=20,
        n_states=8,
        state_labels=list("abcdefgh"),
        n_sequences_per_group=20,
        n_replications=1,
        output_dir=str(tmp_path / "results" / "supplementary_data_generating_checks"),
        verbose=False,
    )
    fake = _minimal_supplementary_check_results(config)

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run_event_order(self, *args, **kwargs):
            return fake["event_order"]

        def run_event_timing(self, *args, **kwargs):
            return fake["event_timing"]

        def run_event_inter_duration(self, *args, **kwargs):
            return fake["event_inter_duration"]

        def run_small_perturbation_token(self, *args, **kwargs):
            return fake["small_perturbation_token"]

        def run_small_perturbation_boundary(self, *args, **kwargs):
            return fake["small_perturbation_boundary"]

    monkeypatch.setattr(mod, "SupplementaryDgChecksRunner", FakeRunner)
    monkeypatch.setattr(
        mod,
        "SimulationConfig",
        lambda **kwargs: config,
    )

    output_dir = tmp_path / "results" / "supplementary_data_generating_checks"
    assert not output_dir.exists()

    mod.run_supplementary_data_generating_checks(overwrite=True)

    assert (output_dir / "run_metadata.json").is_file()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "results.json").is_file()


def test_supplementary_checks_second_run_without_overwrite_raises(tmp_path, monkeypatch):
    mod = _load_supplementary_checks_module()
    output_dir = tmp_path / "results" / "supplementary_data_generating_checks"
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "_BUNDLE_DIR", tmp_path)

    with pytest.raises(FileExistsError, match="results.json"):
        mod.run_supplementary_data_generating_checks(overwrite=False)


def test_supplementary_checks_metadata_matches_source_constants():
    meta = supplementary_dg_checks_metadata_extra()
    assert meta["event_timing_early_range_python_index"] == [3, 6]
    assert meta["event_timing_late_range_python_index"] == [11, 14]
    assert meta["inter_event_short_gap"] == [1, 3]
    assert meta["inter_event_long_gap"] == [6, 8]
    assert meta["strand_seed_offsets"]["supplementary_event_timing"] == 2000


def test_supplementary_checks_raw_plot_reads_config_sample_size(tmp_path):
    sys.path.insert(0, str(_BUNDLE / "pipeline"))
    from _sample_size import resolve_n_total

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"n_sequences_per_group": 250}), encoding="utf-8")
    assert resolve_n_total(n_per_group=None, config_path=cfg) == 500
