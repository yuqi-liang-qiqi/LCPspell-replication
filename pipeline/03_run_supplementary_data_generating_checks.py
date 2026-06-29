"""
Run supplementary data-generating checks (event-based + small perturbation).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_BUNDLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUNDLE_DIR))

from pipeline.supplementary_dg_checks_schema import validate_supplementary_dg_checks_results
from simulation import __version__ as SIMULATION_VERSION
from simulation.supplementary_dg_checks.extension_experiments import (
    EVENT_TIMING_EARLY_RANGE,
    EVENT_TIMING_LATE_RANGE,
    INTER_EVENT_LONG_GAP,
    INTER_EVENT_SHORT_GAP,
    SupplementaryDgChecksRunner,
    supplementary_dg_checks_metadata_extra,
)
from simulation.config import SimulationConfig
from simulation.run_metadata import build_run_metadata


def _to_json_stats(stats: dict) -> dict:
    return {"mean": float(stats["mean"]), "std": float(stats["std"])}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp_path.replace(path)


def _save_results(
    results: dict,
    output_dir: str,
    *,
    overwrite: bool = False,
    filename: str = "results.json",
) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / filename
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Pass --overwrite to replace it."
        )
    as_json = {
        strand: {method: _to_json_stats(s) for method, s in strand_res.items()}
        for strand, strand_res in results.items()
    }
    _atomic_write_json(out, as_json)
    return out


def run_supplementary_data_generating_checks(*, overwrite: bool = False) -> dict:
    t0 = time.time()
    output_dir = _BUNDLE_DIR / "results" / "supplementary_data_generating_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    if results_path.exists() and not overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Pass --overwrite to replace it."
        )

    print("=" * 80)
    print("Supplementary data-generating checks: event-based + small perturbation")
    print("=" * 80)

    config = SimulationConfig(
        total_length=20,
        n_states=8,
        state_labels=list("abcdefgh"),
        n_sequences_per_group=2000,
        n_replications=30,
        random_seed=42,
        output_dir=str(output_dir),
        verbose=False,
    )
    runner = SupplementaryDgChecksRunner(config=config)
    results = {}

    print("\n[1/5] event-order strand ...")
    results["event_order"] = runner.run_event_order()
    print("[2/5] event-timing strand ...")
    results["event_timing"] = runner.run_event_timing(
        early_range=EVENT_TIMING_EARLY_RANGE,
        late_range=EVENT_TIMING_LATE_RANGE,
    )
    print("[3/5] inter-event-duration strand ...")
    results["event_inter_duration"] = runner.run_event_inter_duration(
        short_gap=INTER_EVENT_SHORT_GAP,
        long_gap=INTER_EVENT_LONG_GAP,
    )
    print("[4/5] small perturbation (token) strand ...")
    results["small_perturbation_token"] = runner.run_small_perturbation_token()
    print("[5/5] small perturbation (boundary) strand ...")
    results["small_perturbation_boundary"] = runner.run_small_perturbation_boundary()

    validate_supplementary_dg_checks_results(results)

    run_metadata = build_run_metadata(
        simulation_version=SIMULATION_VERSION,
        bundle_dir=_BUNDLE_DIR,
        extra={
            "experiment": "supplementary_data_generating_checks",
            "random_seed": config.random_seed,
            "n_sequences_per_group": config.n_sequences_per_group,
            "n_replications": config.n_replications,
            **supplementary_dg_checks_metadata_extra(),
        },
    )
    _atomic_write_json(output_dir / "run_metadata.json", run_metadata)
    _atomic_write_json(output_dir / "config.json", config.to_dict())
    out = _save_results(results, output_dir=config.output_dir, overwrite=overwrite)

    elapsed = time.time() - t0
    print(f"\nSaved supplementary data-generating-check results: {out}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run supplementary data-generating checks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing results.json in the output directory.",
    )
    args = parser.parse_args()
    run_supplementary_data_generating_checks(overwrite=args.overwrite)
