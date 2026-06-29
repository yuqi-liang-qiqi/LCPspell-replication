"""
Paper sensitivity simulation (main.tex Section 3).

Two studies, one entry point:
  Study 1 (Strands 1--3, Panels A--F): group-level pseudo-R²
  Study 2 (Strands 4--7, Panels G--J): normalized paired contrast

Usage:
  python3 simulation/run_main.py --norm none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_BUNDLE_DIR = Path(__file__).resolve().parent.parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

from simulation import __version__ as SIMULATION_VERSION
from simulation.config import SimulationConfig
from simulation.framework import SimulationFramework
from simulation.run_metadata import build_run_metadata
from simulation.studies.registry import (
    EXPECTED_RESULTS_JSON_KEYS,
    METRIC_CHANCE_CORRECTED_PSEUDO_R2,
    METRIC_NORMALIZED_PAIRED_CONTRAST,
    SEQUENCING_PATTERN_KEYS,
    STUDY1_SEQUENCING_PATTERNS,
    STUDY2_EXPERIMENTS,
    STUDY2_JSON_KEYS,
    normalize_directional_duration_mode,
    resolve_results_subdirectory,
)

# Study 1 (3 steps) + Study 2 (4 panels) + save
TOTAL_STEPS = 8

STUDY1_STAT_KEYS = frozenset({"mean", "std"})
STUDY2_STAT_KEYS = frozenset({"mean", "std", "win_mean", "win_std"})


def _abort_on_strand_error(exc: Exception) -> None:
    import traceback

    traceback.print_exc()
    raise exc


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp_path.replace(path)


def _validate_method_stats(stats: dict, *, expected_keys: frozenset[str], context: str) -> None:
    actual = set(stats)
    if actual != expected_keys:
        missing = expected_keys - actual
        extra = actual - expected_keys
        raise RuntimeError(
            f"Invalid stats schema for {context}. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for key, value in stats.items():
        if not np.isfinite(float(value)):
            raise RuntimeError(
                f"Non-finite stat for {context}.{key}: {value!r}"
            )


def _validate_all_results(
    all_results: dict,
    *,
    expected_methods: set[str],
) -> None:
    actual_keys = set(all_results)
    expected_keys = EXPECTED_RESULTS_JSON_KEYS
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            "Refusing to save results.json with invalid top-level keys. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )

    sequencing = all_results.get("sequencing")
    if not isinstance(sequencing, dict):
        raise RuntimeError(
            "Refusing to save incomplete results.json: 'sequencing' is missing or invalid."
        )
    actual_patterns = set(sequencing)
    missing_patterns = SEQUENCING_PATTERN_KEYS - actual_patterns
    extra_patterns = actual_patterns - SEQUENCING_PATTERN_KEYS
    if missing_patterns or extra_patterns:
        raise RuntimeError(
            "Refusing to save incomplete results.json. "
            f"Missing sequencing patterns={sorted(missing_patterns)}, "
            f"extra={sorted(extra_patterns)}"
        )

    for strand_key, results in all_results.items():
        if strand_key == "sequencing":
            for pattern_key, pattern_results in results.items():
                actual_methods = set(pattern_results)
                if actual_methods != expected_methods:
                    raise RuntimeError(
                        f"Inconsistent methods in sequencing.{pattern_key}. "
                        f"Missing={sorted(expected_methods - actual_methods)}, "
                        f"extra={sorted(actual_methods - expected_methods)}"
                    )
                for method, stats in pattern_results.items():
                    _validate_method_stats(
                        stats,
                        expected_keys=STUDY1_STAT_KEYS,
                        context=f"sequencing.{pattern_key}.{method}",
                    )
            continue

        actual_methods = set(results)
        if actual_methods != expected_methods:
            raise RuntimeError(
                f"Inconsistent methods in {strand_key}. "
                f"Missing={sorted(expected_methods - actual_methods)}, "
                f"extra={sorted(actual_methods - expected_methods)}"
            )
        expected = STUDY2_STAT_KEYS if strand_key in STUDY2_JSON_KEYS else STUDY1_STAT_KEYS
        for method, stats in results.items():
            _validate_method_stats(
                stats,
                expected_keys=expected,
                context=f"{strand_key}.{method}",
            )


def _progress(step_done: int, current_name: Optional[str], started_at: Optional[float] = None):
    done_str = f"Overall progress: {step_done}/{TOTAL_STEPS} completed"
    if current_name:
        print(f"\n>>> {done_str} | Current: {current_name}")
    else:
        print(f"\n>>> {done_str}")
    if started_at is not None:
        elapsed = time.time() - started_at
        print(f"    Elapsed for this step: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    sys.stdout.flush()


def run_paper_experiment(
    norm_mode: str = "none",
    directional_duration_mode: str = "matched",
    overwrite: bool = False,
):
    script_start = time.time()

    if norm_mode == "none":
        distance_norm = "none"
    elif norm_mode == "elzinga":
        distance_norm = "ElzingaStuder"
    elif norm_mode == "builtin":
        distance_norm = "auto"
    else:
        raise ValueError(f"Unsupported norm_mode: {norm_mode!r}. Use none, builtin, or elzinga.")

    directional_duration_mode = normalize_directional_duration_mode(
        directional_duration_mode
    )
    results_subdir = resolve_results_subdirectory(
        norm_mode,
        directional_duration_mode=directional_duration_mode,
    )
    out_dir = _BUNDLE_DIR / "results" / results_subdir
    results_path = out_dir / "results.json"
    if results_path.exists() and not overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Pass --overwrite to replace it."
        )

    n_sequences_per_group = 2000
    n_replications = 30

    print("=" * 80)
    print("LCPspell paper simulation")
    print("=" * 80)
    print("Study 1 (A–F): timing, sequencing, duration → chance-corrected pseudo-R²")
    print("Study 2 (G–J): early-vs-late directional contrasts → normalized paired contrast")
    print(
        f"Sample size: {n_sequences_per_group} per group (Study 1) "
        f"or matched early/late draws/replication (Study 2); {n_replications} replications"
    )
    print("Expected runtime: several hours (hardware-dependent)")
    print("=" * 80)
    _progress(0, "Initialize configuration")

    config = SimulationConfig(
        total_length=20,
        n_states=5,
        n_sequences_per_group=n_sequences_per_group,
        n_replications=n_replications,
        random_seed=42,
        output_dir=str(out_dir),
        verbose=False,
        distance_norm=distance_norm,
        directional_duration_mode=directional_duration_mode,
    )

    methods = config.get_distance_methods()
    print(f"\nNormalization: {norm_mode} (distance_norm={distance_norm})")
    print(f"Distance methods: {len(methods)}")
    print(f"Directional duration mode: {directional_duration_mode}")
    print()

    framework = SimulationFramework(config)
    all_results: dict = {}
    steps_done = 0

    # ------------------------------------------------------------------
    # Study 1 — Strand 1: Timing
    # ------------------------------------------------------------------
    t1_start = time.time()
    _progress(steps_done, f"Study 1 / Strand 1 timing ({n_replications} reps)")
    print("\n[Study 1] Strand 1 — timing (focal-state occupancy)")
    print("-" * 80)
    try:
        all_results["timing"] = framework.run_timing_sensitivity(
            focal_state="c",
            time_group1=6,
            time_group2_range=(8, 14),
            n_replications=n_replications,
        )
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        _abort_on_strand_error(e)
    steps_done += 1
    _progress(steps_done, None, t1_start)

    # ------------------------------------------------------------------
    # Study 1 — Strand 2: Sequencing (4 patterns)
    # ------------------------------------------------------------------
    t2_start = time.time()
    _progress(steps_done, f"Study 1 / Strand 2 sequencing ({n_replications} reps each)")
    print("\n[Study 1] Strand 2 — sequencing (4 patterns)")
    print("-" * 80)

    sequencing_results = {}
    for i, (pattern_key, dss1, dss2, pattern_name) in enumerate(
        STUDY1_SEQUENCING_PATTERNS, 1
    ):
        print(f"\n  [{i}/4] {pattern_name}: {dss1} vs {dss2}")
        sys.stdout.flush()
        try:
            sequencing_results[pattern_key] = framework.run_sequencing_sensitivity(
                dss_group1=dss1,
                dss_group2=dss2,
                n_replications=n_replications,
            )
            print(f"  OK [{i}/4] {pattern_name}")
        except Exception as e:
            print(f"  ERROR: {e}")
            _abort_on_strand_error(e)
        sys.stdout.flush()

    all_results["sequencing"] = sequencing_results
    steps_done += 1
    _progress(steps_done, None, t2_start)

    # ------------------------------------------------------------------
    # Study 1 — Strand 3: Duration
    # ------------------------------------------------------------------
    t3_start = time.time()
    _progress(steps_done, f"Study 1 / Strand 3 duration ({n_replications} reps)")
    print("\n[Study 1] Strand 3 — duration")
    print("-" * 80)
    try:
        all_results["duration"] = framework.run_duration_sensitivity(
            focal_state="b",
            duration_group1=4,
            duration_group2=14,
            dss_templates=[["a", "b", "c"], ["c", "b", "a"]],
            n_replications=n_replications,
        )
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        _abort_on_strand_error(e)
    steps_done += 1
    _progress(steps_done, None, t3_start)

    # ------------------------------------------------------------------
    # Study 2 — Strands 4--7 (4 early-vs-late panels)
    # ------------------------------------------------------------------
    for panel_idx, (result_key, method_name, description) in enumerate(
        STUDY2_EXPERIMENTS, start=1
    ):
        t_start = time.time()
        _progress(
            steps_done,
            f"Study 2 / Panel {chr(70 + panel_idx)} {description} ({n_replications} reps)",
        )
        print(f"\n[Study 2] {description}")
        print("-" * 80)
        print("Matched early-vs-late pairs · metric: normalized aggregate paired contrast")
        runner = getattr(framework, method_name)
        try:
            all_results[result_key] = runner(n_replications=n_replications)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            _abort_on_strand_error(e)
        steps_done += 1
        _progress(steps_done, None, t_start)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    t_save_start = time.time()
    _progress(steps_done, "Save results")
    print("\n" + "=" * 80)
    print("Save results")
    print("=" * 80)

    expected_methods = set(methods)
    _validate_all_results(all_results, expected_methods=expected_methods)

    os.makedirs(config.output_dir, exist_ok=True)
    results_path = Path(config.output_dir) / "results.json"
    if results_path.exists() and not overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Pass --overwrite to replace it."
        )

    def to_json_stats(stats: dict) -> dict:
        out = {"mean": float(stats["mean"]), "std": float(stats["std"])}
        if "win_mean" in stats:
            out["win_mean"] = float(stats["win_mean"])
            out["win_std"] = float(stats["win_std"])
        return out

    json_results = {}
    for strand, results in all_results.items():
        if strand == "sequencing":
            json_results[strand] = {
                pattern: {method: to_json_stats(s) for method, s in pattern_results.items()}
                for pattern, pattern_results in results.items()
            }
        else:
            json_results[strand] = {
                method: to_json_stats(s) for method, s in results.items()
            }

    run_metadata = build_run_metadata(
        simulation_version=SIMULATION_VERSION,
        bundle_dir=_BUNDLE_DIR,
        extra={
            "studies": {
                "study1": {
                    "panels": "A-F",
                    "strands": [1, 2, 3],
                    "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
                    "unit": "sequences per group",
                },
                "study2": {
                    "panels": "G-J",
                    "strands": [4, 5, 6, 7],
                    "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
                    "unit": "matched early/late pairs per replication",
                    "estimand": "early_vs_late_directional_contrast",
                },
            },
            "directional_duration_mode": config.directional_duration_mode,
            "directional_pair_chunk_size": config.directional_pair_chunk_size,
            "normalization_mode": norm_mode,
            "distance_norm": distance_norm,
            "random_seed": config.random_seed,
            "n_sequences_per_group": n_sequences_per_group,
            "n_replications": n_replications,
            "expcost_values": config.expcost_values,
            "results_subdirectory": results_subdir,
        },
    )

    metadata_path = Path(config.output_dir) / "run_metadata.json"
    config_path = Path(config.output_dir) / "config.json"
    _atomic_write_json(metadata_path, run_metadata)
    _atomic_write_json(config_path, config.to_dict())
    _atomic_write_json(results_path, json_results)

    print(f"OK results:  {results_path}")
    print(f"OK metadata: {metadata_path}")
    print(f"OK config:   {config_path}")

    steps_done += 1
    _progress(steps_done, None, t_save_start)
    total_elapsed = time.time() - script_start
    print(
        f"\n>>> All done. Total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min)"
    )

    print(
        f"\nNext: python3 lcpspell-paper-replication/pipeline/01_aggregate_json_to_csv.py "
        f"--from-norm {norm_mode}"
    )
    print(
        "      python3 lcpspell-paper-replication/pipeline/02_plot_sensitivity_profiles.py"
    )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper simulation (Study 1 + Study 2).")
    parser.add_argument(
        "--norm",
        choices=["none", "builtin", "elzinga"],
        default="none",
    )
    parser.add_argument(
        "--directional-duration-mode",
        choices=[
            "matched",
            "shared_spell_mismatch_compensated",
            "shared_spell_mismatch",
            "background_noise",
        ],
        default="matched",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing results.json in the output directory.",
    )
    args = parser.parse_args()

    try:
        run_paper_experiment(
            norm_mode=args.norm,
            directional_duration_mode=args.directional_duration_mode,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
