"""Schema validation for supplementary data-generating-check results."""

from __future__ import annotations

import numpy as np

from simulation.config import SimulationConfig

STRAND_LABELS = {
    "event_order": "Event-order sensitivity",
    "event_timing": "Event-timing sensitivity",
    "event_inter_duration": "Inter-event-duration sensitivity",
    "small_perturbation_token": "Small perturbation (single-token)",
    "small_perturbation_boundary": "Small perturbation (boundary shift)",
}

EXPECTED_STRAND_KEYS = frozenset(STRAND_LABELS)
STAT_KEYS = frozenset({"mean", "std"})
EXPECTED_METHODS = frozenset(SimulationConfig().get_distance_methods())


def validate_supplementary_dg_checks_results(results: dict) -> None:
    actual = set(results)
    missing = EXPECTED_STRAND_KEYS - actual
    extra = actual - EXPECTED_STRAND_KEYS
    if missing or extra:
        raise ValueError(
            "Invalid supplementary data-generating-check results schema. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for strand, method_stats in results.items():
        if not isinstance(method_stats, dict) or not method_stats:
            raise ValueError(f"{strand} must contain method statistics.")

        actual_methods = set(method_stats)
        missing_methods = EXPECTED_METHODS - actual_methods
        extra_methods = actual_methods - EXPECTED_METHODS
        if missing_methods or extra_methods:
            raise ValueError(
                f"Incomplete method set for {strand}. "
                f"Missing={sorted(missing_methods)}, extra={sorted(extra_methods)}"
            )

        for method, stats in method_stats.items():
            if not isinstance(stats, dict):
                raise ValueError(f"{strand}.{method} stats must be an object.")
            if set(stats) != STAT_KEYS:
                raise ValueError(
                    f"Invalid stats schema for {strand}.{method}: {sorted(stats)}"
                )
            for key, value in stats.items():
                if not np.isfinite(float(value)):
                    raise ValueError(
                        f"Non-finite value for {strand}.{method}.{key}: {value!r}"
                    )
