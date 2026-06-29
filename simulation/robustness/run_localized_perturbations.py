#!/usr/bin/env python3
"""
Entry point for optional robustness checks (not in main-text Strands 1--7).

Includes:
- LocalizedPerturbationMixin: fixed-window localized perturbation location checks

Archived exploratory PersistentPatternMixin lives under archive/exploratory/ and is
not used for paper results.
"""

from __future__ import annotations

try:
    from ..config import SimulationConfig
    from ..framework import SimulationFramework
    from .localized_perturbation_location import LocalizedPerturbationMixin
except ImportError:
    from config import SimulationConfig
    from framework import SimulationFramework
    from robustness.localized_perturbation_location import LocalizedPerturbationMixin


class RobustnessFramework(LocalizedPerturbationMixin, SimulationFramework):
    """Framework for optional robustness checks outside the main paper pipeline."""


def main() -> None:
    config = SimulationConfig(verbose=True)
    framework = RobustnessFramework(config)
    print(
        "RobustnessFramework ready. Example calls:\n"
        "  framework.run_localized_calendar_early_vs_late()\n"
        "  framework.run_localized_spell_order_early_vs_late()"
    )


if __name__ == "__main__":
    main()
