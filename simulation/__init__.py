"""
LCPspell paper replication — simulation engine (main.tex Section 3).
"""

from .config import SimulationConfig, EXPCOST_MIN, EXPCOST_MAX
from .sequence_generator import SequenceGenerator
from .distance_computer import DistanceComputer
from .framework import SimulationFramework
from .evaluation import (
    compute_pseudo_r2,
    compute_disc_pseudo_r2,
    disc_from_chance_corrected,
    expected_disc_baseline,
    evaluate_simulation_strand,
    aggregate_replications,
    compute_directional_pair_scores,
    compute_early_late_directional_scores,
    aggregate_directional_replications,
)
from .method_labels import method_display_name

__version__ = "1.4.0"

__all__ = [
    "SimulationConfig",
    "EXPCOST_MIN",
    "EXPCOST_MAX",
    "SequenceGenerator",
    "DistanceComputer",
    "SimulationFramework",
    "compute_pseudo_r2",
    "compute_disc_pseudo_r2",
    "disc_from_chance_corrected",
    "expected_disc_baseline",
    "evaluate_simulation_strand",
    "aggregate_replications",
    "compute_directional_pair_scores",
    "compute_early_late_directional_scores",
    "aggregate_directional_replications",
    "method_display_name",
]
