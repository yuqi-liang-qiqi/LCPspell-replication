"""
Paper simulation studies (main.tex Section 3).

Study 1 — group-level strands (timing, sequencing, duration):
    chance-corrected pseudo-R²

Study 2 — pair-level strands (divergence, convergence):
    normalized paired contrast
"""

from .registry import (
    EXPECTED_RESULTS_JSON_KEYS,
    METRIC_CHANCE_CORRECTED_PSEUDO_R2,
    METRIC_NORMALIZED_PAIRED_CONTRAST,
    PANEL_ORDER,
    PANEL_SPECS,
    RESULTS_DIRS,
    SEQUENCING_PATTERN_KEYS,
    STUDY1_SEQUENCING_PATTERNS,
    STUDY2_EXPERIMENTS,
    STUDY2_JSON_KEYS,
)
from .study1_group_level import GroupLevelStudyMixin
from .study2_directional_pairs import DirectionalPairStudyMixin

__all__ = [
    "GroupLevelStudyMixin",
    "DirectionalPairStudyMixin",
    "PANEL_SPECS",
    "PANEL_ORDER",
    "EXPECTED_RESULTS_JSON_KEYS",
    "SEQUENCING_PATTERN_KEYS",
    "STUDY1_SEQUENCING_PATTERNS",
    "STUDY2_EXPERIMENTS",
    "STUDY2_JSON_KEYS",
    "METRIC_CHANCE_CORRECTED_PSEUDO_R2",
    "METRIC_NORMALIZED_PAIRED_CONTRAST",
    "RESULTS_DIRS",
]
