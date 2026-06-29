"""
Canonical panel order and display labels for sensitivity-profile figures.

Re-exports from simulation.studies.registry (single source of truth).
"""

from simulation.studies.registry import (
    EXPECTED_PANEL_LETTERS,
    METRIC_CHANCE_CORRECTED_PSEUDO_R2,
    METRIC_NORMALIZED_PAIRED_CONTRAST,
    METRIC_SHORT_YLABEL,
    METRIC_YLABEL,
    PANEL_METRICS,
    PANEL_ORDER,
    PANEL_SPECS,
    PANEL_STUDY,
    PANEL_TITLES,
    STUDY1_PANEL_LABELS,
    STUDY2_PANEL_LABELS,
    ZOOMED_STRANDS,
)

__all__ = [
    "EXPECTED_PANEL_LETTERS",
    "PANEL_ORDER",
    "PANEL_TITLES",
    "PANEL_SPECS",
    "PANEL_METRICS",
    "PANEL_STUDY",
    "STUDY1_PANEL_LABELS",
    "STUDY2_PANEL_LABELS",
    "ZOOMED_STRANDS",
    "METRIC_CHANCE_CORRECTED_PSEUDO_R2",
    "METRIC_NORMALIZED_PAIRED_CONTRAST",
    "METRIC_YLABEL",
    "METRIC_SHORT_YLABEL",
]
