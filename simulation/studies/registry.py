"""
Paper simulation registry: Study 1 (group-level) vs Study 2 (pair-level).

Single source of truth for JSON keys, panel labels (A--J), and evaluation metrics.
Used by run_main.py and the figure pipeline.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

StudyId = Literal["study1", "study2"]
MetricId = Literal["chance_corrected_pseudo_r2", "normalized_paired_contrast"]

METRIC_CHANCE_CORRECTED_PSEUDO_R2: MetricId = "chance_corrected_pseudo_r2"
METRIC_NORMALIZED_PAIRED_CONTRAST: MetricId = "normalized_paired_contrast"

METRIC_YLABEL = {
    METRIC_CHANCE_CORRECTED_PSEUDO_R2: "Chance-corrected pseudo-$R^2$",
    METRIC_NORMALIZED_PAIRED_CONTRAST: "Normalized paired contrast $S_{\\mathrm{pair}}$",
}

METRIC_SHORT_YLABEL = {
    METRIC_CHANCE_CORRECTED_PSEUDO_R2: "Chance-corrected pseudo-R²",
    METRIC_NORMALIZED_PAIRED_CONTRAST: "Normalized paired contrast",
}


class PanelSpec(TypedDict):
    panel: str
    study: StudyId
    strand_number: int
    strand_label: str
    short_title: str
    json_key: str
    metric: MetricId
    sequencing_pattern: str | None


# Panels A--J in figure order (main.tex Figure sensitivity profiles).
PANEL_SPECS: list[PanelSpec] = [
    {
        "panel": "A",
        "study": "study1",
        "strand_number": 1,
        "strand_label": "Timing Sensitivity",
        "short_title": "Timing",
        "json_key": "timing",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": None,
    },
    {
        "panel": "B",
        "study": "study1",
        "strand_number": 2,
        "strand_label": "Sequencing Sensitivity (Complete Reversal)",
        "short_title": "Sequencing (Complete Reversal)",
        "json_key": "sequencing",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": "complete_reversal",
    },
    {
        "panel": "C",
        "study": "study1",
        "strand_number": 2,
        "strand_label": "Sequencing Sensitivity (Local Permutation)",
        "short_title": "Sequencing (Local Permutation)",
        "json_key": "sequencing",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": "local_permutation",
    },
    {
        "panel": "D",
        "study": "study1",
        "strand_number": 2,
        "strand_label": "Sequencing Sensitivity (Early Swap)",
        "short_title": "Sequencing (Early Swap)",
        "json_key": "sequencing",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": "early_swap",
    },
    {
        "panel": "E",
        "study": "study1",
        "strand_number": 2,
        "strand_label": "Sequencing Sensitivity (Late Swap)",
        "short_title": "Sequencing (Late Swap)",
        "json_key": "sequencing",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": "late_swap",
    },
    {
        "panel": "F",
        "study": "study1",
        "strand_number": 3,
        "strand_label": "Duration Sensitivity",
        "short_title": "Duration",
        "json_key": "duration",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "sequencing_pattern": None,
    },
    {
        "panel": "G",
        "study": "study2",
        "strand_number": 4,
        "strand_label": "Calendar-Time Divergence (Early vs Late)",
        "short_title": "Divergence: Early vs Late (Calendar)",
        "json_key": "sensitivity_divergence_calendar",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "sequencing_pattern": None,
    },
    {
        "panel": "H",
        "study": "study2",
        "strand_number": 5,
        "strand_label": "Calendar-Time Convergence (Early vs Late)",
        "short_title": "Convergence: Early vs Late (Calendar)",
        "json_key": "sensitivity_convergence_calendar",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "sequencing_pattern": None,
    },
    {
        "panel": "I",
        "study": "study2",
        "strand_number": 6,
        "strand_label": "Spell-Order Divergence (Early vs Late)",
        "short_title": "Divergence: Early vs Late (Spell Order)",
        "json_key": "sensitivity_divergence_spell_order",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "sequencing_pattern": None,
    },
    {
        "panel": "J",
        "study": "study2",
        "strand_number": 7,
        "strand_label": "Spell-Order Convergence (Early vs Late)",
        "short_title": "Convergence: Early vs Late (Spell Order)",
        "json_key": "sensitivity_convergence_spell_order",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "sequencing_pattern": None,
    },
]

PANEL_ORDER = [spec["strand_label"] for spec in PANEL_SPECS]
EXPECTED_PANEL_LETTERS = [spec["panel"] for spec in PANEL_SPECS]
PANEL_TITLES = {spec["strand_label"]: spec["short_title"] for spec in PANEL_SPECS}
PANEL_METRICS = {spec["strand_label"]: spec["metric"] for spec in PANEL_SPECS}
PANEL_STUDY = {spec["strand_label"]: spec["study"] for spec in PANEL_SPECS}

STUDY1_PANEL_LABELS = [s["strand_label"] for s in PANEL_SPECS if s["study"] == "study1"]
STUDY2_PANEL_LABELS = [s["strand_label"] for s in PANEL_SPECS if s["study"] == "study2"]

STUDY1_JSON_KEYS = frozenset({"timing", "sequencing", "duration"})
STUDY2_JSON_KEYS = frozenset(
    spec["json_key"] for spec in PANEL_SPECS if spec["study"] == "study2"
)
EXPECTED_RESULTS_JSON_KEYS = STUDY1_JSON_KEYS | STUDY2_JSON_KEYS

SEQUENCING_PATTERN_KEYS = frozenset(
    spec["sequencing_pattern"]
    for spec in PANEL_SPECS
    if spec["sequencing_pattern"] is not None
)

# Strands with small values: adaptive y-axis in bar plots.
ZOOMED_STRANDS = frozenset(
    {
        "Timing Sensitivity",
        *STUDY2_PANEL_LABELS,
    }
)

# Study 2 runner registry for run_main.py: (json_key, framework_method_name, log label)
STUDY2_EXPERIMENTS: list[tuple[str, str, str]] = [
    (
        "sensitivity_divergence_calendar",
        "run_sensitivity_divergence_calendar",
        "Strand 4 — calendar-time divergence (early vs late)",
    ),
    (
        "sensitivity_convergence_calendar",
        "run_sensitivity_convergence_calendar",
        "Strand 5 — calendar-time convergence (early vs late)",
    ),
    (
        "sensitivity_divergence_spell_order",
        "run_sensitivity_divergence_spell_order",
        "Strand 6 — spell-order divergence (early vs late)",
    ),
    (
        "sensitivity_convergence_spell_order",
        "run_sensitivity_convergence_spell_order",
        "Strand 7 — spell-order convergence (early vs late)",
    ),
]

STUDY1_SEQUENCING_PATTERNS: list[tuple[str, list[str], list[str], str]] = [
    (
        "complete_reversal",
        ["a", "b", "c", "d", "e"],
        ["e", "d", "c", "b", "a"],
        "complete reversal",
    ),
    (
        "local_permutation",
        ["a", "b", "c", "a"],
        ["a", "c", "b", "a"],
        "local permutation",
    ),
    (
        "early_swap",
        ["a", "b", "c", "d", "e"],
        ["b", "a", "c", "d", "e"],
        "early swap",
    ),
    (
        "late_swap",
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "e", "d"],
        "late swap",
    ),
]

# Separate result directories per normalization mode.
RESULTS_DIRS = {
    "none": "main_raw",
    "builtin": "robustness_builtin",
    "elzinga": "robustness_elzinga",
}

# Non-default directional duration modes write to separate result trees.
DIRECTIONAL_DURATION_RESULTS_DIRS = {
    "shared_spell_mismatch_compensated": "robustness_directional_duration_mismatch",
    "background_noise": "robustness_directional_background_noise",
}


def normalize_directional_duration_mode(mode: str) -> str:
    if mode == "shared_spell_mismatch":
        return "shared_spell_mismatch_compensated"
    return mode


def resolve_results_subdirectory(
    norm_mode: str,
    *,
    directional_duration_mode: str = "matched",
) -> str:
    """Map norm mode + duration mode to a results/ subdirectory name."""
    duration_mode = normalize_directional_duration_mode(directional_duration_mode)
    if duration_mode != "matched" and norm_mode != "none":
        raise ValueError(
            "Non-default directional duration modes are currently defined "
            "only for --norm none."
        )
    if duration_mode != "matched":
        if duration_mode not in DIRECTIONAL_DURATION_RESULTS_DIRS:
            raise ValueError(f"Unknown directional_duration_mode: {duration_mode!r}")
        return DIRECTIONAL_DURATION_RESULTS_DIRS[duration_mode]

    return RESULTS_DIRS[norm_mode]

# Paper strand table (strands 1--7) for documentation cross-references.
STRAND_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "paper_name": "Timing",
        "study": "study1",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "json_key": "timing",
        "panel_label": "Timing Sensitivity",
    },
    2: {
        "paper_name": "Sequencing",
        "study": "study1",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "json_key": "sequencing",
        "sequencing_patterns": {
            spec["sequencing_pattern"]: spec["strand_label"]
            for spec in PANEL_SPECS
            if spec["sequencing_pattern"] is not None
        },
    },
    3: {
        "paper_name": "Duration",
        "study": "study1",
        "metric": METRIC_CHANCE_CORRECTED_PSEUDO_R2,
        "json_key": "duration",
        "panel_label": "Duration Sensitivity",
    },
    4: {
        "paper_name": "Divergence (calendar time)",
        "study": "study2",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "json_key": "sensitivity_divergence_calendar",
        "panel_label": "Calendar-Time Divergence (Early vs Late)",
        "contrast": "early_vs_late",
    },
    5: {
        "paper_name": "Convergence (calendar time)",
        "study": "study2",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "json_key": "sensitivity_convergence_calendar",
        "panel_label": "Calendar-Time Convergence (Early vs Late)",
        "contrast": "early_vs_late",
    },
    6: {
        "paper_name": "Divergence (spell order)",
        "study": "study2",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "json_key": "sensitivity_divergence_spell_order",
        "panel_label": "Spell-Order Divergence (Early vs Late)",
        "contrast": "early_vs_late",
    },
    7: {
        "paper_name": "Convergence (spell order)",
        "study": "study2",
        "metric": METRIC_NORMALIZED_PAIRED_CONTRAST,
        "json_key": "sensitivity_convergence_spell_order",
        "panel_label": "Spell-Order Convergence (Early vs Late)",
        "contrast": "early_vs_late",
    },
}

LOCALIZED_ROBUSTNESS_KEYS = (
    "localized_calendar_early_vs_late",
    "localized_spell_order_early_vs_late",
)
