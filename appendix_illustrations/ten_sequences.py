"""
Canonical illustration sequences for main.tex Figure~\\ref{fig:ten-sequences}.

State codes: E = employed, U = unemployed, I = inactive.
Time window: six yearly observations (columns ``1``--``6``).

Appendix distance tests use numeric states E=1, U=2 on subsets {1,2} and {9,10}.
"""

from __future__ import annotations

import pandas as pd

from sequenzo import SequenceData

TIME_COLS = ["1", "2", "3", "4", "5", "6"]
STATES = ["E", "U", "I"]
LABELS = ["Employed", "Unemployed", "Inactive"]

# Sampled from paper figure ``index_plot`` (Employed / Unemployed / Inactive).
FIGURE_COLORS = ["#b898c5", "#bed9f1", "#d1e5d3"]
COLOR_EMPLOYED, COLOR_UNEMPLOYED, COLOR_INACTIVE = FIGURE_COLORS
INDEX_PLOT_KWARGS = {
    "sort_by": "unsorted",
    "sort_by_ids": list(range(1, 11)),
    "show_sequence_ids": True,
    "xlabel": "Year",
    "ylabel": "Sequences",
    "show_title": False,
    "figsize": (7, 5),
    "dpi": 400,
    # Thicker bars + one blank row => gap ~30% of bar height (matches ``index_plot``).
    "sequence_rows": 3,
    "sequence_gap": 1,
}

# Letter-coded rows (main text, fig:ten-sequences).
TEN_SEQUENCES: dict[int, list[str]] = {
    1: ["U", "U", "U", "U", "U", "U"],
    2: ["E", "E", "E", "E", "E", "E"],
    3: ["U", "U", "E", "E", "U", "U"],
    4: ["E", "E", "U", "U", "E", "E"],
    5: ["E", "U", "U", "U", "U", "U"],
    6: ["U", "E", "E", "E", "E", "E"],
    7: ["E", "E", "U", "E", "E", "I"],
    8: ["E", "U", "E", "U", "E", "U"],
    9: ["E", "E", "E", "U", "E", "E"],
    10: ["E", "U", "U", "U", "E", "E"],
}

# Numeric encoding for appendix distance regression tests.
STATE_E = 1
STATE_U = 2
TAU = 6.0


def build_ten_sequences_dataframe() -> pd.DataFrame:
    """Wide state-sequence table with one row per illustration sequence."""
    rows = [
        {"id": seq_id, **dict(zip(TIME_COLS, states))}
        for seq_id, states in TEN_SEQUENCES.items()
    ]
    return pd.DataFrame(rows)


def plot_ten_sequences_index(
    output_path: str | None = None,
    *,
    sequence_data: SequenceData | None = None,
    show: bool = True,
):
    """Render the main-text ten-sequence index plot with paper colors and row gaps."""
    import matplotlib.pyplot as plt
    from sequenzo import plot_sequence_index

    seqdata = sequence_data or build_ten_sequence_data()
    kwargs = dict(INDEX_PLOT_KWARGS)
    if output_path is not None:
        kwargs["save_as"] = output_path
    plot_sequence_index(seqdata=seqdata, show=show, **kwargs)
    if output_path is not None:
        plt.close("all")


def build_ten_sequence_data(
    *,
    custom_colors: list[str] | None = FIGURE_COLORS,
) -> SequenceData:
    """``SequenceData`` for all ten main-text illustration sequences."""
    df = build_ten_sequences_dataframe()
    kwargs: dict = {
        "time": TIME_COLS,
        "id_col": "id",
        "states": STATES,
        "labels": LABELS,
    }
    if custom_colors is not None:
        kwargs["custom_colors"] = custom_colors
    return SequenceData(df, **kwargs)


def _numeric_subset(rows: list[list[int]], ids: list[int]) -> SequenceData:
    raw = pd.DataFrame(rows, columns=TIME_COLS)
    raw.insert(0, "id", ids)
    return SequenceData(
        raw,
        time=TIME_COLS,
        id_col="id",
        states=[STATE_E, STATE_U],
        labels=["Employed", "Unemployed"],
        custom_colors=[COLOR_EMPLOYED, COLOR_UNEMPLOYED],
    )


def build_sequences_9_10() -> SequenceData:
    """Seq. 9: E E E U E E; Seq. 10: E U U U E E (appendix distances)."""
    return _numeric_subset(
        [
            [STATE_E, STATE_E, STATE_E, STATE_U, STATE_E, STATE_E],
            [STATE_E, STATE_U, STATE_U, STATE_U, STATE_E, STATE_E],
        ],
        [9, 10],
    )


def build_sequences_1_2() -> SequenceData:
    """Seq. 1: U x6; Seq. 2: E x6 (appendix distances)."""
    return _numeric_subset(
        [
            [STATE_U] * 6,
            [STATE_E] * 6,
        ],
        [1, 2],
    )
