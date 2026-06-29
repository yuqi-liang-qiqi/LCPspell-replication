"""
@Author  : Yuqi Liang 梁彧祺
@File    : optional_compact_labels_preview.py
@Time    : 20/02/2026 22:00
@Desc    :
Development utility only. Not part of the replication workflow.

Optional two-line x-axis label preview (representative parameters only).
Default input: ``lcpspell-paper-replication/results/main_raw/results.json``.
Set ``RESULTS_JSON`` to override.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

_BUNDLE = Path(__file__).resolve().parents[2]

# Override with full path to a results.json if needed
RESULTS_JSON = os.environ.get(
    "RESULTS_JSON",
    str(_BUNDLE / "results" / "main_raw" / "results.json"),
)

REP_PARAMS = {"0.00", "0.50", "1.00", "2.00"}


def pick_methods(method_stats: dict[str, dict], prefix: str) -> list[tuple[str, float]]:
    out = []
    for key, stats in method_stats.items():
        if key.startswith(prefix):
            p = key.split("_expcost_")[-1]
            if p in REP_PARAMS:
                out.append((p, float(stats["mean"])))
    out.sort(key=lambda x: float(x[0]))
    return out


def build_labels(params: list[str], family_name: str) -> list[str]:
    labels = []
    for i, p in enumerate(params):
        bottom = family_name if i == 0 else ""
        labels.append(f"{p}\n{bottom}")
    return labels


def main():
    source_json = Path(RESULTS_JSON)
    out = _BUNDLE / "figures_preview" / "preview_compact_xticks.png"

    if not source_json.is_file():
        raise FileNotFoundError(
            f"Missing {source_json}. Run simulation/run_main.py --norm none first, "
            "or set RESULTS_JSON to an explicit results.json path."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(source_json.read_text(encoding="utf-8"))
    strand = data["sequencing"]["complete_reversal"]

    lcp = pick_methods(strand, "LCPspell_expcost_")
    rlcp = pick_methods(strand, "RLCPspell_expcost_")
    omn = pick_methods(strand, "OMspellRS_expcost_")

    blocks = [
        ("LCPspell", lcp, "#10B981"),
        ("RLCPspell", rlcp, "#8B5CF6"),
        ("OMspellRS", omn, "#F59E0B"),
    ]

    x_vals, y_vals, labels, colors = [], [], [], []
    x = 0
    for family, vals, color in blocks:
        if not vals:
            continue
        params = [p for p, _ in vals]
        means = [m for _, m in vals]
        block_labels = build_labels(params, family)
        for p_lab, m in zip(block_labels, means):
            x_vals.append(x)
            y_vals.append(m)
            labels.append(p_lab)
            colors.append(color)
            x += 1
        x += 1

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x_vals, y_vals, color=colors, alpha=0.82)
    plt.xticks(x_vals, labels, rotation=90, fontsize=9)
    plt.ylabel("Mean pseudo-R2")
    plt.title("Preview: representative parameters with compact two-line labels")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=240)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
