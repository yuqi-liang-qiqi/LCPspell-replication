# Appendix illustration distances & sequence data

Numeric worked examples and **reconstructible sequence data** for the paper's illustration figures.

## Contents

| File | Purpose |
|------|---------|
| [`ten_sequences.py`](ten_sequences.py) | Canonical DataFrame / `SequenceData` for all ten main-text sequences |
| [`illustration_sequences.ipynb`](illustration_sequences.ipynb) | Interactive rebuild of Figure~1 index plot + appendix subsets |
| [`test_appendix_sequences.py`](test_appendix_sequences.py) | Regression tests for appendix distance values |

## Main-text figure (`fig:ten-sequences`)

Ten sequences over six years; states **E** (employed), **U** (unemployed), **I** (inactive).

| ID | State sequence |
|----|----------------|
| 1 | `U U U U U U` |
| 2 | `E E E E E E` |
| 3 | `U U E E U U` |
| 4 | `E E U U E E` |
| 5 | `E U U U U U` |
| 6 | `U E E E E E` |
| 7 | `E E U E E I` |
| 8 | `E U E U E U` |
| 9 | `E E E U E E` |
| 10 | `E U U U E E` |

Paper PDF: `../../index_plot.pdf` (repo root). Regenerated plot: `figures/ten_sequences_index.pdf`.

## Appendix distance pairs

| IDs | Role |
|-----|------|
| 9, 10 | Same spell order E→U→E, different durations (`app:om-omspell-illustration`, `app:lcp-variants`) |
| 1, 2 | Single-spell U vs E (different-state substitution) |

Time window: 6 positions; `tau = 6` for reference-scaled measures.

## Requirements

- Python 3.10+
- `sequenzo` (same package as main simulations)
- `pytest` (tests); `jupyter` (notebook)

## Quick start

From this folder:

```bash
# Notebook: DataFrame → SequenceData → index plot
jupyter notebook illustration_sequences.ipynb

# Or non-interactive plot export
python3 -c "from pathlib import Path; Path('figures').mkdir(exist_ok=True); \
from ten_sequences import plot_ten_sequences_index; plot_ten_sequences_index('figures/ten_sequences_index.pdf')"

# Appendix distance regression tests
python3 -m pytest test_appendix_sequences.py -v
```

## Appendix labels in `main.tex`

- `app:om-omspell-illustration` — OM, OMspell, OMspellRS on sequences 9–10 and 1–2
- `app:lcp-variants` — LCP family on the same pairs
