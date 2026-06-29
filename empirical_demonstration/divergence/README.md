# Divergence empirical demonstration (Section 6.1)

Pairfam family-by-month clustering: Hamming, OM, OMspell, OMspellRS, **LCP**, and LCPspell (`expcost` 0.5 and 2).

Convergence counterpart (mvad, RLCP / RLCPspell): see `../convergence/`.

## Folder layout

| Path | Content |
|------|---------|
| `*.pdf` (this folder) | **Main text**: weighted PAM (`PAMonce`) index plots |
| `ward/` | **Appendix**: Ward linkage (`ward_d`) index plots |
| `ward/auto_norm/` | Optional replication output with `norm='auto'` (Ward only; not shown in the paper) |

## Run

Main-text PAM results (default; `norm='none'`):

```bash
python3 pairfam_family_by_month_cluster_batch.py
```

Appendix Ward linkage (already generated under `ward/`):

```bash
python3 pairfam_family_by_month_cluster_batch.py --clustering-method ward
```

Optional normalized distances (Ward, method-specific `norm='auto'`; not cited in the paper):

```bash
python3 pairfam_family_by_month_cluster_batch.py --norm auto --clustering-method ward
```

Optional calibration run (OMspellRS `expcost=132` ≈ OMspell `expcost=0.5` on 264-month window):

```bash
python3 pairfam_family_by_month_cluster_batch.py --method OMspellRS --expcosts 132 --norm none
```

Outputs one sequence-index PDF per configuration. Rebuild Sequenzo after C++ changes:

```bash
pip install -e /path/to/Sequenzo
```

The batch script uses a **headless** matplotlib backend (`Agg`). Main-text partitions use weighted **PAM** (`PAMonce` via `sequenzo.clustering.KMedoids`); appendix figures use classic **Ward** linkage (`ward_d`). OM on ~1k monthly sequences is slow; expect several minutes per spell-based method.
