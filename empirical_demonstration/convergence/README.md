# Convergence empirical demonstration (mvad)

Northern Ireland **school-to-work** trajectories (`mvad`, \citep{mcvicar2002predicting}): a
complement to the pairfam **divergence** illustration in `../divergence/`.

Many individuals begin in different early states (school, training, joblessness) but
later move into similar activity states (employment, further education). Suffix-based
**RLCP** and **RLCPspell** therefore provide a natural convergence-oriented contrast
to the prefix-based **LCP** / **LCPspell** analysis on pairfam.

## Methods

Hamming, OM, OMspell, OMspellRS, **RLCP**, and **RLCPspell** (`expcost` 0.5 and 2).

Settings mirror the divergence batch: `indel=1`, `sm='CONSTANT'`; observation window
$T=70$ months (Sep.93--Jun.99); main outputs use **raw distances** (`norm='none'`).

## Folder layout

| Path | Content |
|------|---------|
| `*.pdf` (this folder) | **Main text**: weighted PAM (`PAMonce`) index plots |
| `ward/` | **Appendix**: Ward linkage (`ward_d`) index plots |
| `ward/auto_norm/` | Optional replication output with `norm='auto'` (Ward only; not shown in the paper) |

## Run

Main-text PAM results (default):

```bash
cd lcpspell-paper-replication/empirical_demonstration/convergence
python3 mvad_school_to_work_cluster_batch.py
```

Appendix Ward linkage (already generated under `ward/`):

```bash
python3 mvad_school_to_work_cluster_batch.py --clustering-method ward
```

Optional normalized distances (Ward; not cited in the paper):

```bash
python3 mvad_school_to_work_cluster_batch.py --norm auto --clustering-method ward
```

Single method smoke test:

```bash
python3 mvad_school_to_work_cluster_batch.py --method RLCPspell --expcosts 0.5
```

Rebuild Sequenzo after C++ changes:

```bash
pip install -e /path/to/Sequenzo
```

## Outputs

One sequence-index PDF per configuration (four-cluster solution by default), e.g.
`RLCPspell-expcost-0.5.pdf`, `RLCP.pdf`, `HAM.pdf`.

## Interpretation (expected contrast)

- **Whole-trajectory measures** (Hamming, OM, OMspell, OMspellRS) group individuals
  by broad profiles over the full 70-month window.
- **RLCP** emphasizes how long trajectories share the same **terminal calendar-time**
  states; clusters should differ in when heterogeneous early pathways align on
  common late states (e.g.\ employment).
- **RLCPspell** compares **terminal spell order**; it is less sensitive to exact
  month-to-month timing of late transitions and can separate recurrent spell-order
  patterns at the end of trajectories that RLCP does not isolate as clearly.

Compare visually with `../divergence/` (pairfam, LCP / LCPspell) to see how prefix-
versus suffix-based measures highlight different substantive features.
