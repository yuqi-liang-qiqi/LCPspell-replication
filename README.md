# Representing Trajectories from State Sequences to Spell Sequences: Measuring Divergence and Convergence with Spell-Based Longest Common Prefixes

Replication materials for the simulation sensitivity analysis, empirical clustering demonstrations, and online-supplement figures described in the manuscript. The sections below follow the paper's headings so you can match the PDF to files in this repository.

---

## How this repository relates to Sequenzo

The distance methods used in the paper ([Optimal Matching](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/OMdistance.cpp), [OMspell](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/OMspellDistance.cpp), [OMspellRS](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/OMspellRSDistance.cpp), [LCP](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/LCPdistance.cpp), [LCPmst](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/LCPmstDistance.cpp), [LCPprod](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/LCPprodDistance.cpp), and [LCPspell](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/dissimilarity_measures/src/LCPspellDistance.cpp)) are implemented and maintained in [the Sequenzo Python package](https://github.com/Liang-Team/Sequenzo/tree/main). The same package also contains several built-in example datasets (such as [pairfam](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/datasets/pairfam_activity_by_month.csv) and [mvad](https://github.com/Liang-Team/Sequenzo/blob/main/sequenzo/datasets/mvad.csv), for further details of provided datasets, please see [here](https://sequenzo.yuqi-liang.tech/en/datasets/CO2-emissions)) used in the appendix index plots. To run any script in this repository, install Sequenzo in your Python environment and import from it. All simulation and empirical code here calls Sequenzo's `get_distance_matrix()`, clustering utilities, and data loaders.

The folder `[dissimilarity_measures_source_code/](dissimilarity_measures_source_code/)` contains a read-only reference copy of selected C++ files that mirror the Sequenzo implementation. It is there so readers can compare the paper's notation with the code while reading; it is not a second software package and you do not need to build it. If you want to rerun distances or regenerate figures, use Sequenzo.

This repository holds what is specific to the paper: the simulation design (Studies 1 and 2), precomputed JSON results, the figure pipeline, worked appendix examples, and the pairfam analytic dataset used in the divergence demonstration. 

Tested with: Sequenzo 0.1.40, Python 3.10+ (see `results/*/run_metadata.json` for the environment used to produce the bundled outputs).

---



## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install sequenzo==0.1.40       # or: pip install -e /path/to/Sequenzo
```

---



## What to run, and how long it takes

**Regenerate the main sensitivity figures from bundled JSON, which will take a few minutes.** The repository already includes `results/main_raw/results.json` and the robustness variants. You only need to aggregate and plot:

```bash
./reproduce.sh
```

Outputs land in `figures/`. This is enough to verify the main-text and appendix sensitivity panels without recomputing distances.

**Check appendix distance arithmetic and the ten-sequence illustration (under a minute after setup).**

```bash
python3 -m pytest appendix_illustrations/test_appendix_sequences.py -v
```

**Rerun the full simulation from scratch (several hours per normalization mode).** Each production run draws thousands of sequences per replication across thirty replications and all panels A–J:

```bash
python3 simulation/run_main.py --norm none --overwrite
python3 simulation/run_main.py --norm builtin --overwrite
python3 simulation/run_main.py --norm elzinga --overwrite
```

Then run `./reproduce.sh`, or the pipeline steps listed under [Simulation](#simulation-sensitivity-to-sequence-variation-divergence-and-convergence) below.

**Regenerate empirical cluster index plots (tens of minutes per batch; OM on monthly pairfam data is slow).** See [Empirical demonstration](#empirical-demonstration) and the README files in `empirical_demonstration/divergence/` and `empirical_demonstration/convergence/`.

---



## Directory layout

```
.
├── simulation/                      # Study 1 & 2 generators; the entry point is run_main.py
├── pipeline/                        # JSON → CSV → PDF figures
├── results/                         # Bundled JSON: main runs, normalization robustness, supplementary data-generating checks
├── figures/                         # Paper figures produced by the pipeline
├── tables/                          # LaTeX tables (e.g. empirical cluster-quality diagnostics)
├── empirical_demonstration/         # pairfam (divergence) and mvad (convergence) clustering scripts
│   └── data/                        # Datasets; also available inside Sequenzo
├── appendix_illustrations/          # Ten-sequence data and appendix distance tests
├── dissimilarity_measures_source_code/   # Reference C++ only; run code via Sequenzo
└── dev/                             # Optional local tests (not required for replication)
```

---



## Guide by manuscript section



### Conceptual framework of spell-based sequence comparison

**Ten illustration sequences (main-text figure).** Sequence data and the index plot live in `appendix_illustrations/ten_sequences.py`; a regenerated PDF is in `appendix_illustrations/figures/ten_sequences_index.pdf`. The interactive notebook `appendix_illustrations/illustration_sequences.ipynb` rebuilds the plot step by step.

**Spell-based distance measures (Sections 2–3 in the PDF).** The measures themselves are in Sequenzo. The online supplement section *Worked examples of spell-based distance measures* is checked by `appendix_illustrations/test_appendix_sequences.py` (OM, OMspell, OMspellRS, LCP, LCPmst, LCPprod, LCPspell on the sequence pairs discussed in the appendix).

---



### Simulation: Sensitivity to Sequence Variation, Divergence, and Convergence

**Simulation design and strand table.** Studies 1 and 2 are implemented in `simulation/studies/study1_group_level.py` and `simulation/studies/study2_directional_pairs.py`. Panel definitions (A–J), JSON keys, and axis labels are centralized in `simulation/studies/registry.py`, which also defines the strand designs summarized in the simulation table in the paper.


| Study                             | Panels | Metric                     |
| --------------------------------- | ------ | -------------------------- |
| Study 1 — group-level strands     | A–F    | Chance-corrected pseudo-R² |
| Study 2 — early-vs-late contrasts | G–J    | Normalized paired contrast |


One production run executes Study 1, then Study 2:

```bash
python3 simulation/run_main.py --norm none
```

Bundled outputs are keyed by strand name (for example `timing`, `sequencing.*`, `duration` in Study 1; `sensitivity_*_divergence` and related keys in Study 2). See `registry.py` for the full mapping.

**Results of sensitivity patterns across distance measures (main-text figure).** The 2×5 panel figure is `figures/fig_sensitivity_profiles.pdf`, built from `results/main_raw/results.json` via `pipeline/01_aggregate_json_to_csv.py` and `pipeline/02_plot_sensitivity_profiles.py`. Row 1 shows Study 1 panels A–E; row 2 shows panel F and Study 2 panels G–J.


| Normalization                   | JSON directory                | Aggregated CSV                                 | Figure                                         |
| ------------------------------- | ----------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Raw (`none`) — main text        | `results/main_raw/`           | `pipeline/data/aggregated_results.csv`         | `figures/fig_sensitivity_profiles.pdf`         |
| Built-in (`builtin`) — appendix | `results/robustness_builtin/` | `pipeline/data/aggregated_results_builtin.csv` | `figures/fig_sensitivity_profiles_builtin.pdf` |
| Elzinga (`elzinga`): appendix   | `results/robustness_elzinga/` | `pipeline/data/aggregated_results_elzinga.csv` | `figures/fig_sensitivity_profiles_elzinga.pdf` |


Manual steps (if you prefer not to use `./reproduce.sh`):

```bash
python3 pipeline/01_aggregate_json_to_csv.py --from-norm none
python3 pipeline/02_plot_sensitivity_profiles.py
```

---



### Empirical demonstration

**Divergence: pairfam family formation (LCP / LCPspell, PAM).** Analytic data: `empirical_demonstration/data/pairfam_activity_by_month.csv` (derived from the public pairfam release; see the manuscript for the citation). Per-method index-plot PDFs: `empirical_demonstration/divergence/`. The multi-method panel shown in the main PDF is assembled from those PDFs when compiling the manuscript; no single script here writes that composite.

```bash
python3 empirical_demonstration/divergence/pairfam_family_by_month_cluster_batch.py
```

**Convergence: mvad school-to-work (RLCP / RLCPspell, PAM).** The scripts load mvad through Sequenzo; a frozen copy is in `empirical_demonstration/data/mvad.csv`. Output PDFs: `empirical_demonstration/convergence/`.

```bash
python3 empirical_demonstration/convergence/mvad_school_to_work_cluster_batch.py
```

---



### Online supplement: Robustness checks of simulation results

**Alternative distance scaling (built-in normalization).** `figures/fig_sensitivity_profiles_builtin.pdf`: rerun simulation with `--norm builtin`, then `01_aggregate --from-norm builtin` and `02_plot_sensitivity_profiles.py --output-suffix _builtin`.

**Elzinga–Studer reference rescaling.** `figures/fig_sensitivity_profiles_elzinga.pdf` — same pipeline with `--norm elzinga` and `--output-suffix _elzinga`.

**Alternative evaluation scoring for Study 1 (raw pseudo-R²).** `figures/fig_sensitivity_raw_pseudor2_core.pdf`: `pipeline/02_plot_sensitivity_raw_pseudor2.py` on the main aggregated CSV (panels A–F only).

**Supplementary data-generating checks.** The main simulations generate trajectories from predefined DSS patterns and spell durations. The supplement asks whether the same qualitative conclusions hold when sequence differences are produced from event times or from minimal local changes instead, following the event-based and small-perturbation designs in Studer and Ritschard (2016). Bundled results are in `results/supplementary_data_generating_checks/`. The chance-corrected figure is `figures/fig_supplementary_data_generating_checks.pdf`; the companion panel using uncorrected discrepancy-based pseudo-R² is `figures/fig_supplementary_data_generating_checks_raw.pdf`. To rebuild: `pipeline/03_run_supplementary_data_generating_checks.py`, then `04_plot_supplementary_data_generating_checks.py` and `04_plot_supplementary_data_generating_checks_raw.py`.

Optional localized-perturbation scripts that do not enter the published figures: `simulation/robustness/`.

Built-in normalization rules discussed in the supplement refer to Sequenzo's `norm="auto"` behaviour; the prose and tables are in the manuscript, not regenerated here.

---



### Online supplement: Supplementary empirical demonstrations and clustering robustness

**Ward linkage figures (appendix).** PAM results are in `empirical_demonstration/divergence/` and `empirical_demonstration/convergence/`; Ward counterparts are in the `ward/` subfolders. Regenerate with `--clustering-method ward` on the batch scripts above. The folder `empirical_demonstration/divergence/ward/auto_norm/` holds optional Ward runs with `norm='auto'`; they are not cited in the paper.

**Cluster quality diagnostics and linkage robustness (appendix tables).** LaTeX tables in `tables/tab_empirical_cluster_quality.tex` and `tables/tab_empirical_linkage_robustness.tex`, generated by:

```bash
python3 empirical_demonstration/empirical_clustering_diagnostics.py
```

**Supplementary empirical illustrations of sequence variation (appendix index plots).** Most datasets are built into Sequenzo and plotted with its sequence-index tools; the RCA metropolitan patent trajectories are an exception and are described in the manuscript only. This repository does not bundle a separate regeneration script for that full index-plot gallery.

---



## Notes

- The primary simulation benchmark uses raw distances (`--norm none`). Chance-corrected pseudo-R² is scale-invariant; built-in and Elzinga normalizations are appendix robustness checks.
- Spell methods use `duration_ref = T` (the observation window). The sensitivity grid varies `expcost` only.
- Development utilities under `dev/` are for local checks and are not part of `./reproduce.sh`.

