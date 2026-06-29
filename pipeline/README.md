# Figure pipeline

Maps simulation JSON → CSV → figures in `../figures/`.

## Story line (matches main.tex)

1. **Study 1 (Panels A–F):** `chance_corrected_pseudo_r2` on group-level strands  
2. **Study 2 (Panels G–J):** `normalized_paired_contrast` on matched early-vs-late pairs  

Panel order, labels, and metrics are defined once in `simulation/studies/registry.py` and re-exported by `strand_layout.py`.

## Scripts

| Step | Script | Output |
|------|--------|--------|
| Aggregate | `01_aggregate_json_to_csv.py` | `data/aggregated_results*.csv` |
| Main figure | `02_plot_sensitivity_profiles.py` | `figures/fig_sensitivity_profiles.pdf` |
| Appendix R²_disc | `02_plot_sensitivity_raw_pseudor2.py` | Study 1 panels only |
| Supplementary data-generating checks | `03_run_supplementary_data_generating_checks.py` → `04_plot_supplementary_data_generating_checks.py` + `04_plot_supplementary_data_generating_checks_raw.py` | Event-based and perturbation appendix figures |

```bash
# From repo root
python3 lcpspell-paper-replication/pipeline/01_aggregate_json_to_csv.py --from-norm none
python3 lcpspell-paper-replication/pipeline/02_plot_sensitivity_profiles.py

# Or regenerate everything from existing JSON:
./lcpspell-paper-replication/reproduce.sh
```

## CSV columns

| Column | Meaning |
|--------|---------|
| `panel` | A–J |
| `study` | `study1` or `study2` |
| `metric` | `chance_corrected_pseudo_r2` or `normalized_paired_contrast` |
| `mean_score` | Primary score (mean across replications) |
| `sd_score` | Standard deviation across replications |
| `mean_r2` | Legacy alias of `mean_score` (Study 1 scripts) |
