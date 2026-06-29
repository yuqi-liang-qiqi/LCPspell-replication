# Aggregated simulation CSVs

| CSV | Source JSON | Role |
|-----|-------------|------|
| `aggregated_results.csv` | `results/main_raw/results.json` | Main text (`norm=none`) |
| `aggregated_results_builtin.csv` | `results/robustness_builtin/results.json` | Appendix RC1 (`norm=auto`) |
| `aggregated_results_elzinga.csv` | `results/robustness_elzinga/results.json` | Appendix RC2 (Elzinga–Studer) |

Generate with:

```bash
python3 lcpspell-paper-replication/pipeline/01_aggregate_json_to_csv.py --from-norm none
python3 lcpspell-paper-replication/pipeline/01_aggregate_json_to_csv.py --from-norm builtin
python3 lcpspell-paper-replication/pipeline/01_aggregate_json_to_csv.py --from-norm elzinga
```

## Columns (Study 1 + Study 2)

| Column | Meaning |
|--------|---------|
| `panel` | A–J (paper panel letter) |
| `study` | `study1` (A–F) or `study2` (G–J) |
| `metric` | `chance_corrected_pseudo_r2` or `normalized_paired_contrast` |
| `mean_score` | Primary score averaged over replications |
| `sd_score` | Standard deviation across replications |
| `mean_r2` | Legacy alias of `mean_score` (backward compatibility) |
