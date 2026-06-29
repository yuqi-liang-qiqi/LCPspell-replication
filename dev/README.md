# Development utilities

**Not part of the replication workflow.** These scripts are for local checks and previews only.

| Path | Purpose |
|------|---------|
| `tests/test_timing_and_convergence_generators.py` | Pytest invariants for generators and evaluation |
| `smoke_validate_distances.py` | Pre-flight distance-matrix sanity checks |
| `previews/optional_compact_labels_preview.py` | X-axis label layout preview |
| `previews/quick_sequencing_check.py` | One-panel sequencing sanity plot |

```bash
python3 -m pytest lcpspell-paper-replication/dev/tests/
python3 lcpspell-paper-replication/dev/smoke_validate_distances.py
```
