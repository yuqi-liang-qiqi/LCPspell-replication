# Simulation engine (main.tex Section 3)

Two studies share one distance-method grid but use **different experimental units and metrics**.

## Study 1 — group-level strands (Panels A–F)

| Panel | Strand | JSON key | Metric |
|-------|--------|----------|--------|
| A | Timing | `timing` | chance-corrected pseudo-R² |
| B–E | Sequencing (4 patterns) | `sequencing.*` | chance-corrected pseudo-R² |
| F | Duration | `duration` | chance-corrected pseudo-R² |

**Code:** `studies/study1_group_level.py` (`GroupLevelStudyMixin`)

## Study 2 — early-vs-late directional contrasts (Panels G–J)

| Panel | Strand | JSON key | Estimand |
|-------|--------|----------|----------|
| G | Calendar divergence | `sensitivity_divergence_calendar` | early pair distance > late |
| H | Calendar convergence | `sensitivity_convergence_calendar` | early pair distance < late |
| I | Spell-order divergence | `sensitivity_divergence_spell_order` | early pair distance > late |
| J | Spell-order convergence | `sensitivity_convergence_spell_order` | early pair distance < late |

**Code:** `studies/study2_directional_pairs.py` (`DirectionalPairStudyMixin`)

Each draw samples a **nested matched quadruple** from a shared latent background:
early and late focal pairs differ only in where the shared segment occurs. The
score is a **normalized aggregate early-vs-late contrast** between mean focal
within-pair distances.

### Duration modes (spell-order strands only)

| Mode | Output directory |
|------|------------------|
| `matched` (default) | `results/main_raw/` (with norm mode) |
| `shared_spell_mismatch_compensated` | `results/robustness_directional_duration_mismatch/` (`--norm none` only) |
| `background_noise` | `results/robustness_directional_background_noise/` (`--norm none` only) |

## Package layout

```
simulation/
  run_main.py              # Entry point (Study 1 then Study 2)
  framework.py             # Shared infrastructure + mixin composition
  config.py                # Global parameters
  evaluation.py            # pseudo-R² and paired-contrast metrics
  run_metadata.py          # Sequenzo version, git commit, platform
  studies/
    registry.py            # Panel A–J specs, JSON keys, metrics
    study1_group_level.py
    study2_directional_pairs.py
  supplementary_dg_checks/   # Supplementary data-generating checks (event-based + perturbation)
  robustness/            # Optional localized perturbation checks
```

## Run

```bash
python3 simulation/run_main.py --norm none
```

## Tests

```bash
python3 -m pytest ../dev/tests/
```
