# Full Reproduction Guide

For the complete command reference, see `readme_CL.txt` in the repository root.

## Prerequisites

```bash
conda activate sni
pip install -r requirements_core.txt  # main paper
pip install -r requirements_extra.txt  # supplement (HyperImpute, SHAP, etc.)
```

## Quick Path (Recommended)

```bash
# Smoke test first
python scripts/paper_quickstart.py --smoke-test --outdir results_quick

# Full reproduction (all 7 stages, one command)
python scripts/run_all_experiments.py \
    --outdir results_all \
    --skip-existing \
    --n-jobs-cpu 8 \
    --n-jobs-gpu 1

# Or: main paper only (SNI + classic baselines + postprocess)
python scripts/run_all_experiments.py --outdir results_all --stages 1 2 7 --skip-existing

# Preview all commands without executing
python scripts/run_all_experiments.py --outdir results_all --dry-run
```

## Manual Reproduction (Step by Step)

### Part 1: SNI Main Experiments

```bash
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_v03_main.csv \
    --outdir results_sni_v03_main \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_v03_main \
    --outdir results_sni_v03_main/_summary
```

### Part 2: Baseline Experiments

```bash
# Classic baselines (CPU, fast)
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_options/manifest_baselines_main_all.csv \
    --outdir results_baselines_main \
    --n-jobs -1 --skip-existing --default-use-gpu false

# New baselines (HyperImpute/TabCSDI, slower)
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_new.csv \
    --outdir results_baselines_new \
    --n-jobs 1 --default-timeout 1800 --skip-existing
```

### Part 3: Post-Processing

```bash
python scripts/paper_quickstart.py --postprocess-only \
    --sni-results results_sni_v03_main \
    --baseline-results results_baselines_main \
    --outdir results_paper
```

### Part 4: Extended Experiments

```bash
# Sanity check (dependency recovery)
python scripts/synth_generate_s5.py --outdir data/synth_s5
python scripts/sanity_check_v2_s5.py --data-dir data/synth_s5 --outdir results_sanity

# Interpretability audit
python ext1/scripts/exp1_audit_story_leakage.py --outdir results_ext1_audit
python ext1/scripts/exp2_downstream_task_validation.py --outdir results_ext1_downstream

# Advanced analysis
python ext2/scripts/exp3_per_class_categorical.py --outdir results_ext2_perclass
python ext2/scripts/exp4_shap_comparison.py --outdir results_ext2_shap

# Attention Rollout / Flow (MAR requires --mar-driver-cols)
python ext2/scripts/exp4b_attention_rollout.py \
    --mechanism MAR --missing-rate 0.30 \
    --mar-driver-cols HR PULSE \
    --seed 2026 --outdir results_ext2_attn

# D stability across seeds (MAR requires --mar-driver-cols)
python ext2/scripts/exp4c_d_stability.py \
    --mechanism MAR --missing-rate 0.30 \
    --mar-driver-cols HR PULSE \
    --missing-seed 2026 --outdir results_ext2_dstab

# Significance tests (--results-dir should point to the root containing all result subdirs)
python ext2/scripts/exp5_significance_tests.py \
    --results-dir results_all \
    --outdir results_ext2_significance

python ext2/scripts/exp6_mimic_mortality_impute_predict.py --outdir results_ext2_mimic
```

> **Note**: For MAR mechanism experiments (exp4b, exp4c), `--mar-driver-cols` is required when
> `strict_mar=True` (the default). Use the appropriate driver columns for your dataset
> (e.g., `HR PULSE` for MIMIC).

## Estimated Runtimes

| Experiment | Manifest | Approx. Time | n-jobs |
|-----------|----------|-------------|--------|
| SNI Main | `manifest_sni_v03_main.csv` | 1-2h | -1 |
| SNI Ablation | `manifest_sni_v03_ablation_lambda.csv` | 2-4h | -1 |
| Baselines Classic | `manifest_baselines_main_all.csv` | 30min-1h | -1 |
| Baselines New | `manifest_baselines_new.csv` | 2-4h | 1 |
| Sanity Check | (generated) | 30min | -1 |
| Ext1 | (scripts) | 1-2h | 1 |
| Ext2 | (scripts) | 2-4h | 1 |

## Troubleshooting

- **OOM with GPU baselines**: Use `--n-jobs 1` for GAIN/MIWAE/TabCSDI
- **HyperImpute timeout**: Increase `--default-timeout` (default 1800s)
- **Missing dependencies**: Run `pip install -r requirements_core.txt` (main) and `pip install -r requirements_extra.txt` (supplement)
- **Resume interrupted runs**: Use `--skip-existing` flag
