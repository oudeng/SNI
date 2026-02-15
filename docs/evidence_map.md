# Evidence Map: Paper Artifact to Experiment Traceability

This document maps each paper artifact (table/figure) to the specific manifest, runner command, and output path.

## Main Paper

| Paper Artifact | Manifest | Runner Command | Output Path | Key Intermediates |
|:---|:---|:---|:---|:---|
| Table I (Avg Rank) | `manifest_sni_v03_main.csv` + `manifest_baselines_main_all.csv` | `paper_quickstart.py --main` | `results_paper/tables/Table_I_avg_rank.tex` | `summary_ranks_agg.csv` |
| Table II (MIMIC) | same | same | `results_paper/tables/Table_II_MIMIC.tex` | `summary_agg.csv` filtered by MIMIC |
| Table III (Synth dep.) | synth manifests | `scripts/sanity_check_v2_s5.py` | `results_paper/tables/Table_III_synth.tex` | `dependency_matrix.csv` |
| Table IV (Proxy audit) | ext1 scripts | `ext1/scripts/exp1_audit_story_leakage.py` | `results_paper/tables/Table_IV_proxy.tex` | -- |
| Table V (Downstream) | ext1/ext2 | `ext1/scripts/exp2_downstream_task_validation.py` | `results_paper/tables/Table_V_downstream.tex` | -- |
| Fig. 3 (Compact) | `manifest_sni_v03_main.csv` + baselines | `paper_quickstart.py --main` | `results_paper/figures/Fig3_compact_main.png` | `merged_summary_agg.csv` |
| Fig. 4 (Rate sweep) | `manifest_sni_rate_sweep.csv` + baselines | `viz_compact_rate_sweep.py` | `results_paper/figures/Fig4_rate_sweep.png` | -- |
| Fig. 5 (MCAR vs MAR) | main manifests | `viz_mcar_mar_robustness.py` | `results_paper/figures/Fig5_robustness.png` | -- |
| Fig. 6 (MIMIC dep. net) | MIMIC main runs | `viz_dependency_network.py` | `results_paper/figures/Fig6_dep_network.png` | `dependency_matrix.csv` |

## Supplementary Materials

| Paper Artifact | Manifest | Runner Command | Key Notes |
|:---|:---|:---|:---|
| Table S7 (SHAP vs D) | main runs | `ext2/scripts/exp4_shap_comparison.py` | Uses SHAP library |
| Table S7.B (Attn rollout) | main runs | `ext2/scripts/exp4b_attention_rollout.py` | D vs Attention Rollout/Flow; requires `--mar-driver-cols` for MAR |
| Table S7.C (D stability) | main runs | `ext2/scripts/exp4c_d_stability.py` | Cross-seed stability; requires `--mar-driver-cols` for MAR |
| Table S8 (Significance) | main + baselines | `ext2/scripts/exp5_significance_tests.py` | Wilcoxon tests; `--results-dir` should point to `results_all/` |
| Table S9 (Per-class) | main runs | `ext2/scripts/exp3_per_class_categorical.py` | Per-class breakdown |
| Table VI (MIMIC mortality) | MIMIC runs | `ext2/scripts/exp6_mimic_mortality_impute_predict.py` | Impute-then-predict |

## Reproduction Commands

### One-command full reproduction (recommended)
```bash
# Run ALL experiments (all 7 stages) with one command
python scripts/run_all_experiments.py \
    --outdir results_all \
    --skip-existing \
    --n-jobs-cpu 8 \
    --n-jobs-gpu 1

# Or run specific stages: SNI (1) + baselines (2,3) + postprocess (7)
python scripts/run_all_experiments.py --outdir results_all --stages 1 2 3 7 --skip-existing

# Preview all commands without executing
python scripts/run_all_experiments.py --outdir results_all --dry-run
```

### Manual step-by-step (main paper only)
```bash
# Step 1: Run SNI experiments
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_v03_main.csv \
    --outdir results_sni_v03_main \
    --n-jobs 8

# Step 2: Run baseline experiments
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_options/manifest_baselines_main_all.csv \
    --outdir results_baselines_main \
    --n-jobs 8

# Step 3: Aggregate and generate outputs
python scripts/paper_quickstart.py --postprocess-only \
    --sni-results results_sni_v03_main \
    --baseline-results results_baselines_main \
    --outdir results_paper
```
