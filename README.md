# SNI: Statistical–Neural Interaction （SNI）

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Reproducibility Package**

This repository contains the complete experimental code for the paper, **Statistical-Neural Interaction Networks for Interpretable Imputation of Mixed Numerical and Categorical Tabular Data**.

It enables full reproduction of all experiments, tables, and figures presented in the manuscript.

Statistical--Neural Interaction (SNI), an interpretable mixed-type imputation framework that couples correlation-derived statistical priors with neural feature attention through a Controllable-Prior Feature Attention (CPFA) module. CPFA learns head-wise prior-strength coefficients $\{\lambda_h\}$ that softly regularize attention toward the prior while allowing data-driven deviations when nonlinear patterns appear to be present in the data. Beyond imputation, SNI aggregates attention maps into a directed feature-dependency matrix that summarizes which variables the imputer relied on, without requiring post-hoc explainers.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Repository Structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [Experiment Design](#5-experiment-design)
6. [Quick Start](#6-quick-start)
7. [Full Reproduction Guide](#7-full-reproduction-guide)
8. [Output Files](#8-output-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Changelog](#10-changelog)

---

## 1. Overview

### Core Components

| Directory/File | Description |
|----------------|-------------|
| `SNI_v0_2/` | Core SNI algorithm package v0.2 (importable module) |
| `SNI_v0_3/` | Core SNI algorithm package v0.3 — learnable per-head λ, categorical balance mode, convergence monitoring |
| `baselines/` | Baseline imputation methods (MeanMode, KNN, MICE, MissForest, GAIN, MIWAE, HyperImpute, TabCSDI) |
| `scripts/` | Experiment runners, aggregation, visualization, and table generation |
| `ext1/` | Extended experiments: interpretability audit & downstream task validation |
| `ext2/` | Extended experiments: per-class breakdown, SHAP comparison, attention rollout, D stability, significance tests, Impute→Predict |
| `data/` | Datasets and experiment manifests |
| `configs/` | Configuration files |

### Design Principles

1. **Reproducibility First**: All outputs (metrics, figures, tables) are saved as CSV/PNG/TEX files
2. **Manifest-Driven Experiments**: Each experiment is defined in a CSV manifest file
3. **Parallel Execution**: Multi-core parallel processing for efficient large-scale experiments
4. **Fixed Random Seeds**: Deterministic results across runs

### SNI Variants

| Variant | Description |
|---------|-------------|
| `SNI` | Default (Normal) - with learned prior |
| `NoPrior` | λ=0, attention-only without prior |
| `HardPrior` | Fixed large λ, strong prior |
| `SNI-M` | Mask-aware with missingness indicator embedding |
| `SNI+KNN` | Post-processing with kNN for imbalanced categorical features |

---

## 2. Installation

### Requirements

- Python ≥ 3.9
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Create and activate environment
conda create -n sni python=3.10
conda activate sni

# Install dependencies
pip install -r requirements.txt

# (Optional) Install extra dependencies for Ext2
pip install shap scipy xgboost

# Verify installation
python -c "from SNI_v0_3.imputer import SNIImputer; print('SNI v0.3 ready!')"
```

### Dependencies Summary

- `torch>=1.12`: Deep learning framework
- `scikit-learn>=1.0`: ML utilities and MICE
- `pandas>=1.4`: Data manipulation
- `numpy>=1.21`: Numerical computing
- `joblib`: Parallel execution
- `matplotlib>=3.5`: Visualization
- `hyperimpute>=0.1.0`: HyperImpute baseline (Jarrett et al., ICML 2022)
- `shap` (optional): TreeSHAP for Ext2 Exp4
- `scipy` (optional): Wilcoxon tests for Ext2 Exp5
- `xgboost` (optional): XGBoost classifier for Ext2 Exp6

---

## 3. Repository Structure

```
.
├── SNI_v0_2/                   # Core SNI algorithm (v0.2)
│   ├── imputer.py              # Main imputer class
│   ├── cpfa.py                 # Cross-Feature Attention module
│   ├── metrics.py              # Evaluation metrics
│   └── ...
├── SNI_v0_3/                   # Core SNI algorithm (v0.3) ← NEW
│   ├── imputer.py              # Learnable per-head λ, convergence monitoring
│   ├── cpfa.py                 # CPFA with cat_balance_mode support
│   ├── metrics.py              # Extended metrics + augment_summary_with_imputer_stats()
│   └── ...
├── baselines/                  # Baseline methods
│   ├── MeanMode_v1.py          # Mean/Mode imputation
│   ├── KNN_v1.py               # K-Nearest Neighbors
│   ├── MICE_v3.py              # Multiple Imputation by Chained Equations
│   ├── MissForest_v2.py        # Random Forest-based imputation
│   ├── GAIN_v5.py              # Generative Adversarial Imputation Nets
│   ├── MIWAE_v3.py             # Missing data IWAE
│   ├── HyperImpute_v1.py       # AutoML-based imputation (Jarrett, ICML 2022) ← NEW
│   ├── TabCSDI_v1.py           # Score-based diffusion (Zheng, NeurIPS TRL 2022) ← NEW
│   ├── tabcsdi_core/           # TabCSDI internal modules ← NEW
│   │   ├── diff_models.py      # Denoising transformer
│   │   └── main_model.py       # Diffusion sampling loop
│   └── registry.py             # Unified baseline interface (8 methods)
├── scripts/
│   ├── run_experiment.py       # Single experiment runner (supports --sni-version v0.2|v0.3)
│   ├── run_manifest_parallel.py    # Parallel batch runner for SNI
│   ├── run_manifest_baselines.py   # Parallel batch runner for baselines (8 methods)
│   ├── aggregate_results.py    # Results aggregation (mean±std)
│   ├── make_latex_table.py     # LaTeX table generation
│   ├── synth_generate.py       # Synthetic data generator
│   ├── sanity_check_v2.py      # Sanity check experiments
│   └── viz_*.py                # Visualization scripts
├── ext1/                       # Extended experiments (Ext1)
│   └── scripts/
│       ├── exp1_audit_story_leakage.py         # Interpretability audit (+ continuous proxy)
│       └── exp2_downstream_task_validation.py  # Downstream task validation
├── ext2/                       # Extended experiments (Ext2)
│   └── scripts/
│       ├── exp3_per_class_categorical.py       # Per-class breakdown (Table S9)
│       ├── exp4_shap_comparison.py             # SHAP vs SNI D comparison (Table S7)
│       ├── exp4b_attention_rollout.py          # D vs rollout vs flow (Table S7.B) ← NEW
│       ├── exp4c_d_stability.py                # Cross-seed D stability (Table S7.C) ← NEW
│       ├── exp5_significance_tests.py          # Wilcoxon significance tests (Table S8)
│       └── exp6_mimic_mortality_impute_predict.py  # MIMIC Impute→Predict (Table VI)
├── data/
│   ├── *_complete.csv              # Complete datasets
│   ├── {Dataset}/                  # Missing datasets with masks
│   ├── synth_s5/                   # Synthetic datasets for sanity check
│   ├── manifest_sni_*.csv          # SNI experiment manifests (v0.2)
│   ├── manifest_sni_v03_*.csv      # SNI experiment manifests (v0.3) ← NEW
│   ├── manifest_baselines_*.csv    # Baseline experiment manifests
│   └── manifest_baselines_new.csv  # HyperImpute + TabCSDI manifest ← NEW
├── configs/
│   └── datasets.yaml           # Dataset configurations (+ v0.3 balance settings)
├── requirements.txt
└── README.md
```

---

## 4. Datasets

### Real-World Datasets

| Dataset | Samples | Features | Domain |
|---------|---------|----------|--------|
| MIMIC | 5,000 | 8 | Healthcare (ICU) |
| eICU | 5,000 | 9 | Healthcare (ICU) |
| NHANES | 7,500 | 9 | Public Health |
| Concrete | 1,030 | 9 | Engineering |
| ComCri | 2,000 | 8 | Materials Science |
| AutoMPG | 392 | 8 | Automotive |

### Missingness Mechanisms

| Mechanism | Description | Rate |
|-----------|-------------|------|
| MCAR | Missing Completely At Random | 10%, 30%, 50% |
| MAR | Missing At Random | 10%, 30%, 50% |
| MNAR | Missing Not At Random | 10%, 30%, 50% |

### Synthetic Datasets (Sanity Check)

Three synthetic settings with **known ground-truth dependency structures**:

| Setting | Description | Purpose |
|---------|-------------|---------|
| `linear_gaussian` | Linear dependencies with Gaussian noise | Baseline validation |
| `nonlinear_mixed` | Nonlinear + product interactions | Test non-correlation dependencies |
| `interaction_xor` | XOR-like interactions with zero marginal correlation | Stress test for correlation-based methods |

---

## 5. Experiment Design

### 5.1 Main Experiments

**Objective**: Compare SNI with baselines under MCAR and MAR at 30% missingness.

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_sni_main.csv` | 60 | SNI on 6 datasets × 2 mechanisms × 5 seeds |
| `manifest_baselines_main.csv` | 300+ | 6 baselines × same settings |

### 5.2 Ablation Study (Table 4)

**Objective**: Evaluate the contribution of each SNI component.

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_sni_ablation.csv` | 90 | SNI/NoPrior/HardPrior × 6 datasets × 5 seeds |

### 5.3 MNAR Robustness (Table 5)

**Objective**: Evaluate performance under MNAR with varying missing rates.

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_sni_mnar.csv` | 270 | SNI/SNI-M × 6 datasets × 3 rates × 5 seeds |
| `manifest_baselines_mnar.csv` | 810+ | Baselines × same settings |

### 5.4 Deep Baselines (Table 6)

**Objective**: Compare with deep learning baselines (GAIN, MIWAE).

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_baselines_deep.csv` | 540 | GAIN/MIWAE × 6 datasets × 3 mechanisms × 3 rates × 5 seeds |

### 5.4b New Baselines: HyperImpute & TabCSDI

**Objective**: Compare with modern AutoML and diffusion-based imputation methods.

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_baselines_new.csv` | 120 | HyperImpute/TabCSDI × 6 datasets × 2 mechanisms × 5 seeds |

| Method | Reference | Key Idea |
|--------|-----------|----------|
| **HyperImpute** | Jarrett et al., ICML 2022 | AutoML column-wise imputer with iterative refinement |
| **TabCSDI** | Zheng & Charoenphakdee, NeurIPS TRL 2022 | Score-based conditional diffusion for tabular data |

### 5.4c SNI v0.3 Experiments

**Objective**: Evaluate the new SNI v0.3 features (learnable per-head λ, categorical balance mode).

| Manifest | Experiments | Description |
|----------|-------------|-------------|
| `manifest_sni_v03_main.csv` | 60 | SNI v0.3 on 6 datasets × 2 mechanisms × 5 seeds |
| `manifest_sni_v03_ablation_lambda.csv` | 50 | Fixed-λ grid (0.1–5.0) on MIMIC/NHANES × 5 seeds |

### 5.5 Sanity Check: Dependency Recovery

**Objective**: Validate that the learned dependency matrix D recovers true causal structure.

**Method**: 
1. Generate synthetic data with known DAG structure
2. Compare dependency recovery metrics: AUROC, AUPRC, Precision@K, Hub Spearman
3. Evaluate SNI vs NoPrior vs PriorOnly (correlation-based)

**Key Insight**: `interaction_xor` setting specifically tests scenarios where marginal correlations are near-zero but true dependencies exist, demonstrating SNI's advantage over correlation-only methods.

### 5.6 Ext1 — Interpretability Audit Story (Leakage / Proxy Injection)

**Objective**: Demonstrate that the SNI dependency network can be directly used for real-data auditing — detecting field leakage or proxy dominance.

**Method**:
1. Inject a proxy column (e.g. `ALARM_LEAK = ALARM`) into a real dataset (e.g. MIMIC), keeping the proxy always observed.
2. Run SNI and export the dependency matrix; check whether `D[ALARM, ALARM_LEAK]` exhibits single-source dominance (weight ≈ 1).
3. Re-run SNI **without** the proxy column (same missing mask) and compare target-column Macro-F1 / Accuracy, confirming the proxy genuinely drove performance.

**Actionable Outputs**:
- `audit_report.md` — one-paragraph summary: "who dominates whom, potential leakage, performance drop after removal."
- `audit_top_sources.csv` — Top-5 source features per target (directly usable as a supplementary table).
- `audit_flags.csv` — automatic flags for targets where `top1_weight ≥ 0.6` (single-source dominance risk).

**Script**: `ext1/scripts/exp1_audit_story_leakage.py`

### 5.7 Ext1 — Downstream Task Validation (Impute → Predict)

**Objective**: Verify that cell-wise imputation quality (RMSE / F1) translates to actual downstream task utility.

**Method**:
1. From a **complete** dataset, introduce missingness to **feature columns only** (target *y* is never missing).
2. Impute features with multiple methods (SNI, MissForest, MeanMode, MICE, HyperImpute, TabCSDI).
3. Train a downstream model (LogisticRegression / Ridge) on the imputed features.
4. Compare downstream performance (Accuracy / Macro-F1 / AUC or RMSE / R²) and cross-seed stability (std).
5. Optionally compute a simple `group_gap` (max − min across a sensitive attribute such as `gender_std`) as evidence of reduced bias.

**Default Setting**: NHANES → predict `metabolic_score` (multi-class), MAR @ 30%, fairness column `gender_std`.

**Outputs**:
- `metrics_per_seed.csv` — per seed × method downstream metrics.
- `metrics_summary.csv` — aggregated mean ± std (table-ready).

**Script**: `ext1/scripts/exp2_downstream_task_validation.py`

### 5.8 Ext2 — Per-Class Breakdown on Imbalanced Categorical Targets

**Objective**: Report per-class Precision / Recall / F1 on masked entries for imbalanced categorical variables, revealing whether imputation collapses minority classes.

**Method**:
1. Inject missingness (default **strict MAR**, 30%) into features.
2. Impute using selected methods (SNI, MissForest, MeanMode).
3. Report per-class Precision / Recall / F1 on **masked entries only**.
4. Includes robust handling for mask semantics and float/int categorical codes.

**Default Setting**: MIMIC-IV ALARM column, strict MAR @ 30%.

**Outputs**:
- `perclass_metrics.csv` — per-class metrics for each method.
- `perclass_summary.csv` — aggregated summary across seeds.
- `collapse_flags.csv` — flags for classes that collapsed during imputation.

**Script**: `ext2/scripts/exp3_per_class_categorical.py`

### 5.9 Ext2 — SNI D vs SHAP on MissForest

**Objective**: Compare SNI's intrinsic reliance matrix **D** with post-hoc SHAP importance from MissForest, demonstrating that SNI's built-in interpretability captures similar feature-importance signals without needing external explainers.

**Method**:
1. Run **SNI** once → save reliance matrix **D**.
2. Run **MissForest** once → obtain imputed table.
3. For each target (default: `ALARM`, `SBP` if present):
   - **SNI**: report top-k features from D row.
   - **MissForest**: fit a RandomForest surrogate and compute **TreeSHAP** on the target-masked rows.
   - Report Spearman correlation between D and SHAP (optional).

**Outputs**:
- `table_S7_top_features.csv`.
- `shap_importances.csv` — full SHAP importance table.
- `spearman_d_vs_shap.csv` — Spearman rank correlation between D and SHAP.
- `d_matrix.csv` — full dependency matrix.

**Script**: `ext2/scripts/exp4_shap_comparison.py`
**Extra Dependency**: `pip install shap`

### 5.9b Ext2 — D vs Attention Rollout / Flow (Table S7.B)

**Objective**: Compare three aggregation strategies for extracting feature importance from multi-head attention weights (Abnar & Zuidema, 2020).

| Method | Description |
|--------|-------------|
| D (head-mean) | Default SNI: average over heads, row-normalize |
| Rollout (residual-aware) | A' = 0.5 * mean(A) + 0.5 * I, row-normalize |
| Flow (max-head) | Per-source max across heads, row-normalize |

**Outputs**: `table_S7B_aggregation_comparison.csv`, `spearman_aggregation.csv`, `attention_per_head.csv`

**Script**: `ext2/scripts/exp4b_attention_rollout.py`

### 5.9c Ext2 — Cross-Seed D Stability (Table S7.C)

**Objective**: Assess whether the learned dependency matrix D is reproducible across different random seeds, or an artifact of initialization noise.

**Method**: Fix the missingness pattern (same missing-seed), run SNI with K different model seeds, collect K dependency matrices, compute pairwise Spearman for each target's D row.

**Outputs**: `table_S7C_d_stability.csv`, `pairwise_spearman.csv`, `d_matrices/seed*.csv`

**Script**: `ext2/scripts/exp4c_d_stability.py`

### 5.10 Ext2 — Wilcoxon Significance Tests

**Objective**: Provide statistical evidence that SNI's improvements over baselines are significant, not due to random seed variation.

**Method** (two modes):
- **across_settings** : one paired Wilcoxon signed-rank test per (metric, baseline) across all dataset×mechanism settings.
- **per_setting** (optional): per dataset×mechanism×metric tests across seeds.

**Outputs**:
- `wilcoxon_across_settings.csv`
- `wilcoxon_per_setting.csv` — detailed per-setting tests.
- `wilcoxon_summary.csv` — aggregated summary.

**Script**: `ext2/scripts/exp5_significance_tests.py`  
**Extra Dependency**: `pip install scipy`

### 5.11 Ext2 — MIMIC-IV Impute→Predict (Main Table VI)

**Objective**: Evaluate whether imputation quality translates to downstream predictive performance on a clinically relevant task (MIMIC-IV in-hospital mortality / ALARM prediction).

**Method**:
1. Inject strict MAR missingness into **feature columns only** (label always observed).
2. Impute with each method (SNI, MissForest, MeanMode, HyperImpute, TabCSDI).
3. Train **Logistic Regression** and **XGBoost** on imputed features.
4. Report AUROC / AUPRC / Accuracy / F1.

**Outputs**:
- `per_seed_metrics.csv` — per seed × method × model metrics.
- `table_VI_summary.csv` — mean±std over seeds, ready for Table VI.

**Script**: `ext2/scripts/exp6_mimic_mortality_impute_predict.py`  
**Extra Dependency**: `pip install xgboost`

---

## 6. Quick Start

### Run a Single Experiment (v0.3)

```bash
python scripts/run_experiment.py \
  --input-complete data/MIMIC_complete.csv \
  --input-missing data/MIMIC/MIMIC_MAR_30per.csv \
  --categorical-vars SpO2 ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --variant SNI \
  --seed 8 \
  --sni-version v0.3 \
  --outdir results/test_run
```

### Run Parallel Batch Experiments

```bash
# SNI v0.3 experiments (8 parallel processes)
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_v03_main.csv \
    --outdir results_sni_v03_main \
    --n-jobs 8 \
    --sni-version v0.3

# Baseline experiments (including HyperImpute + TabCSDI)
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_new.csv \
    --outdir results_baselines_new \
    --n-jobs 4 \
    --default-use-gpu true
```

---

## 7. Full Reproduction Guide

### Step 1: SNI Main Experiments

```bash
# Main experiments (MCAR/MAR @ 30%)
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_main.csv \
    --outdir results_sni_main \
    --n-jobs -1

# Aggregate and generate tables
python scripts/aggregate_results.py \
    --results-root results_sni_main \
    --outdir results_sni_main/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --outdir results_sni_main/_tables
```

### Step 2: Ablation Study

```bash
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_ablation.csv \
    --outdir results_sni_ablation \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_ablation \
    --outdir results_sni_ablation/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_ablation/_summary/summary_agg.csv \
    --outdir results_sni_ablation/_tables
```

### Step 3: MNAR Robustness

```bash
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_mnar.csv \
    --outdir results_sni_mnar \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_mnar \
    --outdir results_sni_mnar/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_mnar/_summary/summary_agg.csv \
    --outdir results_sni_mnar/_tables
```

### Step 4: Baseline Comparisons

```bash
# Classic baselines (CPU-based)
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_main.csv \
    --outdir results_baselines_main \
    --n-jobs -1 \
    --default-use-gpu false

# MNAR baselines
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_mnar.csv \
    --outdir results_baselines_mnar \
    --n-jobs -1 \
    --default-use-gpu false

# Deep baselines (GPU recommended, sequential due to memory)
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_deep.csv \
    --outdir results_baselines_deep \
    --n-jobs 1 \
    --default-use-gpu true

# Aggregate all baseline results
for dir in results_baselines_main results_baselines_mnar results_baselines_deep; do
    python scripts/aggregate_results.py \
        --results-root $dir \
        --outdir $dir/_summary
    python scripts/make_latex_table.py \
        --summary-csv $dir/_summary/summary_agg.csv \
        --outdir $dir/_tables
done
```

### Step 5: Sanity Check

```bash
# Generate synthetic data (if not pre-generated)
mkdir -p data/synth_s5

for setting in linear_gaussian nonlinear_mixed; do
    for seed in 2025 2026 2027 2028 2029; do
        python scripts/synth_generate_s5.py \
            --outdir data/synth_s5 \
            --setting $setting \
            --seed $seed \
            --n 2000 --n-cont 10 --n-cat 2 \
            --mechanism MAR --missing-rate 0.30 \
            --driver-cols x0
    done
done

# Run sanity check (interaction_xor is auto-generated)
mkdir -p results_sanity_s5

python scripts/sanity_check_v2_s5.py \
    --data-dir data/synth_s5 \
    --outdir results_sanity_s5 \
    --settings linear_gaussian nonlinear_mixed interaction_xor \
    --seeds 2025 2026 2027 2028 2029 \
    --mechanism MAR --missing-rate 0.30 \
    --use-gpu true \
    --epochs 60 --num-heads 4 --emb-dim 32 --batch-size 64 --max-iters 3
```

**Outputs**:
- `metrics_per_run.csv`: Per-seed metrics for each method
- `table_S21.csv`: Aggregated results (mean±std)
- `table_S21.tex`: LaTeX table for supplementary material
- `D_*.csv`: Learned dependency matrices for inspection

### Step 6: Ext1 — Interpretability Audit Story

```bash
python ext1/scripts/exp1_audit_story_leakage.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars SpO2 ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --audit-target ALARM \
  --mechanism MAR --missing-rate 0.30 \
  --seed 2026 \
  --outdir results_ext1/audit_mimic_alarm \
  --run-without-leak true \
  --use-gpu false
```

**Outputs**:
- `results_ext1/audit_mimic_alarm/audit_report.md` — one-paragraph audit summary
- `results_ext1/audit_mimic_alarm/with_leak/dependency_matrix.csv` — full dependency matrix
- `results_ext1/audit_mimic_alarm/with_leak/audit_top_sources.csv` — Top-5 sources per target
- `results_ext1/audit_mimic_alarm/audit_comparison.csv` — with-leak vs without-leak comparison

### Step 7: Ext1 — Downstream Task Validation

```bash
python ext1/scripts/exp2_downstream_task_validation.py \
  --input-complete data/NHANES_complete.csv \
  --dataset-name NHANES \
  --target-col metabolic_score \
  --categorical-cols gender_std age_band \
  --continuous-cols waist_circumference systolic_bp diastolic_bp triglycerides hdl_cholesterol fasting_glucose age bmi hba1c \
  --mechanism MAR --missing-rate 0.30 \
  --mar-driver-cols age gender_std \
  --fairness-col gender_std \
  --imputers SNI MissForest MeanMode MICE HyperImpute TabCSDI \
  --seeds 1 2 3 5 8 \
  --outdir results_ext1/downstream_nhanes \
  --sni-use-gpu false \
  --baseline-use-gpu false \
  --save-missing true \
  --save-imputed false
```

**Outputs**:
- `results_ext1/downstream_nhanes/metrics_per_seed.csv` — per seed × method downstream metrics
- `results_ext1/downstream_nhanes/metrics_summary.csv` — aggregated mean ± std

### Step 8: Ext2 — Per-Class Breakdown

```bash
python ext2/scripts/exp3_per_class_categorical.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE SpO2 \
  --mechanisms MAR \
  --missing-rate 0.30 \
  --mar-driver-cols HR SpO2 \
  --methods SNI MissForest MeanMode \
  --seeds 1 2 3 5 8 \
  --outdir results_ext2/table_S9_perclass_alarm \
  --use-gpu false
```

**Outputs**:
- `results_ext2/table_S9_perclass_alarm/perclass_metrics.csv` — per-class metrics
- `results_ext2/table_S9_perclass_alarm/perclass_summary.csv` — aggregated summary
- `results_ext2/table_S9_perclass_alarm/collapse_flags.csv` — class collapse flags

### Step 9: Ext2 — SHAP vs SNI D Comparison

```bash
python ext2/scripts/exp4_shap_comparison.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars ALARM SpO2 \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --mechanism MAR --missing-rate 0.30 \
  --mar-driver-cols HR PULSE \
  --seed 2026 \
  --targets ALARM SBP \
  --top-k 10 \
  --shap-max-eval 512 \
  --outdir results_ext2/table_S7_shap_vs_D/MIMIC \
  --use-gpu false
```

**Outputs**:
- `results_ext2/table_S7_shap_vs_D/MIMIC/table_top_features.csv`
- `results_ext2/table_S7_shap_vs_D/MIMIC/shap_importances.csv` — full SHAP importances
- `results_ext2/table_S7_shap_vs_D/MIMIC/spearman_d_vs_shap.csv` — rank correlation
- `results_ext2/table_S7_shap_vs_D/MIMIC/d_matrix.csv` — dependency matrix

### Step 10: Ext2 — Wilcoxon Significance Tests

```bash
# Recommended: across-settings tests
python ext2/scripts/exp5_significance_tests.py \
  --results-dir . \
  --datasets MIMIC eICU NHANES ComCri AutoMPG Concrete \
  --mechanisms MCAR MAR \
  --metrics NRMSE R2 Spearman_rho Macro_F1 \
  --reference-method SNI \
  --baselines MissForest MIWAE GAIN KNN MICE MeanMode HyperImpute TabCSDI \
  --mode across_settings \
  --alpha 0.05 \
  --outdir results_ext2/significance

# Optional: also produce per-setting tests across seeds
python ext2/scripts/exp5_significance_tests.py \
  --results-dir . \
  --datasets MIMIC eICU NHANES ComCri AutoMPG Concrete \
  --mechanisms MCAR MAR \
  --metrics NRMSE R2 Spearman_rho Macro_F1 \
  --reference-method SNI \
  --baselines MissForest MIWAE GAIN KNN MICE MeanMode HyperImpute TabCSDI \
  --mode both \
  --alpha 0.05 \
  --outdir results_ext2/significance
```

**Outputs**:
- `results_ext2/significance/wilcoxon_across_settings.csv`
- `results_ext2/significance/wilcoxon_per_setting.csv` — per-setting tests
- `results_ext2/significance/wilcoxon_summary.csv` — aggregated summary

### Step 11: Ext2 — MIMIC-IV Impute→Predict (Table VI)

```bash
python ext2/scripts/exp6_mimic_mortality_impute_predict.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC_alarm_predict \
  --label-col ALARM \
  --binarize-threshold 34 \
  --categorical-vars SpO2 \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --mechanism MAR --missing-rate 0.30 \
  --mar-driver-cols HR PULSE \
  --imputers SNI MissForest MeanMode HyperImpute TabCSDI \
  --models LR XGB \
  --seeds 1 2 3 5 8 \
  --outdir results_ext2/table_VI_mimic_alarm \
  --use-gpu false
```

**Outputs**:
- `results_ext2/table_VI_mimic_alarm/per_seed_metrics.csv` — per seed × method × model
- `results_ext2/table_VI_mimic_alarm/table_VI_summary.csv` — mean±std

---

## 8. Output Files

### Per-Experiment Outputs

| File | Description |
|------|-------------|
| `imputed.csv` | Final imputed dataset |
| `metrics_per_feature.csv` | Per-feature imputation metrics |
| `metrics_summary.json/csv` | Aggregated metrics (continuous/categorical) |
| `dependency_matrix.csv` | Learned dependency matrix D |
| `dependency_matrix.png` | Heatmap visualization |
| `run_config.json` | Complete run configuration |
| `convergence_curve.csv` | (v0.3) Per-iteration loss values |
| `lambda_per_head.csv` | (v0.3) Learned λ values per head per target |
| `lambda_values.json` | (v0.3) Final λ snapshot |

### Aggregated Outputs

| File | Description |
|------|-------------|
| `summary_agg.csv` | Mean±std across seeds |
| `table_*.tex` | LaTeX-formatted tables |
| `parallel_run_meta.json` | Parallel execution metadata |

### Ext1 Audit Outputs

| File | Description |
|------|-------------|
| `audit_report.md` | One-paragraph audit summary (who dominates whom, leakage risk, performance impact) |
| `audit_top_sources.csv` | Top-5 source features per target |
| `audit_flags.csv` | Automatic flags for single-source dominance (`top1_weight ≥ 0.6`) |
| `audit_comparison.csv` | Performance comparison: with-leak vs without-leak |

### Ext1 Downstream Outputs

| File | Description |
|------|-------------|
| `metrics_per_seed.csv` | Per seed × method downstream performance |
| `metrics_summary.csv` | Aggregated mean ± std across seeds |

### Ext2 Per-Class Outputs

| File | Description |
|------|-------------|
| `perclass_metrics.csv` | Per-class Precision / Recall / F1 for each method and seed |
| `perclass_summary.csv` | Aggregated per-class summary across seeds |
| `collapse_flags.csv` | Flags for classes collapsed during imputation |

### Ext2 SHAP Comparison Outputs

| File | Description |
|------|-------------|
| `table_top_features.csv` | Top-k features from SNI D and MissForest SHAP (Table S7-ready) |
| `shap_importances.csv` | Full SHAP importance table |
| `spearman_d_vs_shap.csv` | Spearman rank correlation between D and SHAP |
| `d_matrix.csv` | Full SNI dependency matrix |

### Ext2 Attention Rollout Outputs (S7.B)

| File | Description |
|------|-------------|
| `table_S7B_aggregation_comparison.csv` | Top-k feature ranking under D / Rollout / Flow |
| `spearman_aggregation.csv` | Pairwise Spearman among three aggregation methods |
| `attention_per_head.csv` | Raw per-head attention weights |

### Ext2 D Stability Outputs (S7.C)

| File | Description |
|------|-------------|
| `table_S7C_d_stability.csv` | Mean ± std Spearman across seed pairs (stability index) |
| `pairwise_spearman.csv` | Per (seed_a, seed_b, target) Spearman values |
| `d_matrices/seed*.csv` | Individual D matrices per seed |

### Ext2 Significance Test Outputs

| File | Description |
|------|-------------|
| `wilcoxon_across_settings.csv` | Paired Wilcoxon tests across all settings (Table S8-ready) |
| `wilcoxon_per_setting.csv` | Per dataset×mechanism×metric tests across seeds |
| `wilcoxon_summary.csv` | Aggregated significance summary |

### Ext2 Impute→Predict Outputs

| File | Description |
|------|-------------|
| `per_seed_metrics.csv` | Per seed × method × model downstream metrics |
| `table_VI_summary.csv` | Mean±std over seeds (Table VI-ready) |

---

## 9. Troubleshooting

### GPU Memory Issues

Reduce parallel jobs or switch to CPU:
```bash
python scripts/run_manifest_parallel.py \
    --manifest data/manifest.csv \
    --outdir results \
    --n-jobs 4 \
    --default-use-gpu false
```

### Resume from Failures

Use `--skip-existing` to continue from last checkpoint:
```bash
python scripts/run_manifest_parallel.py \
    --manifest data/manifest.csv \
    --outdir results \
    --n-jobs 8 \
    --skip-existing
```

### Check Progress

```bash
# Completed experiments
find results -name "metrics_summary.json" | wc -l

# Failed experiments
find results -name "error.log" | wc -l

# View error logs
find results -name "error.log" -exec echo "=== {} ===" \; -exec cat {} \;
```

### Parallel Strategy Guide

| Server Config | Recommended `--n-jobs` | Notes |
|---------------|------------------------|-------|
| 32-core CPU + 1 GPU | 8 | Share GPU across workers |
| 32-core CPU + 4 GPU | 16-24 | Monitor GPU memory |
| 32-core CPU only | 24-32 | Add `--default-use-gpu false` |
| 8-core laptop | 2-4 | Reserve cores for system |

---

## 10. Changelog

### 2026-02-11
- 🚀 **SNI v0.3**: Major algorithmic upgrade
  - Learnable per-head λ coefficients (replacing fixed scalar)
  - Categorical balance mode (`inverse_freq`, `sqrt_inverse_freq`) for imbalanced targets
  - Convergence monitoring with early stopping
  - Runtime tracking (`runtime_seconds` in metrics)
  - New output artifacts: `convergence_curve.csv`, `lambda_per_head.csv`, `lambda_values.json`
- ✨ Added `baselines/HyperImpute_v1.py`: AutoML-based imputation (Jarrett et al., ICML 2022)
- ✨ Added `baselines/TabCSDI_v1.py`: Score-based conditional diffusion (Zheng & Charoenphakdee, NeurIPS TRL 2022)
- ✨ Added `ext2/scripts/exp4b_attention_rollout.py`: D vs attention rollout/flow comparison (Table S7.B)
- ✨ Added `ext2/scripts/exp4c_d_stability.py`: Cross-seed D stability analysis (Table S7.C)
- ✨ Added manifests: `manifest_sni_v03_main.csv`, `manifest_sni_v03_ablation_lambda.csv`, `manifest_baselines_new.csv`
- 📝 All scripts updated: imports SNI_v0_2 → SNI_v0_3, baselines expanded to 8 methods
- 📝 `configs/datasets.yaml`: Added `class_weights` and `recommended_cat_balance_mode` per dataset
- 📝 `exp1`: Added continuous proxy injection support (`--leak-noise-std`)
- 📝 `exp3`: Added `--sni-cat-balance-modes` for balanced mode comparison
- 📝 `exp4`: Added auto-discussion generation for ρ(D, SHAP) differences

### 2026-02-10
- ✨ Added `ext2/`: Additional experiments for paper placeholders
  - `exp3_per_class_categorical.py`: Per-class breakdown for imbalanced categorical targets
  - `exp4_shap_comparison.py`: SNI D vs SHAP on MissForest comparison
  - `exp5_significance_tests.py`: Paired Wilcoxon signed-rank significance tests
  - `exp6_mimic_mortality_impute_predict.py`: MIMIC-IV Impute→Predict with LR/XGBoost

### 2026-02-04
- ✨ Added `ext1/`: Extended experiments for reviewer response
  - `exp1_audit_story_leakage.py`: Interpretability audit via proxy injection on real data
  - `exp2_downstream_task_validation.py`: Downstream Impute → Predict validation with fairness gap

### 2026-01-16
- ✨ Added `synth_generate_s5.py`: Synthetic data generator for sanity check
- ✨ Added `sanity_check_v2_s5.py`: Dependency recovery experiments with `interaction_xor` setting
- 📝 Comprehensive README update for reproducibility

### 2026-01-01
- ✨ Added `run_manifest_parallel.py`: Multi-core parallel execution
- ✨ Added `split_manifest_runner.py`: Batch script generator for clusters
- ✨ Support `--skip-existing` for checkpoint recovery
- ✨ Support `--row-start/--row-end` for partial runs

### 2025-12-31
- 🎉 Initial release with complete experiment manifests

---

## License

This code is provided for peer review purposes. MIT License.
