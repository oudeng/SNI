# SNI: Statistical–Neural Interaction （SNI）

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ZENODO](https://zenodo.org/badge/DOI/10.5281/zenodo.18286410.svg)](https://zenodo.org/records/18286410)

**Reproducibility Package**

This repository contains the complete experimental code for the paper, Statistical-Neural Interaction Networks for Interpretable Mixed-Type Data Imputation.

It enables full reproduction of all experiments, tables, and figures presented in the manuscript.

Statistical--Neural Interaction (SNI), an interpretable mixed-type imputation framework that couples correlation-derived statistical priors with neural feature attention through a Controllable-Prior Feature Attention (CPFA) module. CPFA learns head-wise prior-strength coefficients $\{\lambda_h\}$ that softly regularize attention toward the prior while allowing data-driven deviations when nonlinear patterns appear to be present in the data. Beyond imputation, SNI aggregates attention maps into a directed feature-dependency matrix that summarizes which variables the imputer relied on, without requiring post-hoc explainers.

[![](https://github.com/oudeng/SNI/blob/main/Graphical_Abstract/graphical_abstract.pdf)]

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
| `SNI_v0_2/` | Core SNI algorithm package (importable module) |
| `baselines/` | Baseline imputation methods (MeanMode, KNN, MICE, MissForest, GAIN, MIWAE) |
| `scripts/` | Experiment runners, aggregation, visualization, and table generation |
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

# Verify installation
python -c "from SNI_v0_2.imputer import SNIImputer; print('SNI ready!')"
```

### Dependencies Summary

- `torch>=1.12`: Deep learning framework
- `scikit-learn>=1.0`: ML utilities and MICE
- `pandas>=1.4`: Data manipulation
- `numpy>=1.21`: Numerical computing
- `joblib`: Parallel execution
- `matplotlib>=3.5`: Visualization

---

## 3. Repository Structure

```
.
├── SNI_v0_2/                   # Core SNI algorithm
│   ├── imputer.py              # Main imputer class
│   ├── cpfa.py                 # Cross-Feature Attention module
│   ├── metrics.py              # Evaluation metrics
│   └── ...
├── baselines/                  # Baseline methods
│   ├── MeanMode_v1.py          # Mean/Mode imputation
│   ├── KNN_v1.py               # K-Nearest Neighbors
│   ├── MICE_v3.py              # Multiple Imputation by Chained Equations
│   ├── MissForest_v2.py        # Random Forest-based imputation
│   ├── GAIN_v5.py              # Generative Adversarial Imputation Nets
│   ├── MIWAE_v3.py             # Missing data IWAE
│   └── registry.py             # Unified baseline interface
├── scripts/
│   ├── run_experiment.py       # Single experiment runner
│   ├── run_manifest_parallel.py    # Parallel batch runner for SNI
│   ├── run_manifest_baselines.py   # Parallel batch runner for baselines
│   ├── aggregate_results.py    # Results aggregation (mean±std)
│   ├── make_latex_table.py     # LaTeX table generation
│   ├── synth_generate_s5.py    # Synthetic data generator
│   ├── sanity_check_v2_s5.py   # Sanity check experiments (Section S5)
│   └── viz_*.py                # Visualization scripts
├── data/
│   ├── *_complete.csv          # Complete datasets
│   ├── {Dataset}/              # Missing datasets with masks
│   ├── synth_s5/               # Synthetic datasets for sanity check
│   ├── manifest_sni_*.csv      # SNI experiment manifests
│   └── manifest_baselines_*.csv # Baseline experiment manifests
├── configs/
│   └── datasets.yaml           # Dataset configurations
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

### 5.1 Main Experiments (Table 2-3)

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

### 5.5 Sanity Check: Dependency Recovery (Section S5)

**Objective**: Validate that the learned dependency matrix D recovers true causal structure.

**Method**: 
1. Generate synthetic data with known DAG structure
2. Compare dependency recovery metrics: AUROC, AUPRC, Precision@K, Hub Spearman
3. Evaluate SNI vs NoPrior vs PriorOnly (correlation-based)

**Key Insight**: `interaction_xor` setting specifically tests scenarios where marginal correlations are near-zero but true dependencies exist, demonstrating SNI's advantage over correlation-only methods.

---

## 6. Quick Start

### Run a Single Experiment

```bash
python scripts/run_experiment.py \
  --input-complete data/MIMIC_complete.csv \
  --input-missing data/MIMIC/MIMIC_MAR_30per.csv \
  --categorical-vars SpO2 \
  --continuous-vars RESP ABP SBP DBP HR PULSE ALARM \
  --variant SNI \
  --seed 8 \
  --outdir results/test_run
```

### Run Parallel Batch Experiments

```bash
# SNI experiments (8 parallel processes)
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_main.csv \
    --outdir results_sni_main \
    --n-jobs 8

# Baseline experiments
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_main.csv \
    --outdir results_baselines_main \
    --n-jobs 8
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

### Step 5: Sanity Check (Section S5)

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

### Aggregated Outputs

| File | Description |
|------|-------------|
| `summary_agg.csv` | Mean±std across seeds |
| `table_*.tex` | LaTeX-formatted tables |
| `parallel_run_meta.json` | Parallel execution metadata |

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
