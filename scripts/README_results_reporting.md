# Results Aggregation & Paper-Ready Reporting

This document describes the results aggregation and reporting workflow. **Most of these commands are already included in `readme_CL.txt`** — this document provides additional context and advanced options.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Result Directories Reference](#2-result-directories-reference)
3. [Aggregation Script](#3-aggregation-script)
4. [LaTeX Table Generation](#4-latex-table-generation)
5. [Visualization Scripts](#5-visualization-scripts)
6. [One-Command Report Bundle](#6-one-command-report-bundle)
7. [Advanced Options](#7-advanced-options)

---

## 1. Overview

The reporting pipeline consists of three stages:

```
Raw Results                    Aggregated Results              Final Outputs
(per-experiment)               (summary statistics)            (tables + figures)
─────────────────────────────────────────────────────────────────────────────────
results_*/*/                   results_*/_summary/             results_*/_tables/
├── metrics_summary.json  ───► ├── summary_all.csv        ───► ├── table_*.tex
├── imputed.csv                ├── summary_agg.csv             └── flat_table.tex
├── dependency_matrix.csv      ├── summary_overall_by_algo.csv
└── ...                        ├── summary_ranks.csv           figs/
                               └── summary_wins.csv        ───► ├── heatmap_*.png
                                                               ├── overall_bar_*.png
                                                               └── ...
```

---

## 2. Result Directories Reference

After running `readme_CL.txt`, you will have:

| Experiment | Results Directory | Summary Directory | Tables Directory |
|------------|-------------------|-------------------|------------------|
| SNI Main | `results_sni_main/` | `results_sni_main/_summary/` | `results_sni_main/_tables/` |
| SNI Ablation | `results_sni_ablation/` | `results_sni_ablation/_summary/` | `results_sni_ablation/_tables/` |
| SNI MNAR | `results_sni_mnar/` | `results_sni_mnar/_summary/` | `results_sni_mnar/_tables/` |
| Baselines Main | `results_baselines_main/` | `results_baselines_main/_summary/` | `results_baselines_main/_tables/` |
| Baselines MNAR | `results_baselines_mnar/` | `results_baselines_mnar/_summary/` | `results_baselines_mnar/_tables/` |
| Baselines Deep | `results_baselines_deep/` | `results_baselines_deep/_summary/` | `results_baselines_deep/_tables/` |
| Sanity Check | `results_sanity_s5/` | (inline) | (inline) |

---

## 3. Aggregation Script

### Basic Usage (already in readme_CL.txt)

```bash
# SNI Main experiments
python scripts/aggregate_results.py \
    --results-root results_sni_main \
    --outdir results_sni_main/_summary

# Ablation study
python scripts/aggregate_results.py \
    --results-root results_sni_ablation \
    --outdir results_sni_ablation/_summary

# MNAR experiments
python scripts/aggregate_results.py \
    --results-root results_sni_mnar \
    --outdir results_sni_mnar/_summary

# Baselines
python scripts/aggregate_results.py \
    --results-root results_baselines_main \
    --outdir results_baselines_main/_summary

python scripts/aggregate_results.py \
    --results-root results_baselines_mnar \
    --outdir results_baselines_mnar/_summary

python scripts/aggregate_results.py \
    --results-root results_baselines_deep \
    --outdir results_baselines_deep/_summary
```

### Output Files

| File | Description |
|------|-------------|
| `summary_all.csv` | One row per experiment run (raw data) |
| `summary_agg.csv` | One row per (dataset, mechanism, rate, algo), with `*_mean` and `*_std` |
| `summary_overall_by_algo.csv` | Algorithm-level mean±std across all tasks |
| `summary_ranks.csv` | Per-task ranking of algorithms |
| `summary_ranks_agg.csv` | Mean rank per algorithm |
| `summary_wins.csv` | Win count per algorithm per metric |
| `summary_rel_to_ref.csv` | Relative performance vs reference algorithm (default: SNI) |

---

## 4. LaTeX Table Generation

### Basic Usage (already in readme_CL.txt)

```bash
# Generate tables for all tasks
python scripts/make_latex_table.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --outdir results_sni_main/_tables
```

### Filter by Mechanism/Rate

```bash
# Only MAR @ 30% (common paper setting)
python scripts/make_latex_table.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --outdir results_sni_main/_tables/MAR30 \
    --task-filter "mechanism=='MAR' and rate=='30per'" \
    --also-write-flat false

# Only MCAR @ 30%
python scripts/make_latex_table.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --outdir results_sni_main/_tables/MCAR30 \
    --task-filter "mechanism=='MCAR' and rate=='30per'" \
    --also-write-flat false
```

### Output Format

Each per-task table is saved as:
- `table_<dataset>_<mechanism>_<rate>.tex`

Best values per metric are **bolded** (ties allowed).

---

## 5. Visualization Scripts

> **Note**: See `scripts/viz_README.md` for detailed visualization workflow.

### Quick Comprehensive Figures

```bash
# Generate all figures for main experiments
python scripts/viz_make_figures.py \
    --summary-dir results_sni_main/_summary \
    --outdir figs/comprehensive/main \
    --dpi 600
```

### Merge Multiple Result Sets

```bash
# Merge SNI + Baselines results
python scripts/viz_make_figures.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --summary-csv results_baselines_main/_summary/summary_agg.csv \
    --summary-csv results_baselines_deep/_summary/summary_agg.csv \
    --outdir figs/merged \
    --write-merged \
    --dpi 600
```

---

## 6. One-Command Report Bundle

For convenience, you can generate all reports at once:

```bash
python scripts/report_generate_all.py \
    --results-main results_sni_main \
    --results-ablation results_sni_ablation \
    --results-mnar results_sni_mnar \
    --out-root reports
```

This creates:

```
reports/
├── main/
│   ├── _summary/          # Aggregated CSVs
│   ├── _figs/             # Comprehensive figures
│   └── _tables/           # LaTeX tables
│       └── MAR30/         # MAR@30% specific tables
├── ablation/
│   ├── _summary/
│   ├── _figs/
│   └── _tables/
└── mnar/
    ├── _summary/
    ├── _figs/
    └── _tables/
```

> **Note**: This is separate from the individual `results_*/_summary` directories created by `readme_CL.txt`. Use this if you want a consolidated report bundle.

---

## 7. Advanced Options

### Custom Reference Algorithm

```bash
# Compare relative to MissForest instead of SNI
python scripts/aggregate_results.py \
    --results-root results_sni_main \
    --outdir results_sni_main/_summary \
    --ref-algo MissForest
```

### Include Deep Baselines in Combined Analysis

```bash
# Merge all results for comprehensive comparison
python scripts/viz_make_figures.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --summary-csv results_sni_ablation/_summary/summary_agg.csv \
    --summary-csv results_baselines_main/_summary/summary_agg.csv \
    --summary-csv results_baselines_deep/_summary/summary_agg.csv \
    --outdir figs/all_combined \
    --write-merged \
    --top-k 15 \
    --dpi 600
```

### Filter Specific Algorithms

```bash
# Generate figures for specific algorithms only
python scripts/viz_make_figures.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --summary-csv results_baselines_main/_summary/summary_agg.csv \
    --outdir figs/main_top_methods \
    --top-k 6 \
    --mechanisms MCAR,MAR \
    --rates 30per
```

---

## Summary: What's in readme_CL.txt vs This Document

| Task | In readme_CL.txt? | This Document |
|------|-------------------|---------------|
| Run experiments | ✅ Yes | - |
| Aggregate results | ✅ Yes | Additional options |
| Generate LaTeX tables | ✅ Yes | Filter options |
| Comprehensive figures | ❌ No | ✅ See viz_README.md |
| Paper-ready figures | ❌ No | ✅ See viz_README.md |
| One-command bundle | ❌ No | ✅ report_generate_all.py |

---

## Workflow Recommendation

1. **Run all experiments** following `readme_CL.txt` (PART 1-3)
2. **Aggregation and tables** are automatically done in `readme_CL.txt`
3. **Additional visualizations**: Follow `scripts/viz_README.md` for paper figures
4. **(Optional)** Use `report_generate_all.py` for a consolidated report bundle