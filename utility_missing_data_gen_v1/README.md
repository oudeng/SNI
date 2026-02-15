# Missing Data Generator

A standalone toolkit for generating controlled missing data patterns in tabular datasets. This module supports MCAR, MAR, and MNAR missingness mechanisms with calibrated missing rates.

## Overview

This toolkit generates missing data from complete datasets following rigorous statistical procedures. The generated missing patterns are reproducible and can be used for fair comparison across multiple imputation methods.

## Files

| File | Description |
|------|-------------|
| `missing_data_generator.py` | Core module: MCAR/MAR/MNAR generation, rate calibration, column typing |
| `generate_missing_data.py` | CLI controller: batch generation, naming conventions, export utilities |
| `01_add_ID.py` | Utility for adding unique ID columns to datasets |

## Quick Start

```bash
python generate_missing_data.py \
  --input /path/to/complete.csv \
  --output-dir /path/to/output \
  --dataset DatasetName \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --save-mask --save-metadata
```

## Missingness Mechanisms

### MCAR (Missing Completely at Random)
Values are missing independently of any data values. Each cell has an equal probability of being missing.

### MAR (Missing at Random)
Missingness probability depends on observed values. For example, older patients may have more missing lab values.

### MNAR (Missing Not at Random)
Missingness depends on the unobserved values themselves. For example, extreme measurements may be more likely to be missing due to sensor limitations.

## Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input` | Input CSV path (complete data) | `data/MIMIC_complete.csv` |
| `--output-dir` | Output directory | `data/MIMIC/` |
| `--dataset` | Dataset name (for file naming) | `MIMIC` |
| `--mechanisms` | Mechanisms to generate | `MCAR MAR MNAR` |
| `--rates` | Target missing rates | `0.1 0.3 0.5` |
| `--seed` | Random seed | `2025` |
| `--categorical-cols` | Categorical column names | `"SpO2,ALARM"` |
| `--continuous-cols` | Continuous column names | `"HR,BP"` |
| `--exclude-cols` | Columns to exclude | `"ID,subject_id"` |
| `--mar-driver-cols` | MAR driver columns | `"age,bp"` |
| `--tolerance` | Missing rate tolerance | `0.01` |
| `--min-missing-per-col` | Minimum missing per column | `1` |
| `--rate-style` | Filename rate format | `per`, `float`, `p` |
| `--save-mask` | Save binary mask | (flag) |
| `--save-metadata` | Save generation metadata | (flag) |
| `--overwrite` | Overwrite existing files | (flag) |

## Output Files

For each mechanism × rate combination:

```
{dataset}_{mechanism}_{rate}.csv       # Missing data (NaN values)
{dataset}_{mechanism}_{rate}_mask.npy  # Binary mask (True = missing)
{dataset}_{mechanism}_{rate}_meta.json # Metadata
```

### Filename Rate Styles

| Style | Example |
|-------|---------|
| `per` (default) | `MIMIC_MAR_30per.csv` |
| `float` | `MIMIC_MAR_0.3.csv` |
| `p` | `MIMIC_MAR_0p3.csv` |

## Usage Examples

### Generate all patterns for MIMIC dataset

```bash
python generate_missing_data.py \
  --input data/MIMIC_complete.csv \
  --output-dir data/MIMIC/ \
  --dataset MIMIC \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "SpO2,ALARM" \
  --exclude-cols ID \
  --save-mask --save-metadata
```

### Generate patterns for Concrete dataset (all continuous)

```bash
python generate_missing_data.py \
  --input data/Concrete_complete.csv \
  --output-dir data/Concrete/ \
  --dataset Concrete \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "" \
  --exclude-cols ID \
  --save-mask --save-metadata
```

### Add ID column to raw dataset

```bash
python 01_add_ID.py \
  --input data/raw.csv \
  --output data/with_id.csv
```

## Important Notes

### Column Type Specification

Many datasets store categorical features as integers (e.g., `SpO2=97/98/99/100`). Always explicitly specify categorical columns to avoid misclassification:

```bash
--categorical-cols "SpO2,Sex,DIQ010"
```

### Rate Calibration

The generator uses propensity-based calibration to achieve target missing rates while preserving the missingness structure:
- If missing rate is too low: prioritize masking cells with higher propensity
- If missing rate is too high: restore cells with lower propensity

This approach maintains mechanism consistency better than random adjustment.

### Reproducibility

The `--save-mask` and `--save-metadata` options are strongly recommended for reproducibility:
- **mask.npy**: Exact binary missing pattern
- **meta.json**: Seed, actual rates, column types, per-column rates

## Metadata JSON Structure

```json
{
  "seed": 2025,
  "mechanism": "MAR",
  "target_rate": 0.3,
  "actual_rate": 0.298,
  "column_types": {
    "SpO2": "categorical",
    "HR": "continuous"
  },
  "per_column_missing_rate": {
    "SpO2": 0.31,
    "HR": 0.28
  },
  "mar_driver_cols": ["age", "bp"],
  "excluded_cols": ["ID"]
}
```

## Integration with SNI Experiments

For fair comparison across imputation methods:

1. Generate missing patterns once with fixed seed
2. Save both CSV and mask files
3. All methods read the same missing CSV (or apply the same mask)
4. Evaluation uses the mask to identify originally missing positions

This ensures all methods are evaluated on identical missing patterns.
