# SNI: Statistical-Neural Interaction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproducibility package for **Statistical-Neural Interaction Networks for Interpretable Imputation of Mixed Numerical and Categorical Tabular Data**.

SNI couples correlation-derived statistical priors with neural feature attention through a Controllable-Prior Feature Attention (CPFA) module, providing both strong imputation performance and an intrinsic dependency matrix for interpretability.

## Quick Install

```bash
conda create -n sni python=3.10 && conda activate sni
pip install -r requirements_core.txt       # main paper
pip install -r requirements_extra.txt      # supplement (optional)
```

> **GPU note**: The default `pip install torch` installs CPU-only PyTorch on most platforms.
> For GPU acceleration (used by GAIN, MIWAE, TabCSDI, and SNI), install PyTorch with CUDA
> support following the [official instructions](https://pytorch.org/get-started/locally/).

## Quick Start

```bash
# 1. Smoke test (verify installation, ~5 min on CPU)
python scripts/paper_quickstart.py --smoke-test --outdir results_quick

# 2. Run ALL paper experiments (one command, supports resume)
python scripts/run_all_experiments.py --outdir results_all --skip-existing --n-jobs-cpu 8

# 3. Run selected stages only (e.g., SNI + baselines + postprocess)
python scripts/run_all_experiments.py --outdir results_all --stages 1 2 7 --skip-existing

# 4. Post-process existing results only
python scripts/paper_quickstart.py --postprocess-only \
    --sni-results results_sni --baseline-results results_baselines --outdir results_paper
```

## Repository Structure

```
SNI_v0_3/                      # Core SNI algorithm (v0.3, paper final version)
baselines/                     # 8 baseline methods (MeanMode, KNN, MICE, MissForest, GAIN, MIWAE, HyperImpute, TabCSDI)
scripts/                       # Experiment runners, aggregation, visualization
ext1/                          # Extended: interpretability audit, downstream validation
ext2/                          # Extended: per-class, SHAP, significance, Impute-Predict
data/                          # 6 datasets + experiment manifests
configs/                       # Dataset configs + paper profiles (main/supplement/all)
docs/                          # Detailed documentation
utility_missing_data_gen_v1/   # Missing data generator (MCAR/MAR/MNAR patterns)
```

## Results Output

```
results_all/                       # Full experiment output (run_all_experiments.py)
├── sni_v03_main/                  # SNI main experiments
├── baselines_main/                # Classic baselines (MCAR/MAR)
├── baselines_deep/                # Deep baselines (GAIN, MIWAE)
├── baselines_new/                 # New baselines (HyperImpute, TabCSDI)
├── ext1/ ext2/                    # Extension experiments
├── agg_*/                         # Per-group aggregated summaries
├── merged/merged_summary_agg.csv  # Merged metrics across all groups
├── tables/                        # LaTeX tables (main profile)
├── tables_supplement/             # LaTeX tables (supplement profile)
├── figures/                       # PNG figures (main profile)
└── figures_supplement/            # PNG figures (supplement profile)
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Quick start commands |
| [docs/full_reproduction.md](docs/full_reproduction.md) | Step-by-step full reproduction |
| [docs/evidence_map.md](docs/evidence_map.md) | Paper artifact traceability |
| [docs/baselines.md](docs/baselines.md) | Baseline methods and fairness |
| [docs/results_structure.md](docs/results_structure.md) | Output directory structure |
| [docs/reviewer_risk_checklist.md](docs/reviewer_risk_checklist.md) | Reviewer risk checklist |
| [readme_CL.txt](readme_CL.txt) | Complete command reference |

## Datasets

| Dataset | Samples | Features | Domain |
|---------|---------|----------|--------|
| MIMIC | 2,052 | 8 | Healthcare (ICU) |
| eICU | 1,430 | 20 | Healthcare (ICU) |
| NHANES | 2,274 | 12 | Public Health |
| Concrete | 1,030 | 9 | Engineering |
| ComCri | 1,994 | 10 | Social Science |
| AutoMPG | 392 | 8 | Automotive |

Missingness mechanisms: MCAR, MAR, MNAR at 10%, 30%, 50% rates.

## Citation

```bibtex
@article{SNI2025,
  title={Statistical-Neural Interaction Networks for Interpretable Imputation of Mixed Numerical and Categorical Tabular Data},
  author={Ou Deng, Shoji Nishimura, Atsushi Ogihara and Qun Jin},
  journal={arXiv},
  year={2026}
}
```

## License

MIT License
