# Baseline Methods

## Overview

All baselines are implemented in `baselines/` with a unified interface via `baselines/registry.py`.

| Method | Type | Implementation | File |
|--------|------|---------------|------|
| Mean/Mode | Statistical | scikit-learn SimpleImputer | `MeanMode_v1.py` |
| KNN | Statistical | scikit-learn KNNImputer | `KNN_v1.py` |
| MICE | Statistical | scikit-learn IterativeImputer | `MICE_v3.py` |
| MissForest | Ensemble | scikit-learn + custom wrapper | `MissForest_v2.py` |
| GAIN | Deep (GAN) | In-repo PyTorch implementation | `GAIN_v5.py` |
| MIWAE | Deep (VAE) | In-repo PyTorch implementation | `MIWAE_v3.py` |
| HyperImpute | AutoML | In-repo simplified implementation | `HyperImpute_v1.py` |
| TabCSDI | Deep (Diffusion) | In-repo simplified implementation | `TabCSDI_v1.py` |

## Fairness Principles

1. **Same hardware**: All methods run on the same machine; GPU usage is recorded per-run
2. **Same seeds**: Manifest-specified seeds ensure reproducibility
3. **Runtime reporting**: `runtime_sec` is recorded for every experiment
4. **Budget transparency**: HyperImpute and TabCSDI budget parameters are recorded in `metrics_summary.json` under `budget_params`

## Budget Settings

### HyperImpute
- Default timeout: 1800s (configurable via `--default-timeout`)
- Optimizer: AutoML search (bayesian)

### TabCSDI
- Epochs: as specified in manifest (default: 100)
- Diffusion steps: as specified in manifest (default: 50)
- Architecture: configurable d_model, n_layers

### GPU Usage
- Only GAIN, MIWAE, and TabCSDI support GPU acceleration
- GPU usage is recorded per-run in `metrics_summary.json`
- All statistical baselines (MeanMode, KNN, MICE, MissForest) run CPU-only

## Self-Test

Run the baseline self-test to verify all implementations work:

```bash
python scripts/baseline_selftest.py
```

This generates toy data and runs a quick sanity check.
