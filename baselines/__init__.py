"""Baseline imputers for SNI experiments.

This package contains baseline imputation engines used in the paper:

- Mean/Mode
- KNN (Gower distance)
- MICE
- MissForest
- GAIN
- MIWAE

Each baseline is wrapped with a consistent interface:

    imputer = build_baseline_imputer(method, categorical_vars, continuous_vars, seed=..., use_gpu=...)
    X_imputed = imputer.impute(X_complete, X_missing)

The wrappers also provide small safety features:
- enforce categorical category sets derived from *_complete.csv
- optional fallback filling if a method leaves some NaNs
"""

from .registry import build_baseline_imputer, list_baselines

__all__ = ["build_baseline_imputer", "list_baselines"]
