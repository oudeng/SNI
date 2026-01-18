from __future__ import annotations

"""Baseline registry.

This file wraps the baseline scripts into a single consistent API.

Updated: 2025-01-04
- MICE: v2 → v3 (PMM implementation)
- MissForest: v1 → v2 (stopping criterion γ)
- MIWAE: v2 → v3 (proper variance, importance weighting)
- GAIN: v3 → v5 (official hyperparameters)
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .utils import fallback_fillna, set_categories_from_complete

# Updated baseline implementations (aligned with official papers)
from .GAIN_v5 import GAINImputer
from .KNN_v1 import knnImputer
from .MeanMode_v1 import MeanModeImputer
from .MICE_v3 import MICEImputer
from .MissForest_v2 import MissForestImputer
from .MIWAE_v3 import MIWAEImputer


def _filter_kwargs(kwargs: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
    """Return a new kwargs dict containing only the allowed keys with non-NaN values."""
    out: Dict[str, Any] = {}
    for k in allowed:
        if k not in kwargs:
            continue
        v = kwargs[k]
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        out[k] = v
    return out


class BaseBaseline:
    method: str

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class MeanModeBaseline(BaseBaseline):
    categorical_vars: List[str]
    continuous_vars: List[str]

    def __post_init__(self):
        self.method = "MeanMode"
        self._impl = MeanModeImputer(self.categorical_vars, self.continuous_vars)

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        X_imp, _ = self._impl.impute(X_complete, X_missing)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


@dataclass
class KNNBaseline(BaseBaseline):
    categorical_vars: List[str]
    continuous_vars: List[str]
    k: int = 5

    def __post_init__(self):
        self.method = "KNN"
        self._impl = knnImputer(self.categorical_vars, self.continuous_vars, k=int(self.k))

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        X_imp, _ = self._impl.impute(X_complete, X_missing)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


@dataclass
class MICEBaseline(BaseBaseline):
    """
    MICE v3 - with Predictive Mean Matching (PMM)
    
    Reference: Van Buuren & Groothuis-Oudshoorn (2011), JSS 45(3)
    """
    categorical_vars: List[str]
    continuous_vars: List[str]
    max_iter: int = 5           # Official default: maxit=5
    seed: int = 42
    # PMM parameters (official defaults)
    donors: int = 5             # Donor pool size (official default: 5)
    matchtype: int = 1          # Type I matching (official default)
    ridge: float = 1e-5         # Ridge regularization

    def __post_init__(self):
        self.method = "MICE"
        self._impl = MICEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            max_iter=int(self.max_iter),
            seed=int(self.seed),
            donors=int(self.donors),
            matchtype=int(self.matchtype),
            ridge=float(self.ridge),
        )

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


@dataclass
class MissForestBaseline(BaseBaseline):
    """
    MissForest v2 - with stopping criterion γ
    
    Reference: Stekhoven & Bühlmann (2012), Bioinformatics 28(1)
    """
    categorical_vars: List[str]
    continuous_vars: List[str]
    n_estimators: int = 100     # Official default: ntree=100
    max_iter: int = 10          # Official default: maxiter=10
    seed: int = 42
    n_jobs: int = -1
    verbose: bool = False
    decreasing: bool = False    # Variable ordering (official default: ascending by missing count)

    def __post_init__(self):
        self.method = "MissForest"
        self._impl = MissForestImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            n_estimators=int(self.n_estimators),
            max_iter=int(self.max_iter),
            seed=int(self.seed),
            n_jobs=int(self.n_jobs),
            verbose=bool(self.verbose),
            decreasing=bool(self.decreasing),
        )

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        # Ensure category mapping uses the complete label set.
        _, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xm)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


@dataclass
class GAINBaseline(BaseBaseline):
    """
    GAIN v4 - with official hyperparameters
    
    Reference: Yoon et al. (2018), ICML
    Official code: https://github.com/jsyoon0823/GAIN
    """
    categorical_vars: List[str]
    continuous_vars: List[str]
    seed: int = 42
    use_gpu: bool = False
    # Official hyperparameters
    hidden_dim: int = 256       # Official default
    batch_size: int = 128       # Official default
    hint_rate: float = 0.9      # Official default
    alpha: float = 100.0        # Official default (reconstruction loss weight)
    iterations: int = 10000     # Official default
    learning_rate: float = 1e-3 # Official default

    def __post_init__(self):
        self.method = "GAIN"
        self._impl = GAINImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dim=int(self.hidden_dim),
            batch_size=int(self.batch_size),
            hint_rate=float(self.hint_rate),
            alpha=float(self.alpha),
            iterations=int(self.iterations),
            learning_rate=float(self.learning_rate),
            seed=int(self.seed),
            use_gpu=bool(self.use_gpu),
        )

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


@dataclass
class MIWAEBaseline(BaseBaseline):
    """
    MIWAE v3 - with proper variance learning and importance weighting
    
    Reference: Mattei & Frellsen (2019), ICML
    Official code: https://github.com/pamattei/miwae
    """
    categorical_vars: List[str]
    continuous_vars: List[str]
    seed: int = 42
    use_gpu: bool = False
    # Architecture (paper Section 4.3)
    hidden_dims: str = "128,128,128"  # 3 layers, 128 units each
    latent_dim: int = 10              # Paper UCI experiments
    # Training
    num_iw_samples: int = 20          # K=20 (paper UCI)
    num_impute_samples: int = 10000   # L=10000 (paper imputation)
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 500
    min_epochs: int = 200
    # Variance constraint (paper Section 4.3)
    min_variance: float = 0.01

    def __post_init__(self):
        self.method = "MIWAE"
        # Parse hidden_dims string to list
        if isinstance(self.hidden_dims, str):
            hidden_dims_list = [int(x.strip()) for x in self.hidden_dims.split(",")]
        else:
            hidden_dims_list = list(self.hidden_dims)
        
        self._impl = MIWAEImputer(
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            hidden_dims=hidden_dims_list,
            latent_dim=int(self.latent_dim),
            num_iw_samples=int(self.num_iw_samples),
            num_impute_samples=int(self.num_impute_samples),
            lr=float(self.lr),
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            min_epochs=int(self.min_epochs),
            seed=int(self.seed),
            use_gpu=bool(self.use_gpu),
            min_variance=float(self.min_variance),
        )

    def impute(self, X_complete: pd.DataFrame, X_missing: pd.DataFrame) -> pd.DataFrame:
        Xc, Xm = set_categories_from_complete(X_complete, X_missing, self.categorical_vars)
        X_imp, _ = self._impl.impute(Xc, Xm)
        X_imp = fallback_fillna(X_imp, X_complete, self.categorical_vars, self.continuous_vars)
        return X_imp


_REGISTRY = {
    "MeanMode": MeanModeBaseline,
    "KNN": KNNBaseline,
    "MICE": MICEBaseline,
    "MissForest": MissForestBaseline,
    "GAIN": GAINBaseline,
    "MIWAE": MIWAEBaseline,
}


def list_baselines() -> List[str]:
    return sorted(_REGISTRY.keys())


def build_baseline_imputer(
    method: str,
    categorical_vars: List[str],
    continuous_vars: List[str],
    *,
    seed: int = 42,
    use_gpu: bool = False,
    **kwargs: Any,
) -> BaseBaseline:
    """Build a baseline imputer by name."""
    m = str(method).strip()
    if m not in _REGISTRY:
        raise KeyError(f"Unknown baseline method '{method}'. Available: {list_baselines()}")

    cls = _REGISTRY[m]

    # Extract per-method kwargs (updated for v3/v4 implementations)
    if m == "KNN":
        allowed = ["k"]
    elif m == "MICE":
        # v3 parameters
        allowed = ["max_iter", "donors", "matchtype", "ridge"]
    elif m == "MissForest":
        # v2 parameters
        allowed = ["n_estimators", "max_iter", "n_jobs", "verbose", "decreasing"]
    elif m == "GAIN":
        # v4 parameters
        allowed = ["hidden_dim", "batch_size", "hint_rate", "alpha", "iterations", "learning_rate"]
    elif m == "MIWAE":
        # v3 parameters
        allowed = ["hidden_dims", "latent_dim", "num_iw_samples", "num_impute_samples", 
                   "lr", "batch_size", "epochs", "min_epochs", "min_variance"]
    else:
        allowed = []

    filtered = _filter_kwargs(kwargs, allowed)

    # Baselines that accept seed/use_gpu
    if m in {"MICE", "MissForest", "GAIN", "MIWAE"}:
        filtered.update({"seed": int(seed)})
    if m in {"GAIN", "MIWAE"}:
        filtered.update({"use_gpu": bool(use_gpu)})

    return cls(categorical_vars=categorical_vars, continuous_vars=continuous_vars, **filtered)