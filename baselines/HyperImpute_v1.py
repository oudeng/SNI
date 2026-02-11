# HyperImpute_v1.py
# -*- coding: utf-8 -*-
"""
HyperImpute - Automated imputation via AutoML column-wise model selection.

Wrapper around the ``hyperimpute`` package that provides a consistent
interface matching the other baselines in this repository.

HyperImpute automatically selects the best imputation model per column
via Hyperband-based search over a configurable set of learners.

Reference:
    Jarrett, D., Cebere, B. C., Liu, T., Curth, A., & van der Schaar, M. (2022).
    HyperImpute: Generalized Iterative Imputation with Automatic Model Selection.
    ICML 2022.

Package: pip install hyperimpute
"""

from __future__ import annotations

import signal
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class HyperImputeImputer:
    """
    HyperImpute imputer using the official ``hyperimpute`` package.

    Key characteristics:
    - AutoML-based: selects the best model per column via Hyperband
    - Does not distinguish continuous/categorical internally (auto-detected)
    - We encode categoricals to integer codes before calling the plugin
      and decode back afterward (consistent with MICE/MissForest)
    """

    def __init__(
        self,
        categorical_vars: Optional[List[str]] = None,
        continuous_vars: Optional[List[str]] = None,
        seed: int = 42,
        timeout: int = 600,
        optimizer: str = "hyperband",
        classifier_seed: Optional[List[str]] = None,
        regression_seed: Optional[List[str]] = None,
    ):
        """
        Args:
            categorical_vars: Column names of categorical features.
            continuous_vars: Column names of continuous features.
            seed: Random seed for reproducibility.
            timeout: Maximum runtime in seconds (default 600).
            optimizer: Search strategy ('hyperband' or 'simple').
            classifier_seed: Classifier candidates for categorical columns.
            regression_seed: Regressor candidates for continuous columns.
        """
        self.categorical_vars = list(categorical_vars or [])
        self.continuous_vars = list(continuous_vars or [])
        self.seed = seed
        self.timeout = timeout
        self.optimizer = optimizer
        self.classifier_seed = classifier_seed or [
            "logistic_regression",
            "random_forests",
            "xgboost",
        ]
        self.regression_seed = regression_seed or [
            "linear_regression",
            "random_forests",
            "xgboost",
        ]
        self.models_: Dict = {}

    def impute(
        self,
        X_complete: pd.DataFrame,
        X_missing: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run HyperImpute on *X_missing*.

        Args:
            X_complete: Ground-truth DataFrame (used only for category mapping,
                        NOT for supervision — consistent with other baselines).
            X_missing: DataFrame with NaN at missing positions.

        Returns:
            X_imputed: Imputed DataFrame (same shape/columns as X_missing).
            models_: Empty dict (HyperImpute manages models internally).
        """
        try:
            from hyperimpute.plugins.imputers import Imputers
        except ImportError as exc:
            raise ImportError(
                "HyperImpute requires the 'hyperimpute' package. "
                "Install it with: pip install hyperimpute"
            ) from exc

        df = X_missing.copy()

        # --- Encode categorical columns to integer codes ---
        category_mappings: Dict[str, list] = {}
        for col in self.categorical_vars:
            if col not in df.columns:
                continue
            # Build categories from complete data to ensure full label set
            if col in X_complete.columns:
                all_cats = X_complete[col].dropna().unique().tolist()
                try:
                    all_cats = sorted(all_cats)
                except Exception:
                    pass
            else:
                all_cats = df[col].dropna().unique().tolist()

            category_mappings[col] = all_cats
            cat_type = pd.CategoricalDtype(categories=all_cats)
            df[col] = df[col].astype(cat_type).cat.codes.astype(float)
            # cat.codes returns -1 for NaN — convert back to NaN
            df.loc[df[col] < 0, col] = np.nan

        # --- Ensure continuous columns are numeric ---
        for col in self.continuous_vars:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Set random seeds ---
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass

        # --- Build HyperImpute plugin ---
        plugin = Imputers().get(
            "hyperimpute",
            optimizer=self.optimizer,
            classifier_seed=self.classifier_seed,
            regression_seed=self.regression_seed,
            random_state=self.seed,
        )

        # --- Run with timeout ---
        df_imputed_values = self._run_with_timeout(plugin, df)

        # --- Build output DataFrame ---
        df_imputed = pd.DataFrame(
            df_imputed_values,
            columns=df.columns,
            index=df.index,
        )

        # --- Decode categorical columns back to original labels ---
        for col in self.categorical_vars:
            if col not in df_imputed.columns:
                continue
            if col not in category_mappings:
                continue

            original_categories = category_mappings[col]
            max_code = len(original_categories) - 1

            # Round to nearest integer code and clip
            codes = df_imputed[col].values.astype(float)
            codes = np.round(codes).astype(int)
            codes = np.clip(codes, 0, max_code)

            decoded = [original_categories[int(c)] for c in codes]
            df_imputed[col] = decoded

        # --- Ensure continuous columns are float ---
        for col in self.continuous_vars:
            if col in df_imputed.columns:
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

        return df_imputed, self.models_

    def _run_with_timeout(self, plugin, df: pd.DataFrame) -> np.ndarray:
        """Run HyperImpute with a timeout guard."""

        class TimeoutError(Exception):
            pass

        def _handler(signum, frame):
            raise TimeoutError(
                f"HyperImpute exceeded timeout of {self.timeout}s"
            )

        # Try signal-based timeout (Unix only)
        use_signal = hasattr(signal, "SIGALRM")

        if use_signal:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.timeout)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                result = plugin.fit_transform(df)

            if hasattr(result, "values"):
                return result.values
            return np.asarray(result)

        except TimeoutError:
            warnings.warn(
                f"HyperImpute timed out after {self.timeout}s. "
                "Returning partially imputed result with mean/mode fallback.",
                RuntimeWarning,
            )
            # Fallback: fill remaining NaN with mean/mode
            df_fallback = df.copy()
            for col in df_fallback.columns:
                if df_fallback[col].isna().any():
                    if col in self.categorical_vars:
                        mode_val = df_fallback[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    else:
                        fill_val = df_fallback[col].mean(skipna=True)
                        if pd.isna(fill_val):
                            fill_val = 0.0
                    df_fallback[col] = df_fallback[col].fillna(fill_val)
            return df_fallback.values

        except Exception as exc:
            warnings.warn(
                f"HyperImpute failed: {exc}. "
                "Returning mean/mode fallback imputation.",
                RuntimeWarning,
            )
            df_fallback = df.copy()
            for col in df_fallback.columns:
                if df_fallback[col].isna().any():
                    if col in self.categorical_vars:
                        mode_val = df_fallback[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    else:
                        fill_val = df_fallback[col].mean(skipna=True)
                        if pd.isna(fill_val):
                            fill_val = 0.0
                    df_fallback[col] = df_fallback[col].fillna(fill_val)
            return df_fallback.values

        finally:
            if use_signal:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


# Backward-compatible alias
hyperImputeImputer = HyperImputeImputer


if __name__ == "__main__":
    # Simple test
    print("HyperImpute v1 test")
    print("=" * 60)

    np.random.seed(42)
    n = 200

    x1 = np.random.randn(n) * 10 + 50
    x2 = np.random.randn(n) * 5 + 20
    x3 = x1 * 0.5 + x2 * 0.3 + np.random.randn(n) * 2

    cat1 = np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(["X", "Y"], n)

    df_complete = pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "cat1": cat1, "cat2": cat2}
    )

    df_missing = df_complete.copy()
    miss_rate = 0.2

    for col in df_missing.columns:
        miss_idx = np.random.choice(n, size=int(n * miss_rate), replace=False)
        df_missing.loc[miss_idx, col] = np.nan

    print(f"Missing rate: {df_missing.isna().mean().mean():.1%}")
    print(f"Per-column NaN count: {df_missing.isna().sum().to_dict()}")

    imputer = HyperImputeImputer(
        categorical_vars=["cat1", "cat2"],
        continuous_vars=["x1", "x2", "x3"],
        seed=42,
        timeout=120,
    )

    df_imputed, _ = imputer.impute(df_complete, df_missing)

    print("\n" + "=" * 60)
    print("Imputation evaluation")
    print("=" * 60)

    for col in ["x1", "x2", "x3"]:
        mask = df_missing[col].isna()
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = pd.to_numeric(df_imputed.loc[mask, col], errors="coerce").values
            rmse = np.sqrt(np.mean((true_vals - imp_vals) ** 2))
            nrmse = rmse / np.std(true_vals) if np.std(true_vals) > 0 else rmse
            print(f"  {col}: NRMSE = {nrmse:.4f}")

    for col in ["cat1", "cat2"]:
        mask = df_missing[col].isna()
        if mask.sum() > 0:
            true_vals = df_complete.loc[mask, col].values
            imp_vals = df_imputed.loc[mask, col].values
            pfc = np.mean(true_vals != imp_vals)
            print(f"  {col}: PFC = {pfc:.4f}")
