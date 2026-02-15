#!/usr/bin/env python3
"""Baseline self-test: generate toy data, run MeanMode, and verify metrics.

This script is a quick sanity check that:
  1. Generates 100x5 toy mixed-type data (3 continuous, 2 categorical columns)
  2. Introduces 20% MCAR missing values
  3. Runs MeanMode imputation (always available, fast)
  4. Computes evaluation metrics via evaluate_imputation
  5. Prints sanity check results and baseline implementation version

Usage:
    python scripts/baseline_selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.MeanMode_v1 import MeanModeImputer
from SNI_v0_3.metrics import evaluate_imputation


def _generate_toy_data(
    n_rows: int = 100,
    seed: int = 42,
) -> tuple:
    """Generate toy mixed-type data.

    Returns:
        X_complete: DataFrame with 3 continuous and 2 categorical columns (no missing)
        continuous_vars: list of continuous column names
        categorical_vars: list of categorical column names
    """
    rng = np.random.RandomState(seed)

    # 3 continuous columns
    c1 = rng.randn(n_rows) * 10 + 50       # mean=50, std=10
    c2 = rng.exponential(scale=5, size=n_rows)
    c3 = rng.uniform(0, 100, size=n_rows)

    # 2 categorical columns (integer-coded)
    cat1 = rng.choice([0, 1, 2], size=n_rows, p=[0.5, 0.3, 0.2])
    cat2 = rng.choice([10, 20, 30, 40], size=n_rows, p=[0.4, 0.3, 0.2, 0.1])

    continuous_vars = ["cont_A", "cont_B", "cont_C"]
    categorical_vars = ["cat_X", "cat_Y"]

    X_complete = pd.DataFrame({
        "cont_A": c1,
        "cont_B": c2,
        "cont_C": c3,
        "cat_X": cat1.astype(int),
        "cat_Y": cat2.astype(int),
    })

    return X_complete, continuous_vars, categorical_vars


def _introduce_mcar(
    X_complete: pd.DataFrame,
    rate: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Introduce MCAR missing values at the given rate."""
    rng = np.random.RandomState(seed)
    X_missing = X_complete.copy()
    mask = rng.rand(*X_complete.shape) < rate
    # Convert to float for continuous, object for categorical to allow NaN
    for col in X_missing.columns:
        col_mask = mask[:, X_complete.columns.get_loc(col)]
        if col_mask.any():
            X_missing.loc[col_mask, col] = np.nan
    return X_missing


def main() -> None:
    print("=" * 60)
    print("Baseline Self-Test")
    print("=" * 60)

    # Step 1: Generate toy data
    print("\n[1/4] Generating 100x5 toy mixed-type data...")
    X_complete, continuous_vars, categorical_vars = _generate_toy_data(n_rows=100, seed=42)
    print(f"       Shape: {X_complete.shape}")
    print(f"       Continuous: {continuous_vars}")
    print(f"       Categorical: {categorical_vars}")

    # Step 2: Introduce MCAR missing
    print("\n[2/4] Introducing 20% MCAR missing values...")
    X_missing = _introduce_mcar(X_complete, rate=0.2, seed=42)
    total_cells = X_missing.shape[0] * X_missing.shape[1]
    n_missing = int(X_missing.isna().sum().sum())
    actual_rate = n_missing / total_cells
    print(f"       Missing cells: {n_missing}/{total_cells} ({actual_rate:.1%})")

    # Step 3: Run MeanMode imputation
    print("\n[3/4] Running MeanMode imputation...")
    imputer = MeanModeImputer(
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )
    X_imputed, _ = imputer.impute(X_complete, X_missing)

    # Verify no NaN remains
    remaining_nan = int(X_imputed.isna().sum().sum())
    print(f"       Remaining NaN after imputation: {remaining_nan}")

    # Step 4: Evaluate
    print("\n[4/4] Computing evaluation metrics...")
    eval_result = evaluate_imputation(
        X_imputed=X_imputed,
        X_complete=X_complete,
        X_missing=X_missing,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )

    # Print per-feature metrics
    print("\n--- Per-Feature Metrics ---")
    if not eval_result.per_feature.empty:
        print(eval_result.per_feature.to_string(index=False))
    else:
        print("  (no per-feature metrics computed)")

    # Print summary metrics
    print("\n--- Summary Metrics ---")
    for key, val in sorted(eval_result.summary.items()):
        print(f"  {key:30s}: {val}")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    # Check 1: No remaining NaN
    checks_total += 1
    if remaining_nan == 0:
        print("  [PASS] No remaining NaN after imputation")
        checks_passed += 1
    else:
        print(f"  [FAIL] {remaining_nan} NaN values remain after imputation")

    # Check 2: NRMSE is finite and reasonable (< 2.0 for mean imputation)
    checks_total += 1
    nrmse = eval_result.summary.get("cont_NRMSE", float("nan"))
    if np.isfinite(nrmse) and 0 < nrmse < 2.0:
        print(f"  [PASS] cont_NRMSE = {nrmse:.4f} (finite and < 2.0)")
        checks_passed += 1
    else:
        print(f"  [FAIL] cont_NRMSE = {nrmse} (expected finite and < 2.0)")

    # Check 3: Categorical accuracy is reasonable (> 0 for mode imputation)
    checks_total += 1
    acc = eval_result.summary.get("cat_Accuracy", float("nan"))
    if np.isfinite(acc) and acc > 0:
        print(f"  [PASS] cat_Accuracy = {acc:.4f} (finite and > 0)")
        checks_passed += 1
    else:
        print(f"  [FAIL] cat_Accuracy = {acc} (expected finite and > 0)")

    # Check 4: Output shape matches
    checks_total += 1
    if X_imputed.shape == X_complete.shape:
        print(f"  [PASS] Output shape matches: {X_imputed.shape}")
        checks_passed += 1
    else:
        print(f"  [FAIL] Shape mismatch: imputed={X_imputed.shape}, complete={X_complete.shape}")

    # Final summary
    print(f"\n--- Result: {checks_passed}/{checks_total} checks passed ---")
    print(f'baseline_impl: "in-repo v1"')

    if checks_passed == checks_total:
        print("\n[PASS] All sanity checks passed.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] {checks_total - checks_passed} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
