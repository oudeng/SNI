#!/usr/bin/env python3
"""exp6_mimic_mortality_impute_predict.py

Ext2 / Experiment 6: Downstream Impute→Predict on MIMIC-IV in-hospital mortality
-------------------------------------------------------------------------------

This script fills the main-paper placeholder:

  - Table VI (Ext2 planned): MIMIC-IV in-hospital mortality, Impute→Predict
    (strict MAR 30% on features; label always observed; LR + XGBoost)

Protocol (aligned with v4.2 Table VI)
-------------------------------------
For each random seed:
  1) Split complete data into train/test (stratified by label).
  2) Inject strict MAR missingness at the specified rate into *feature columns only*.
     (Label is always observed.)
  3) Impute missing features with each imputer (SNI / MissForest / MeanMode).
  4) Train downstream classifiers on the imputed train features and evaluate on
     the imputed test features:
       - Logistic Regression
       - XGBoost (XGBClassifier)
  5) Report AUROC, AUPRC, Accuracy, F1.

Outputs
-------
<outdir>/
  per_seed_metrics.csv      : seed × imputer × model × metrics
  table_VI_summary.csv      : mean±std over seeds (ready for Table VI)

Notes
-----
- This script assumes you have an authorized, preprocessed "complete" CSV with
  the mortality label column already constructed (no missing values).
- Missingness is injected only for features; label is never masked.

Example (Table VI)
------------------
python ext2/scripts/exp6_mimic_mortality_impute_predict.py \
  --input-complete data/MIMIC_mortality_complete.csv \
  --dataset-name MIMIC_mortality \
  --label-col mortality \
  --categorical-vars ALARM SpO2 ... \
  --continuous-vars RESP ABP SBP DBP HR PULSE ... \
  --mechanism MAR --missing-rate 0.30 \
  --imputers SNI MissForest MeanMode HyperImpute TabCSDI \
  --models LR XGB \
  --seeds 1 2 3 5 8 \
  --outdir results_ext2/table_VI_mimic_mortality \
  --use-gpu false
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so internal modules can be imported.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for _candidate in (str(_PROJECT_ROOT), os.getcwd()):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ---------------------------------------------------------------------------
# Missingness generator (project internal)
# ---------------------------------------------------------------------------
_HAS_MISSING_GEN = False
try:
    from utility_missing_data_gen_v1.missing_data_generator import generate_missing_dataset  # type: ignore
    _HAS_MISSING_GEN = True
except Exception as e:  # pragma: no cover
    print(f"[WARN] Could not import missing_data_generator: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        f1_score,
    )
except ImportError:  # pragma: no cover
    print("[ERROR] scikit-learn not found. Install via: pip install scikit-learn", file=sys.stderr)
    raise

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None


# ===================================================================== #
#                               Helpers                                 #
# ===================================================================== #

def _align_mask_to_nan(mask: pd.DataFrame, df_missing: pd.DataFrame) -> pd.DataFrame:
    """Return boolean mask where True indicates missing positions."""
    mask_bool = mask.astype(bool)
    nan_bool = df_missing.isna()
    agree = (mask_bool == nan_bool).to_numpy().sum()
    agree_inv = ((~mask_bool) == nan_bool).to_numpy().sum()
    return ~mask_bool if agree_inv > agree else mask_bool


def _inject_missing(
    df_complete_features: pd.DataFrame,
    mechanism: str,
    missing_rate: float,
    seed: int,
    driver_col: str | None = None,
    mar_driver_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject missingness into feature columns using the project generator."""
    if not _HAS_MISSING_GEN:
        raise RuntimeError("missing_data_generator not available.")

    if driver_col is not None and driver_col in df_complete_features.columns:
        df_gen = df_complete_features[[driver_col] + [c for c in df_complete_features.columns if c != driver_col]].copy()
    else:
        df_gen = df_complete_features.copy()

    kwargs: Dict[str, Any] = dict(
        mechanism=mechanism,
        rate=missing_rate,
        seed=seed,
        allow_input_missing=False,
    )
    if mar_driver_cols is not None:
        kwargs["mar_driver_cols"] = mar_driver_cols

    result = generate_missing_dataset(df_gen, **kwargs)

    df_missing = result.data_missing.copy()
    mask_df = pd.DataFrame(result.mask.astype(int), index=df_gen.index, columns=df_gen.columns)

    # dtype friendliness
    for col in df_missing.columns:
        if isinstance(df_missing[col].dtype, pd.CategoricalDtype) or pd.api.types.is_extension_array_dtype(df_missing[col]):
            df_missing[col] = df_missing[col].astype(object)
    for col in df_missing.columns:
        if df_missing[col].dtype == object:
            numeric = pd.to_numeric(df_missing[col], errors="coerce")
            orig_notna = df_missing[col].notna().sum()
            if orig_notna > 0 and numeric.notna().sum() == orig_notna:
                df_missing[col] = numeric

    mask_missing = _align_mask_to_nan(mask_df, df_missing)

    # enforce driver always observed (if requested)
    if driver_col is not None and driver_col in df_missing.columns:
        df_missing[driver_col] = df_gen[driver_col]
        mask_missing[driver_col] = False

    # Return aligned to original column order (without driver if it was moved)
    df_missing = df_missing[df_gen.columns]
    mask_missing = mask_missing[df_gen.columns]

    return df_missing, mask_missing


def _run_sni(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    use_gpu: bool,
) -> pd.DataFrame:
    from SNI_v0_3 import SNIImputer  # type: ignore
    from SNI_v0_3.imputer import SNIConfig  # type: ignore

    cfg = SNIConfig(seed=seed, use_gpu=use_gpu)
    imputer = SNIImputer(
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        config=cfg,
    )
    return imputer.impute(
        X_missing=df_missing,
        X_complete=df_complete,
        mask_df=mask_missing.astype(int),
    )


def _run_baseline(
    method_name: str,
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
) -> pd.DataFrame:
    from baselines import build_baseline_imputer  # type: ignore

    imputer = build_baseline_imputer(
        method_name,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        seed=seed,
    )
    return imputer.impute(df_complete, df_missing)


def _make_preprocessor(categorical_vars: List[str], continuous_vars: List[str]) -> ColumnTransformer:
    cat = OneHotEncoder(handle_unknown="ignore")
    cont = StandardScaler(with_mean=False)
    return ColumnTransformer(
        transformers=[
            ("cat", cat, categorical_vars),
            ("cont", cont, continuous_vars),
        ],
        remainder="drop",
    )


def _fit_and_eval(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    class_weight: str = "balanced",
) -> Dict[str, float]:
    pre = _make_preprocessor(categorical_vars, continuous_vars)

    if model_name == "LR":
        cw = "balanced" if class_weight == "balanced" else None
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=cw,
            random_state=seed,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
    elif model_name == "XGB":
        if xgb is None:
            raise RuntimeError("xgboost not installed. Please: pip install xgboost")
        # scale_pos_weight for imbalance
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = float(neg / max(pos, 1))
        clf = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "AUROC": float(roc_auc_score(y_test, proba)),
        "AUPRC": float(average_precision_score(y_test, proba)),
        "Accuracy": float(accuracy_score(y_test, pred)),
        "F1": float(f1_score(y_test, pred)),
    }
    return metrics


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ext2/Exp6: MIMIC mortality Impute→Predict")
    parser.add_argument("--input-complete", required=True, help="Complete CSV with features + label")
    parser.add_argument("--dataset-name", default="MIMIC_mortality")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--categorical-vars", nargs="+", required=True)
    parser.add_argument("--continuous-vars", nargs="+", required=True)
    parser.add_argument("--driver-col", default=None, help="Optional always-observed driver (e.g., ID) for strict MAR")
    parser.add_argument("--mechanism", default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--imputers", nargs="+", default=["SNI", "MissForest", "MeanMode", "HyperImpute", "TabCSDI"])
    parser.add_argument("--models", nargs="+", default=["LR", "XGB"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--outdir", default="results_ext2/mimic_mortality_impute_predict")
    parser.add_argument("--use-gpu", default="false")
    parser.add_argument("--mar-driver-cols", nargs="+", default=None,
                        help="Fully-observed driver columns for strict MAR.")
    parser.add_argument("--binarize-threshold", type=float, default=None,
                        help="If set, binarize label-col: values >= threshold → 1, else → 0. "
                             "Useful when reusing a multi-class column (e.g., ALARM) as binary label.")
    args = parser.parse_args()

    use_gpu = args.use_gpu.lower() in ("true", "1", "yes")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_complete)

    # Basic checks
    if args.label_col not in df.columns:
        raise SystemExit(f"[ERROR] label column not found: {args.label_col}")

    # --- Binarize label if requested ---
    if args.binarize_threshold is not None:
        raw = df[args.label_col]
        thr = args.binarize_threshold
        df[args.label_col] = (raw >= thr).astype(int)
        n_pos = int(df[args.label_col].sum())
        n_tot = len(df)
        print(f"[Exp6] Binarized {args.label_col}: >= {thr} → 1  "
              f"(pos={n_pos}, neg={n_tot - n_pos}, rate={n_pos/n_tot:.3f})")

    # --- Ensure label-col is NOT in feature lists ---
    args.categorical_vars = [v for v in args.categorical_vars if v != args.label_col]
    args.continuous_vars = [v for v in args.continuous_vars if v != args.label_col]

    feature_cols = args.categorical_vars + args.continuous_vars
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[Exp6][WARN] Columns NOT in CSV (skipped): {missing_cols}")
        print(f"[Exp6][INFO] Available columns: {sorted(df.columns.tolist())}")
        args.categorical_vars = [v for v in args.categorical_vars if v in df.columns]
        args.continuous_vars = [v for v in args.continuous_vars if v in df.columns]
        feature_cols = args.categorical_vars + args.continuous_vars
        if not feature_cols:
            raise SystemExit("[Exp6][ERROR] No valid feature columns remain.")

    y = df[args.label_col].to_numpy()
    # ensure binary 0/1
    if set(pd.unique(y)) - {0, 1}:
        # try to coerce
        y = pd.Series(y).astype(int).to_numpy()
    y = y.astype(int)

    rows: List[Dict[str, Any]] = []

    print(f"[Exp6] Dataset: {args.dataset_name}")
    print(f"[Exp6] Label: {args.label_col} (pos rate={y.mean():.3f})")
    print(f"[Exp6] Mechanism: {args.mechanism} @ rate={args.missing_rate}")
    print(f"[Exp6] Imputers: {args.imputers}")
    print(f"[Exp6] Models: {args.models}")
    print(f"[Exp6] Seeds: {args.seeds}")
    print(f"[Exp6] Output: {outdir}")

    for seed in args.seeds:
        print(f"\n--- seed={seed} ---")

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=args.test_size,
            random_state=seed,
            stratify=y,
        )

        df_train_complete = df.iloc[train_idx].reset_index(drop=True)
        df_test_complete = df.iloc[test_idx].reset_index(drop=True)

        y_train = df_train_complete[args.label_col].to_numpy(dtype=int)
        y_test = df_test_complete[args.label_col].to_numpy(dtype=int)

        # Inject missingness into features only (label untouched)
        train_missing_all, train_mask_all = _inject_missing(
            df_train_complete[feature_cols].copy(),
            mechanism=args.mechanism,
            missing_rate=args.missing_rate,
            seed=seed * 100 + 1,
            driver_col=args.driver_col,
            mar_driver_cols=args.mar_driver_cols,
        )
        test_missing_all, test_mask_all = _inject_missing(
            df_test_complete[feature_cols].copy(),
            mechanism=args.mechanism,
            missing_rate=args.missing_rate,
            seed=seed * 100 + 2,
            driver_col=args.driver_col,
            mar_driver_cols=args.mar_driver_cols,
        )

        for imputer_name in args.imputers:
            print(f"  Imputer: {imputer_name}")

            if imputer_name.upper() == "SNI":
                X_train_imp = _run_sni(
                    df_missing=train_missing_all,
                    df_complete=df_train_complete[feature_cols],
                    mask_missing=train_mask_all,
                    categorical_vars=args.categorical_vars,
                    continuous_vars=args.continuous_vars,
                    seed=seed,
                    use_gpu=use_gpu,
                )
                X_test_imp = _run_sni(
                    df_missing=test_missing_all,
                    df_complete=df_test_complete[feature_cols],
                    mask_missing=test_mask_all,
                    categorical_vars=args.categorical_vars,
                    continuous_vars=args.continuous_vars,
                    seed=seed,
                    use_gpu=use_gpu,
                )
            else:
                X_train_imp = _run_baseline(
                    method_name=imputer_name,
                    df_missing=train_missing_all,
                    df_complete=df_train_complete[feature_cols],
                    categorical_vars=args.categorical_vars,
                    continuous_vars=args.continuous_vars,
                    seed=seed,
                )
                X_test_imp = _run_baseline(
                    method_name=imputer_name,
                    df_missing=test_missing_all,
                    df_complete=df_test_complete[feature_cols],
                    categorical_vars=args.categorical_vars,
                    continuous_vars=args.continuous_vars,
                    seed=seed,
                )

            for model_name in args.models:
                print(f"    Model: {model_name} ...", end=" ", flush=True)
                metrics = _fit_and_eval(
                    model_name=model_name,
                    X_train=X_train_imp,
                    y_train=y_train,
                    X_test=X_test_imp,
                    y_test=y_test,
                    categorical_vars=args.categorical_vars,
                    continuous_vars=args.continuous_vars,
                    seed=seed,
                    class_weight=args.class_weight,
                )
                print("done")

                row = {
                    "dataset": args.dataset_name,
                    "mechanism": args.mechanism,
                    "missing_rate": float(args.missing_rate),
                    "seed": int(seed),
                    "imputer": imputer_name,
                    "downstream_model": "Logistic regression" if model_name == "LR" else "XGBoost",
                }
                row.update(metrics)
                rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(outdir / "per_seed_metrics.csv", index=False)
    print(f"\n[Exp6] Saved: {outdir / 'per_seed_metrics.csv'} (n={len(df_out)})")

    # Table VI summary (mean±std over seeds)
    if len(df_out) > 0:
        summary = (
            df_out.groupby(["imputer", "downstream_model"], dropna=False)
            .agg(
                AUROC_mean=("AUROC", "mean"),
                AUROC_std=("AUROC", "std"),
                AUPRC_mean=("AUPRC", "mean"),
                AUPRC_std=("AUPRC", "std"),
                Accuracy_mean=("Accuracy", "mean"),
                Accuracy_std=("Accuracy", "std"),
                F1_mean=("F1", "mean"),
                F1_std=("F1", "std"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / "table_VI_summary.csv", index=False)
        print(f"[Exp6] Saved: {outdir / 'table_VI_summary.csv'}")

    print("[Exp6] Done.")


if __name__ == "__main__":
    main()