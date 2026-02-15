#!/usr/bin/env python3
"""exp3_per_class_categorical.py

Ext2 / Experiment 3: Per-class breakdown for imbalanced categorical targets
--------------------------------------------------------------------------

This script produces per-class Precision / Recall / F1 (and support) for
categorical targets under injected missingness. It is designed to fill the
Supplementary placeholder:

  - Table S9 (MIMIC-IV ALARM, strict MAR, 30% missingness): per-class breakdown

It can also be used for other datasets/targets (e.g., eICU categorical fields)
when reviewer feedback requests class-wise diagnostics beyond Macro-F1 / κ.

Key robustness fixes vs the original ext2 draft
-----------------------------------------------
1) Mask semantics auto-detection:
   Different missingness generators use different conventions (1=missing vs
   1=observed). We automatically align the mask to match the NaN pattern in
   the generated missing table.

2) Categorical value canonicalization:
   Some imputers output float-coded categories (e.g., 1.0) while the complete
   data uses int-coded categories (e.g., 1). We canonicalize category values
   to avoid spurious "unknown" labels.

3) Unknown predictions handling:
   Predictions that still cannot be mapped to a known class are coerced to the
   global majority class (so they are counted as errors rather than being
   silently ignored by sklearn when labels are restricted).

Outputs
-------
<outdir>/
  perclass_metrics.csv   : seed × method × mechanism × target × class metrics
  perclass_summary.csv   : aggregated across seeds (mean±std)
  collapse_flags.csv     : rows where F1==0 (class-level collapse diagnostic)

Example (Table S9: MIMIC-IV ALARM, strict MAR)
----------------------------------------------
python ext2/scripts/exp3_per_class_categorical.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE SpO2 TEMP ... \
  --mechanisms MAR \
  --missing-rate 0.30 \
  --methods SNI MissForest MeanMode \
  --seeds 1 2 3 5 8 \
  --outdir results_ext2/table_S9_perclass_alarm \
  --use-gpu false
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so internal modules can be imported.
# ext2/scripts/ -> ext2/ -> project root
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
    from sklearn.metrics import precision_recall_fscore_support
except ImportError:  # pragma: no cover
    print("[ERROR] scikit-learn not found. Install via: pip install scikit-learn", file=sys.stderr)
    raise


# ===================================================================== #
#                               Helpers                                 #
# ===================================================================== #

def _canonicalize_category(v: Any) -> str:
    """Convert category values to a stable, comparable string representation."""
    if v is None:
        return ""
    # pandas NA / numpy nan
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    # numpy scalar -> python scalar
    if isinstance(v, (np.generic,)):
        v = v.item()

    # numeric
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v) and abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return str(float(v))

    # string
    if isinstance(v, str):
        return v.strip()

    return str(v).strip()


def _align_mask_to_nan_pattern(
    mask: pd.DataFrame,
    df_missing: pd.DataFrame,
    mode: str = "auto",
) -> pd.DataFrame:
    """Return a boolean mask where True indicates *missing entries*.

    If mode == 'auto', decide whether to invert based on agreement with NaNs.
    """
    mask_bool = mask.astype(bool)
    nan_bool = df_missing.isna()

    if mode == "missing_is_1":
        return mask_bool
    if mode == "observed_is_1":
        return ~mask_bool

    # auto: choose orientation that best matches NaN locations
    agree = (mask_bool == nan_bool).to_numpy().sum()
    agree_inv = ((~mask_bool) == nan_bool).to_numpy().sum()

    if agree_inv > agree:
        return ~mask_bool
    return mask_bool


def _generate_missing_data(
    df: pd.DataFrame,
    mechanism: str,
    missing_rate: float,
    seed: int,
    mask_mode: str = "auto",
    mar_driver_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper around generate_missing_dataset → (df_missing, mask_missing_bool)."""
    if not _HAS_MISSING_GEN:
        raise RuntimeError("missing_data_generator is not available in this environment.")

    kwargs: dict[str, Any] = dict(
        mechanism=mechanism,
        rate=missing_rate,
        seed=seed,
        allow_input_missing=False,
    )
    if mar_driver_cols is not None:
        kwargs["mar_driver_cols"] = mar_driver_cols

    result = generate_missing_dataset(df, **kwargs)

    # Result mask -> DataFrame
    mask_df = pd.DataFrame(result.mask.astype(int), index=df.index, columns=df.columns)
    df_miss = result.data_missing.copy()

    # Make dtypes assignment-friendly
    for col in df_miss.columns:
        if isinstance(df_miss[col].dtype, pd.CategoricalDtype) or pd.api.types.is_extension_array_dtype(df_miss[col]):
            df_miss[col] = df_miss[col].astype(object)

    # Try to restore numeric dtype where safe
    for col in df_miss.columns:
        if df_miss[col].dtype == object:
            numeric = pd.to_numeric(df_miss[col], errors="coerce")
            orig_notna = df_miss[col].notna().sum()
            if orig_notna > 0 and numeric.notna().sum() == orig_notna:
                df_miss[col] = numeric

    mask_missing = _align_mask_to_nan_pattern(mask_df, df_miss, mode=mask_mode)
    return df_miss, mask_missing


def _run_sni(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask_missing: pd.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
    seed: int,
    use_gpu: bool,
    cat_balance_mode: str = "none",
) -> pd.DataFrame:
    """Run SNI imputation and return imputed DataFrame."""
    from SNI_v0_3 import SNIImputer  # type: ignore
    from SNI_v0_3.imputer import SNIConfig  # type: ignore

    cfg = SNIConfig(seed=seed, use_gpu=use_gpu, cat_balance_mode=cat_balance_mode)
    imputer = SNIImputer(
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        config=cfg,
    )
    imputed_df = imputer.impute(
        X_missing=df_missing,
        X_complete=df_complete,
        mask_df=mask_missing.astype(int),
    )
    return imputed_df


def _run_baseline(
    method_name: str,
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
    seed: int,
) -> pd.DataFrame:
    """Run a baseline imputer and return imputed DataFrame.

    NOTE: We rely on the project baseline factory to ensure consistency with
    the main experiments.
    """
    from baselines import build_baseline_imputer  # type: ignore

    imputer = build_baseline_imputer(
        method_name,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        seed=seed,
    )
    return imputer.impute(df_complete, df_missing)


def _per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> list[dict[str, Any]]:
    """Compute per-class P/R/F1 and support."""
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        zero_division=0.0,
    )
    rows: list[dict[str, Any]] = []
    for i, cls in enumerate(classes):
        rows.append(
            {
                "class": cls,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(supp[i]),
            }
        )
    return rows


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ext2/Exp3: Per-class categorical breakdown")
    parser.add_argument("--input-complete", required=True, help="Path to complete CSV")
    parser.add_argument("--dataset-name", required=True, help="Dataset name (for logging/output)")
    parser.add_argument("--categorical-vars", nargs="+", required=True)
    parser.add_argument("--continuous-vars", nargs="+", required=True)
    parser.add_argument("--mechanisms", nargs="+", default=["MAR"], help="MCAR / MAR / MNAR")
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--methods", nargs="+", default=["SNI", "MissForest"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    parser.add_argument("--outdir", default="results_ext2/perclass")
    parser.add_argument("--use-gpu", default="false")
    parser.add_argument(
        "--mask-mode",
        default="auto",
        choices=["auto", "missing_is_1", "observed_is_1"],
        help="How to interpret the generator mask.",
    )
    parser.add_argument(
        "--mar-driver-cols",
        nargs="+",
        default=None,
        help="Fully-observed driver columns for strict MAR (e.g., HR SpO2).",
    )
    parser.add_argument(
        "--sni-cat-balance-modes",
        nargs="+",
        default=["none"],
        help="SNI v0.3 cat_balance_mode values to test. "
             "Use 'none inverse_freq sqrt_inverse_freq' to compare all three.",
    )
    args = parser.parse_args()

    use_gpu = args.use_gpu.lower() in ("true", "1", "yes")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_complete = pd.read_csv(args.input_complete)

    # --- Column validation: warn and drop variables not in the CSV ----------
    available_cols = set(df_complete.columns)
    cat_valid = [v for v in args.categorical_vars if v in available_cols]
    cat_missing = [v for v in args.categorical_vars if v not in available_cols]
    cont_valid = [v for v in args.continuous_vars if v in available_cols]
    cont_missing = [v for v in args.continuous_vars if v not in available_cols]

    if cat_missing or cont_missing:
        print(f"[Exp3][WARN] The following requested columns are NOT in the CSV "
              f"and will be skipped: {cat_missing + cont_missing}")
        print(f"[Exp3][INFO] Available columns in CSV ({len(available_cols)}): "
              f"{sorted(available_cols)}")
        if cat_missing:
            print(f"  Categorical dropped: {cat_missing}")
        if cont_missing:
            print(f"  Continuous  dropped: {cont_missing}")

    args.categorical_vars = cat_valid
    args.continuous_vars = cont_valid
    all_vars = args.categorical_vars + args.continuous_vars

    if not args.categorical_vars:
        print("[Exp3][ERROR] No valid categorical variables remain. Exiting.")
        sys.exit(1)
    if not all_vars:
        print("[Exp3][ERROR] No valid variables remain at all. Exiting.")
        sys.exit(1)

    print(f"[Exp3] Dataset: {args.dataset_name}")
    print(f"[Exp3] Mechanisms: {args.mechanisms} @ rate={args.missing_rate}")
    print(f"[Exp3] Methods: {args.methods}")
    print(f"[Exp3] Seeds: {args.seeds}")
    print(f"[Exp3] Output: {outdir}")

    rows: list[dict[str, Any]] = []

    for mechanism in args.mechanisms:
        for seed in args.seeds:
            print(f"\n--- {mechanism} / seed={seed} ---")

            df_missing, mask_missing = _generate_missing_data(
                df_complete[all_vars].copy(),
                mechanism=mechanism,
                missing_rate=args.missing_rate,
                seed=seed,
                mask_mode=args.mask_mode,
                mar_driver_cols=args.mar_driver_cols,
            )

            # Build list of (display_name, runner) pairs.
            # For SNI, expand with each cat_balance_mode variant.
            run_configs: list[tuple[str, str, str]] = []  # (display_name, method_type, balance_mode)
            for method in args.methods:
                if method.upper() == "SNI":
                    for bm in args.sni_cat_balance_modes:
                        bm = bm.strip()
                        suffix = f"(bal={bm})" if bm != "none" else ""
                        display = f"SNI{suffix}" if suffix else "SNI"
                        run_configs.append((display, "SNI", bm))
                else:
                    run_configs.append((method, "BASELINE", "none"))

            for display_name, method_type, balance_mode in run_configs:
                t0 = time.time()
                print(f"  Running {display_name}...", end=" ", flush=True)

                if method_type == "SNI":
                    imputed = _run_sni(
                        df_missing=df_missing,
                        df_complete=df_complete[all_vars],
                        mask_missing=mask_missing,
                        categorical_vars=args.categorical_vars,
                        continuous_vars=args.continuous_vars,
                        seed=seed,
                        use_gpu=use_gpu,
                        cat_balance_mode=balance_mode,
                    )
                else:
                    imputed = _run_baseline(
                        method_name=display_name,
                        df_missing=df_missing,
                        df_complete=df_complete[all_vars],
                        categorical_vars=args.categorical_vars,
                        continuous_vars=args.continuous_vars,
                        seed=seed,
                    )

                method = display_name  # use display name in output
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")

                # Per-class for each categorical target
                for target in args.categorical_vars:
                    miss_idx = mask_missing[target].to_numpy().astype(bool)
                    n_eval = int(miss_idx.sum())
                    if n_eval == 0:
                        continue

                    y_true_raw = df_complete.loc[miss_idx, target].to_numpy()
                    y_pred_raw = imputed.loc[miss_idx, target].to_numpy()

                    # Canonicalize
                    y_true = np.array([_canonicalize_category(v) for v in y_true_raw])
                    y_pred = np.array([_canonicalize_category(v) for v in y_pred_raw])

                    # Define class set from complete data (canonicalized)
                    classes = [_canonicalize_category(v) for v in pd.unique(df_complete[target])]
                    classes = [c for c in classes if c != ""]
                    # Stable order: sort as strings
                    classes = sorted(set(classes), key=lambda x: (len(x), x))

                    if len(classes) == 0:
                        continue

                    # Majority class for coercing unknown predictions
                    majority = (
                        pd.Series([_canonicalize_category(v) for v in df_complete[target]])
                        .value_counts()
                        .idxmax()
                    )

                    unknown = ~np.isin(y_pred, classes)
                    n_unknown = int(unknown.sum())
                    if n_unknown > 0:
                        y_pred = y_pred.copy()
                        y_pred[unknown] = majority

                    # Encode
                    label_map = {c: i for i, c in enumerate(classes)}
                    y_true_enc = np.array([label_map[c] for c in y_true], dtype=int)
                    y_pred_enc = np.array([label_map[c] for c in y_pred], dtype=int)

                    for r in _per_class_rows(y_true_enc, y_pred_enc, classes):
                        r.update(
                            {
                                "dataset": args.dataset_name,
                                "mechanism": mechanism,
                                "missing_rate": float(args.missing_rate),
                                "seed": int(seed),
                                "method": method,
                                "target": target,
                                "n_eval": n_eval,
                                "unknown_pred_n": n_unknown,
                                "unknown_pred_rate": float(n_unknown / max(n_eval, 1)),
                            }
                        )
                        rows.append(r)

    df_res = pd.DataFrame(rows)
    df_res.to_csv(outdir / "perclass_metrics.csv", index=False)
    print(f"\n[Exp3] Saved: {outdir / 'perclass_metrics.csv'} (n={len(df_res)})")

    if len(df_res) > 0:
        summary = (
            df_res.groupby(["dataset", "mechanism", "missing_rate", "method", "target", "class"], dropna=False)
            .agg(
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                support_mean=("support", "mean"),
                n_eval_mean=("n_eval", "mean"),
                unknown_pred_rate_mean=("unknown_pred_rate", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / "perclass_summary.csv", index=False)
        print(f"[Exp3] Saved: {outdir / 'perclass_summary.csv'}")

        flags = df_res[df_res["f1"] == 0.0].copy()
        flags.to_csv(outdir / "collapse_flags.csv", index=False)
        print(f"[Exp3] Saved: {outdir / 'collapse_flags.csv'} (n={len(flags)})")

    print("[Exp3] Done.")


if __name__ == "__main__":
    main()