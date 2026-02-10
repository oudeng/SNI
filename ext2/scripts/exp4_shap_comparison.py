#!/usr/bin/env python3
"""exp4_shap_comparison.py

Ext2 / Experiment 4: SNI reliance (D) vs post-hoc SHAP on MissForest
--------------------------------------------------------------------

This script fills the Supplementary placeholder:

  - Table S7: "Post-hoc explanation comparison (SHAP on MissForest) — planned"

Paper requirement (v4.2)
------------------------
Table S7 explicitly requests SHAP computed on MissForest imputers (not SHAP on
SNI/CPFA). The original ext2 draft compared D vs KernelSHAP on CPFA, which is
(1) mismatched to the placeholder and (2) extremely slow.

What this script does (fast + aligned with Table S7)
----------------------------------------------------
1) Inject strict MAR missingness at 30% into a *complete* table.
2) Run SNI to obtain its intrinsic reliance matrix D.
3) Run MissForest to obtain an imputed table.
4) For each selected target (default: ALARM, SBP on MIMIC-IV):
   - SNI: take the D row for the target and report top-k source features.
   - MissForest: fit a RandomForest surrogate model on observed target rows
     using the MissForest-imputed features, then compute TreeSHAP on the
     target-masked rows, reporting top-k features by mean(|SHAP|).

Outputs
-------
<outdir>/
  d_matrix.csv
  shap_importances.csv            : target × source mean(|SHAP|)
  spearman_d_vs_shap.csv          : Spearman ρ between D and SHAP per target
  table_S7_top_features.csv       : directly usable to populate Table S7

Example (MIMIC-IV, Table S7)
----------------------------
python ext2/scripts/exp4_shap_comparison.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars ALARM SpO2 ... \
  --continuous-vars RESP ABP SBP DBP HR PULSE ... \
  --mechanism MAR --missing-rate 0.30 \
  --seed 2026 \
  --targets ALARM SBP \
  --top-k 10 \
  --shap-max-eval 512 \
  --outdir results_ext2/table_S7_shap_vs_D/MIMIC \
  --use-gpu false
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    print("[ERROR] scipy not found. Install via: pip install scipy", file=sys.stderr)
    raise

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None
    print("[ERROR] shap not found. Install via: pip install shap", file=sys.stderr)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover
    print("[ERROR] scikit-learn not found. Install via: pip install scikit-learn", file=sys.stderr)
    raise


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


def _generate_missing_data(
    df: pd.DataFrame, mechanism: str, missing_rate: float, seed: int,
    mar_driver_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not _HAS_MISSING_GEN:
        raise RuntimeError("missing_data_generator not available in this environment.")

    kwargs: Dict[str, Any] = dict(
        mechanism=mechanism,
        rate=missing_rate,
        seed=seed,
        allow_input_missing=False,
    )
    if mar_driver_cols is not None:
        kwargs["mar_driver_cols"] = mar_driver_cols

    result = generate_missing_dataset(df, **kwargs)
    mask_df = pd.DataFrame(result.mask.astype(int), index=df.index, columns=df.columns)
    df_miss = result.data_missing.copy()

    # Make dtypes assignment-friendly
    for col in df_miss.columns:
        if isinstance(df_miss[col].dtype, pd.CategoricalDtype) or pd.api.types.is_extension_array_dtype(df_miss[col]):
            df_miss[col] = df_miss[col].astype(object)

    for col in df_miss.columns:
        if df_miss[col].dtype == object:
            numeric = pd.to_numeric(df_miss[col], errors="coerce")
            orig_notna = df_miss[col].notna().sum()
            if orig_notna > 0 and numeric.notna().sum() == orig_notna:
                df_miss[col] = numeric

    mask_missing = _align_mask_to_nan(mask_df, df_miss)
    return df_miss, mask_missing


def _run_sni(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    use_gpu: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run SNI imputation and return (imputed_df, D_matrix_or_None)."""
    from SNI_v0_2 import SNIImputer  # type: ignore
    from SNI_v0_2.imputer import SNIConfig  # type: ignore

    cfg = SNIConfig(seed=seed, use_gpu=use_gpu)
    imputer = SNIImputer(
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        config=cfg,
    )
    imputed = imputer.impute(
        X_missing=df_missing,
        X_complete=df_complete,
        mask_df=mask_missing.astype(int),
    )
    # D matrix is computed on demand, not set automatically by impute()
    try:
        dep_mat = imputer.compute_dependency_matrix()
    except Exception as e:
        print(f"[WARN] compute_dependency_matrix() failed: {e}", file=sys.stderr)
        dep_mat = None
    return imputed, dep_mat


def _run_missforest(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
) -> pd.DataFrame:
    """Run MissForest via the project's baseline factory."""
    from baselines import build_baseline_imputer  # type: ignore

    imputer = build_baseline_imputer(
        "MissForest",
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        seed=seed,
    )
    return imputer.impute(df_complete, df_missing)


def _tree_shap_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    task: str,
    seed: int,
    max_eval: int = 512,
) -> np.ndarray:
    """Fit a RF surrogate and compute mean(|SHAP|) on X_eval.

    Returns importance array shape (n_features,).
    """
    if shap is None:
        raise RuntimeError("shap is required. Install via: pip install shap")

    # Subsample evaluation to control runtime
    if len(X_eval) > max_eval:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_eval), size=max_eval, replace=False)
        X_eval = X_eval.iloc[idx].copy()

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        if isinstance(shap_values, list):
            # Old SHAP: list[n_classes] each (n_samples, n_features)
            arr = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (n_samples, n_features)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New SHAP: (n_samples, n_features, n_classes) → average over classes
            arr = np.abs(shap_values).mean(axis=2)  # (n_samples, n_features)
        else:
            arr = np.abs(shap_values)  # (n_samples, n_features)

        importance = arr.mean(axis=0)

    else:
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)
        importance = np.abs(shap_values).mean(axis=0)

    return importance.astype(float)


def _top_k_features(scores: pd.Series, k: int) -> List[str]:
    return [str(x) for x in scores.sort_values(ascending=False).head(k).index.tolist()]


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ext2/Exp4: SHAP (MissForest) vs D (SNI)")
    parser.add_argument("--input-complete", required=True, help="Path to complete CSV")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--categorical-vars", nargs="+", required=True)
    parser.add_argument("--continuous-vars", nargs="+", required=True)
    parser.add_argument("--mechanism", default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--targets", nargs="+", default=None, help="Targets to report (default: ALARM SBP if present else all)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shap-max-eval", type=int, default=512, help="Max rows for SHAP evaluation per target")
    parser.add_argument("--outdir", default="results_ext2/shap_vs_d")
    parser.add_argument("--use-gpu", default="false")
    parser.add_argument("--mar-driver-cols", nargs="+", default=None,
                        help="Fully-observed driver columns for strict MAR.")
    args = parser.parse_args()

    if shap is None:
        raise SystemExit("[ERROR] shap not installed. Please: pip install shap")

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
        print(f"[Exp4][WARN] Columns NOT in CSV (skipped): {cat_missing + cont_missing}")
        print(f"[Exp4][INFO] Available columns: {sorted(available_cols)}")

    args.categorical_vars = cat_valid
    args.continuous_vars = cont_valid
    all_vars = args.categorical_vars + args.continuous_vars

    if not all_vars:
        print("[Exp4][ERROR] No valid variables remain. Exiting.")
        sys.exit(1)

    # Decide targets
    if args.targets is None:
        default_targets = []
        for t in ("ALARM", "SBP"):
            if t in all_vars:
                default_targets.append(t)
        targets = default_targets if default_targets else all_vars
    else:
        targets = args.targets

    print(f"[Exp4] Dataset: {args.dataset_name}")
    print(f"[Exp4] Mechanism: {args.mechanism} @ rate={args.missing_rate}")
    print(f"[Exp4] Seed: {args.seed}")
    print(f"[Exp4] Targets: {targets}")
    print(f"[Exp4] Output: {outdir}")

    # 1) Generate missing data
    df_missing, mask_missing = _generate_missing_data(
        df_complete[all_vars].copy(),
        mechanism=args.mechanism,
        missing_rate=args.missing_rate,
        seed=args.seed,
        mar_driver_cols=args.mar_driver_cols,
    )

    # 2) SNI
    print("[Exp4] Running SNI...")
    sni_imputed, dep_mat = _run_sni(
        df_missing=df_missing,
        df_complete=df_complete[all_vars],
        mask_missing=mask_missing,
        categorical_vars=args.categorical_vars,
        continuous_vars=args.continuous_vars,
        seed=args.seed,
        use_gpu=use_gpu,
    )

    if dep_mat is not None:
        dep_mat.to_csv(outdir / "d_matrix.csv")
        print(f"[Exp4] Saved D matrix: {outdir / 'd_matrix.csv'}")
    else:
        print("[WARN] SNI dependency_matrix_ not found. D-related outputs will be skipped.", file=sys.stderr)

    # 3) MissForest
    print("[Exp4] Running MissForest...")
    mf_imputed = _run_missforest(
        df_missing=df_missing,
        df_complete=df_complete[all_vars],
        categorical_vars=args.categorical_vars,
        continuous_vars=args.continuous_vars,
        seed=args.seed,
    )

    # 4) Per target: D top-k, SHAP top-k, Spearman
    shap_rows: List[Dict[str, Any]] = []
    spearman_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []

    for target in targets:
        if target not in all_vars:
            print(f"[Exp4] Skip target={target} (not in vars)")
            continue

        src_features = [c for c in all_vars if c != target]

        # Identify masked rows for this target (evaluation subset)
        masked_rows = mask_missing[target].to_numpy().astype(bool)
        observed_rows = ~masked_rows
        if masked_rows.sum() == 0:
            print(f"[Exp4] Skip target={target} (no masked rows)")
            continue

        # -----------------------
        # MissForest TreeSHAP
        # -----------------------
        task = "classification" if target in args.categorical_vars else "regression"

        X_train = mf_imputed.loc[observed_rows, src_features]
        X_eval = mf_imputed.loc[masked_rows, src_features]

        if task == "classification":
            # Use ground-truth labels from complete data for training the surrogate
            y_train_raw = df_complete.loc[observed_rows, target].to_numpy()
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw.astype(str))
            importance = _tree_shap_importance(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                task="classification",
                seed=args.seed,
                max_eval=args.shap_max_eval,
            )
        else:
            y_train = df_complete.loc[observed_rows, target].to_numpy(dtype=float)
            importance = _tree_shap_importance(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                task="regression",
                seed=args.seed,
                max_eval=args.shap_max_eval,
            )

        shap_series = pd.Series(importance, index=src_features, dtype=float)
        shap_series = shap_series / (shap_series.sum() + 1e-12)

        for src, val in shap_series.items():
            shap_rows.append(
                {
                    "dataset": args.dataset_name,
                    "mechanism": args.mechanism,
                    "missing_rate": float(args.missing_rate),
                    "seed": int(args.seed),
                    "target": target,
                    "source": src,
                    "shap_importance": float(val),
                }
            )

        mf_top = _top_k_features(shap_series, args.top_k)
        table_rows.append(
            {
                "Target": target,
                "Method": "MissForest (TreeSHAP surrogate)",
                "Top relied-upon features": ", ".join(mf_top),
                "Notes": f"mean(|SHAP|) on target-masked rows (n={int(masked_rows.sum())})",
            }
        )

        # -----------------------
        # SNI D top-k + Spearman
        # -----------------------
        if dep_mat is not None and target in dep_mat.index:
            # Align feature set
            common = [c for c in src_features if c in dep_mat.columns]
            d_row = dep_mat.loc[target, common].astype(float)
            # Normalize
            d_row = d_row / (d_row.sum() + 1e-12)

            d_top = _top_k_features(d_row, args.top_k)
            table_rows.append(
                {
                    "Target": target,
                    "Method": "SNI (D)",
                    "Top relied-upon features": ", ".join(d_top),
                    "Notes": "attention-derived reliance matrix row-normalized",
                }
            )

            # Spearman ρ between D and SHAP over common sources
            shap_aligned = shap_series.loc[common]
            rho, p = spearmanr(d_row.to_numpy(), shap_aligned.to_numpy())
            spearman_rows.append(
                {
                    "dataset": args.dataset_name,
                    "mechanism": args.mechanism,
                    "missing_rate": float(args.missing_rate),
                    "seed": int(args.seed),
                    "target": target,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n_sources": int(len(common)),
                }
            )
            print(f"[Exp4] target={target}: Spearman ρ={rho:.3f} (p={p:.3g}, n={len(common)})")
        else:
            print(f"[Exp4] target={target}: no D row available, skipping D comparison")

    # Save outputs
    pd.DataFrame(shap_rows).to_csv(outdir / "shap_importances.csv", index=False)
    pd.DataFrame(spearman_rows).to_csv(outdir / "spearman_d_vs_shap.csv", index=False)
    pd.DataFrame(table_rows).to_csv(outdir / "table_S7_top_features.csv", index=False)

    print(f"[Exp4] Saved: {outdir / 'shap_importances.csv'}")
    print(f"[Exp4] Saved: {outdir / 'spearman_d_vs_shap.csv'}")
    print(f"[Exp4] Saved: {outdir / 'table_S7_top_features.csv'}")
    print("[Exp4] Done.")


if __name__ == "__main__":
    main()