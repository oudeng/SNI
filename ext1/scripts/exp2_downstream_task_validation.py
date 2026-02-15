#!/usr/bin/env python3
"""exp2_downstream_task_validation.py

Ext1 / Experiment 2 (Downstream task validation)
================================================

Motivation
----------
Many reviewers look for *utility* beyond per-cell imputation scores.
This experiment measures how imputations impact an actual downstream model.

Protocol (high-level)
---------------------
For each seed:
1) Start from a **complete** dataset.
2) Generate missingness **only on feature columns** (target y is always observed).
3) Impute features with multiple methods (SNI + baselines).
4) Train a predictive model on the imputed features and evaluate on a hold-out
   test split.
5) Optionally compute a simple group-gap metric ("bias") along a fairness column
   (e.g., gender).

Outputs
-------
<outdir>/
  metrics_per_seed.csv
  metrics_summary.csv
  (optional) imputed/<method>/seed<seed>.csv
  (optional) missing/seed<seed>_missing.csv + metadata/mask

Example (NHANES → predict metabolic_score)
-----------------------------------------
python ext1/scripts/exp2_downstream_task_validation.py \
  --input-complete data/NHANES_complete.csv \
  --dataset-name NHANES \
  --target-col metabolic_score \
  --categorical-cols gender_std age_band \
  --continuous-cols waist_circumference systolic_bp diastolic_bp triglycerides hdl_cholesterol fasting_glucose age bmi hba1c \
  --mechanism MAR --missing-rate 0.30 \
  --mar-driver-cols age gender_std \
  --fairness-col gender_std \
  --imputers SNI MissForest MeanMode MICE HyperImpute TabCSDI \
  --seeds 1 2 3 5 8 \
  --outdir results_ext1/downstream_nhanes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Path setup: allow imports from repo root + missingness generator
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

GEN_DIR = REPO_ROOT / "utility_missing_data_gen_v1"
sys.path.insert(0, str(GEN_DIR))

from missing_data_generator import generate_missing_dataset

from baselines import build_baseline_imputer, list_baselines
from SNI_v0_3 import SNIImputer
from SNI_v0_3.imputer import SNIConfig
from SNI_v0_3.dataio import infer_schema_from_complete, cast_dataframe_to_schema

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    out: List[str] = []
    for x in xs:
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        out.extend(parts)
    # de-duplicate (keep order)
    seen = set()
    uniq: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _infer_task_type(y: pd.Series, user_task: Optional[str]) -> str:
    if user_task:
        t = str(user_task).lower().strip()
        if t in {"classification", "regression"}:
            return t
        raise ValueError(f"Unknown --task {user_task}. Use classification/regression.")

    # heuristic
    if pd.api.types.is_numeric_dtype(y):
        nunq = int(pd.Series(y).nunique(dropna=True))
        if nunq <= 20:
            return "classification"
        return "regression"

    return "classification"


def _build_model_pipeline(
    *,
    task: str,
    categorical_cols: List[str],
    continuous_cols: List[str],
    seed: int,
    class_weight_balanced: bool,
) -> Pipeline:
    """Return sklearn Pipeline(preprocess -> model)."""
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("cont", StandardScaler(), continuous_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        model = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            multi_class="auto",
            class_weight=("balanced" if class_weight_balanced else None),
            random_state=int(seed),
        )
    else:
        model = Ridge(alpha=1.0, random_state=int(seed))

    return Pipeline([("prep", preprocess), ("model", model)])


def _classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            # roc_auc_score supports binary/multiclass with different options
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # binary
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["auc"] = float(roc_auc_score(y_true, proba_pos))
            else:
                out["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            # Not fatal
            pass

    return out


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _group_gap(
    *,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> Dict[str, float]:
    """Compute a simple performance gap across groups.

    For classification: gap in macro-F1 (max - min).
    For regression:     gap in RMSE (max - min).

    This is intentionally simple (reviewer-friendly) and avoids heavy fairness
    frameworks.
    """
    out: Dict[str, float] = {}
    groups = pd.Series(group).dropna().unique().tolist()
    if len(groups) < 2:
        return out

    per_g = []
    for g in groups:
        idx = group == g
        if idx.sum() < 5:
            continue
        if task == "classification":
            f1g = float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
            per_g.append((g, f1g))
        else:
            rmseg = float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
            per_g.append((g, rmseg))

    if len(per_g) < 2:
        return out

    vals = [v for _, v in per_g]
    out["group_gap"] = float(max(vals) - min(vals))
    out["group_min"] = float(min(vals))
    out["group_max"] = float(max(vals))
    out["n_groups_eval"] = int(len(per_g))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ext1-Exp2: Downstream task validation (impute → predict).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--input-complete", type=str, required=True, help="Complete CSV.")
    ap.add_argument("--dataset-name", type=str, default="DATA")
    ap.add_argument("--target-col", type=str, required=True, help="Downstream label column.")

    ap.add_argument("--categorical-cols", nargs="+", default=[], help="Categorical feature columns.")
    ap.add_argument("--continuous-cols", nargs="+", default=[], help="Continuous feature columns.")

    ap.add_argument("--mechanism", type=str, default="MAR", choices=["MCAR", "MAR", "MNAR"], help="Missingness mechanism on features.")
    ap.add_argument("--missing-rate", type=float, default=0.30)
    ap.add_argument("--mar-driver-cols", nargs="+", default=None, help="(MAR) strict drivers; kept observed.")

    ap.add_argument("--task", type=str, default=None, help="classification or regression. If omitted, inferred from y.")
    ap.add_argument("--test-size", type=float, default=0.30)

    ap.add_argument("--fairness-col", type=str, default=None, help="Optional group column for a simple bias gap metric.")

    ap.add_argument(
        "--imputers",
        nargs="+",
        default=["SNI", "MissForest", "MeanMode", "HyperImpute", "TabCSDI"],
        help=(
            "Imputation methods. 'SNI' or any baseline in baselines/registry.py ("
            + ",".join(list_baselines())
            + "). Special: COMPLETE, DROP_ROWS."
        ),
    )

    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8])

    # SNI knobs (small-ish defaults; can match paper settings if desired)
    ap.add_argument("--sni-use-gpu", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--sni-epochs", type=int, default=80)
    ap.add_argument("--sni-max-iters", type=int, default=2)
    ap.add_argument("--sni-emb-dim", type=int, default=64)
    ap.add_argument("--sni-num-heads", type=int, default=8)
    ap.add_argument("--sni-hidden-dims", type=str, default="128,64")
    ap.add_argument("--sni-lr", type=float, default=2e-4)
    ap.add_argument("--sni-batch-size", type=int, default=128)

    ap.add_argument("--baseline-use-gpu", type=str, default="false", choices=["true", "false"], help="GPU for GAIN/MIWAE/TabCSDI baselines.")

    ap.add_argument("--class-weight-balanced", type=str, default="true", choices=["true", "false"], help="Use class_weight='balanced' for classification model.")

    ap.add_argument("--save-imputed", type=str, default="false", choices=["true", "false"], help="Save imputed feature CSVs per seed/method.")
    ap.add_argument("--save-missing", type=str, default="true", choices=["true", "false"], help="Save generated missing feature CSV + metadata.")

    ap.add_argument("--outdir", type=str, required=True)

    args = ap.parse_args()

    outdir = _ensure_outdir(Path(args.outdir))

    cat_cols = _parse_list(args.categorical_cols)
    cont_cols = _parse_list(args.continuous_cols)
    if not cat_cols and not cont_cols:
        raise ValueError("Provide --categorical-cols and/or --continuous-cols")

    target = str(args.target_col).strip()
    fairness_col = str(args.fairness_col).strip() if args.fairness_col else None

    # Guardrail: target must NOT be included as a feature.
    if target in set(cat_cols + cont_cols):
        raise ValueError(
            f"target-col='{target}' is included in feature columns. "
            "For downstream validation we keep y fully observed and ONLY mask/impute X. "
            "Please remove the target from --categorical-cols/--continuous-cols."
        )

    imputers = [str(m).strip() for m in args.imputers]

    df = pd.read_csv(args.input_complete)

    # Validate columns
    for c in cat_cols + cont_cols + [target] + ([fairness_col] if fairness_col else []):
        if c and c not in df.columns:
            raise KeyError(f"Column '{c}' not found in {args.input_complete}")

    # Features / label
    X_complete = df[cat_cols + cont_cols].copy()
    y = df[target].copy()

    task = _infer_task_type(y, args.task)

    # For fairness gap computation we want the group values from the original complete data
    group_all = df[fairness_col].to_numpy() if fairness_col else None

    # Missingness generation config
    mechanism = str(args.mechanism).upper()
    rate = float(args.missing_rate)

    mar_drivers = _parse_list(args.mar_driver_cols) if args.mar_driver_cols else []
    if mechanism == "MAR" and len(mar_drivers) == 0:
        # default drivers: fairness_col (if provided) + first continuous
        drivers = []
        if fairness_col and fairness_col in X_complete.columns:
            drivers.append(fairness_col)
        if len(cont_cols) > 0 and cont_cols[0] not in drivers:
            drivers.append(cont_cols[0])
        if len(drivers) == 0:
            drivers = [X_complete.columns[0]]
        mar_drivers = drivers

    exclude_cols = []
    if fairness_col and fairness_col in X_complete.columns:
        # Keep sensitive attribute observed (common review-safe setting).
        # Note: if fairness_col is NOT included as a feature column, we still
        # can compute group metrics (using the complete data), but we must NOT
        # pass it into the missingness generator (it would raise).
        exclude_cols.append(fairness_col)

    # SNI config
    sni_hidden_dims = tuple(int(x) for x in str(args.sni_hidden_dims).split(",") if str(x).strip())
    sni_use_gpu = args.sni_use_gpu == "true"

    baseline_use_gpu = args.baseline_use_gpu == "true"

    class_weight_balanced = args.class_weight_balanced == "true"

    save_imputed = args.save_imputed == "true"
    save_missing = args.save_missing == "true"

    rows_out: List[Dict[str, object]] = []

    # Prepare global folders
    if save_imputed:
        _ensure_outdir(outdir / "imputed")
    if save_missing:
        _ensure_outdir(outdir / "missing")
        _ensure_outdir(outdir / "metadata")

    # Build schema for stable dtypes (important for int-coded categoricals)
    schema = infer_schema_from_complete(
        X_complete,
        categorical_vars=cat_cols,
        continuous_vars=cont_cols,
    )
    X_complete_cast = cast_dataframe_to_schema(X_complete, schema)

    for seed in args.seeds:
        seed = int(seed)

        # 1) Generate missingness on features only
        gen_res = generate_missing_dataset(
            df=pd.concat([X_complete_cast], axis=1),
            mechanism=mechanism,
            rate=rate,
            seed=seed,
            dataset_name=str(args.dataset_name),
            categorical_cols=cat_cols,
            continuous_cols=cont_cols,
            exclude_cols=exclude_cols,
            mar_driver_cols=mar_drivers if mechanism == "MAR" else None,
            strict_mar=(mechanism == "MAR"),
        )
        X_missing = gen_res.data_missing[cat_cols + cont_cols].copy()
        X_missing = cast_dataframe_to_schema(X_missing, schema)

        if save_missing:
            X_missing.to_csv(outdir / "missing" / f"seed{seed}_features_missing.csv", index=False)
            _save_json(outdir / "metadata" / f"seed{seed}_missing_meta.json", gen_res.metadata)

        # 2) Create a fixed train/test split (same indices for all imputers)
        idx = np.arange(len(X_complete_cast))

        if task == "classification":
            train_idx, test_idx = train_test_split(
                idx,
                test_size=float(args.test_size),
                random_state=seed,
                stratify=y,
            )
        else:
            train_idx, test_idx = train_test_split(
                idx,
                test_size=float(args.test_size),
                random_state=seed,
            )

        # 3) Evaluate each imputer
        for method in imputers:
            m = method.strip()

            t_imp0 = time.time()

            if m.upper() == "COMPLETE":
                X_imp = X_complete_cast.copy()
                imp_runtime = 0.0

            elif m.upper() == "DROP_ROWS":
                # complete-case analysis (drop any row with missing in any feature)
                keep_mask = ~X_missing.isna().any(axis=1)
                X_imp = X_missing.loc[keep_mask].copy()
                # labels/groups must match the kept rows
                y_keep = y.loc[keep_mask].reset_index(drop=True)
                group_keep = group_all[keep_mask.to_numpy()] if group_all is not None else None
                imp_runtime = float(time.time() - t_imp0)

            elif m.upper() == "SNI":
                cfg = SNIConfig(
                    seed=seed,
                    use_gpu=sni_use_gpu,
                    variant="SNI",
                    epochs=int(args.sni_epochs),
                    max_iters=int(args.sni_max_iters),
                    emb_dim=int(args.sni_emb_dim),
                    num_heads=int(args.sni_num_heads),
                    hidden_dims=sni_hidden_dims,
                    lr=float(args.sni_lr),
                    batch_size=int(args.sni_batch_size),
                )
                imputer = SNIImputer(categorical_vars=cat_cols, continuous_vars=cont_cols, config=cfg)
                X_imp = imputer.impute(X_missing=X_missing, X_complete=None, mask_df=X_missing.isna())
                X_imp = cast_dataframe_to_schema(X_imp, schema)
                imp_runtime = float(time.time() - t_imp0)

            else:
                # baseline
                if m not in list_baselines():
                    raise KeyError(
                        f"Unknown imputer '{m}'. Allowed: SNI, COMPLETE, DROP_ROWS, baselines={list_baselines()}"
                    )
                baseline = build_baseline_imputer(
                    m,
                    categorical_vars=cat_cols,
                    continuous_vars=cont_cols,
                    seed=seed,
                    use_gpu=baseline_use_gpu,
                )
                X_imp = baseline.impute(X_complete_cast, X_missing)
                X_imp = cast_dataframe_to_schema(X_imp, schema)
                imp_runtime = float(time.time() - t_imp0)

            # Optional save
            if save_imputed and m.upper() != "DROP_ROWS":
                mdir = _ensure_outdir(outdir / "imputed" / m)
                X_imp.to_csv(mdir / f"seed{seed}.csv", index=False)

            # 4) Train/eval downstream model
            t_fit0 = time.time()

            pipe = _build_model_pipeline(
                task=task,
                categorical_cols=cat_cols,
                continuous_cols=cont_cols,
                seed=seed,
                class_weight_balanced=class_weight_balanced,
            )

            if m.upper() == "DROP_ROWS":
                # split within remaining rows (not strictly comparable, but a useful baseline)
                idx2 = np.arange(len(X_imp))
                if task == "classification":
                    tr2, te2 = train_test_split(
                        idx2,
                        test_size=float(args.test_size),
                        random_state=seed,
                        stratify=y_keep,
                    )
                else:
                    tr2, te2 = train_test_split(idx2, test_size=float(args.test_size), random_state=seed)

                X_train = X_imp.iloc[tr2]
                y_train = y_keep.iloc[tr2]
                X_test = X_imp.iloc[te2]
                y_test = y_keep.iloc[te2]
                group_test = group_keep[te2] if group_keep is not None else None

            else:
                X_train = X_imp.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X_imp.iloc[test_idx]
                y_test = y.iloc[test_idx]
                group_test = group_all[test_idx] if group_all is not None else None

            pipe.fit(X_train, y_train)

            if task == "classification":
                y_pred = pipe.predict(X_test)
                y_proba = None
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None
                metrics = _classification_metrics(y_test, y_pred, y_proba=y_proba)

                gap = {}
                if group_test is not None:
                    gap = _group_gap(task=task, y_true=np.asarray(y_test), y_pred=np.asarray(y_pred), group=np.asarray(group_test))

            else:
                y_pred = pipe.predict(X_test)
                metrics = _regression_metrics(y_test, y_pred)

                gap = {}
                if group_test is not None:
                    gap = _group_gap(task=task, y_true=np.asarray(y_test, dtype=float), y_pred=np.asarray(y_pred, dtype=float), group=np.asarray(group_test))

            fit_runtime = float(time.time() - t_fit0)

            row = {
                "dataset": args.dataset_name,
                "target": target,
                "task": task,
                "mechanism": mechanism,
                "missing_rate": rate,
                "seed": seed,
                "imputer": m,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "impute_runtime_sec": float(imp_runtime),
                "fit_runtime_sec": float(fit_runtime),
                **metrics,
                **gap,
            }
            rows_out.append(row)

            print(f"[OK] seed={seed} imputer={m} metrics={{{', '.join([f'{k}={row[k]:.4f}' for k in metrics.keys()])}}}")

    # Save per-seed
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(outdir / "metrics_per_seed.csv", index=False)

    # Aggregate summary
    metric_cols = [c for c in df_out.columns if c not in {
        "dataset", "target", "task", "mechanism", "missing_rate", "seed", "imputer"
    }]

    # mean/std for numeric cols
    agg = (
        df_out
        .groupby(["dataset", "target", "task", "mechanism", "missing_rate", "imputer"], dropna=False)[metric_cols]
        .agg(["mean", "std"])\
        .reset_index()
    )

    # flatten columns
    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]

    agg.to_csv(outdir / "metrics_summary.csv", index=False)

    print(f"[DONE] Wrote: {outdir / 'metrics_per_seed.csv'}")
    print(f"[DONE] Wrote: {outdir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
