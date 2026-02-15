#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from SNI_v0_3 import SNIImputer
from SNI_v0_3.dataio import cast_dataframe_to_schema, load_complete_and_missing
from SNI_v0_3.imputer import SNIConfig
from SNI_v0_3.metrics import evaluate_imputation, augment_summary_with_imputer_stats

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message=r".*\[IterativeImputer\] Early stopping criterion not reached.*"
)

def _parse_list(arg: Optional[List[str]]) -> List[str]:
    if arg is None:
        return []
    # allow "A,B,C" or space separated
    out = []
    for x in arg:
        parts = [p.strip() for p in x.split(",") if p.strip()]
        out.extend(parts)
    return out


def _load_mask(mask_file: str, columns: List[str]) -> pd.DataFrame:
    m = pd.read_csv(mask_file)
    # try align columns
    if set(columns).issubset(set(m.columns)):
        m = m[columns]
    # accept 0/1 or True/False
    m_bool = m.astype(int).astype(bool)
    return m_bool


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_dependency_matrix(df: pd.DataFrame, out_png: Path, dpi: int = 600) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Source feature (attended to)")
    ax.set_ylabel("Target feature (imputed)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention mass")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _knn_postprocess_categorical(
    X_imputed: pd.DataFrame,
    X_missing: pd.DataFrame,
    dependency_matrix: pd.DataFrame,
    target: str,
    top_m: int = 5,
    k: int = 5,
) -> pd.Series:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    miss_mask = X_missing[target].isna()
    if int(miss_mask.sum()) == 0:
        return X_imputed[target]

    # pick top-m sources
    w = dependency_matrix.loc[target].drop(index=target)
    top_sources = w.sort_values(ascending=False).head(top_m).index.tolist()

    # build design matrix
    X_cond = X_imputed[top_sources].copy()
    for c in X_cond.columns:
        if not np.issubdtype(X_cond[c].dtype, np.number):
            X_cond[c] = pd.factorize(X_cond[c].astype(str))[0]
    X_cond = X_cond.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # y (keep original dtype; LabelEncoder-style stringification here would
    # re-introduce artifacts like "82.0" for integer-coded categories)
    y_obs = X_imputed.loc[~miss_mask, target].astype(object)
    X_obs = X_cond.loc[~miss_mask]
    X_miss = X_cond.loc[miss_mask]

    if len(y_obs) < k:
        # fallback: do nothing
        return X_imputed[target]

    scaler = StandardScaler()
    X_obs_s = scaler.fit_transform(X_obs.values)
    X_miss_s = scaler.transform(X_miss.values)

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(X_obs_s, y_obs.values)
    y_pred = clf.predict(X_miss_s)

    out = X_imputed[target].copy()
    # Preserve dtype (prefer Int64 for integer-coded categorical vars).
    try:
        if pd.api.types.is_integer_dtype(out.dtype) or str(out.dtype) == "Int64":
            y_pred_s = pd.Series(y_pred, index=out.index[miss_mask])
            y_pred_s = pd.to_numeric(y_pred_s, errors="coerce").round().astype("Int64")
            out.loc[miss_mask] = y_pred_s.values
        else:
            out.loc[miss_mask] = y_pred
    except Exception:
        out.loc[miss_mask] = y_pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-complete", type=str, required=True, help="Complete ground-truth CSV.")
    ap.add_argument("--input-missing", type=str, required=True, help="Missing CSV (NaN positions are evaluated).")
    ap.add_argument("--mask-file", type=str, default=None, help="Optional mask CSV (1=eval position).")
    ap.add_argument("--categorical-vars", nargs="+", default=[], help="Categorical variable names.")
    ap.add_argument("--continuous-vars", nargs="+", default=[], help="Continuous variable names.")
    ap.add_argument("--variant", type=str, default="SNI", choices=["SNI", "NoPrior", "HardPrior", "SNI-M", "SNI+KNN"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-gpu", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--hidden-dims", type=str, default="64,32")
    ap.add_argument("--mask-fraction", type=float, default=0.15)
    ap.add_argument("--use-stat-refine", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--hard-prior-lambda", type=float, default=10.0)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--save-imputed", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--save-dependency-fig", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--knn-top-m", type=int, default=5)
    ap.add_argument("--knn-mode-threshold", type=float, default=0.5)
    # v0.3 parameters
    ap.add_argument("--cat-balance-mode", type=str, default="none",
                     choices=["none", "inverse_freq", "sqrt_inverse_freq"],
                     help="(v0.3) Categorical class-balance weighting mode")
    ap.add_argument("--cat-lr-mult", type=float, default=1.0,
                     help="(v0.3) LR multiplier for categorical head parameters")
    ap.add_argument("--lambda-mode", type=str, default="learned",
                     choices=["learned", "fixed"],
                     help="(v0.3) Lambda mode: 'learned' (default) or 'fixed'")
    ap.add_argument("--lambda-fixed-value", type=float, default=1.0,
                     help="(v0.3) Fixed lambda value when --lambda-mode=fixed")

    args = ap.parse_args()

    sni_version = "v0.3"

    outdir = _ensure_outdir(args.outdir)

    cat_vars = _parse_list(args.categorical_vars)
    cont_vars = _parse_list(args.continuous_vars)
    all_vars = cat_vars + cont_vars
    if len(all_vars) == 0:
        raise ValueError("Provide --categorical-vars and/or --continuous-vars.")

    # Load with a stable schema derived from *_complete.csv.
    # This keeps integer-coded categoricals as Int64 (nullable integer) and
    # prevents float coercion (e.g., 82 -> 82.0), eliminating pandas dtype warnings.
    X_complete, X_missing, schema = load_complete_and_missing(
        input_complete=args.input_complete,
        input_missing=args.input_missing,
        categorical_vars=cat_vars,
        continuous_vars=cont_vars,
    )

    if args.mask_file is not None:
        mask_df = _load_mask(args.mask_file, all_vars)
    else:
        mask_df = X_missing.isna()

    # Build SNIConfig — v0.3 adds cat_balance_mode, cat_lr_mult, lambda_mode, lambda_fixed_value
    cfg_kwargs = dict(
        alpha0=args.alpha0,
        gamma=args.gamma,
        max_iters=args.max_iters,
        tol=args.tol,
        use_stat_refine=(args.use_stat_refine == "true"),
        mask_fraction=args.mask_fraction,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",") if x.strip()),
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        variant=args.variant if args.variant != "SNI+KNN" else "SNI",
        hard_prior_lambda=args.hard_prior_lambda,
        seed=args.seed,
        use_gpu=(args.use_gpu == "true"),
    )

    cfg_kwargs.update(
        cat_balance_mode=args.cat_balance_mode,
        cat_lr_mult=args.cat_lr_mult,
        lambda_mode=args.lambda_mode,
        lambda_fixed_value=args.lambda_fixed_value,
    )

    cfg = SNIConfig(**cfg_kwargs)

    imputer = SNIImputer(categorical_vars=cat_vars, continuous_vars=cont_vars, config=cfg)

    t0 = time.time()
    X_imp = imputer.impute(X_missing=X_missing, X_complete=X_complete, mask_df=mask_df)
    runtime_sec = float(time.time() - t0)

    # Ensure clean output dtypes (e.g., categorical Int64 instead of float64).
    X_imp = cast_dataframe_to_schema(X_imp, schema)

    # optional SNI+KNN post-processing
    if args.variant == "SNI+KNN" and len(cat_vars) > 0:
        D = imputer.compute_dependency_matrix()
        for f in cat_vars:
            # decide whether to post-process based on mode frequency on complete
            mode_freq = float(X_complete[f].astype(str).value_counts(normalize=True).iloc[0])
            if mode_freq >= args.knn_mode_threshold:
                X_imp[f] = _knn_postprocess_categorical(
                    X_imputed=X_imp,
                    X_missing=X_missing,
                    dependency_matrix=D,
                    target=f,
                    top_m=args.knn_top_m,
                    k=args.knn_k,
                )

    # evaluation
    eval_res = evaluate_imputation(
        X_imputed=X_imp,
        X_complete=X_complete,
        X_missing=X_missing,
        categorical_vars=cat_vars,
        continuous_vars=cont_vars,
        mask_df=mask_df,
    )

    # save artifacts
    if args.save_imputed == "true":
        X_imp.to_csv(outdir / "imputed.csv", index=False)

    eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)

    summary = dict(eval_res.summary)
    summary.update(
        {
            "variant": args.variant,
            "sni_version": sni_version,
            "seed": args.seed,
            "runtime_sec": runtime_sec,
            "n_rows": int(len(X_imp)),
            "n_cols": int(len(X_imp.columns)),
        }
    )

    summary = augment_summary_with_imputer_stats(summary, imputer)

    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
    _save_json(outdir / "metrics_summary.json", summary)

    try:
        imputer.save_convergence_curve(outdir / "convergence_curve.csv")
    except Exception:
        pass
    try:
        imputer.save_lambda_per_head(outdir / "lambda_per_head.csv")
    except Exception:
        pass
    try:
        imputer.save_lambda_values(outdir / "lambda_values.json")
    except Exception:
        pass

    # save per-target attention maps & lambda traces (for debugging / paper figures)
    attn_dir = outdir / "attention_maps"
    attn_dir.mkdir(exist_ok=True)
    for k, mat in imputer.attention_maps.items():
        try:
            pd.DataFrame(mat).to_csv(attn_dir / f"{k}.csv", index=False, header=False)
        except Exception:
            pass

    lam_dir = outdir / "lambda_traces"
    lam_dir.mkdir(exist_ok=True)
    for k, trace in imputer.lambda_trace_per_head.items():
        try:
            df_l = pd.DataFrame(trace, columns=[f"head{i}" for i in range(len(trace[0]))] if len(trace)>0 else None)
            df_l.insert(0, "epoch", range(len(df_l)))
            df_l.to_csv(lam_dir / f"{k}.csv", index=False)
        except Exception:
            pass

    # dependency matrix & network edges
    D = imputer.compute_dependency_matrix()
    D.to_csv(outdir / "dependency_matrix.csv", index=True)

    edges = imputer.export_dependency_network_edges(tau=0.15)
    edges.to_csv(outdir / "dependency_network_edges.csv", index=False)

    if args.save_dependency_fig == "true":
        _plot_dependency_matrix(D, outdir / "dependency_matrix.png", dpi=args.dpi)

    # config snapshot
    run_cfg = {
        "input_complete": args.input_complete,
        "input_missing": args.input_missing,
        "mask_file": args.mask_file,
        "categorical_vars": cat_vars,
        "continuous_vars": cont_vars,
        "variant": args.variant,
        "sni_version": sni_version,
        "seed": args.seed,
        "runtime_sec": runtime_sec,
        "cfg": cfg.__dict__,
    }
    _save_json(outdir / "run_config.json", run_cfg)

    print(f"[DONE] outdir={outdir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
