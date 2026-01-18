#!/usr/bin/env python3
"""
SNI v0.2 Experiment Runner
Runs SNI imputation experiments from a manifest CSV file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from SNI_v0_2 import SNIImputer
from SNI_v0_2.dataio import cast_dataframe_to_schema, load_complete_and_missing
from SNI_v0_2.imputer import SNIConfig
from SNI_v0_2.metrics import evaluate_imputation

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message=r".*\[IterativeImputer\] Early stopping criterion not reached.*"
)


def _split_vars(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        parts.extend([p for p in chunk.split() if p.strip()])
    return parts


def _as_bool(x) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser(description="Run SNI v0.2 experiments from manifest")
    ap.add_argument("--manifest", type=str, required=True, help="CSV manifest (one experiment per row)")
    ap.add_argument("--outdir", type=str, required=True, help="Root output directory")
    ap.add_argument("--default-use-gpu", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing results")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    required = {"exp_id", "input_complete", "input_missing", "variant", "seed", "categorical_vars", "continuous_vars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    summaries = []

    for i, row in df.iterrows():
        exp_id = str(row["exp_id"])
        outdir = out_root / exp_id
        
        # Skip existing
        if args.skip_existing and (outdir / "metrics_summary.json").exists():
            print(f"[SKIP] {exp_id} already exists")
            continue
            
        outdir.mkdir(parents=True, exist_ok=True)

        cat_vars = _split_vars(row["categorical_vars"])
        cont_vars = _split_vars(row["continuous_vars"])
        all_vars = cat_vars + cont_vars

        # Load data
        X_complete, X_missing, schema = load_complete_and_missing(
            input_complete=str(row["input_complete"]),
            input_missing=str(row["input_missing"]),
            categorical_vars=cat_vars,
            continuous_vars=cont_vars,
        )

        mask_df = X_missing.isna()
        if "mask_file" in df.columns and pd.notna(row.get("mask_file", None)):
            mask_df = pd.read_csv(row["mask_file"])[all_vars].astype(int).astype(bool)

        # Build config with v0.2 optimized defaults
        cfg = SNIConfig(
            alpha0=float(row.get("alpha0", 1.0)),
            gamma=float(row.get("gamma", 0.9)),
            max_iters=int(row.get("max_iters", 2)),          # v0.2: 2 (was 3)
            tol=float(row.get("tol", 1e-4)),
            use_stat_refine=_as_bool(row.get("use_stat_refine", True)),
            mask_fraction=float(row.get("mask_fraction", 0.15)),
            hidden_dims=tuple(int(x) for x in str(row.get("hidden_dims", "64,32")).split(",") if str(x).strip()),
            emb_dim=int(row.get("emb_dim", 32)),
            num_heads=int(row.get("num_heads", 4)),
            lr=float(row.get("lr", 1e-3)),
            epochs=int(row.get("epochs", 50)),               # v0.2: 50 (was 80)
            batch_size=int(row.get("batch_size", 128)),      # v0.2: 128 (was 64)
            early_stopping_patience=int(row.get("early_stopping_patience", 10)),
            variant=str(row["variant"]),
            hard_prior_lambda=float(row.get("hard_prior_lambda", 10.0)),
            seed=int(row["seed"]),
            use_gpu=_as_bool(row.get("use_gpu", args.default_use_gpu)),
        )

        # Handle SNI+KNN variant
        train_variant = cfg.variant if cfg.variant != "SNI+KNN" else "SNI"
        cfg.variant = train_variant

        imputer = SNIImputer(cat_vars, cont_vars, config=cfg)

        t0 = time.time()
        X_imp = imputer.impute(X_missing=X_missing, X_complete=X_complete, mask_df=mask_df)
        runtime = float(time.time() - t0)

        X_imp = cast_dataframe_to_schema(X_imp, schema)

        # Evaluate
        eval_res = evaluate_imputation(
            X_imputed=X_imp,
            X_complete=X_complete,
            X_missing=X_missing,
            categorical_vars=cat_vars,
            continuous_vars=cont_vars,
            mask_df=mask_df,
        )

        # Save results
        X_imp.to_csv(outdir / "imputed.csv", index=False)
        eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)

        summary = dict(eval_res.summary)
        summary.update({
            "exp_id": exp_id,
            "variant": str(row["variant"]),
            "seed": int(row["seed"]),
            "runtime_sec": runtime,
        })
        pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
        (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # Dependency matrix
        D = imputer.compute_dependency_matrix()
        D.to_csv(outdir / "dependency_matrix.csv", index=True)

        summaries.append(summary)

        print(f"[OK] {exp_id} done in {runtime:.1f}s -> {outdir}")

    # Global summary
    if len(summaries) > 0:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(out_root / "summary_all_runs.csv", index=False)
        print(f"[DONE] wrote {out_root / 'summary_all_runs.csv'}")


if __name__ == "__main__":
    main()
