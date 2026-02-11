#!/usr/bin/env python3
"""exp1_audit_story_leakage.py

Ext1 / Experiment 1 (Interpretability audit story):
--------------------------------------------------

Goal
----
Provide a *reviewer-friendly*, real-data audit story for the learned dependency
network D:

1) We inject a *known-leakage/proxy* column into a real dataset (e.g., MIMIC),
   then generate missingness on other columns.
2) Run SNI and export the dependency matrix/network.
3) Produce an *audit report* that flags suspicious "single-source dominance" and
   highlights the injected proxy as the top dependency for the audited target.
4) (Optional) Re-run without the proxy to show that the imputation metric on the
   audited target drops, i.e., the proxy was truly driving performance.

Why this helps
--------------
It demonstrates how D can be used as a *practical data-auditing tool*:
- Detect label leakage / duplicate columns
- Identify dominant drivers of imputations
- Provide actionable "remove/lock" recommendations

Outputs
-------
<outdir>/
  with_leak/
    imputed.csv
    metrics_summary.{csv,json}
    metrics_per_feature.csv
    dependency_matrix.csv
    dependency_matrix.png
    dependency_network_edges.csv
    audit_top_sources.csv
    audit_flags.csv
  without_leak/              (if --run-without-leak true)
    ... (same)
  audit_report.md
  audit_comparison.csv

Example (categorical proxy: ALARM_LEAK = ALARM)
-------------------------------------------------
python ext1/scripts/exp1_audit_story_leakage.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars SpO2 ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --audit-target ALARM \
  --mechanism MAR --missing-rate 0.30 \
  --seed 2026 \
  --outdir results_ext1/audit_mimic_alarm

Example (continuous proxy: SBP_LEAK = SBP + noise)
---------------------------------------------------
python ext1/scripts/exp1_audit_story_leakage.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars SpO2 ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --audit-target SBP \
  --leak-source SBP --leak-col-name SBP_LEAK \
  --leak-noise-std 0.5 \
  --mechanism MAR --missing-rate 0.30 \
  --seed 2026 \
  --outdir results_ext1/audit_mimic_sbp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Path setup: allow imports from repo root + missingness generator
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

GEN_DIR = REPO_ROOT / "utility_missing_data_gen_v1"
sys.path.insert(0, str(GEN_DIR))

from SNI_v0_3 import SNIImputer
from SNI_v0_3.imputer import SNIConfig
from SNI_v0_3.metrics import evaluate_imputation
from SNI_v0_3.dataio import infer_schema_from_complete, cast_dataframe_to_schema

from missing_data_generator import generate_missing_dataset


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


def _plot_dependency_matrix(df: pd.DataFrame, out_png: Path, dpi: int = 600) -> None:
    fig = plt.figure(figsize=(9, 7))
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


def _topk_sources(D: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    rows = []
    for tgt in D.index:
        s = D.loc[tgt].copy()
        if tgt in s.index:
            s = s.drop(index=tgt)
        top = s.sort_values(ascending=False).head(k)
        rec: Dict[str, object] = {"target": tgt}
        for i, (src, w) in enumerate(top.items(), start=1):
            rec[f"top{i}_src"] = src
            rec[f"top{i}_w"] = float(w)
        rec["top1_w"] = float(top.iloc[0]) if len(top) > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _run_sni_once(
    *,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    use_gpu: bool,
    variant: str,
    epochs: int,
    max_iters: int,
    emb_dim: int,
    num_heads: int,
    hidden_dims: Tuple[int, ...],
    lr: float,
    batch_size: int,
    outdir: Path,
    tau_edges: float,
    dpi: int,
) -> Dict[str, object]:
    """Run SNI (single run) and write standard artifacts."""
    _ensure_outdir(outdir)

    # Stable schema from complete
    schema = infer_schema_from_complete(
        X_complete,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )
    Xc = cast_dataframe_to_schema(X_complete, schema)
    Xm = cast_dataframe_to_schema(X_missing, schema)

    mask_df = Xm.isna()

    cfg = SNIConfig(
        seed=int(seed),
        use_gpu=bool(use_gpu),
        variant=str(variant),
        epochs=int(epochs),
        max_iters=int(max_iters),
        emb_dim=int(emb_dim),
        num_heads=int(num_heads),
        hidden_dims=tuple(int(x) for x in hidden_dims),
        lr=float(lr),
        batch_size=int(batch_size),
    )

    imputer = SNIImputer(categorical_vars=categorical_vars, continuous_vars=continuous_vars, config=cfg)

    t0 = time.time()
    X_imp = imputer.impute(X_missing=Xm, X_complete=None, mask_df=mask_df)
    runtime = float(time.time() - t0)

    # cast back to schema
    X_imp = cast_dataframe_to_schema(X_imp, schema)

    # evaluate (only true missing positions)
    eval_res = evaluate_imputation(
        X_imputed=X_imp,
        X_complete=Xc,
        X_missing=Xm,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        mask_df=mask_df,
    )

    # Save outputs (same style as scripts/run_experiment.py)
    X_imp.to_csv(outdir / "imputed.csv", index=False)
    eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)

    summary = dict(eval_res.summary)
    summary.update(
        {
            "variant": variant,
            "seed": int(seed),
            "runtime_sec": runtime,
            "n_rows": int(len(X_imp)),
            "n_cols": int(len(X_imp.columns)),
        }
    )
    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
    _save_json(outdir / "metrics_summary.json", summary)

    # dependency matrix + edges
    D = imputer.compute_dependency_matrix()
    D.to_csv(outdir / "dependency_matrix.csv", index=True)

    edges = imputer.export_dependency_network_edges(tau=float(tau_edges))
    edges.to_csv(outdir / "dependency_network_edges.csv", index=False)

    _plot_dependency_matrix(D, outdir / "dependency_matrix.png", dpi=int(dpi))

    # audit top-k
    topk_df = _topk_sources(D, k=5)
    topk_df.to_csv(outdir / "audit_top_sources.csv", index=False)

    # quick flags: dominance
    flags = topk_df[topk_df["top1_w"] >= 0.6].copy()
    flags.to_csv(outdir / "audit_flags.csv", index=False)

    return {
        "summary": summary,
        "per_feature": eval_res.per_feature,
        "dependency_matrix": D,
        "topk": topk_df,
        "runtime_sec": runtime,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Ext1-Exp1: Real-data interpretability audit story via leakage/proxy injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-complete", type=str, required=True, help="Complete ground-truth CSV.")
    ap.add_argument("--dataset-name", type=str, default="DATA", help="Name for reporting only.")

    ap.add_argument("--categorical-vars", nargs="+", default=[], help="Categorical variable names (space or comma separated).")
    ap.add_argument("--continuous-vars", nargs="+", default=[], help="Continuous variable names (space or comma separated).")

    ap.add_argument("--audit-target", type=str, required=True, help="Which target column to audit (e.g., ALARM).")
    ap.add_argument("--leak-source", type=str, default=None, help="Which source column to copy as the proxy. Default: same as --audit-target")
    ap.add_argument("--leak-col-name", type=str, default=None, help="Name of the injected proxy column. Default: <audit_target>_LEAK")
    ap.add_argument("--leak-noise-std", type=float, default=0.0,
                     help="(Continuous proxy only) Add Gaussian noise N(0, std) to the proxy copy. "
                          "0 = perfect duplicate; >0 = realistic noisy proxy.")

    ap.add_argument("--mechanism", type=str, default="MAR", choices=["MCAR", "MAR", "MNAR"], help="Missingness mechanism.")
    ap.add_argument("--missing-rate", type=float, default=0.30, help="Missing rate.")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--mar-driver-cols", nargs="+", default=None, help="(MAR) driver columns. If empty, use the first continuous var.")

    ap.add_argument("--run-without-leak", type=str, default="true", choices=["true", "false"], help="Also run a control without the proxy column.")

    # SNI hyperparams (keep modest defaults so it runs on CPU too)
    ap.add_argument("--use-gpu", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--variant", type=str, default="SNI", choices=["SNI", "NoPrior", "HardPrior", "SNI-M"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--hidden-dims", type=str, default="128,64")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=128)

    ap.add_argument("--tau-edges", type=float, default=0.15, help="Edge threshold for exporting dependency edges.")
    ap.add_argument("--dpi", type=int, default=600)

    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")

    args = ap.parse_args()

    out_root = _ensure_outdir(Path(args.outdir))

    cat_vars = _parse_list(args.categorical_vars)
    cont_vars = _parse_list(args.continuous_vars)
    if not cat_vars and not cont_vars:
        raise ValueError("Provide --categorical-vars and/or --continuous-vars")

    audit_target = str(args.audit_target).strip()
    leak_source = str(args.leak_source).strip() if args.leak_source else audit_target
    leak_col = str(args.leak_col_name).strip() if args.leak_col_name else f"{audit_target}_LEAK"

    # Load complete data
    df = pd.read_csv(args.input_complete)

    # Basic checks
    for c in [leak_source, audit_target]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in complete CSV.")

    # Build augmented complete table
    df_aug = df.copy()
    df_aug[leak_col] = df_aug[leak_source].copy()

    # Optional: add Gaussian noise for continuous proxies to simulate a
    # realistic "noisy duplicate" rather than an exact copy.
    leak_noise_std = float(args.leak_noise_std)
    leak_is_continuous = leak_source in cont_vars

    if leak_is_continuous and leak_noise_std > 0:
        rng = np.random.default_rng(int(args.seed))
        noise = rng.normal(loc=0.0, scale=leak_noise_std, size=len(df_aug))
        df_aug[leak_col] = df_aug[leak_col].astype(float) + noise

    # Update var lists: proxy inherits the type of its source column
    cat_aug = list(cat_vars)
    cont_aug = list(cont_vars)
    if leak_source in cat_vars and leak_col not in cat_aug:
        cat_aug.append(leak_col)
    elif leak_source in cont_vars and leak_col not in cont_aug:
        cont_aug.append(leak_col)
    else:
        # default: treat as categorical (safe for int-coded classes)
        if leak_col not in cat_aug:
            cat_aug.append(leak_col)

    all_vars_aug = cat_aug + cont_aug

    # Generate missingness on augmented data, but keep the proxy column always observed
    exclude_cols = [leak_col]

    mechanism = str(args.mechanism).upper()

    mar_drivers = _parse_list(args.mar_driver_cols) if args.mar_driver_cols else []
    if mechanism == "MAR" and len(mar_drivers) == 0:
        # strict MAR: pick the first continuous feature (not the proxy) if possible
        candidates = [c for c in cont_aug if c != leak_col]
        if len(candidates) == 0:
            # fallback: pick first categorical feature
            candidates = [c for c in cat_aug if c != leak_col]
        if len(candidates) == 0:
            raise ValueError("Cannot auto-pick MAR driver; please set --mar-driver-cols")
        mar_drivers = [candidates[0]]

    gen_res = generate_missing_dataset(
        df_aug,
        mechanism=mechanism,
        rate=float(args.missing_rate),
        seed=int(args.seed),
        dataset_name=str(args.dataset_name),
        categorical_cols=cat_aug,
        continuous_cols=cont_aug,
        exclude_cols=exclude_cols,
        mar_driver_cols=mar_drivers if mechanism == "MAR" else None,
        strict_mar=(mechanism == "MAR"),
    )

    df_missing_aug = gen_res.data_missing

    # Persist the generated missing dataset + metadata for reproducibility
    _ensure_outdir(out_root / "_generated")
    df_missing_aug.to_csv(out_root / "_generated" / f"{args.dataset_name}_{mechanism}_{args.missing_rate:.2f}_seed{args.seed}_with_leak.csv", index=False)
    _save_json(out_root / "_generated" / "missing_generation_meta.json", gen_res.metadata)

    # Prepare X for SNI (drop non-feature cols)
    X_complete_aug = df_aug[all_vars_aug].copy()
    X_missing_aug = df_missing_aug[all_vars_aug].copy()

    use_gpu = args.use_gpu == "true"
    hidden_dims = tuple(int(x) for x in str(args.hidden_dims).split(",") if str(x).strip())

    # Run with leak
    run_with_dir = out_root / "with_leak"
    res_with = _run_sni_once(
        X_complete=X_complete_aug,
        X_missing=X_missing_aug,
        categorical_vars=cat_aug,
        continuous_vars=cont_aug,
        seed=int(args.seed),
        use_gpu=use_gpu,
        variant=str(args.variant),
        epochs=int(args.epochs),
        max_iters=int(args.max_iters),
        emb_dim=int(args.emb_dim),
        num_heads=int(args.num_heads),
        hidden_dims=hidden_dims,
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        outdir=run_with_dir,
        tau_edges=float(args.tau_edges),
        dpi=int(args.dpi),
    )

    # Optional run without leak
    res_wo = None
    run_without = args.run_without_leak == "true"
    if run_without:
        vars_wo = [v for v in all_vars_aug if v != leak_col]
        cat_wo = [v for v in cat_aug if v != leak_col]
        cont_wo = [v for v in cont_aug if v != leak_col]

        X_complete_wo = df_aug[vars_wo].copy()
        X_missing_wo = df_missing_aug[vars_wo].copy()

        run_wo_dir = out_root / "without_leak"
        res_wo = _run_sni_once(
            X_complete=X_complete_wo,
            X_missing=X_missing_wo,
            categorical_vars=cat_wo,
            continuous_vars=cont_wo,
            seed=int(args.seed),
            use_gpu=use_gpu,
            variant=str(args.variant),
            epochs=int(args.epochs),
            max_iters=int(args.max_iters),
            emb_dim=int(args.emb_dim),
            num_heads=int(args.num_heads),
            hidden_dims=hidden_dims,
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            outdir=run_wo_dir,
            tau_edges=float(args.tau_edges),
            dpi=int(args.dpi),
        )

    # Build comparison table (focus on audited target)
    def _extract_target_metric(per_feature: pd.DataFrame, target: str) -> Dict[str, float]:
        if per_feature is None or per_feature.empty:
            return {}
        df_t = per_feature[per_feature["feature"] == target]
        if df_t.empty:
            return {}
        row = df_t.iloc[0].to_dict()
        keep = {}
        for k in ["Accuracy", "Macro-F1", "Cohen_kappa", "RMSE", "NRMSE", "MAE", "R2", "Spearman"]:
            if k in row and pd.notna(row[k]):
                keep[k] = float(row[k])
        keep["n_eval"] = int(row.get("n_eval", 0))
        keep["type"] = str(row.get("type", ""))
        return keep

    comp_rows = []
    # with leak
    tgt_with = _extract_target_metric(res_with["per_feature"], audit_target)
    D_with = res_with["dependency_matrix"]
    leak_w = float(D_with.loc[audit_target, leak_col]) if (audit_target in D_with.index and leak_col in D_with.columns) else float("nan")
    comp_rows.append({"setting": "with_leak", "leak_edge_weight": leak_w, **tgt_with})

    if res_wo is not None:
        tgt_wo = _extract_target_metric(res_wo["per_feature"], audit_target)
        comp_rows.append({"setting": "without_leak", "leak_edge_weight": float("nan"), **tgt_wo})

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_root / "audit_comparison.csv", index=False)

    # Write a small markdown report
    lines = []
    lines.append(f"# Ext1-Exp1 Audit Report: Leakage/Proxy Injection\n")
    lines.append(f"- Dataset: **{args.dataset_name}**")
    lines.append(f"- Complete CSV: `{args.input_complete}`")
    lines.append(f"- Missingness: **{mechanism}** @ rate={args.missing_rate:.2f}, seed={args.seed}")
    if mechanism == "MAR":
        lines.append(f"- MAR drivers (strict): {mar_drivers}")
    lines.append(f"- Audited target: **{audit_target}**")
    proxy_type = "continuous" if leak_is_continuous else "categorical"
    noise_note = f", noise σ={leak_noise_std:.3f}" if (leak_is_continuous and leak_noise_std > 0) else ""
    lines.append(f"- Injected proxy: **{leak_col} = {leak_source}** ({proxy_type}{noise_note}, kept always observed)")
    lines.append("")

    lines.append("## Key result: does D flag the proxy as dominant?\n")
    if np.isfinite(leak_w):
        lines.append(f"For target **{audit_target}**, the dependency weight on **{leak_col}** is **{leak_w:.3f}** (row-normalized).")
        lines.append("A large value (e.g., >0.6) indicates the imputation of the target is dominated by this single source.")
    else:
        lines.append("(Could not compute leak edge weight; check that audit_target and leak_col are included in the variable list.)")

    # Top-5 sources table for the audited target
    topk_with = res_with["topk"]
    row_t = topk_with[topk_with["target"] == audit_target]
    if not row_t.empty:
        r = row_t.iloc[0].to_dict()
        lines.append("\nTop-5 sources for the audited target (with leak):")
        for i in range(1, 6):
            s = r.get(f"top{i}_src")
            w = r.get(f"top{i}_w")
            if pd.isna(s) or pd.isna(w):
                continue
            lines.append(f"- top{i}: {s} (w={float(w):.3f})")

    if res_wo is not None:
        topk_wo = res_wo["topk"]
        row_t2 = topk_wo[topk_wo["target"] == audit_target]
        if not row_t2.empty:
            r2 = row_t2.iloc[0].to_dict()
            lines.append("\nTop-5 sources for the audited target (without leak):")
            for i in range(1, 6):
                s = r2.get(f"top{i}_src")
                w = r2.get(f"top{i}_w")
                if pd.isna(s) or pd.isna(w):
                    continue
                lines.append(f"- top{i}: {s} (w={float(w):.3f})")

    lines.append("\n## Metric comparison on audited target (missing positions only)\n")
    lines.append(comp_df.to_markdown(index=False))

    lines.append("\n## Files\n")
    lines.append(f"- With leak: `{(out_root / 'with_leak').as_posix()}`")
    if res_wo is not None:
        lines.append(f"- Without leak: `{(out_root / 'without_leak').as_posix()}`")
    lines.append(f"- Comparison CSV: `{(out_root / 'audit_comparison.csv').as_posix()}`")

    (out_root / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] Audit story written to: {out_root}")


if __name__ == "__main__":
    main()
