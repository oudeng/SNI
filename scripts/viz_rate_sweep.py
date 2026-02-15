#!/usr/bin/env python3
"""Make a single publication-ready missing-rate sweep figure.

Typical usage (from repo root):
  python scripts/viz_rate_sweep.py \
    --summary-csv results_sni_rate_sweep/_summary/summary_agg.csv \
    --summary-csv results_baselines_rate_sweep_missforest/_summary/summary_agg.csv \
    --outdir figs \
    --mechanism MAR \
    --metric cont_NRMSE_mean \
    --algos SNI,MissForest \
    --aggregate mean

It merges multiple summary_agg.csv files, deduplicates by (dataset, mechanism, rate, algo)
(first occurrence wins), and plots metric vs missing rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reporting_constants import ALGO_COLORS, ALGO_ORDER_MAIN


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rf(x: str) -> float:
    x = str(x)
    if x.endswith("per"):
        try:
            return float(x[:-3]) / 100.0
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def load_merge(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "algo" not in df.columns:
            if "variant" in df.columns:
                df["algo"] = df["variant"].astype(str)
            elif "method" in df.columns:
                df["algo"] = df["method"].astype(str)
        for c in ["dataset", "mechanism", "rate", "algo"]:
            df[c] = df[c].astype(str)
        if "rate_float" not in df.columns:
            df["rate_float"] = df["rate"].map(_rf)
        df["__source__"] = p.as_posix()
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["dataset", "mechanism", "rate", "algo"], keep="first").copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", action="append", required=True, help="summary_agg.csv path (repeatable)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mechanism", default="MAR", help="MCAR/MAR/MNAR")
    ap.add_argument("--metric", default="cont_NRMSE_mean", help="e.g., cont_NRMSE_mean, cont_R2_mean")
    ap.add_argument("--algos", default="SNI,MissForest", help="Comma-separated algos to plot")
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean", help="Aggregate across datasets")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--outfile", default="rate_sweep_mar_contnrmse", help="Output basename (no extension)")
    ap.add_argument("--include-algos", type=str, default=None, help="Comma-separated algo filter.")
    args = ap.parse_args()

    paths = [Path(p) for p in args.summary_csv]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    df = load_merge(paths)

    # --- algo filter (--include-algos overrides --algos) ---
    if args.include_algos:
        _algo_list = [a.strip() for a in args.include_algos.split(",") if a.strip()]
        df = df[df["algo"].isin(_algo_list)].copy()
        args.algos = ",".join(_algo_list)

    df = df[df["mechanism"].astype(str) == str(args.mechanism)].copy()
    df = df.dropna(subset=["rate_float", args.metric])

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    df = df[df["algo"].isin(algos)].copy()
    if df.empty:
        raise ValueError("No rows after filtering. Check mechanism/metric/algos.")

    # aggregate over datasets for each algo x rate
    if args.aggregate == "mean":
        agg = df.groupby(["algo", "rate_float"], as_index=False)[args.metric].mean()
    else:
        agg = df.groupby(["algo", "rate_float"], as_index=False)[args.metric].median()

    fig = plt.figure(figsize=(6.8, 4.0))
    ax = fig.add_subplot(111)
    for algo in algos:
        sub = agg[agg["algo"] == algo].sort_values("rate_float")
        ax.plot(sub["rate_float"].values, sub[args.metric].values, marker="o", label=algo)

    ax.set_xlabel("Missing rate")
    ax.set_ylabel(args.metric.replace("_mean", ""))
    ax.set_title(f"{args.mechanism}: {args.metric.replace('_mean','')} vs missing rate (aggregate={args.aggregate})")
    ax.legend(loc="best", fontsize=9)

    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    outbase = outdir / args.outfile
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] wrote {outbase}.png/.pdf")


if __name__ == "__main__":
    main()
