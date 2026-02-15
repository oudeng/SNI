#!/usr/bin/env python3
"""
viz_mcar_mar_robustness.py

Compact figure for "mechanism robustness" (MCAR vs MAR) at a fixed missing rate.
We quantify robustness as the absolute performance shift between MCAR and MAR.

Panels:
  A: |Δ NRMSE| between MCAR and MAR (lower is more robust)
  B: |Δ Macro-F1| between MCAR and MAR (lower is more robust; only datasets with categorical vars)

Input:
  merged_summary_agg.csv (or concatenated summary_agg.csv files)
  Must contain: dataset, mechanism, rate, algo, cont_NRMSE_mean, cat_Macro-F1_mean

Example:
python scripts/viz_mcar_mar_robustness.py \
  --csv figs/merged_summary_agg.csv \
  --rate 30per \
  --outdir figs/viz_mcar_mar_robustness \
  --outfile Fig_MCARvsMAR_Robustness

This is main-text friendly because it compresses "mechanism-agnostic robustness"
into a single plot, rather than showing many per-dataset charts.
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.reporting_constants import ALGO_COLORS, ALGO_ORDER_MAIN


# Overlay centralized palette with legacy entries
COLORS: Dict[str, str] = {**{
    "SNI": "#D62728",
    "MissForest": "#2CA02C",
    "MIWAE": "#1F77B4",
    "GAIN": "#9467BD",
    "KNN": "#8C564B",
    "MICE": "#E377C2",
    "MeanMode": "#7F7F7F",
}, **ALGO_COLORS}

ALGO_ORDER = ALGO_ORDER_MAIN


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_rate_to_float(rate: str) -> float:
    s = str(rate).strip()
    m = re.match(r"^(\d+)\s*per$", s)
    if m:
        return float(m.group(1)) / 100.0
    return float(s)


def attach_rate_float(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rate_float" in df.columns:
        if df["rate_float"].isna().any() and "rate_float_mean" in df.columns:
            df["rate_float"] = df["rate_float"].fillna(df["rate_float_mean"])
    elif "rate_float_mean" in df.columns:
        df["rate_float"] = df["rate_float_mean"]
    else:
        df["rate_float"] = df["rate"].apply(parse_rate_to_float)
    df["rate_float"] = df["rate_float"].astype(float)
    return df


def load_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["__source__"] = os.path.basename(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = attach_rate_float(df)
    key = ["dataset", "mechanism", "rate", "algo"]
    metric_cols = ["cont_NRMSE_mean", "cat_Macro-F1_mean"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    keep = key + ["rate_float"] + metric_cols
    df = df[keep].copy()
    df = df.groupby(key, as_index=False).agg({**{c: "mean" for c in metric_cols}, "rate_float": "mean"})
    return df


def compute_abs_delta(df: pd.DataFrame, metric: str, rate: str) -> pd.DataFrame:
    sub = df[(df["rate"]==rate) & (df["mechanism"].isin(["MCAR","MAR"]))].copy()
    # pivot -> columns MCAR/MAR
    piv = sub.pivot_table(index=["dataset","algo"], columns="mechanism", values=metric, aggfunc="mean")
    if "MCAR" not in piv.columns or "MAR" not in piv.columns:
        return pd.DataFrame(columns=["dataset","algo","abs_delta"])
    out = (piv["MAR"] - piv["MCAR"]).abs().reset_index().rename(columns={0:"abs_delta"})
    out = out.rename(columns={metric:"abs_delta"}) if metric in out.columns else out
    out = out.rename(columns={"MAR":"MAR", "MCAR":"MCAR"})
    out = out.dropna(subset=["abs_delta"])
    return out


def plot_box(ax, deltas: pd.DataFrame, metric_label: str):
    if deltas.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {metric_label}", ha="center", va="center")
        return

    deltas = deltas.copy()
    deltas["algo"] = pd.Categorical(deltas["algo"], categories=ALGO_ORDER, ordered=True)
    deltas = deltas.sort_values("algo")

    algos = [a for a in ALGO_ORDER if a in deltas["algo"].unique().tolist()]
    data = [deltas.loc[deltas["algo"]==a, "abs_delta"].to_numpy() for a in algos]

    bp = ax.boxplot(data, tick_labels=algos, patch_artist=True, showfliers=False)

    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(COLORS.get(algo, "#CCCCCC"))
        patch.set_alpha(0.55)
        patch.set_linewidth(0.8)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--rate", default="30per")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--outfile", default="Fig_MCARvsMAR_Robustness")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--include-algos", type=str, default=None, help="Comma-separated algo filter.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = load_csvs(args.csv)

    # --- algo filter ---
    if args.include_algos:
        _algo_list = [a.strip() for a in args.include_algos.split(",") if a.strip()]
        df = df[df["algo"].isin(_algo_list)].copy()

    d_nrmse = compute_abs_delta(df, "cont_NRMSE_mean", args.rate)
    d_f1    = compute_abs_delta(df, "cat_Macro-F1_mean", args.rate)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    plot_box(axes[0], d_nrmse, r"$|\Delta|$ NRMSE (MCAR vs MAR, ↓)")
    axes[0].set_title("A  Mechanism robustness (Continuous)")

    plot_box(axes[1], d_f1, r"$|\Delta|$ Macro-F1 (MCAR vs MAR, ↓)")
    axes[1].set_title("B  Mechanism robustness (Categorical)")

    plt.tight_layout()
    out_pdf = os.path.join(args.outdir, f"{args.outfile}.pdf")
    out_png = os.path.join(args.outdir, f"{args.outfile}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")


if __name__ == "__main__":
    main()