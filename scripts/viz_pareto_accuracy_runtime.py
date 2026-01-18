#!/usr/bin/env python3
"""
viz_pareto_accuracy_runtime.py

A compact, main-text-friendly figure showing the trade-off between
imputation accuracy and computational cost.

Panels:
  A: cont_NRMSE vs runtime (lower-left is better)
  B: cat_Macro-F1 vs runtime (upper-left is better)

Input:
  merged_summary_agg.csv (or concatenated summary_agg.csv files)
  Must contain: dataset, mechanism, rate, algo, runtime_sec_mean,
                cont_NRMSE_mean, cat_Macro-F1_mean (may be NaN for all-cont datasets).

Example:
python scripts/viz_pareto_accuracy_runtime.py \
  --csv figs/merged_summary_agg.csv \
  --rate 30per \
  --mechanisms MCAR MAR \
  --outdir figs/viz_pareto_accuracy_runtime \
  --outfile Fig_Pareto_AccuracyRuntime

Notes:
  - We aggregate each algorithm over all (dataset, mechanism) scenarios at the specified rate.
  - Error bars indicate std across scenarios (not seed std).
  - Runtime axis is log-scaled by default.
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS: Dict[str, str] = {
    "SNI": "#D62728",
    "MissForest": "#2CA02C",
    "MIWAE": "#1F77B4",
    "GAIN": "#9467BD",
    "KNN": "#8C564B",
    "MICE": "#E377C2",
    "MeanMode": "#7F7F7F",
}

ALGO_ORDER = ["SNI", "MissForest", "KNN", "MICE", "MeanMode", "MIWAE", "GAIN"]


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
    # dedupe by keys
    key = ["dataset", "mechanism", "rate", "algo"]
    metric_cols = [c for c in df.columns if c.endswith("_mean")]
    # Add runtime col if not already present
    if "runtime_sec_mean" in df.columns and "runtime_sec_mean" not in metric_cols:
        metric_cols.append("runtime_sec_mean")

    # Remove duplicates and ensure columns exist
    metric_cols = list(dict.fromkeys(metric_cols))  # preserve order, remove dups
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Build unique keep list
    keep = key + ["rate_float"]
    for c in metric_cols:
        if c not in keep:
            keep.append(c)

    df = df[keep].copy()

    # Convert metric columns to numeric, coercing errors to NaN
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rate_float"] = pd.to_numeric(df["rate_float"], errors="coerce")

    # Filter metric_cols to only those that have at least some non-NaN values
    numeric_metric_cols = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

    # Build aggregation dict, avoiding duplicates
    agg_dict = {c: "mean" for c in numeric_metric_cols}
    if "rate_float" not in agg_dict:
        agg_dict["rate_float"] = "mean"

    df = df.groupby(key, as_index=False).agg(agg_dict)
    return df


def summarize_algo(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """
    Aggregate each algo over scenarios: mean/std of runtime and y_col.
    """
    sub = df.dropna(subset=[y_col, "runtime_sec_mean"]).copy()
    g = sub.groupby("algo", as_index=False).agg(
        y_mean=(y_col, "mean"),
        y_std=(y_col, "std"),
        rt_mean=("runtime_sec_mean", "mean"),
        rt_std=("runtime_sec_mean", "std"),
        n=("runtime_sec_mean", "count"),
    )
    return g


def pareto_front(points: List[Tuple[float, float]], maximize_y: bool) -> List[int]:
    """
    Return indices of non-dominated points for (x=runtime, y=metric).
    For cont_NRMSE: minimize y -> maximize_y=False but we invert y in dominance check.
    For Macro-F1: maximize y -> maximize_y=True.
    """
    idxs = list(range(len(points)))
    keep = []
    for i,(xi,yi) in enumerate(points):
        dominated = False
        for j,(xj,yj) in enumerate(points):
            if j==i:
                continue
            # runtime: smaller is better
            better_or_eq_x = xj <= xi
            if maximize_y:
                better_or_eq_y = yj >= yi
                strictly_better = (xj < xi) or (yj > yi)
            else:
                # minimize y
                better_or_eq_y = yj <= yi
                strictly_better = (xj < xi) or (yj < yi)
            if better_or_eq_x and better_or_eq_y and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def plot_panel(ax, g: pd.DataFrame, ylabel: str, maximize_y: bool, logx: bool = True, highlight: List[str] = ["SNI","MissForest"]):
    # order
    g = g.copy()
    g["algo"] = pd.Categorical(g["algo"], categories=ALGO_ORDER, ordered=True)
    g = g.sort_values("algo")

    # Pareto
    pts = list(zip(g["rt_mean"].to_list(), g["y_mean"].to_list()))
    pareto_idx = set(pareto_front(pts, maximize_y=maximize_y))

    handles = []
    labels = []

    for i,row in enumerate(g.itertuples(index=False)):
        algo = row.algo
        x = row.rt_mean
        y = row.y_mean
        xerr = row.rt_std
        yerr = row.y_std

        color = COLORS.get(algo, None)
        is_pareto = i in pareto_idx
        is_hi = algo in highlight

        # For log scale, use asymmetric xerr to avoid extending to zero/negative
        # Limit left error bar so that x - xerr_left >= x * 0.1 (minimum 10% of x)
        if logx and np.isfinite(xerr):
            xerr_left = min(xerr, x * 0.9)  # don't go below 10% of x
            xerr_right = xerr
            xerr_asym = [[xerr_left], [xerr_right]]
        else:
            xerr_asym = xerr if np.isfinite(xerr) else None

        eb = ax.errorbar(
            x, y,
            xerr=xerr_asym,
            yerr=yerr if np.isfinite(yerr) else None,
            fmt="o",
            markersize=7 if is_hi else 6,
            linewidth=1.2,
            capsize=2.5,
            color=color,
            ecolor=color,
            alpha=0.95 if is_hi else 0.75,
            zorder=3 if is_hi else 2,
            label=algo,
        )

        # Collect for legend (only first occurrence)
        if algo not in labels:
            handles.append(eb)
            labels.append(algo)

        if is_pareto:
            # add a ring to indicate Pareto
            ax.scatter([x],[y], s=120, facecolors="none", edgecolors=color, linewidths=1.6, zorder=4)

    if logx:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlabel("Runtime (sec, log scale)" if logx else "Runtime (sec)")
    ax.set_ylabel(ylabel)

    return handles, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="merged_summary_agg.csv or summary_agg.csv (repeatable).")
    ap.add_argument("--rate", default="30per", help="Rate to filter (default: 30per).")
    ap.add_argument("--mechanisms", nargs="*", default=["MCAR","MAR"], help="Mechanisms to include (default: MCAR MAR).")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--outfile", default="Fig_Pareto_AccuracyRuntime")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--logx", action="store_true", default=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = load_csvs(args.csv)
    df = df[(df["rate"]==args.rate) & (df["mechanism"].isin(args.mechanisms))].copy()

    # Panel A: cont_NRMSE
    gA = summarize_algo(df, "cont_NRMSE_mean")

    # Panel B: cat_Macro-F1
    gB = summarize_algo(df, "cat_Macro-F1_mean")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    h1, l1 = plot_panel(axes[0], gA, ylabel="NRMSE (↓)", maximize_y=False, logx=args.logx)
    axes[0].set_title("A  Accuracy–Runtime (Continuous)")

    if len(gB) > 0:
        h2, l2 = plot_panel(axes[1], gB, ylabel="Macro-F1 (↑)", maximize_y=True, logx=args.logx)
        axes[1].set_title("B  Accuracy–Runtime (Categorical)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No categorical metrics available", ha="center", va="center")
        h2, l2 = [], []

    # Combine handles/labels, remove duplicates while preserving order
    all_labels = []
    all_handles = []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in all_labels:
            all_labels.append(l)
            all_handles.append(h)

    # Add legend at bottom
    fig.legend(all_handles, all_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(all_labels), 7), frameon=False, columnspacing=1.5, handletextpad=0.5)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_pdf = os.path.join(args.outdir, f"{args.outfile}.pdf")
    out_png = os.path.join(args.outdir, f"{args.outfile}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")


if __name__ == "__main__":
    main()