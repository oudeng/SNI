#!/usr/bin/env python3
"""
viz_compact_rate_sweep.py

Generate a compact missing-rate sensitivity plot (single figure, 2 panels):
  - Left: MCAR
  - Right: MAR

Designed for paper main text: concise, dataset-mean curves with error bars
(std across datasets), for a small set of key methods (default: SNI vs MissForest).

Input:
  One or more CSV files in the "summary_agg.csv" schema, OR a pre-merged CSV.
  Must contain columns:
    dataset, mechanism, rate, algo, <metric>

Example:
python scripts/viz_compact_rate_sweep.py \
  --csv figs/merged_summary_agg.csv \
  --csv results_sni_rate_sweep/_summary/summary_agg.csv \
  --csv results_baselines_rate_sweep_missforest/_summary/summary_agg.csv \
  --metric cont_NRMSE_mean \
  --algos SNI MissForest \
  --rates 10per 30per 50per \
  --outdir figs/viz_compact_rate_sweep \
  --outfile Fig_RateSweep_NRMSE

Outputs:
  <outdir>/<outfile>.pdf  (vector)
  <outdir>/<outfile>.png  (dpi>=600)

Notes:
  - If some rates are missing (e.g., only 30per exists), the script will still plot
    whatever is available and print a warning.
  - Error bars are std across datasets (not seed std), giving a quick sense of
    across-domain stability.
"""

from __future__ import annotations
import argparse
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Consistent color mapping (match your viz_merged_summary_agg.py if desired)
COLORS: Dict[str, str] = {
    "SNI": "#D62728",
    "MissForest": "#2CA02C",
    "MIWAE": "#1F77B4",
    "GAIN": "#9467BD",
    "KNN": "#8C564B",
    "MICE": "#E377C2",
    "MeanMode": "#7F7F7F",
}

DEFAULT_ALGOS = ["SNI", "MissForest"]


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
        # prefer explicit float, fall back to rate_float_mean
        if df["rate_float"].isna().any() and "rate_float_mean" in df.columns:
            df["rate_float"] = df["rate_float"].fillna(df["rate_float_mean"])
    elif "rate_float_mean" in df.columns:
        df["rate_float"] = df["rate_float_mean"]
    else:
        df["rate_float"] = df["rate"].apply(parse_rate_to_float)
    df["rate_float"] = df["rate_float"].astype(float)
    return df


def load_and_merge_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["__source__"] = os.path.basename(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = attach_rate_float(df)

    # Deduplicate by (dataset, mechanism, rate, algo) if multiple sources exist
    # Use mean for metrics (safe if duplicates are truly identical)
    key = ["dataset", "mechanism", "rate", "algo"]
    metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    # Add runtime cols if not already present
    for rt_col in ["runtime_sec_mean", "runtime_sec_std"]:
        if rt_col in df.columns and rt_col not in metric_cols:
            metric_cols.append(rt_col)

    # Remove duplicates and ensure columns exist
    metric_cols = list(dict.fromkeys(metric_cols))  # preserve order, remove dups
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Build unique keep_cols list
    keep_cols = key + ["rate_float"]
    for c in metric_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols].copy()

    # Convert metric columns to numeric, coercing errors to NaN
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rate_float"] = pd.to_numeric(df["rate_float"], errors="coerce")

    # Filter metric_cols to only those that are numeric and have at least some non-NaN values
    numeric_metric_cols = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

    # Build aggregation dict, avoiding duplicates (rate_float might already be in metric_cols)
    agg_dict = {c: "mean" for c in numeric_metric_cols}
    if "rate_float" not in agg_dict:
        agg_dict["rate_float"] = "mean"

    df = df.groupby(key, as_index=False).agg(agg_dict)
    return df


def aggregate_dataset_mean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For each (mechanism, algo, rate_float), compute mean/std across datasets.
    """
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")

    g = df.groupby(["mechanism", "algo", "rate_float"], as_index=False)[metric].agg(["mean", "std"]).reset_index()
    g = g.rename(columns={"mean": "y_mean", "std": "y_std"})
    return g


def _ylabel(metric: str) -> str:
    if metric.startswith("cont_NRMSE"):
        return "NRMSE (↓)"
    if metric.startswith("cont_R2"):
        return r"$R^2$ (↑)"
    if metric.startswith("cont_Spearman"):
        return r"Spearman $\rho$ (↑)"
    if metric.startswith("cat_Macro-F1"):
        return "Macro-F1 (↑)"
    if metric.startswith("cat_Accuracy"):
        return "Accuracy (↑)"
    return metric


def plot_rate_sweep(df_agg: pd.DataFrame, metric: str, mechanisms: List[str], algos: List[str],
                    out_pdf: str, out_png: str, dpi: int = 600, title: str | None = None) -> None:

    fig, axes = plt.subplots(1, len(mechanisms), figsize=(7.6, 3.2), sharey=True)
    if len(mechanisms) == 1:
        axes = [axes]

    for ax, mech in zip(axes, mechanisms):
        sub = df_agg[df_agg["mechanism"] == mech].copy()
        if sub.empty:
            ax.set_title(f"{mech} (no data)")
            ax.axis("off")
            continue

        for algo in algos:
            s = sub[sub["algo"] == algo].sort_values("rate_float")
            if s.empty:
                continue
            x = s["rate_float"].to_numpy() * 100
            y = s["y_mean"].to_numpy()
            yerr = s["y_std"].to_numpy()

            ax.errorbar(
                x, y, yerr=yerr,
                marker="o",
                linewidth=1.6,
                capsize=3,
                label=algo,
                color=COLORS.get(algo, None),
            )

        ax.set_title(mech)
        ax.set_xlabel("Missing rate (%)")
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].set_ylabel(_ylabel(metric))

    # Legend once
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=min(len(labels), 4), frameon=False, columnspacing=1.0, handletextpad=0.4)

    if title:
        fig.suptitle(title, y=1.08, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="Path to merged_summary_agg.csv or one/more summary_agg.csv files (repeatable).")
    ap.add_argument("--metric", default="cont_NRMSE_mean",
                    help="Metric column to plot (e.g., cont_NRMSE_mean, cont_R2_mean, cat_Macro-F1_mean).")
    ap.add_argument("--algos", nargs="*", default=DEFAULT_ALGOS,
                    help="Algorithms to include (default: SNI MissForest).")
    ap.add_argument("--mechanisms", nargs="*", default=["MCAR", "MAR"],
                    help="Mechanisms/panels (default: MCAR MAR).")
    ap.add_argument("--rates", nargs="*", default=["10per", "30per", "50per"],
                    help="Rates to include, e.g., 10per 30per 50per.")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--outfile", default="Fig_RateSweep")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = load_and_merge_csvs(args.csv)
    df = df[df["mechanism"].isin(args.mechanisms)].copy()
    df = df[df["algo"].isin(args.algos)].copy()
    df = df[df["rate"].isin(args.rates)].copy()

    # Warn if rate coverage is incomplete
    avail_rates = sorted(df["rate"].unique().tolist())
    wanted_rates = list(args.rates)
    missing_rates = [r for r in wanted_rates if r not in avail_rates]
    if missing_rates:
        print(f"[WARN] Missing requested rates in data: {missing_rates}. Available: {avail_rates}")

    df_agg = aggregate_dataset_mean(df, args.metric)

    out_pdf = os.path.join(args.outdir, f"{args.outfile}.pdf")
    out_png = os.path.join(args.outdir, f"{args.outfile}.png")

    title = None
    plot_rate_sweep(
        df_agg,
        metric=args.metric,
        mechanisms=args.mechanisms,
        algos=args.algos,
        out_pdf=out_pdf,
        out_png=out_png,
        dpi=args.dpi,
        title=title,
    )
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")


if __name__ == "__main__":
    main()