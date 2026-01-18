#!/usr/bin/env python3
"""
viz_rate_sweep_old1.py

Merge:
  - SNI main @30% (MCAR/MAR)
  - SNI sweep @10/50%
  - MissForest main @30%
  - MissForest sweep @10/50%

and plot metric vs missingness rate (PDF + 600dpi PNG).

Usage:
python scripts/viz_rate_sweep_old1.py \
  --sni-main results_sni_main/_summary/summary_agg.csv \
  --sni-sweep results_sni_rate_sweep/_summary/summary_agg.csv \
  --mf-main  results_baselines_main/_summary/summary_agg.csv \
  --mf-sweep results_baselines_rate_sweep_missforest/_summary/summary_agg.csv \
  --mechanism MAR \
  --metric cont_NRMSE \
  --aggregate dataset_mean \
  --outdir figs/viz_rate_sweep_old1


"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_rate_to_float(rate: str) -> float:
    m = re.match(r"^\s*(\d+)\s*per\s*$", str(rate))
    if m: return float(m.group(1))/100.0
    return float(rate)

def get_rate_float(df: pd.DataFrame) -> pd.Series:
    if "rate_float_mean" in df.columns: return df["rate_float_mean"].astype(float)
    if "rate_float" in df.columns: return df["rate_float"].astype(float)
    return df["rate"].apply(parse_rate_to_float).astype(float)

def metric_cols(metric: str):
    return f"{metric}_mean", f"{metric}_std"

def load_algo(path: str, algo: str):
    df = pd.read_csv(path)
    df = df[df["algo"]==algo].copy()
    df["rate_float"] = get_rate_float(df)
    return df

def aggregate_dataset_mean(df: pd.DataFrame, mean_col: str):
    g = df.groupby(["algo","rate_float"], as_index=False)[mean_col].agg(["mean","std"]).reset_index()
    g = g.rename(columns={"mean":"y_mean","std":"y_std"})
    return g

def plot_curve(df_plot, ylabel, title, out_pdf, out_png, dpi=600):
    plt.figure()
    for algo, sub in df_plot.groupby("algo"):
        sub = sub.sort_values("rate_float")
        x = sub["rate_float"].to_numpy()*100
        y = sub["y_mean"].to_numpy()
        yerr = sub["y_std"].to_numpy()
        plt.errorbar(x,y,yerr=yerr,marker="o",capsize=3,label=algo)
    plt.xlabel("Missing rate (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sni-main", required=True)
    ap.add_argument("--sni-sweep", required=True)
    ap.add_argument("--mf-main", required=True)
    ap.add_argument("--mf-sweep", required=True)
    ap.add_argument("--mechanism", choices=["MCAR","MAR"], default="MAR")
    ap.add_argument("--metric", default="cont_NRMSE",
                    choices=["cont_NRMSE","cont_R2","cont_Spearman","cat_Macro-F1","cat_Cohen_kappa"])
    ap.add_argument("--aggregate", choices=["dataset_mean","per_dataset"], default="dataset_mean")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    mean_col, std_col = metric_cols(args.metric)

    sni = pd.concat([
        load_algo(args.sni_main, "SNI"),
        load_algo(args.sni_sweep, "SNI")
    ], ignore_index=True)
    mf = pd.concat([
        load_algo(args.mf_main, "MissForest"),
        load_algo(args.mf_sweep, "MissForest")
    ], ignore_index=True)

    sni = sni[sni["mechanism"]==args.mechanism].dropna(subset=[mean_col])
    mf  = mf [mf ["mechanism"]==args.mechanism].dropna(subset=[mean_col])

    ds_common = sorted(set(sni["dataset"]).intersection(set(mf["dataset"])))
    df = pd.concat([sni[sni["dataset"].isin(ds_common)], mf[mf["dataset"].isin(ds_common)]], ignore_index=True)

    ensure_dir(args.outdir)

    slug = args.metric.lower().replace("-","").replace("_","")
    out_pdf = os.path.join(args.outdir, f"rate_sweep_{args.mechanism.lower()}_{slug}.pdf")
    out_png = os.path.join(args.outdir, f"rate_sweep_{args.mechanism.lower()}_{slug}.png")

    ylabel = args.metric
    if args.metric=="cont_NRMSE": ylabel="NRMSE (↓)"
    if args.metric=="cat_Macro-F1": ylabel="Macro-F1 (↑)"

    if args.aggregate=="dataset_mean":
        agg = aggregate_dataset_mean(df, mean_col)
        title = f"Rate sweep ({args.mechanism}) — {ylabel} (avg over {len(ds_common)} datasets)"
        plot_curve(agg, ylabel, title, out_pdf, out_png, dpi=args.dpi)
    else:
        for ds in ds_common:
            sub = df[df["dataset"]==ds]
            agg = aggregate_dataset_mean(sub, mean_col)
            plot_curve(agg, ylabel, f"{ds} — Rate sweep ({args.mechanism})", 
                       os.path.join(args.outdir, f"rate_sweep_{ds}_{args.mechanism.lower()}_{slug}.pdf"),
                       os.path.join(args.outdir, f"rate_sweep_{ds}_{args.mechanism.lower()}_{slug}.png"),
                       dpi=args.dpi)

if __name__=="__main__":
    main()