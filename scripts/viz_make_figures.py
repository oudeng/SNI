#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reporting_constants import ALGO_COLORS, ALGO_ORDER_MAIN
from scripts.profile_utils import load_profile, filter_dataframe

try:
    import seaborn as sns  # optional
except Exception:
    sns = None


# --------------------------- helpers --------------------------- #

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in str(s))


def _save_fig(fig: plt.Figure, outbase: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    """Pick *_mean columns (continuous/categorical + runtime)."""
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    keep: List[str] = []
    for c in mean_cols:
        base = c[:-5]
        if base.startswith("cont_") or base.startswith("cat_") or base in {"runtime_sec"}:
            keep.append(c)
    return keep


def _metric_direction(metric_base: str) -> str:
    if metric_base.endswith(("NRMSE", "MAE", "RMSE")):
        return "min"
    if metric_base.endswith(("R2", "Spearman", "Accuracy", "Macro-F1", "Kappa", "Cohen_kappa")):
        return "max"
    if metric_base.endswith("MB"):
        return "absmin"
    if metric_base == "runtime_sec":
        return "min"
    return "min"


def _best_idx(values: np.ndarray, direction: str) -> int:
    x = values.astype(float)
    if direction == "min":
        return int(np.nanargmin(x))
    if direction == "max":
        return int(np.nanargmax(x))
    return int(np.nanargmin(np.abs(x)))


def _read_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source__"] = path.as_posix()
    # normalize algo column
    if "algo" not in df.columns:
        if "variant" in df.columns:
            df["algo"] = df["variant"].astype(str)
        elif "method" in df.columns:
            df["algo"] = df["method"].astype(str)
    # normalize rate_float if missing
    if "rate_float" not in df.columns and "rate" in df.columns:
        # common encoding: '30per' -> 0.3
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
        df["rate_float"] = df["rate"].map(_rf)
    return df


def load_and_merge(paths: List[Path], dedup: bool = True) -> pd.DataFrame:
    """Load many summary_agg.csv files and merge them.

    Dedup key: (dataset, mechanism, rate, algo). First occurrence wins.
    """
    dfs = [_read_one_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure required keys exist if present in any file
    for col in ["dataset", "mechanism", "rate", "algo"]:
        if col not in df.columns:
            raise ValueError(f"Merged dataframe missing required column: {col}")

    # coerce key columns to string
    for col in ["dataset", "mechanism", "rate", "algo"]:
        df[col] = df[col].astype(str)

    if dedup:
        key_cols = ["dataset", "mechanism", "rate", "algo"]
        df = df.drop_duplicates(subset=key_cols, keep="first").copy()

    return df


# --------------------------- plots --------------------------- #

def plot_heatmaps(
    df: pd.DataFrame,
    outdir: Path,
    metrics_mean_cols: List[str],
    dpi: int = 600,
    mechanisms: Optional[List[str]] = None,
    rates: Optional[List[str]] = None,
    top_k_algos: int = 12,
) -> None:
    needed = {"dataset", "mechanism", "rate", "algo"}
    if not needed.issubset(df.columns):
        return

    mechs = sorted(df["mechanism"].dropna().astype(str).unique().tolist())
    if mechanisms:
        mechs = [m for m in mechs if m in mechanisms]
    rts = sorted(df["rate"].dropna().astype(str).unique().tolist())
    if rates:
        rts = [r for r in rts if r in rates]

    for mech in mechs:
        for rate in rts:
            sub = df[(df["mechanism"].astype(str) == mech) & (df["rate"].astype(str) == rate)].copy()
            if sub.empty:
                continue

            # choose top-K algos by cont_NRMSE if available else first metric
            if "cont_NRMSE_mean" in sub.columns:
                score = sub.groupby("algo")["cont_NRMSE_mean"].mean().sort_values()
            else:
                score = sub.groupby("algo")[metrics_mean_cols[0]].mean().sort_values()
            algos = score.index.tolist()[:top_k_algos]
            sub = sub[sub["algo"].isin(algos)]

            for mcol in metrics_mean_cols:
                base = mcol[:-5]
                pivot = sub.pivot_table(index="dataset", columns="algo", values=mcol, aggfunc="mean")
                if pivot.shape[0] < 2 or pivot.shape[1] < 2:
                    continue

                fig = plt.figure(figsize=(1.2 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
                ax = fig.add_subplot(111)

                if sns is not None:
                    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3g", cmap="viridis")
                else:
                    im = ax.imshow(pivot.values, aspect="auto")
                    ax.set_xticks(range(pivot.shape[1]))
                    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
                    ax.set_yticks(range(pivot.shape[0]))
                    ax.set_yticklabels(pivot.index.tolist())
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_title(f"{base} ({mech}, {rate})")
                ax.set_xlabel("Algo")
                ax.set_ylabel("Dataset")

                outbase = outdir / f"heatmap_{_sanitize_filename(base)}_{mech}_{rate}"
                _save_fig(fig, outbase, dpi)


def plot_overall_bar(
    df: pd.DataFrame,
    outdir: Path,
    metric_mean_col: str,
    dpi: int = 600,
    top_k: int = 12,
) -> None:
    if "algo" not in df.columns:
        return

    base = metric_mean_col[:-5]
    direction = _metric_direction(base)

    g = df.groupby("algo")[metric_mean_col].mean().sort_values(ascending=(direction == "min"))
    g = g.head(top_k)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    bar_colors = [ALGO_COLORS.get(a, None) for a in g.index.astype(str).tolist()]
    ax.bar(g.index.astype(str).tolist(), g.values.astype(float), color=bar_colors)
    ax.set_ylabel(base)
    ax.set_title(f"Overall mean across tasks: {base}")
    ax.tick_params(axis="x", rotation=30)
    _save_fig(fig, outdir / f"overall_bar_{_sanitize_filename(base)}", dpi)


def plot_taskwise_bars(
    df: pd.DataFrame,
    outdir: Path,
    metric_mean_col: str,
    dpi: int = 600,
    max_tasks: int = 24,
    top_k_algos: int = 8,
) -> None:
    needed = {"dataset", "mechanism", "rate", "algo"}
    if not needed.issubset(df.columns):
        return

    base = metric_mean_col[:-5]
    direction = _metric_direction(base)

    tasks = (
        df[["dataset", "mechanism", "rate"]]
        .drop_duplicates()
        .astype(str)
        .sort_values(["dataset", "mechanism", "rate"])
        .head(max_tasks)
        .to_dict("records")
    )

    for t in tasks:
        sub = df[
            (df["dataset"].astype(str) == t["dataset"])
            & (df["mechanism"].astype(str) == t["mechanism"])
            & (df["rate"].astype(str) == t["rate"])
        ].copy()
        sub = sub.dropna(subset=[metric_mean_col])
        if sub.empty:
            continue

        sub = sub.sort_values(metric_mean_col, ascending=(direction == "min")).head(top_k_algos)

        fig = plt.figure(figsize=(7, 3.5))
        ax = fig.add_subplot(111)
        task_bar_colors = [ALGO_COLORS.get(a, None) for a in sub["algo"].astype(str).tolist()]
        ax.bar(sub["algo"].astype(str).tolist(), sub[metric_mean_col].astype(float).values, color=task_bar_colors)
        ax.set_ylabel(base)
        ax.set_title(f"{t['dataset']}  {t['mechanism']}  {t['rate']}  ({base})")
        ax.tick_params(axis="x", rotation=30)
        outbase = outdir / f"taskbar_{t['dataset']}_{t['mechanism']}_{t['rate']}_{_sanitize_filename(base)}"
        _save_fig(fig, outbase, dpi)


def plot_rate_curves(
    df: pd.DataFrame,
    outdir: Path,
    metric_mean_col: str,
    dpi: int = 600,
    mechanism: str = "MNAR",
    top_k_algos: int = 6,
) -> None:
    needed = {"dataset", "mechanism", "rate_float", "algo"}
    if not needed.issubset(df.columns):
        return

    base = metric_mean_col[:-5]
    direction = _metric_direction(base)

    sub = df[df["mechanism"].astype(str) == str(mechanism)].copy()
    sub = sub.dropna(subset=["rate_float", metric_mean_col])
    if sub.empty:
        return

    for dataset in sorted(sub["dataset"].astype(str).unique().tolist()):
        dsub = sub[sub["dataset"].astype(str) == dataset].copy()
        if dsub["rate_float"].nunique() < 2:
            continue

        max_r = dsub["rate_float"].max()
        hard = dsub[dsub["rate_float"] == max_r].groupby("algo")[metric_mean_col].mean()
        hard = hard.sort_values(ascending=(direction == "min")).head(top_k_algos)
        algos = hard.index.tolist()
        dsub = dsub[dsub["algo"].isin(algos)]

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        # order algos by ALGO_ORDER_MAIN if possible
        ordered_algos = [a for a in ALGO_ORDER_MAIN if a in algos] + [a for a in algos if a not in ALGO_ORDER_MAIN]
        for algo in ordered_algos:
            a = dsub[dsub["algo"] == algo].sort_values("rate_float")
            ax.plot(a["rate_float"].values, a[metric_mean_col].values, marker="o", label=str(algo),
                    color=ALGO_COLORS.get(algo, None))
        ax.set_xlabel("Missing rate")
        ax.set_ylabel(base)
        ax.set_title(f"{dataset} ({mechanism}) - {base} vs missing rate")
        ax.legend(loc="best", fontsize=8)
        outbase = outdir / f"curve_{dataset}_{mechanism}_{_sanitize_filename(base)}"
        _save_fig(fig, outbase, dpi)


def plot_runtime_tradeoff(df: pd.DataFrame, outdir: Path, dpi: int = 600) -> None:
    if not {"algo", "runtime_sec_mean"}.issubset(df.columns):
        return
    if "cont_NRMSE_mean" not in df.columns:
        return

    sub = df.dropna(subset=["runtime_sec_mean", "cont_NRMSE_mean"]).copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.scatter(
        sub["runtime_sec_mean"].astype(float).values,
        sub["cont_NRMSE_mean"].astype(float).values,
        s=20,
        alpha=0.7,
    )
    ax.set_xlabel("Runtime (sec)  [mean over seeds]")
    ax.set_ylabel("cont_NRMSE  [mean over seeds]")
    ax.set_title("Runtime vs accuracy trade-off (each point = task×algo)")
    _save_fig(fig, outdir / "scatter_runtime_vs_nrmse", dpi)


# --------------------------- main --------------------------- #

def _split_csvs(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()

    # Backward compatible
    ap.add_argument("--summary-csv", action="append", default=None, help="Path to summary_agg.csv. Can be repeated.")
    ap.add_argument("--summary-csvs", type=str, default=None, help="Comma-separated list of summary_agg.csv paths.")
    ap.add_argument("--summary-dir", action="append", default=None, help="Directory containing summary_agg.csv. Can be repeated.")

    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--mechanisms", type=str, default=None, help="Comma-separated mechanisms to include (e.g., MCAR,MAR).")
    ap.add_argument("--rates", type=str, default=None, help="Comma-separated rates to include (e.g., 10per,30per).")
    ap.add_argument("--top-k", type=int, default=12, help="Top-K algos for heatmaps/overall bars.")
    ap.add_argument("--max-task-bars", type=int, default=24, help="Max number of per-task bar charts to write.")
    ap.add_argument("--no-dedup", action="store_true", help="Disable deduplication when merging multiple CSVs.")
    ap.add_argument("--write-merged", action="store_true", help="Write merged dataframe to <outdir>/merged_summary_agg.csv.")
    ap.add_argument("--profile", type=str, default=None, help="Profile spec e.g. configs/paper_profiles.yaml:main")
    ap.add_argument("--include-algos", type=str, default=None, help="Comma-separated algo filter.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    csv_paths: List[Path] = []
    if args.summary_csv:
        csv_paths += [Path(p) for p in args.summary_csv]
    csv_paths += [Path(p) for p in _split_csvs(args.summary_csvs)]
    if args.summary_dir:
        for d in args.summary_dir:
            csv_paths.append(Path(d) / "summary_agg.csv")

    csv_paths = [p for p in csv_paths if p is not None]
    if not csv_paths:
        raise FileNotFoundError("Provide at least one --summary-csv/--summary-dir.")

    missing = [p.as_posix() for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing summary_agg.csv files: " + ", ".join(missing))

    df = load_and_merge(csv_paths, dedup=(not args.no_dedup))

    # --- profile filter ---
    profile = load_profile(args.profile)
    if "algo" in df.columns:
        df = filter_dataframe(df, profile)

    # --- algo filter (CLI override) ---
    if args.include_algos:
        _algo_list = [a.strip() for a in args.include_algos.split(",") if a.strip()]
        df = df[df["algo"].isin(_algo_list)].copy()

    if df.empty:
        print("[WARN] No data after filtering. Nothing to plot.")
        return

    if args.write_merged:
        df.to_csv(outdir / "merged_summary_agg.csv", index=False)

    mechs = None
    if args.mechanisms:
        mechs = [x.strip() for x in args.mechanisms.split(",") if x.strip()]
    rts = None
    if args.rates:
        rts = [x.strip() for x in args.rates.split(",") if x.strip()]

    metrics_mean_cols = _pick_metric_cols(df)
    if not metrics_mean_cols:
        print("[WARN] No *_mean metric columns found in input CSV(s). Skipping figure generation.")
        return

    n_figs = 0

    # 1) Heatmaps per (mech, rate)
    try:
        plot_heatmaps(df, outdir, metrics_mean_cols, dpi=args.dpi, mechanisms=mechs, rates=rts, top_k_algos=args.top_k)
        n_figs += 1
    except Exception as e:
        print(f"[WARN] Heatmaps failed: {e}")

    # 2) Overall bars (across tasks)
    for mcol in metrics_mean_cols:
        try:
            plot_overall_bar(df, outdir, mcol, dpi=args.dpi, top_k=args.top_k)
            n_figs += 1
        except Exception as e:
            print(f"[WARN] Overall bar ({mcol}) failed: {e}")

    # 3) Per-task bars (limited)
    for mcol in metrics_mean_cols:
        try:
            plot_taskwise_bars(df, outdir, mcol, dpi=args.dpi, max_tasks=args.max_task_bars, top_k_algos=min(8, args.top_k))
            n_figs += 1
        except Exception as e:
            print(f"[WARN] Task bars ({mcol}) failed: {e}")

    # 4) Rate curves (if rate_float exists)
    if "rate_float" in df.columns:
        for base in ("cont_NRMSE", "cat_Macro-F1", "cat_Accuracy"):
            mcol = f"{base}_mean"
            if mcol in df.columns:
                try:
                    plot_rate_curves(df, outdir, mcol, dpi=args.dpi, mechanism="MNAR", top_k_algos=min(6, args.top_k))
                    n_figs += 1
                except Exception as e:
                    print(f"[WARN] Rate curves ({mcol}) failed: {e}")

    # 5) Runtime trade-off scatter
    try:
        plot_runtime_tradeoff(df, outdir, dpi=args.dpi)
        n_figs += 1
    except Exception as e:
        print(f"[WARN] Runtime scatter failed: {e}")

    print(f"[DONE] wrote figures to {outdir} ({n_figs} plot groups generated)")


if __name__ == "__main__":
    main()
