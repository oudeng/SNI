#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _format_pm(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}\\pm{std:.{digits}f}"


def _metric_direction(metric: str) -> str:
    if metric.endswith(("NRMSE", "MAE")):
        return "min"
    if metric.endswith(("R2", "Spearman", "Accuracy", "Macro-F1", "Kappa")):
        return "max"
    if metric.endswith("MB"):
        return "absmin"
    if metric == "runtime_sec":
        return "min"
    return "min"


def _best_mask(vals: np.ndarray, direction: str) -> np.ndarray:
    x = vals.astype(float)
    if direction == "min":
        best = np.nanmin(x)
        return np.isclose(x, best, atol=1e-12, rtol=0)
    if direction == "max":
        best = np.nanmax(x)
        return np.isclose(x, best, atol=1e-12, rtol=0)
    best = np.nanmin(np.abs(x))
    return np.isclose(np.abs(x), best, atol=1e-12, rtol=0)


def _escape(s: str) -> str:
    # minimal latex escaping for method names etc
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _render_table(df: pd.DataFrame, caption: str, label: str, col_format: Optional[str] = None) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format=col_format or ("l" * len(df.columns)),
    )
    return "\\begin{table}[t]\n\\centering\n" + latex + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"


def _apply_bold_best(df: pd.DataFrame, metric_cols: List[str], directions: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for m in metric_cols:
        direction = directions[m]
        # strip latex formatting for comparison: extract numeric mean from 'mean\pmstd'
        means = []
        for v in out[m].astype(str).tolist():
            if v.strip() in {"--", "nan"}:
                means.append(np.nan)
                continue
            # split at \pm if exists
            parts = v.split("\\pm")
            try:
                means.append(float(parts[0]))
            except Exception:
                means.append(np.nan)
        means_arr = np.array(means, dtype=float)
        mask = _best_mask(means_arr, direction)
        for i, is_best in enumerate(mask.tolist()):
            if is_best and out.loc[i, m] not in {"--"}:
                out.loc[i, m] = f"\\textbf{{{out.loc[i, m]}}}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", type=str, required=True, help="summary_agg.csv (preferred) or summary_all.csv")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--metrics",
        type=str,
        default="cont_NRMSE,cont_MAE,cont_MB,cont_R2,cont_Spearman,cat_Accuracy,cat_Macro-F1,cat_Kappa,runtime_sec",
        help="Comma-separated base metric names.",
    )
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help="Optional pandas query string, e.g., \"mechanism=='MAR' and rate=='30per'\"",
    )
    ap.add_argument(
        "--by-task",
        type=str,
        default="true",
        choices=["true", "false"],
        help="If true and task columns exist, write one LaTeX table per (dataset, mechanism, rate).",
    )
    ap.add_argument("--caption-template", type=str, default="{dataset}, {mechanism} {rate} (mean$\\pm$std over seeds).")
    ap.add_argument("--label-template", type=str, default="tab:{dataset}_{mechanism}_{rate}")
    ap.add_argument(
        "--also-write-flat",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Also write a flat table_results_agg.tex from the provided CSV (backward compatible).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)

    # If summary_all.csv is provided, convert to the aggregate format we need.
    is_agg = any(c.endswith("_mean") for c in df.columns)
    if not is_agg:
        # need at least algo + task cols
        if "algo" not in df.columns:
            if "variant" in df.columns:
                df["algo"] = df["variant"].astype(str)
            elif "method" in df.columns:
                df["algo"] = df["method"].astype(str)
            else:
                raise ValueError("summary_all.csv must contain 'variant' or 'method' columns.")
        for c in ("dataset", "mechanism", "rate"):
            if c not in df.columns:
                df[c] = None

        # numeric columns
        metric_bases = [m.strip() for m in args.metrics.split(",") if m.strip()]
        metric_bases = [m for m in metric_bases if m in df.columns]
        if not metric_bases:
            raise ValueError("No requested metrics found in summary_all.csv.")

        rows = []
        for (dataset, mechanism, rate, algo), g in df.groupby(["dataset", "mechanism", "rate", "algo"], dropna=False):
            base = {"dataset": dataset, "mechanism": mechanism, "rate": rate, "algo": algo, "n_runs": int(len(g))}
            for m in metric_bases:
                base[f"{m}_mean"] = float(np.nanmean(g[m].values))
                base[f"{m}_std"] = float(np.nanstd(g[m].values))
            rows.append(base)
        df = pd.DataFrame(rows)

    # ensure algo + task cols
    if "algo" not in df.columns:
        if "variant" in df.columns:
            df["algo"] = df["variant"].astype(str)
        elif "method" in df.columns:
            df["algo"] = df["method"].astype(str)
        else:
            raise ValueError("summary_agg.csv must contain 'algo' (or variant/method).")

    metric_bases = [m.strip() for m in args.metrics.split(",") if m.strip()]
    # keep only metrics present
    metric_bases = [m for m in metric_bases if f"{m}_mean" in df.columns]

    # apply task filter
    if args.task_filter:
        df = df.query(args.task_filter)

    # Optional: write the old flat table (one row per group-cols already aggregated).
    if args.also_write_flat == "true":
        # Build a flat table: algo + available mean±std metrics
        flat_rows = []
        for _, r in df.iterrows():
            row = {"algo": _escape(str(r.get("algo", "")))}
            for m in metric_bases:
                row[m] = _format_pm(float(r[f"{m}_mean"]), float(r.get(f"{m}_std", np.nan)), digits=args.digits)
            flat_rows.append(row)
        flat = pd.DataFrame(flat_rows)
        # Bold best overall across rows (not per-task) is not meaningful; keep plain.
        tex = _render_table(
            flat,
            caption="Aggregated results (mean$\\pm$std).",
            label="tab:results_agg",
        )
        (outdir / "table_results_agg.tex").write_text(tex, encoding="utf-8")

    # Write per-task tables
    if args.by_task == "true" and {"dataset", "mechanism", "rate"}.issubset(df.columns):
        directions = {m: _metric_direction(m) for m in metric_bases}
        for (dataset, mechanism, rate), g in df.groupby(["dataset", "mechanism", "rate"], dropna=False):
            g = g.copy()
            # methods rows
            table_rows = []
            for _, r in g.sort_values("algo").iterrows():
                row = {"Method": _escape(str(r["algo"]))}
                for m in metric_bases:
                    row[m] = _format_pm(float(r[f"{m}_mean"]), float(r.get(f"{m}_std", np.nan)), digits=args.digits)
                table_rows.append(row)
            tdf = pd.DataFrame(table_rows)
            tdf = _apply_bold_best(tdf, metric_bases, directions)

            cap = args.caption_template.format(dataset=dataset, mechanism=mechanism, rate=rate)
            lab = args.label_template.format(dataset=dataset, mechanism=mechanism, rate=rate)
            tex = _render_table(tdf, caption=cap, label=lab, col_format="l" + "c" * len(metric_bases))

            fname = f"table_{dataset}_{mechanism}_{rate}.tex"
            (outdir / fname).write_text(tex, encoding="utf-8")

    print(f"[DONE] wrote LaTeX tables to {outdir}")


if __name__ == "__main__":
    main()
