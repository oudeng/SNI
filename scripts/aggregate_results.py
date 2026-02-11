#!/usr/bin/env python3
"""
Aggregate many experiment folders into paper-ready CSV summaries.

Supports:
  - SNI runs: metrics_summary.csv (and also metrics_summary.json as a mirror)
  - Baseline runs: metrics_summary.json

Outputs (to --outdir):
  - summary_all.csv
  - summary_agg.csv                  (task x algo; mean±std across seeds)
  - summary_overall_by_algo.csv      (algo-level mean±std across tasks)
  - summary_ranks.csv                (per-task ranking; long format)
  - summary_ranks_agg.csv            (average ranks across tasks)
  - summary_wins.csv                 (win counts across tasks)
  - summary_rel_to_ref.csv           (per-task improvement vs --ref-algo)

Design notes:
  - We treat each (dataset, mechanism, rate) as one "task".
  - We average across tasks (not across runs) for overall_by_algo to avoid
    tasks with more seeds dominating.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# -------------------------- file discovery / parsing -------------------------- #

def _find_metric_files(root: Path) -> List[Path]:
    """Return metric summary files under *root*.

    Prefer CSV if both exist in the same run directory to avoid duplicates.
    """
    csv_files = list(root.rglob("metrics_summary.csv"))
    csv_parents = {p.parent for p in csv_files}
    json_files = [p for p in root.rglob("metrics_summary.json") if p.parent not in csv_parents]
    return csv_files + json_files


_EXP_ID_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(?P<dataset>[A-Za-z0-9]+)_(?P<mech>MCAR|MAR|MNAR)_(?P<rate>\d+per)_(?P<body>.+?)(?:_s(?P<seed>\d+))?$"
)

def _parse_exp_id(exp_id: Any) -> Dict[str, Any]:
    """Best-effort parsing of exp_id -> dataset / mechanism / rate / seed."""
    if exp_id is None:
        return {}
    s = str(exp_id).strip()
    if not s:
        return {}
    m = _EXP_ID_RE.match(s)
    if not m:
        return {}
    out: Dict[str, Any] = {
        "exp_prefix": m.group("prefix"),
        "dataset": m.group("dataset"),
        "mechanism": m.group("mech"),
        "rate": m.group("rate"),
        "exp_body": m.group("body"),
    }
    seed = m.group("seed")
    if seed is not None and seed.isdigit():
        out["seed_parsed"] = int(seed)
    return out


def _rate_to_float(rate: Any) -> Optional[float]:
    """Convert '30per' -> 0.3 (best-effort)."""
    if rate is None:
        return None
    s = str(rate).strip().lower()
    m = re.match(r"^(\d+)\s*per$", s)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _read_metrics_file(fp: Path) -> Optional[pd.DataFrame]:
    """Read one metrics summary file and return a DataFrame (usually 1 row)."""
    try:
        if fp.suffix.lower() == ".csv":
            return pd.read_csv(fp)
        if fp.suffix.lower() == ".json":
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return pd.DataFrame([data])
            if isinstance(data, list):
                return pd.DataFrame(data)
            return None
        return None
    except Exception:
        return None


def _pick_algo_columns(df: pd.DataFrame) -> str:
    """Pick the algorithm identifier column name."""
    # Prefer 'variant' for SNI; otherwise 'method' for baselines.
    if "variant" in df.columns:
        return "variant"
    if "method" in df.columns:
        return "method"
    # Fallback: try common alternatives
    for c in ("algo", "model", "name"):
        if c in df.columns:
            return c
    # As a last resort, create one.
    df["algo"] = "UNKNOWN"
    return "algo"


# -------------------------- aggregation helpers -------------------------- #

def _numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # Do not aggregate seeds or non-metric numeric-like columns.
    cols = [c for c in cols if c not in {"seed", "seed_parsed", "convergence_iterations"}]
    return cols


def _metric_directions(all_df: pd.DataFrame) -> Dict[str, str]:
    """Infer directions for common metrics.

    Returns: {metric_base_name: 'min'|'max'|'absmin'}
    """
    # Base names used by the project (see paper Table 1/2): cont_* and cat_*.
    directions: Dict[str, str] = {}
    for c in all_df.columns:
        if c.startswith("cont_") or c.startswith("cat_") or c in {"runtime_sec", "runtime_seconds"}:
            if c.endswith(("NRMSE", "MAE")):
                directions[c] = "min"
            elif c.endswith(("R2", "Spearman", "Accuracy", "Macro-F1", "Kappa")):
                directions[c] = "max"
            elif c.endswith("MB"):
                # mean bias: closer to 0 is better
                directions[c] = "absmin"
            elif c in {"runtime_sec", "runtime_seconds"}:
                directions[c] = "min"
    return directions


def _best_value(series: pd.Series, direction: str) -> float:
    x = pd.to_numeric(series, errors="coerce").astype(float).values
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if direction == "min":
        return float(np.nanmin(x))
    if direction == "max":
        return float(np.nanmax(x))
    # absmin
    return float(x[np.nanargmin(np.abs(x))])


def _rank_values(values: pd.Series, direction: str) -> pd.Series:
    """Return ranks where 1 is best."""
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if direction == "min":
        return x.rank(method="average", ascending=True)
    if direction == "max":
        return x.rank(method="average", ascending=False)
    # absmin
    return x.abs().rank(method="average", ascending=True)


# -------------------------- main -------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, required=True, help="Root directory containing run subfolders.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--group-cols",
        type=str,
        default="auto",
        help="Comma-separated columns to group by for summary_agg.csv. Use 'auto' for task+algo grouping.",
    )
    ap.add_argument(
        "--task-cols",
        type=str,
        default="dataset,mechanism,rate",
        help="Task columns (used for overall/ranks).",
    )
    ap.add_argument(
        "--ref-algo",
        type=str,
        default="SNI",
        help="Reference algorithm name for summary_rel_to_ref.csv (only if present).",
    )
    ap.add_argument(
        "--rank-metrics",
        type=str,
        default="auto",
        help="Comma-separated metric base names to rank; 'auto' uses common cont/cat metrics + runtime_sec when present.",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = _find_metric_files(root)
    if len(files) == 0:
        raise FileNotFoundError(f"No metrics_summary.(csv|json) found under {root}")

    rows: List[pd.DataFrame] = []
    for fp in files:
        df = _read_metrics_file(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["__path"] = str(fp)
        # Parse exp_id for task info.
        if "exp_id" in df.columns:
            parsed = _parse_exp_id(df.loc[df.index[0], "exp_id"])
            for k, v in parsed.items():
                if k not in df.columns:
                    df[k] = v
        if "rate" in df.columns and "rate_float" not in df.columns:
            df["rate_float"] = _rate_to_float(df.loc[df.index[0], "rate"])  # type: ignore[index]
        rows.append(df)

    all_df = pd.concat(rows, axis=0, ignore_index=True)
    # Ensure algo column exists
    algo_col = _pick_algo_columns(all_df)
    if algo_col != "algo":
        all_df["algo"] = all_df[algo_col].astype(str)
    else:
        all_df["algo"] = all_df["algo"].astype(str)

    # task cols
    task_cols = [c.strip() for c in args.task_cols.split(",") if c.strip()]
    for c in task_cols:
        if c not in all_df.columns:
            # Keep going; some runs may miss parsed fields.
            all_df[c] = None

    # Save run-level
    all_df.to_csv(outdir / "summary_all.csv", index=False)

    # Decide grouping for summary_agg
    if args.group_cols.strip().lower() == "auto":
        group_cols = task_cols + ["algo"]
        # Keep exp_prefix if present and task cols are missing (rare)
        if "exp_prefix" in all_df.columns and all_df["exp_prefix"].notna().any():
            # Do not add by default; tasks are already unique in your manifests.
            pass
    else:
        group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
        # Convenience fallbacks
        if group_cols == ["variant"] and "variant" not in all_df.columns and "method" in all_df.columns:
            group_cols = ["method"]
        if group_cols == ["method"] and "method" not in all_df.columns and "variant" in all_df.columns:
            group_cols = ["variant"]

    for c in group_cols:
        if c not in all_df.columns:
            raise ValueError(f"group-col '{c}' not found. Available: {sorted(all_df.columns.tolist())}")

    num_cols = _numeric_columns(all_df, exclude=group_cols + ["__path"])
    # Aggregate mean/std across seeds (runs)
    agg_rows = []
    for key, g in all_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: key[i] for i, col in enumerate(group_cols)}
        base["n_runs"] = int(len(g))
        for col in num_cols:
            base[f"{col}_mean"] = float(np.nanmean(g[col].values))
            base[f"{col}_std"] = float(np.nanstd(g[col].values))
        agg_rows.append(base)
    agg_df = pd.DataFrame(agg_rows).sort_values(group_cols)
    agg_df.to_csv(outdir / "summary_agg.csv", index=False)

    # ---------------- overall-by-algo (equal weight per task) ---------------- #
    # Build task+algo means, then average across tasks.
    per_task_cols = task_cols + ["algo"]
    per_task = agg_df.copy()
    # Ensure task cols exist in agg_df
    for c in task_cols:
        if c not in per_task.columns:
            per_task[c] = None

    metric_mean_cols = [c for c in per_task.columns if c.endswith("_mean") and c.replace("_mean", "") in num_cols]
    overall_rows = []
    for algo, g_algo in per_task.groupby("algo"):
        base = {"algo": algo, "n_tasks": int(len(g_algo))}
        for mc in metric_mean_cols:
            vals = g_algo[mc].astype(float).values
            base[f"{mc}_mean_over_tasks"] = float(np.nanmean(vals))
            base[f"{mc}_std_over_tasks"] = float(np.nanstd(vals))
        overall_rows.append(base)
    overall_df = pd.DataFrame(overall_rows).sort_values("algo")
    overall_df.to_csv(outdir / "summary_overall_by_algo.csv", index=False)

    # ----------------------------- ranking & wins ---------------------------- #
    directions = _metric_directions(all_df)
    if args.rank_metrics.strip().lower() == "auto":
        rank_metrics = [m for m in ("cont_NRMSE", "cont_R2", "cont_Spearman", "cat_Macro-F1", "cat_Accuracy", "cat_Kappa", "runtime_sec", "runtime_seconds") if m in all_df.columns]
    else:
        rank_metrics = [m.strip() for m in args.rank_metrics.split(",") if m.strip()]

    # We rank using per-task means (agg_df).
    rank_rows = []
    wins_rows = []
    for metric in rank_metrics:
        mean_col = f"{metric}_mean" if f"{metric}_mean" in per_task.columns else None
        if mean_col is None:
            continue
        direction = directions.get(metric, "min")
        # Per task: rank algos
        for task_key, g_task in per_task.groupby(task_cols, dropna=False):
            g_task = g_task.copy()
            g_task["rank"] = _rank_values(g_task[mean_col], direction)
            # wins
            best = _best_value(g_task[mean_col], direction)
            # For absmin: best is value closest to 0; win if abs equal within tol.
            tol = 1e-12
            if direction == "absmin":
                win_mask = (g_task[mean_col].astype(float).abs() - abs(best)).abs() <= tol
            else:
                win_mask = (g_task[mean_col].astype(float) - best).abs() <= tol

            for _, r in g_task.iterrows():
                rec = {task_cols[i]: task_key[i] for i in range(len(task_cols))}
                rec.update({"algo": r["algo"], "metric": metric, "value": float(r[mean_col]) if pd.notna(r[mean_col]) else float("nan"), "rank": float(r["rank"])})
                rank_rows.append(rec)
            for algo in g_task.loc[win_mask, "algo"].astype(str).tolist():
                wins_rows.append({"metric": metric, "algo": algo, "win": 1})

    ranks_df = pd.DataFrame(rank_rows)
    if not ranks_df.empty:
        ranks_df.to_csv(outdir / "summary_ranks.csv", index=False)
        ranks_agg = ranks_df.groupby(["metric", "algo"], as_index=False).agg(
            n_tasks=("rank", "count"),
            avg_rank=("rank", "mean"),
            std_rank=("rank", "std"),
        )
        ranks_agg.to_csv(outdir / "summary_ranks_agg.csv", index=False)
    else:
        ranks_agg = pd.DataFrame()

    wins_df = pd.DataFrame(wins_rows)
    if not wins_df.empty:
        wins_agg = wins_df.groupby(["metric", "algo"], as_index=False)["win"].sum().rename(columns={"win": "n_wins"})
        wins_agg.to_csv(outdir / "summary_wins.csv", index=False)

    # ------------------------- relative to reference ------------------------- #
    # Compute per-task improvement of ref vs best-other.
    ref = str(args.ref_algo)
    rel_rows = []
    if ref in per_task["algo"].astype(str).unique().tolist():
        for metric in rank_metrics:
            mean_col = f"{metric}_mean"
            if mean_col not in per_task.columns:
                continue
            direction = directions.get(metric, "min")
            for task_key, g_task in per_task.groupby(task_cols, dropna=False):
                g_task = g_task.copy()
                # ref value
                g_ref = g_task[g_task["algo"].astype(str) == ref]
                if g_ref.empty:
                    continue
                ref_val = float(g_ref.iloc[0][mean_col])
                others = g_task[g_task["algo"].astype(str) != ref]
                if others.empty:
                    continue
                best_other = _best_value(others[mean_col], direction)
                if not np.isfinite(ref_val) or not np.isfinite(best_other):
                    continue
                denom = float(abs(best_other) + 1e-12)
                if direction == "min":
                    # positive => ref better
                    rel = (best_other - ref_val) / denom
                    abs_diff = ref_val - best_other
                elif direction == "max":
                    rel = (ref_val - best_other) / denom
                    abs_diff = ref_val - best_other
                else:  # absmin
                    rel = (abs(best_other) - abs(ref_val)) / denom
                    abs_diff = abs(ref_val) - abs(best_other)
                rec = {task_cols[i]: task_key[i] for i in range(len(task_cols))}
                rec.update(
                    {
                        "metric": metric,
                        "ref_algo": ref,
                        "ref_value": ref_val,
                        "best_other_value": best_other,
                        "relative_improvement": float(rel),
                        "abs_diff_signed": float(abs_diff),
                    }
                )
                rel_rows.append(rec)
    rel_df = pd.DataFrame(rel_rows)
    if not rel_df.empty:
        rel_df.to_csv(outdir / "summary_rel_to_ref.csv", index=False)

    print(f"[DONE] wrote summaries to {outdir}")
    print(f"        - summary_all.csv ({len(all_df)} runs)")
    print(f"        - summary_agg.csv ({len(agg_df)} task×algo rows)")


if __name__ == "__main__":
    main()