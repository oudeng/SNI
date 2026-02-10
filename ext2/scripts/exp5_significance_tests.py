#!/usr/bin/env python3
"""exp5_significance_tests.py

Ext2 / Experiment 5: Statistical significance tests (Wilcoxon signed-rank)
-------------------------------------------------------------------------

This script fills the Supplementary placeholder:

  - Table S8: "Statistical significance tests — planned"

The placeholder text suggests paired, non-parametric tests comparing SNI
against strong baselines across dataset–mechanism settings.

What this script produces
-------------------------
It supports TWO complementary test modes (you can run both):

(A) across_settings (recommended for Table S8):
    For each metric and baseline, compute a single Wilcoxon signed-rank test
    over *settings* = (dataset × mechanism). Each setting contributes one
    paired value (mean over seeds) for SNI and for the baseline.

(B) per_setting (optional diagnostic):
    For each dataset × mechanism × metric, run Wilcoxon across *seeds*
    (paired by seed) for SNI vs baseline.

Multiple-comparison correction
------------------------------
Within each (metric × test-mode × dataset×mechanism) group, we apply
Holm–Bonferroni correction across baselines.

Inputs
------
The script tries to discover per-seed metric CSVs from --results-dir.

Supported table formats:
1) Unified table containing columns:
     dataset, mechanism, seed, method, <metric columns...>
2) Per-setting files (one per dataset×mechanism) containing:
     seed, method, <metric columns...>
   In this case, dataset/mechanism are inferred from the path or filename.

Outputs
-------
<outdir>/
  wilcoxon_across_settings.csv    : directly usable for Table S8
  wilcoxon_per_setting.csv        : optional detailed diagnostics
  wilcoxon_summary.csv            : counts of significant better/worse/tie

Example
-------
python ext2/scripts/exp5_significance_tests.py \
  --results-dir . \
  --datasets MIMIC eICU NHANES ComCri AutoMPG Concrete \
  --mechanisms MCAR MAR \
  --metrics NRMSE R2 Spearman_rho Macro_F1 \
  --reference-method SNI \
  --baselines MissForest MIWAE \
  --mode both \
  --alpha 0.05 \
  --outdir results_ext2/significance
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    print("[ERROR] scipy not found. Install via: pip install scipy", file=sys.stderr)
    raise


# ---------------------------------------------------------------------
# Metric direction: True = higher is better, False = lower is better
# ---------------------------------------------------------------------
METRIC_DIRECTION = {
    "NRMSE": False,
    "R2": True,
    "Spearman_rho": True,
    "Spearman": True,
    "Macro_F1": True,
    "Cohens_kappa": True,
}


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _norm_method(s: str) -> str:
    return _norm_token(s)


def _norm_metric(s: str) -> str:
    return _norm_token(s)


def _find_metric_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    """Find the best matching metric column name in df for requested metric."""
    target = _norm_metric(metric)
    candidates = { _norm_metric(c): c for c in df.columns }
    if target in candidates:
        return candidates[target]

    # heuristic aliases — includes cont_/cat_ prefixed names from the project
    aliases = {
        "spearmanrho": ["spearman", "spearmanr", "spearman_rho", "rho",
                        "cont_spearman", "cont_spearman_rho"],
        "macrof1": ["macro_f1", "macro-f1", "f1_macro", "f1macro",
                    "cat_macro-f1", "cat_macro_f1", "cat_macrof1"],
        "cohenskappa": ["kappa", "cohens_kappa", "cohen_kappa",
                        "cat_cohen_kappa", "cat_cohens_kappa"],
        "r2": ["r2score", "r2_score", "cont_r2"],
        "nrmse": ["rmse", "nrmse", "cont_nrmse"],
    }
    for key, al in aliases.items():
        if target == key:
            for a in al:
                na = _norm_metric(a)
                if na in candidates:
                    return candidates[na]
    return None


def _discover_unified_table(results_dir: Path) -> Optional[pd.DataFrame]:
    """Try to find a unified per-seed table under results_dir."""
    preferred_names = [
        "metrics_per_seed.csv",
        "per_seed_metrics.csv",
        "all_metrics_per_seed.csv",
        "all_per_seed_metrics.csv",
        "per_seed.csv",
    ]
    # Search preferred filenames first (fast)
    for name in preferred_names:
        for p in results_dir.rglob(name):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            needed = {"seed", "method"}
            if needed.issubset(set(map(str.lower, df.columns))) or ("seed" in df.columns and "method" in df.columns):
                # If dataset/mechanism not present, we can still use it as partial
                df["_source_path"] = str(p)
                return df

    # Otherwise scan small subset: files containing 'per_seed'
    for p in results_dir.rglob("*per_seed*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "seed" in df.columns and "method" in df.columns:
            df["_source_path"] = str(p)
            return df

    return None


_EXP_ID_RE = re.compile(
    r"^[A-Z]+_(.+?)_(MCAR|MAR|MNAR)_(\d+)per_(.+?)_s(\d+)$"
)


def _parse_exp_id(exp_id: str) -> Optional[Dict[str, str]]:
    """Parse experiment ID like MAIN_MIMIC_MAR_30per_SNI_s5."""
    m = _EXP_ID_RE.match(str(exp_id))
    if m is None:
        return None
    return {
        "dataset": m.group(1),
        "mechanism": m.group(2),
        "rate": m.group(3),
        "method": m.group(4),
        "seed": m.group(5),
    }


def _infer_dataset_mechanism_from_path(path: Path, datasets: List[str], mechanisms: List[str]) -> Tuple[Optional[str], Optional[str]]:
    parts = [p.lower() for p in path.parts]
    ds = None
    mech = None

    # dataset inference
    for d in datasets:
        if d.lower() in parts or d.lower() in path.name.lower():
            ds = d
            break

    # mechanism inference
    for m in mechanisms:
        if m.lower() in parts or re.search(rf"\b{re.escape(m.lower())}\b", path.name.lower()):
            mech = m
            break

    return ds, mech


def _load_baseline_summaries(project_root: Path, mechanisms: List[str]) -> List[pd.DataFrame]:
    """Load baseline summary_all.csv files from results_baselines_* dirs."""
    frames: List[pd.DataFrame] = []
    # Map mechanism groups to result directory patterns
    dir_patterns = [
        "results_baselines_main",       # MCAR, MAR
        "results_baselines_main_all",    # MCAR, MAR (extended)
        "results_baselines_mnar",        # MNAR
        "results_baselines_deep",        # deep methods (MIWAE, GAIN, etc.)
    ]
    for pattern in dir_patterns:
        d = project_root / pattern
        if not d.exists():
            continue
        # Try _summary/summary_all.csv first, then summary_all_runs.csv
        for csv_name in ["_summary/summary_all.csv", "summary_all_runs.csv"]:
            p = d / csv_name
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue

            if "seed" not in df.columns:
                continue

            # Ensure method column
            if "method" not in df.columns and "algo" in df.columns:
                df["method"] = df["algo"]

            # Parse dataset/mechanism from exp_id if not present
            if "dataset" not in df.columns or "mechanism" not in df.columns:
                if "exp_id" in df.columns:
                    parsed = df["exp_id"].map(_parse_exp_id)
                    if "dataset" not in df.columns:
                        df["dataset"] = parsed.map(lambda x: x["dataset"] if x else None)
                    if "mechanism" not in df.columns:
                        df["mechanism"] = parsed.map(lambda x: x["mechanism"] if x else None)

            # Filter to requested mechanisms
            if "mechanism" in df.columns:
                mech_upper = [m.upper() for m in mechanisms]
                df = df[df["mechanism"].str.upper().isin(mech_upper)].copy()

            if len(df) > 0:
                df["_source_path"] = str(p)
                frames.append(df)
                print(f"  [load] {p}: {len(df)} rows")
            break  # only use the first found CSV per directory

    return frames


def _load_sni_results(project_root: Path, mechanisms: List[str]) -> List[pd.DataFrame]:
    """Load SNI per-experiment metrics_summary.csv files."""
    frames: List[pd.DataFrame] = []
    sni_dirs = [
        "results_sni_main",    # MCAR, MAR
        "results_sni_mnar",    # MNAR
    ]
    mech_upper = {m.upper() for m in mechanisms}

    for sni_dir_name in sni_dirs:
        sni_dir = project_root / sni_dir_name
        if not sni_dir.exists():
            continue

        rows: List[pd.DataFrame] = []
        for exp_folder in sorted(sni_dir.iterdir()):
            if not exp_folder.is_dir():
                continue
            csv_path = exp_folder / "metrics_summary.csv"
            if not csv_path.exists():
                continue
            parsed = _parse_exp_id(exp_folder.name)
            if parsed is None:
                continue
            if parsed["mechanism"].upper() not in mech_upper:
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # Add parsed metadata
            df["dataset"] = parsed["dataset"]
            df["mechanism"] = parsed["mechanism"]
            df["seed"] = int(parsed["seed"])
            # Rename 'variant' → 'method' if needed
            if "variant" in df.columns and "method" not in df.columns:
                df["method"] = df["variant"]
            elif "method" not in df.columns:
                df["method"] = parsed["method"]

            rows.append(df)

        if rows:
            combined = pd.concat(rows, ignore_index=True)
            combined["_source_path"] = str(sni_dir)
            frames.append(combined)
            print(f"  [load] {sni_dir}: {len(combined)} rows (SNI)")

    return frames


def _load_all_per_seed(
    results_dir: Path,
    datasets: List[str],
    mechanisms: List[str],
) -> pd.DataFrame:
    """Load per-seed metrics into a single tidy dataframe.

    Supports two modes:
    1) Legacy: results_dir contains unified CSV or per-setting subfolders
    2) Project layout: results_dir is the project root with results_baselines_*/
       and results_sni_* directories
    """
    # --- Try legacy mode first (unified table) ---
    df_unified = _discover_unified_table(results_dir)
    if df_unified is not None:
        cols = set(df_unified.columns)
        if "dataset" not in cols or "mechanism" not in cols:
            src_path = df_unified.get("_source_path")
            if src_path is None:
                df_unified["dataset"] = None
                df_unified["mechanism"] = None
            else:
                ds, mech = _infer_dataset_mechanism_from_path(Path(str(src_path.iloc[0])), datasets, mechanisms)
                df_unified["dataset"] = df_unified.get("dataset", ds)
                df_unified["mechanism"] = df_unified.get("mechanism", mech)
        return df_unified

    # --- Try project layout mode ---
    print("[Exp5] Auto-discovering results from project layout...")
    all_frames: List[pd.DataFrame] = []

    baseline_frames = _load_baseline_summaries(results_dir, mechanisms)
    all_frames.extend(baseline_frames)

    sni_frames = _load_sni_results(results_dir, mechanisms)
    all_frames.extend(sni_frames)

    if not all_frames:
        # --- Legacy fallback: per-setting file scan ---
        rows: List[pd.DataFrame] = []
        patterns = [
            "metrics_per_seed.csv",
            "per_seed_metrics.csv",
            "*per_seed*.csv",
            "*metrics*.csv",
        ]
        for ds in datasets:
            for mech in mechanisms:
                candidates = [
                    results_dir / ds / mech / "metrics_per_seed.csv",
                    results_dir / ds / mech / "per_seed_metrics.csv",
                    results_dir / ds / f"{mech}_metrics_per_seed.csv",
                ]
                found = None
                for p in candidates:
                    if p.exists():
                        found = p
                        break
                if found is None:
                    ds_dir = results_dir / ds
                    if ds_dir.exists():
                        for pat in patterns:
                            for p in ds_dir.rglob(pat):
                                if mech.lower() in p.as_posix().lower():
                                    found = p
                                    break
                            if found is not None:
                                break
                if found is None:
                    continue
                try:
                    df = pd.read_csv(found)
                except Exception:
                    continue
                if "seed" not in df.columns or "method" not in df.columns:
                    continue
                df = df.copy()
                df["dataset"] = ds
                df["mechanism"] = mech
                df["_source_path"] = str(found)
                rows.append(df)

        if not rows:
            raise FileNotFoundError(
                f"Could not find per-seed metric CSVs under {results_dir}. "
                "Please provide the project root or the main results directory "
                "that contains results_baselines_*, results_sni_* subdirectories."
            )
        return pd.concat(rows, ignore_index=True)

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure seed is int
    combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce").astype("Int64")

    # Filter to requested datasets
    if "dataset" in combined.columns:
        ds_upper = {d.lower() for d in datasets}
        combined = combined[combined["dataset"].str.lower().isin(ds_upper)].copy()

    print(f"[Exp5] Loaded {len(combined)} total rows "
          f"({combined['method'].nunique()} methods, "
          f"{combined['dataset'].nunique()} datasets)")

    return combined


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    if m == 0:
        return []
    idx_sorted = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    for rank, idx in enumerate(idx_sorted):
        adj[idx] = min(pvals[idx] * (m - rank), 1.0)
    # monotonicity
    for i in range(1, m):
        idx_i = idx_sorted[i]
        idx_prev = idx_sorted[i - 1]
        adj[idx_i] = max(adj[idx_i], adj[idx_prev])
    return adj.tolist()


def _wilcoxon_safe(diff: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon on diff, robust to all-zero and small n."""
    diff = diff.astype(float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 3:
        return float("nan"), float("nan")
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="Ext2/Exp5: Wilcoxon significance tests")
    parser.add_argument("--results-dir", required=True,
                        help="Project root or directory containing results_baselines_*/results_sni_* subdirs")
    parser.add_argument("--datasets", nargs="+", default=["MIMIC", "eICU", "NHANES", "ComCri", "AutoMPG", "Concrete"])
    parser.add_argument("--mechanisms", nargs="+", default=["MCAR", "MAR"])
    parser.add_argument("--metrics", nargs="+", default=["NRMSE", "R2", "Spearman_rho", "Macro_F1"])
    parser.add_argument("--reference-method", default="SNI")
    parser.add_argument("--baselines", nargs="+", default=["MissForest", "MIWAE", "GAIN", "KNN", "MICE", "MeanMode"])
    parser.add_argument("--mode", default="across_settings", choices=["across_settings", "per_setting", "both"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--outdir", default="results_ext2/significance")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load_all_per_seed(results_dir, datasets=args.datasets, mechanisms=args.mechanisms)

    # Normalize method names for robust matching
    df = df.copy()
    df["method_norm"] = df["method"].astype(str).map(_norm_method)
    df["dataset"] = df["dataset"].astype(str)
    df["mechanism"] = df["mechanism"].astype(str)

    ref_norm = _norm_method(args.reference_method)
    baseline_norms = {b: _norm_method(b) for b in args.baselines}

    # ------------------------------------------------------------------
    # (A) across_settings: one test per (metric, baseline)
    # ------------------------------------------------------------------
    across_rows: List[Dict[str, object]] = []
    if args.mode in ("across_settings", "both"):
        for metric in args.metrics:
            metric_col = _find_metric_column(df, metric)
            if metric_col is None:
                continue

            higher_is_better = METRIC_DIRECTION.get(metric, True)

            # compute setting means (dataset×mechanism×method)
            setting_means = (
                df.groupby(["dataset", "mechanism", "method_norm"], dropna=False)[metric_col]
                .mean()
                .reset_index()
                .rename(columns={metric_col: "metric_mean"})
            )

            # build vectors per baseline
            pvals: List[float] = []
            tmp_rows: List[Dict[str, object]] = []
            for baseline, bnorm in baseline_norms.items():
                ref = setting_means[setting_means["method_norm"] == ref_norm]
                base = setting_means[setting_means["method_norm"] == bnorm]

                merged = ref.merge(base, on=["dataset", "mechanism"], suffixes=("_ref", "_base"))
                if len(merged) < 3:
                    continue

                diff = merged["metric_mean_ref"].to_numpy(dtype=float) - merged["metric_mean_base"].to_numpy(dtype=float)
                if not higher_is_better:
                    diff = -diff  # so positive => SNI better

                stat, p = _wilcoxon_safe(diff)
                pvals.append(p)
                tmp_rows.append(
                    {
                        "metric": metric,
                        "comparison": f"{args.reference_method} vs {baseline}",
                        "test": "Wilcoxon signed-rank (paired across settings)",
                        "n_settings": int(len(merged)),
                        "mean_diff": float(np.nanmean(diff)),
                        "W_statistic": stat,
                        "p_value": p,
                    }
                )

            # Holm correction across baselines for this metric
            if tmp_rows:
                p_adj = _holm_bonferroni(pvals)
                for row, adj in zip(tmp_rows, p_adj):
                    row["p_adjusted"] = float(adj)
                    row["significant"] = bool(adj < args.alpha)
                    across_rows.append(row)

        pd.DataFrame(across_rows).to_csv(outdir / "wilcoxon_across_settings.csv", index=False)
        print(f"[Exp5] Saved: {outdir / 'wilcoxon_across_settings.csv'} (n={len(across_rows)})")

    # ------------------------------------------------------------------
    # (B) per_setting: one test per (dataset, mechanism, metric, baseline)
    # ------------------------------------------------------------------
    per_rows: List[Dict[str, object]] = []
    if args.mode in ("per_setting", "both"):
        for dataset in args.datasets:
            for mechanism in args.mechanisms:
                df_dm = df[(df["dataset"] == str(dataset)) & (df["mechanism"] == str(mechanism))].copy()
                if df_dm.empty:
                    continue

                for metric in args.metrics:
                    metric_col = _find_metric_column(df_dm, metric)
                    if metric_col is None:
                        continue
                    higher_is_better = METRIC_DIRECTION.get(metric, True)

                    # pivot: seed × method_norm
                    piv = df_dm.pivot_table(index="seed", columns="method_norm", values=metric_col, aggfunc="mean")
                    if ref_norm not in piv.columns:
                        continue

                    pvals: List[float] = []
                    tmp_rows = []

                    for baseline, bnorm in baseline_norms.items():
                        if bnorm not in piv.columns:
                            continue
                        pair = piv[[ref_norm, bnorm]].dropna()
                        if len(pair) < 3:
                            continue

                        diff = pair[ref_norm].to_numpy(dtype=float) - pair[bnorm].to_numpy(dtype=float)
                        if not higher_is_better:
                            diff = -diff

                        stat, p = _wilcoxon_safe(diff)
                        pvals.append(p)
                        tmp_rows.append(
                            {
                                "dataset": dataset,
                                "mechanism": mechanism,
                                "metric": metric,
                                "baseline": baseline,
                                "n_seeds": int(len(pair)),
                                "mean_diff": float(np.nanmean(diff)),
                                "W_statistic": stat,
                                "p_value": p,
                            }
                        )

                    if tmp_rows:
                        p_adj = _holm_bonferroni(pvals)
                        for row, adj in zip(tmp_rows, p_adj):
                            row["p_adjusted"] = float(adj)
                            row["significant"] = bool(adj < args.alpha)
                            if row["significant"]:
                                row["direction"] = "SNI_better" if row["mean_diff"] > 0 else "baseline_better"
                            else:
                                row["direction"] = "tie"
                            per_rows.append(row)

        pd.DataFrame(per_rows).to_csv(outdir / "wilcoxon_per_setting.csv", index=False)
        print(f"[Exp5] Saved: {outdir / 'wilcoxon_per_setting.csv'} (n={len(per_rows)})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, object]] = []
    if per_rows:
        df_per = pd.DataFrame(per_rows)
        summary = (
            df_per.groupby(["metric", "direction"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary.to_csv(outdir / "wilcoxon_summary.csv", index=False)
        print(f"[Exp5] Saved: {outdir / 'wilcoxon_summary.csv'}")

    print("[Exp5] Done.")


if __name__ == "__main__":
    main()