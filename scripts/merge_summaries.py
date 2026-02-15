#!/usr/bin/env python3
"""Merge multiple summary_agg.csv files into a single unified summary.

Features:
  - Accepts multiple CSV paths or glob patterns via --inputs
  - Concatenates all input files
  - De-duplicates by (dataset, mechanism, rate, algo), keeping the last occurrence
  - Detects conflicts: same key with metric difference > threshold
  - Writes merge_conflicts.csv and merged_summary_agg.csv

Usage:
    python scripts/merge_summaries.py \
        --inputs results_sni/summary_agg.csv results_baselines/summary_agg.csv \
        --outdir results_merged

    # With glob patterns
    python scripts/merge_summaries.py \
        --inputs "results_*/summary_agg.csv" \
        --outdir results_merged \
        --conflict-threshold 0.05
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _resolve_inputs(patterns: List[str]) -> List[Path]:
    """Resolve a list of file paths or glob patterns to concrete paths."""
    resolved: List[Path] = []
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        # Try glob expansion first
        matches = sorted(glob.glob(pat, recursive=True))
        if matches:
            resolved.extend(Path(m) for m in matches)
        else:
            # Treat as a literal path
            p = Path(pat)
            if p.exists():
                resolved.append(p)
            else:
                print(f"[WARN] File not found and no glob matches: {pat}")
    # De-duplicate while preserving order
    seen = set()
    out: List[Path] = []
    for p in resolved:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single summary CSV and tag it with its source path."""
    df = pd.read_csv(path)
    df["__source__"] = str(path)
    return df


def _detect_conflicts(
    df: pd.DataFrame,
    key_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Detect rows with the same key but metric values differing by more than threshold.

    Returns a DataFrame of conflicting pairs (or empty if none).
    """
    metric_cols = [c for c in df.columns if c.endswith("_mean") and c != "__source__"]
    if not metric_cols:
        return pd.DataFrame()

    # Group by key and look for groups with >1 row (before dedup)
    conflict_rows = []
    for key_vals, grp in df.groupby(key_cols, dropna=False):
        if len(grp) < 2:
            continue
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)

        # Compare all pairs within the group
        indices = grp.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                row_a = grp.loc[indices[i]]
                row_b = grp.loc[indices[j]]
                for mc in metric_cols:
                    va = row_a.get(mc)
                    vb = row_b.get(mc)
                    try:
                        va_f = float(va)
                        vb_f = float(vb)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(va_f) or not np.isfinite(vb_f):
                        continue
                    diff = abs(va_f - vb_f)
                    if diff > threshold:
                        rec = {key_cols[k]: key_vals[k] for k in range(len(key_cols))}
                        rec.update({
                            "metric": mc,
                            "value_a": va_f,
                            "value_b": vb_f,
                            "abs_diff": diff,
                            "source_a": row_a.get("__source__", ""),
                            "source_b": row_b.get("__source__", ""),
                        })
                        conflict_rows.append(rec)

    return pd.DataFrame(conflict_rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge multiple summary_agg.csv files into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more CSV paths or glob patterns (e.g., 'results_*/summary_agg.csv').",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for merged files.",
    )
    ap.add_argument(
        "--conflict-threshold",
        type=float,
        default=0.1,
        help="Absolute metric difference threshold for conflict detection (default: 0.1).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input files
    csv_paths = _resolve_inputs(args.inputs)
    if not csv_paths:
        raise FileNotFoundError(
            f"No input files found for patterns: {args.inputs}"
        )

    print(f"[INFO] Found {len(csv_paths)} input file(s):")
    for p in csv_paths:
        print(f"       - {p}")

    # Read and concatenate
    dfs = [_read_one(p) for p in csv_paths]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[INFO] Concatenated {len(merged)} total rows")

    # Ensure key columns exist
    key_cols = ["dataset", "mechanism", "rate", "algo"]
    for c in key_cols:
        if c not in merged.columns:
            raise ValueError(
                f"Required column '{c}' not found in merged data. "
                f"Available columns: {sorted(merged.columns.tolist())}"
            )

    # Coerce key columns to string for consistent grouping
    for c in key_cols:
        merged[c] = merged[c].astype(str)

    # Detect conflicts before dedup
    conflicts = _detect_conflicts(merged, key_cols, args.conflict_threshold)
    if not conflicts.empty:
        conflicts_path = outdir / "merge_conflicts.csv"
        conflicts.to_csv(conflicts_path, index=False)
        print(f"[WARN] Found {len(conflicts)} metric conflicts (threshold={args.conflict_threshold})")
        print(f"       Written to {conflicts_path}")
    else:
        print(f"[INFO] No metric conflicts detected (threshold={args.conflict_threshold})")

    # De-duplicate: keep last occurrence (later files override earlier ones)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=key_cols, keep="last").copy()
    n_after = len(merged)
    if n_before != n_after:
        print(f"[INFO] De-duplicated: {n_before} -> {n_after} rows ({n_before - n_after} duplicates removed)")

    # Drop internal columns before writing
    drop_cols = [c for c in merged.columns if c.startswith("__")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # Sort for deterministic output
    merged = merged.sort_values(key_cols).reset_index(drop=True)

    # Write output
    out_path = outdir / "merged_summary_agg.csv"
    merged.to_csv(out_path, index=False)

    print(f"[DONE] Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
