#!/usr/bin/env python3
"""
One-command report generator for SNI experiments.

Example:
  python scripts/report_generate_all.py \
    --results-main results_main \
    --results-ablation results_ablation \
    --results-mnar results_mnar \
    --out-root reports

It will create:
  reports/main/_summary + _figs + _tables
  reports/ablation/_summary + _figs + _tables
  reports/mnar/_summary + _figs + _tables

You can still run the underlying scripts manually; this is just a convenience wrapper.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def _suite(name: str, results_root: str, out_root: Path) -> None:
    if results_root is None:
        return
    r = Path(results_root)
    if not r.exists():
        print(f"[WARN] skip {name}: results root not found: {r}")
        return

    suite_dir = out_root / name
    summary_dir = suite_dir / "_summary"
    figs_dir = suite_dir / "_figs"
    tables_dir = suite_dir / "_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    _run(["python", "scripts/aggregate_results.py", "--results-root", str(r), "--outdir", str(summary_dir)])
    _run(["python", "scripts/viz_make_figures.py", "--summary-dir", str(summary_dir), "--outdir", str(figs_dir)])

    # tables: default writes per-task tables
    _run(["python", "scripts/make_latex_table.py", "--summary-csv", str(summary_dir / "summary_agg.csv"), "--outdir", str(tables_dir)])

    # paper-friendly subsets (optional):
    # - MAR 30% (common main table)
    _run([
        "python","scripts/make_latex_table.py",
        "--summary-csv", str(summary_dir / "summary_agg.csv"),
        "--outdir", str(tables_dir / "MAR30"),
        "--task-filter", "mechanism=='MAR' and rate=='30per'",
        "--also-write-flat", "false",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-main", type=str, default=None)
    ap.add_argument("--results-ablation", type=str, default=None)
    ap.add_argument("--results-mnar", type=str, default=None)
    ap.add_argument("--out-root", type=str, default="reports")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _suite("main", args.results_main, out_root)
    _suite("ablation", args.results_ablation, out_root)
    _suite("mnar", args.results_mnar, out_root)

    print(f"[DONE] reports written under: {out_root}")


if __name__ == "__main__":
    main()
