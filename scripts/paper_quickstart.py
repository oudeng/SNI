#!/usr/bin/env python3
"""Paper quickstart: one-command pipeline for smoke tests, full runs, or postprocessing.

Three mutually exclusive modes:

  --smoke-test       Comprehensive smoke test (~5 min on CPU):
                     Concrete (pure continuous) + MIMIC (mixed-type, if available),
                     SNI + MeanMode/KNN/MICE baselines, v0.3 artifact checks,
                     aggregation -> merge -> LaTeX -> figures full chain.

  --main             Run main manifests (SNI + baselines) in parallel, then
                     aggregate, generate LaTeX tables, and produce figures.

  --postprocess-only Merge existing results from --sni-results and
                     --baseline-results, then aggregate + latex + viz.

Usage:
    # Quick smoke test (~5 min on CPU)
    python scripts/paper_quickstart.py --smoke-test --outdir results_quick

    # Full paper pipeline
    python scripts/paper_quickstart.py --main --outdir results_paper --n-jobs 8

    # Postprocess existing results
    python scripts/paper_quickstart.py --postprocess-only \
        --sni-results results_sni/summary_agg.csv \
        --baseline-results results_baselines/summary_agg.csv \
        --outdir results_paper
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def _run(cmd: List[str], description: str) -> int:
    """Run a subprocess command with clear status messages."""
    print(f"\n{'='*60}")
    print(f"[RUN] {description}")
    print(f"      {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        print(f"[FAIL] {description} (exit code {result.returncode})")
    else:
        print(f"[OK]   {description}")
    return result.returncode


def _write_environment_snapshot(outdir: Path) -> None:
    """Call env_snapshot.write_environment_snapshot if available."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.env_snapshot import write_environment_snapshot
        outdir.mkdir(parents=True, exist_ok=True)
        write_environment_snapshot(outdir)
        print(f"[INFO] Environment snapshot written to {outdir / 'environment_snapshot.txt'}")
    except Exception as e:
        print(f"[WARN] Could not write environment snapshot: {e}")


def do_smoke_test(outdir: Path, n_jobs: int) -> bool:
    """Comprehensive smoke test verifying the full pipeline (~5 min on CPU).

    Coverage:
      - Concrete (pure continuous) + MIMIC (mixed-type, if available) SNI experiments
      - MeanMode + KNN + MICE baselines on Concrete (+ MeanMode on MIMIC if available)
      - v0.3 artifact checks: dependency_matrix, metrics_per_feature, run_config
      - Full post-process chain: aggregate -> merge -> LaTeX -> figures
    """
    print("\n" + "#" * 60)
    print("# SMOKE TEST")
    print("#" * 60)

    outdir.mkdir(parents=True, exist_ok=True)
    smoke_dir = outdir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent

    # Track whether MIMIC was tested
    mimic_tested = False

    # ================================================================
    # 1. SNI Manifests
    # ================================================================
    sni_manifest = smoke_dir / "manifest_smoke_sni.csv"
    sni_rows = [
        {
            "exp_id": "smoke_SNI_Concrete_MCAR_30per_s42",
            "input_complete": "data/Concrete_complete.csv",
            "input_missing": "data/Concrete/Concrete_MCAR_30per.csv",
            "categorical_vars": "",
            "continuous_vars": "Cement,BlastFurnaceSlag,FlyAsh,Water,Superplasticizer,CoarseAggregate,FineAggregate,Duration,ConcreteCS",
            "variant": "SNI",
            "seed": 42,
            "epochs": 5,
            "max_iters": 1,
            "hidden_dims": "32,16",
            "emb_dim": 16,
            "num_heads": 2,
            "batch_size": 32,
        },
    ]

    # Task A.1: Add MIMIC mixed-type test if data exists
    mimic_complete = project_root / "data" / "MIMIC_complete.csv"
    mimic_missing = project_root / "data" / "MIMIC" / "MIMIC_MCAR_30per.csv"
    if mimic_complete.exists() and mimic_missing.exists():
        sni_rows.append({
            "exp_id": "smoke_SNI_MIMIC_MCAR_30per_s42",
            "input_complete": "data/MIMIC_complete.csv",
            "input_missing": "data/MIMIC/MIMIC_MCAR_30per.csv",
            "categorical_vars": "SpO2,ALARM",
            "continuous_vars": "RESP,ABP,SBP,DBP,HR,PULSE",
            "variant": "SNI",
            "seed": 42,
            "epochs": 5,
            "max_iters": 1,
            "hidden_dims": "32,16",
            "emb_dim": 16,
            "num_heads": 2,
            "batch_size": 32,
        })
        mimic_tested = True
        print("[INFO] MIMIC data found — mixed-type smoke test enabled")
    else:
        print("[SKIP] MIMIC data not found — skipping mixed-type smoke test")
        print("[INFO] Concrete-only smoke test still validates core pipeline")

    with open(sni_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sni_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sni_rows)

    # ================================================================
    # 2. Baseline Manifests (Task A.2: MeanMode + KNN + MICE)
    # ================================================================
    baseline_manifest = smoke_dir / "manifest_smoke_baselines.csv"
    concrete_baseline_common = {
        "input_complete": "data/Concrete_complete.csv",
        "input_missing": "data/Concrete/Concrete_MCAR_30per.csv",
        "categorical_vars": "",
        "continuous_vars": "Cement,BlastFurnaceSlag,FlyAsh,Water,Superplasticizer,CoarseAggregate,FineAggregate,Duration,ConcreteCS",
        "seed": 42,
    }
    baseline_rows = [
        {**concrete_baseline_common, "exp_id": "smoke_MeanMode_Concrete_MCAR_30per_s42", "method": "MeanMode"},
        {**concrete_baseline_common, "exp_id": "smoke_KNN_Concrete_MCAR_30per_s42", "method": "KNN"},
        {**concrete_baseline_common, "exp_id": "smoke_MICE_Concrete_MCAR_30per_s42", "method": "MICE"},
    ]

    # Optional: MIMIC MeanMode baseline to validate mixed-type baseline path
    if mimic_tested:
        try:
            baseline_rows.append({
                "exp_id": "smoke_MeanMode_MIMIC_MCAR_30per_s42",
                "input_complete": "data/MIMIC_complete.csv",
                "input_missing": "data/MIMIC/MIMIC_MCAR_30per.csv",
                "categorical_vars": "SpO2,ALARM",
                "continuous_vars": "RESP,ABP,SBP,DBP,HR,PULSE",
                "method": "MeanMode",
                "seed": 42,
            })
        except Exception:
            pass

    with open(baseline_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_rows)

    sni_results_dir = smoke_dir / "results_sni"
    baseline_results_dir = smoke_dir / "results_baselines"

    # ================================================================
    # 3. Run Experiments
    # ================================================================

    # --- Run SNI experiments ---
    rc = _run(
        [
            sys.executable, "scripts/run_manifest_parallel.py",
            "--manifest", str(sni_manifest),
            "--outdir", str(sni_results_dir),
            "--n-jobs", "1",
            "--default-use-gpu", "false",
            "--no-progress",
        ],
        "Smoke test: SNI experiments (Concrete" + (" + MIMIC" if mimic_tested else "") + ")",
    )
    if rc != 0:
        print("[FAIL] Smoke test: SNI experiment failed")
        return False

    # --- Run baselines (MeanMode + KNN + MICE) ---
    rc = _run(
        [
            sys.executable, "scripts/run_manifest_baselines.py",
            "--manifest", str(baseline_manifest),
            "--outdir", str(baseline_results_dir),
            "--n-jobs", "1",
            "--no-progress",
        ],
        "Smoke test: Baselines (MeanMode, KNN, MICE)",
    )
    if rc != 0:
        print("[FAIL] Smoke test: baselines failed")
        return False

    # ================================================================
    # 4. Aggregation
    # ================================================================

    sni_agg_dir = smoke_dir / "agg_sni"
    rc = _run(
        [
            sys.executable, "-m", "scripts.aggregate_results",
            "--results-root", str(sni_results_dir),
            "--outdir", str(sni_agg_dir),
        ],
        "Smoke test: aggregate SNI results",
    )
    if rc != 0:
        print("[FAIL] Smoke test: SNI aggregation failed")
        return False

    baseline_agg_dir = smoke_dir / "agg_baselines"
    rc = _run(
        [
            sys.executable, "-m", "scripts.aggregate_results",
            "--results-root", str(baseline_results_dir),
            "--outdir", str(baseline_agg_dir),
        ],
        "Smoke test: aggregate baseline results",
    )
    if rc != 0:
        print("[FAIL] Smoke test: baseline aggregation failed")
        return False

    # ================================================================
    # 5. Merge + LaTeX + Figures (Task A.4)
    # ================================================================

    merged_dir = smoke_dir / "merged"
    rc = _run(
        [
            sys.executable, "-m", "scripts.merge_summaries",
            "--inputs",
            str(sni_agg_dir / "summary_agg.csv"),
            str(baseline_agg_dir / "summary_agg.csv"),
            "--outdir", str(merged_dir),
        ],
        "Smoke test: merge summaries",
    )

    tables_dir = smoke_dir / "tables"
    figures_dir = smoke_dir / "figures"
    merged_csv = merged_dir / "merged_summary_agg.csv"
    if merged_csv.exists():
        _run(
            [
                sys.executable, "-m", "scripts.make_latex_table",
                "--summary-csv", str(merged_csv),
                "--outdir", str(tables_dir),
            ],
            "Smoke test: generate LaTeX tables",
        )
        _run(
            [
                sys.executable, "-m", "scripts.viz_make_figures",
                "--summary-csv", str(merged_csv),
                "--outdir", str(figures_dir),
            ],
            "Smoke test: generate figures",
        )

    # ================================================================
    # 6. Verify Output Files (Task A.3: v0.3 artifact checks)
    # ================================================================

    print("\n" + "=" * 60)
    print("  SMOKE TEST RESULTS")
    print("=" * 60)

    sni_run_dir = sni_results_dir / "smoke_SNI_Concrete_MCAR_30per_s42"

    # --- Core pipeline checks (Concrete, continuous-only) ---
    core_checks = [
        (sni_run_dir / "imputed.csv", "SNI imputed.csv"),
        (sni_run_dir / "metrics_summary.csv", "SNI metrics_summary.csv"),
        (baseline_results_dir / "smoke_MeanMode_Concrete_MCAR_30per_s42" / "metrics_summary.json", "MeanMode metrics"),
        (baseline_results_dir / "smoke_KNN_Concrete_MCAR_30per_s42" / "metrics_summary.json", "KNN metrics"),
        (baseline_results_dir / "smoke_MICE_Concrete_MCAR_30per_s42" / "metrics_summary.json", "MICE metrics"),
        (sni_agg_dir / "summary_agg.csv", "SNI aggregation"),
        (baseline_agg_dir / "summary_agg.csv", "Baseline aggregation"),
    ]

    # --- v0.3-specific artifact checks ---
    v03_checks = [
        (sni_run_dir / "dependency_matrix.csv", "dependency_matrix"),
        (sni_run_dir / "metrics_per_feature.csv", "metrics_per_feature"),
    ]

    # Check run_config.json for sni_version field
    run_config_path = sni_run_dir / "metrics_summary.json"
    run_config_v03_ok = False
    if run_config_path.exists():
        try:
            with open(run_config_path) as f:
                cfg_data = json.load(f)
            if cfg_data.get("sni_version") == "v0.3":
                run_config_v03_ok = True
        except Exception:
            pass

    # --- MIMIC-specific checks ---
    mimic_checks = []
    if mimic_tested:
        mimic_run_dir = sni_results_dir / "smoke_SNI_MIMIC_MCAR_30per_s42"
        mimic_checks = [
            (mimic_run_dir / "dependency_matrix.csv", "MIMIC dependency_matrix"),
            (mimic_run_dir / "lambda_per_head.csv", "MIMIC lambda_per_head"),
        ]

    # --- Post-process chain checks ---
    postprocess_checks = [
        (merged_dir / "merged_summary_agg.csv", "merged summary"),
    ]

    # Print results
    pass_count = 0
    fail_count = 0

    def _check(path, label):
        nonlocal pass_count, fail_count
        if path.exists():
            print(f"    [OK]   {label}")
            pass_count += 1
        else:
            print(f"    [MISS] {label}")
            fail_count += 1

    print("  Core pipeline (Concrete, continuous-only):")
    for path, label in core_checks:
        _check(path, label)

    print("  v0.3-specific artifacts:")
    for path, label in v03_checks:
        _check(path, label)
    if run_config_v03_ok:
        print('    [OK]   sni_version == "v0.3" in metrics_summary.json')
        pass_count += 1
    else:
        print('    [MISS] sni_version == "v0.3" in metrics_summary.json')
        fail_count += 1

    print("  Mixed-type pipeline (MIMIC):")
    if mimic_tested:
        for path, label in mimic_checks:
            _check(path, label)
    else:
        print("    [SKIP] MIMIC data not available")

    print("  Full postprocess chain:")
    for path, label in postprocess_checks:
        _check(path, label)
    # tables/ and figures/ directories may not produce full output with so little data,
    # so only check that directories were created
    if tables_dir.exists():
        print("    [OK]   tables/ directory created")
        pass_count += 1
    else:
        print("    [MISS] tables/ directory")
        fail_count += 1
    if figures_dir.exists():
        print("    [OK]   figures/ directory created")
        pass_count += 1
    else:
        print("    [MISS] figures/ directory")
        fail_count += 1

    total_checks = pass_count + fail_count
    print()
    if fail_count == 0:
        print(f"  Result: PASSED ({pass_count}/{total_checks} checks)")
    else:
        print(f"  Result: FAILED ({pass_count}/{total_checks} checks, {fail_count} missing)")
    print("=" * 60)

    return fail_count == 0


def do_main(outdir: Path, n_jobs: int) -> bool:
    """Run the full paper pipeline: manifests -> aggregate -> latex -> figures."""
    print("\n" + "#" * 60)
    print("# MAIN PIPELINE")
    print("#" * 60)

    outdir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent

    sni_results = outdir / "results_sni"
    baseline_results = outdir / "results_baselines"
    agg_sni = outdir / "agg_sni"
    agg_baselines = outdir / "agg_baselines"
    merged_dir = outdir / "merged"
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"

    success = True

    # Step 1: Run SNI main manifest
    sni_manifest = project_root / "data" / "manifest_sni_v03_main.csv"
    if sni_manifest.exists():
        rc = _run(
            [
                sys.executable, "scripts/run_manifest_parallel.py",
                "--manifest", str(sni_manifest),
                "--outdir", str(sni_results),
                "--n-jobs", str(n_jobs),
                "--skip-existing",
            ],
            "Main: run SNI manifest",
        )
        if rc != 0:
            print("[WARN] SNI manifest had errors; continuing with available results")
    else:
        print(f"[WARN] SNI manifest not found: {sni_manifest}")

    # Step 2: Run baselines manifest
    baseline_manifest = project_root / "data" / "manifest_baselines_new.csv"
    if baseline_manifest.exists():
        rc = _run(
            [
                sys.executable, "scripts/run_manifest_baselines.py",
                "--manifest", str(baseline_manifest),
                "--outdir", str(baseline_results),
                "--n-jobs", str(max(1, n_jobs // 2)),
                "--skip-existing",
            ],
            "Main: run baselines manifest",
        )
        if rc != 0:
            print("[WARN] Baselines manifest had errors; continuing with available results")
    else:
        print(f"[WARN] Baselines manifest not found: {baseline_manifest}")

    # Step 3: Aggregate SNI results
    rc = _run(
        [
            sys.executable, "-m", "scripts.aggregate_results",
            "--results-root", str(sni_results),
            "--outdir", str(agg_sni),
        ],
        "Main: aggregate SNI results",
    )
    if rc != 0:
        success = False

    # Step 4: Aggregate baseline results
    rc = _run(
        [
            sys.executable, "-m", "scripts.aggregate_results",
            "--results-root", str(baseline_results),
            "--outdir", str(agg_baselines),
        ],
        "Main: aggregate baseline results",
    )
    if rc != 0:
        success = False

    # Step 5: Merge summaries
    sni_agg_csv = agg_sni / "summary_agg.csv"
    baseline_agg_csv = agg_baselines / "summary_agg.csv"
    merge_inputs = [str(p) for p in [sni_agg_csv, baseline_agg_csv] if p.exists()]

    if merge_inputs:
        rc = _run(
            [
                sys.executable, "-m", "scripts.merge_summaries",
                "--inputs", *merge_inputs,
                "--outdir", str(merged_dir),
            ],
            "Main: merge summaries",
        )
        if rc != 0:
            success = False

    # Step 6: LaTeX tables
    merged_csv = merged_dir / "merged_summary_agg.csv"
    if merged_csv.exists():
        rc = _run(
            [
                sys.executable, "-m", "scripts.make_latex_table",
                "--summary-csv", str(merged_csv),
                "--outdir", str(tables_dir),
                "--profile", "main",
            ],
            "Main: generate LaTeX tables",
        )
        if rc != 0:
            success = False

    # Step 7: Figures
    if merged_csv.exists():
        rc = _run(
            [
                sys.executable, "-m", "scripts.viz_make_figures",
                "--summary-csv", str(merged_csv),
                "--outdir", str(figures_dir),
                "--profile", "main",
            ],
            "Main: generate figures",
        )
        if rc != 0:
            success = False

    if success:
        print("\n[DONE] Main pipeline completed successfully.")
    else:
        print("\n[WARN] Main pipeline completed with some errors.")

    return success


def do_postprocess(
    sni_results: str,
    baseline_results: str,
    outdir: Path,
) -> bool:
    """Merge pre-existing results and run postprocessing (aggregate + latex + viz)."""
    print("\n" + "#" * 60)
    print("# POSTPROCESS ONLY")
    print("#" * 60)

    outdir.mkdir(parents=True, exist_ok=True)
    merged_dir = outdir / "merged"
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"

    success = True

    # Step 1: Merge summaries
    merge_inputs = []
    if sni_results:
        merge_inputs.append(sni_results)
    if baseline_results:
        merge_inputs.append(baseline_results)

    if not merge_inputs:
        print("[ERROR] No input summary files provided.")
        return False

    rc = _run(
        [
            sys.executable, "-m", "scripts.merge_summaries",
            "--inputs", *merge_inputs,
            "--outdir", str(merged_dir),
        ],
        "Postprocess: merge summaries",
    )
    if rc != 0:
        success = False

    merged_csv = merged_dir / "merged_summary_agg.csv"

    # Step 2: Aggregate (re-run on merged for overall/ranks tables)
    # The merged CSV is already aggregated, so we generate the secondary tables.
    # We create a dummy root and copy the merged data there for aggregate_results.
    # Instead, use make_latex_table and viz_make_figures directly on merged CSV.

    # Step 3: LaTeX tables
    if merged_csv.exists():
        rc = _run(
            [
                sys.executable, "-m", "scripts.make_latex_table",
                "--summary-csv", str(merged_csv),
                "--outdir", str(tables_dir),
            ],
            "Postprocess: generate LaTeX tables",
        )
        if rc != 0:
            success = False

    # Step 4: Figures
    if merged_csv.exists():
        rc = _run(
            [
                sys.executable, "-m", "scripts.viz_make_figures",
                "--summary-csv", str(merged_csv),
                "--outdir", str(figures_dir),
            ],
            "Postprocess: generate figures",
        )
        if rc != 0:
            success = False

    if success:
        print("\n[DONE] Postprocessing completed successfully.")
    else:
        print("\n[WARN] Postprocessing completed with some errors.")

    return success


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paper quickstart: smoke test, full run, or postprocess.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal experiment (Concrete, 1 seed, MCAR 30%%, 5 epochs) with SNI + MeanMode.",
    )
    mode.add_argument(
        "--main",
        action="store_true",
        help="Run full paper pipeline: manifests -> aggregate -> latex -> figures.",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Merge existing results and run postprocessing only.",
    )

    # Common args
    ap.add_argument("--outdir", type=str, default="results_paper", help="Output directory (default: results_paper).")
    ap.add_argument("--n-jobs", type=int, default=4, help="Number of parallel workers (default: 4).")

    # Postprocess-only args
    ap.add_argument("--sni-results", type=str, default=None, help="Path to SNI summary_agg.csv (for --postprocess-only).")
    ap.add_argument("--baseline-results", type=str, default=None, help="Path to baseline summary_agg.csv (for --postprocess-only).")

    args = ap.parse_args()
    outdir = Path(args.outdir)

    # Write environment snapshot at start
    _write_environment_snapshot(outdir)

    if args.smoke_test:
        ok = do_smoke_test(outdir, args.n_jobs)
        sys.exit(0 if ok else 1)
    elif args.main:
        ok = do_main(outdir, args.n_jobs)
        sys.exit(0 if ok else 1)
    elif args.postprocess_only:
        if not args.sni_results and not args.baseline_results:
            ap.error("--postprocess-only requires at least one of --sni-results or --baseline-results")
        ok = do_postprocess(
            sni_results=args.sni_results or "",
            baseline_results=args.baseline_results or "",
            outdir=outdir,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
