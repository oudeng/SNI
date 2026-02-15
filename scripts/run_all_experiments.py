#!/usr/bin/env python3
"""Run ALL paper experiments in stages with one command.

This orchestrator launches every experiment needed for the paper, from SNI main
runs through baselines, extension experiments, and final aggregation / tables /
figures.  Stages can be selected individually via ``--stages``.

Stage overview
--------------
  1  SNI manifests        (1a main, 1b ablation-lambda, 1c MNAR, 1d rate-sweep)
  2  Classical baselines  (2a main, 2b MNAR)
  3  Deep baselines       (3a deep, 3b new)
  4  Sanity check         (4a synth S5)
  5  Ext1 experiments     (5a audit ALARM, 5b audit SBP, 5c downstream)
  6  Ext2 experiments     (6a per-class, 6b SHAP, 6c rollout, 6d stability,
                           6e significance, 6f MIMIC mortality)
  7  Aggregation + LaTeX tables + Figures

Usage
-----
    # Run everything
    python scripts/run_all_experiments.py

    # Run only stages 1 and 7 (SNI + postprocessing)
    python scripts/run_all_experiments.py --stages 1 7

    # Dry run (print commands only)
    python scripts/run_all_experiments.py --dry-run

    # Custom output directory and parallelism
    python scripts/run_all_experiments.py --outdir results_full --n-jobs-cpu 12 --n-jobs-gpu 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root (two levels up from this script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_DONE = "Done"
STATUS_SKIP = "Skip"
STATUS_FAIL = "FAIL"
STATUS_DRY  = "Dry-run"

# Status icons for the summary table
_STATUS_ICON = {
    STATUS_DONE: "\u2705",
    STATUS_SKIP: "\u23ed\ufe0f",
    STATUS_FAIL: "\u274c",
    STATUS_DRY:  "\U0001f4dd",
}


# ---------------------------------------------------------------------------
# Helper: run a single subprocess command
# ---------------------------------------------------------------------------

def _run(
    cmd: List[str],
    description: str,
    *,
    dry_run: bool = False,
    cwd: Optional[str] = None,
) -> int:
    """Run *cmd* as a subprocess, printing a clear status banner.

    In dry-run mode the command is printed but not executed (returns 0).
    """
    print(f"\n{'=' * 64}")
    print(f"[RUN] {description}")
    print(f"      {' '.join(cmd)}")
    print(f"{'=' * 64}")

    if dry_run:
        print("[DRY-RUN] Command not executed.")
        return 0

    result = subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {description} (exit code {result.returncode})")
    else:
        print(f"[OK]   {description}")
    return result.returncode


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _write_environment_snapshot(outdir: Path) -> None:
    """Write an environment snapshot via ``scripts.env_snapshot`` if available."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.env_snapshot import write_environment_snapshot
        outdir.mkdir(parents=True, exist_ok=True)
        write_environment_snapshot(outdir)
        print(f"[INFO] Environment snapshot written to {outdir / 'environment_snapshot.txt'}")
    except Exception as e:
        print(f"[WARN] Could not write environment snapshot: {e}")


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

# Each manifest stage is a tuple:
#   (sub_id, manifest_relpath, results_subdir, runner, gpu_mode, n_jobs_key,
#    description, extra_args)
#
# runner:      "sni" | "baselines"
# gpu_mode:    "auto" | "false" | "true"
# n_jobs_key:  "cpu" | "gpu"

ManifestStageDef = Tuple[str, str, str, str, str, str, str, Dict[str, str]]

MANIFEST_STAGES: List[ManifestStageDef] = [
    # --- Stage 1: SNI ---
    ("1a", "data/manifest_sni_v03_main.csv",
     "sni_v03_main", "sni", "auto", "cpu",
     "SNI v0.3 Main", {}),
    ("1b", "data/manifest_sni_v03_ablation_lambda.csv",
     "sni_v03_ablation_lambda", "sni", "auto", "cpu",
     "SNI v0.3 Ablation Lambda", {}),
    ("1c", "data/manifest_sni_mnar.csv",
     "sni_mnar", "sni", "auto", "cpu",
     "SNI MNAR", {}),
    ("1d", "data/manifest_options/manifest_sni_rate_sweep.csv",
     "sni_rate_sweep", "sni", "auto", "cpu",
     "SNI Rate Sweep", {}),
    # --- Stage 2: Classical baselines ---
    ("2a", "data/manifest_options/manifest_baselines_main_all.csv",
     "baselines_main", "baselines", "false", "cpu",
     "Baselines Main", {}),
    ("2b", "data/manifest_options/manifest_baselines_mnar_all.csv",
     "baselines_mnar", "baselines", "false", "cpu",
     "Baselines MNAR", {}),
    # --- Stage 3: Deep baselines ---
    ("3a", "data/manifest_baselines_deep.csv",
     "baselines_deep", "baselines", "auto", "gpu",
     "Baselines Deep (GAIN/MIWAE)", {}),
    ("3b", "data/manifest_baselines_new.csv",
     "baselines_new", "baselines", "auto", "gpu",
     "Baselines New (HyperImpute/TabCSDI)", {"use_timeout": "true"}),
]


# ---------------------------------------------------------------------------
# Build commands for manifest stages
# ---------------------------------------------------------------------------

def _build_manifest_cmd(
    stage: ManifestStageDef,
    outdir: Path,
    *,
    use_gpu: str,
    n_jobs_cpu: int,
    n_jobs_gpu: int,
    skip_existing: bool,
    timeout: int,
) -> List[str]:
    """Return the subprocess command list for a manifest stage."""
    sub_id, manifest_rel, results_sub, runner, gpu_mode, n_jobs_key, _desc, extras = stage

    results_dir = str(outdir / results_sub)

    # Determine GPU flag
    if gpu_mode == "auto":
        gpu_flag = use_gpu
    else:
        gpu_flag = gpu_mode

    # Determine parallelism
    n_jobs = n_jobs_cpu if n_jobs_key == "cpu" else n_jobs_gpu

    if runner == "sni":
        cmd = [
            sys.executable, "scripts/run_manifest_parallel.py",
            "--manifest", manifest_rel,
            "--outdir", results_dir,
            "--n-jobs", str(n_jobs),
            "--default-use-gpu", gpu_flag,
        ]
        if skip_existing:
            cmd.append("--skip-existing")

    elif runner == "baselines":
        cmd = [
            sys.executable, "scripts/run_manifest_baselines.py",
            "--manifest", manifest_rel,
            "--outdir", results_dir,
            "--n-jobs", str(n_jobs),
            "--default-use-gpu", gpu_flag,
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        # Add timeout for stages that need it (e.g. 3b with HyperImpute)
        if extras.get("use_timeout") == "true":
            cmd.extend(["--default-timeout", str(timeout)])
    else:
        raise ValueError(f"Unknown runner: {runner}")

    return cmd


# ---------------------------------------------------------------------------
# Script-based stage helpers
# ---------------------------------------------------------------------------

def _check_data_exists(rel_paths: List[str]) -> Optional[str]:
    """Return the first missing path (relative to PROJECT_ROOT), or None if all exist."""
    for rp in rel_paths:
        full = PROJECT_ROOT / rp
        if not full.exists():
            return rp
    return None


def _build_script_stages(
    outdir: Path,
    use_gpu: str,
) -> List[Tuple[str, str, List[str], List[str]]]:
    """Return a list of (sub_id, description, cmd, required_data_relpaths)."""

    stages: List[Tuple[str, str, List[str], List[str]]] = []

    # --- 4a: Sanity check ---
    stages.append((
        "4a", "Sanity Check (synth S5)",
        [
            sys.executable, "scripts/sanity_check_v2_s5.py",
            "--data-dir", "data/synth_s5",
            "--outdir", str(outdir / "sanity_s5"),
            "--settings", "linear_gaussian", "nonlinear_mixed", "interaction_xor",
            "--seeds", "2025", "2026", "2027", "2028", "2029",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--use-gpu", use_gpu,
            "--epochs", "50", "--num-heads", "4", "--emb-dim", "32",
            "--batch-size", "128", "--max-iters", "2",
        ],
        ["data/synth_s5"],
    ))

    # --- 5a: Ext1 Audit ALARM ---
    stages.append((
        "5a", "Ext1 Audit ALARM",
        [
            sys.executable, "ext1/scripts/exp1_audit_story_leakage.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "SpO2", "ALARM",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--audit-target", "ALARM",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--seed", "2026",
            "--outdir", str(outdir / "ext1" / "audit_mimic_alarm"),
            "--run-without-leak", "true",
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 5b: Ext1 Audit SBP (continuous proxy) ---
    stages.append((
        "5b", "Ext1 Audit SBP",
        [
            sys.executable, "ext1/scripts/exp1_audit_story_leakage.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "SpO2", "ALARM",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--audit-target", "SBP",
            "--leak-source", "SBP", "--leak-col-name", "SBP_LEAK",
            "--leak-noise-std", "0.5",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--seed", "2026",
            "--outdir", str(outdir / "ext1" / "audit_mimic_sbp"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 5c: Ext1 Downstream ---
    stages.append((
        "5c", "Ext1 Downstream NHANES",
        [
            sys.executable, "ext1/scripts/exp2_downstream_task_validation.py",
            "--input-complete", "data/NHANES_complete.csv",
            "--dataset-name", "NHANES",
            "--target-col", "metabolic_score",
            "--categorical-cols", "gender_std", "age_band",
            "--continuous-cols", "waist_circumference", "systolic_bp",
            "diastolic_bp", "triglycerides", "hdl_cholesterol",
            "fasting_glucose", "age", "bmi", "hba1c",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--mar-driver-cols", "age", "gender_std",
            "--fairness-col", "gender_std",
            "--imputers", "SNI", "MissForest", "MeanMode", "MICE",
            "HyperImpute", "TabCSDI",
            "--seeds", "1", "2", "3", "5", "8",
            "--outdir", str(outdir / "ext1" / "downstream_nhanes"),
            "--sni-use-gpu", "false",
            "--baseline-use-gpu", "false",
            "--save-missing", "true",
            "--save-imputed", "false",
        ],
        ["data/NHANES_complete.csv"],
    ))

    # --- 6a: Ext2 Exp3 Per-class ---
    stages.append((
        "6a", "Ext2 Per-class Categorical",
        [
            sys.executable, "ext2/scripts/exp3_per_class_categorical.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "ALARM",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE", "SpO2",
            "--mechanisms", "MAR",
            "--missing-rate", "0.30",
            "--mar-driver-cols", "HR", "SpO2",
            "--methods", "SNI", "MissForest", "MeanMode",
            "--seeds", "1", "2", "3", "5", "8",
            "--outdir", str(outdir / "ext2" / "table_S9_perclass_alarm"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 6b: Ext2 Exp4 SHAP ---
    stages.append((
        "6b", "Ext2 SHAP Comparison",
        [
            sys.executable, "ext2/scripts/exp4_shap_comparison.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "ALARM", "SpO2",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--mar-driver-cols", "HR", "PULSE",
            "--seed", "2026",
            "--targets", "ALARM", "SBP",
            "--top-k", "10",
            "--shap-max-eval", "512",
            "--outdir", str(outdir / "ext2" / "table_S7_shap_vs_D" / "MIMIC"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 6c: Ext2 Exp4b Attention rollout ---
    stages.append((
        "6c", "Ext2 Attention Rollout",
        [
            sys.executable, "ext2/scripts/exp4b_attention_rollout.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "SpO2", "ALARM",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--mar-driver-cols", "HR", "PULSE",
            "--seed", "2026",
            "--targets", "ALARM", "SBP",
            "--top-k", "5",
            "--outdir", str(outdir / "ext2" / "table_S7B_rollout" / "MIMIC"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 6d: Ext2 Exp4c D stability ---
    stages.append((
        "6d", "Ext2 D Stability",
        [
            sys.executable, "ext2/scripts/exp4c_d_stability.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC",
            "--categorical-vars", "SpO2", "ALARM",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--mar-driver-cols", "HR", "PULSE",
            "--missing-seed", "2026",
            "--seeds", "1", "2", "3", "5", "8",
            "--targets", "ALARM", "SBP",
            "--outdir", str(outdir / "ext2" / "table_S7C_stability" / "MIMIC"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    # --- 6e: Ext2 Exp5 Significance ---
    stages.append((
        "6e", "Ext2 Significance Tests",
        [
            sys.executable, "ext2/scripts/exp5_significance_tests.py",
            "--results-dir", str(outdir),
            "--datasets", "MIMIC", "eICU", "NHANES", "ComCri", "AutoMPG", "Concrete",
            "--mechanisms", "MCAR", "MAR",
            "--metrics", "NRMSE", "R2", "Spearman_rho", "Macro_F1",
            "--reference-method", "SNI",
            "--baselines", "MissForest", "MIWAE", "GAIN", "KNN", "MICE",
            "MeanMode", "HyperImpute", "TabCSDI",
            "--mode", "across_settings",
            "--alpha", "0.05",
            "--outdir", str(outdir / "ext2" / "significance"),
        ],
        [],  # No data check needed — uses existing results
    ))

    # --- 6f: Ext2 Exp6 MIMIC mortality ---
    stages.append((
        "6f", "Ext2 MIMIC Mortality",
        [
            sys.executable, "ext2/scripts/exp6_mimic_mortality_impute_predict.py",
            "--input-complete", "data/MIMIC_complete.csv",
            "--dataset-name", "MIMIC_alarm_predict",
            "--label-col", "ALARM",
            "--binarize-threshold", "34",
            "--categorical-vars", "SpO2",
            "--continuous-vars", "RESP", "ABP", "SBP", "DBP", "HR", "PULSE",
            "--mechanism", "MAR", "--missing-rate", "0.30",
            "--mar-driver-cols", "HR", "PULSE",
            "--imputers", "SNI", "MissForest", "MeanMode", "HyperImpute", "TabCSDI",
            "--models", "LR", "XGB",
            "--seeds", "1", "2", "3", "5", "8",
            "--outdir", str(outdir / "ext2" / "table_VI_mimic_alarm"),
            "--use-gpu", "false",
        ],
        ["data/MIMIC_complete.csv"],
    ))

    return stages


# ---------------------------------------------------------------------------
# Stage 7: Aggregation, merge, tables, figures
# ---------------------------------------------------------------------------

AGG_RESULT_DIRS = [
    "sni_v03_main",
    "sni_v03_ablation_lambda",
    "sni_mnar",
    "sni_rate_sweep",
    "baselines_main",
    "baselines_mnar",
    "baselines_deep",
    "baselines_new",
]


def _run_stage7(
    outdir: Path,
    *,
    dry_run: bool,
) -> List[Tuple[str, str, str]]:
    """Run Stage 7: aggregate, merge, latex tables, and figures.

    Returns a list of (sub_id, description, status) for each sub-step.
    """
    results: List[Tuple[str, str, str]] = []

    # 7a -- Aggregate each result directory
    agg_csvs: List[str] = []
    for rdir in AGG_RESULT_DIRS:
        results_path = outdir / rdir
        agg_path = outdir / f"agg_{rdir}"
        desc = f"Aggregate {rdir}"

        if not results_path.exists():
            print(f"[SKIP] {desc}: results dir not found ({results_path})")
            results.append(("7a", desc, STATUS_SKIP))
            continue

        cmd = [
            sys.executable, "-m", "scripts.aggregate_results",
            "--results-root", str(results_path),
            "--outdir", str(agg_path),
        ]
        rc = _run(cmd, desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        if rc == 0 or dry_run:
            agg_csv = agg_path / "summary_agg.csv"
            agg_csvs.append(str(agg_csv))
            results.append(("7a", desc, STATUS_DRY if dry_run else STATUS_DONE))
        else:
            results.append(("7a", desc, STATUS_FAIL))

    # 7b -- Merge all summary_agg.csv files
    merged_dir = outdir / "merged"
    merged_csv = merged_dir / "merged_summary_agg.csv"
    merge_desc = "Merge summaries"

    # Filter to CSVs that actually exist (unless dry-run)
    if not dry_run:
        agg_csvs = [p for p in agg_csvs if Path(p).exists()]

    if agg_csvs:
        cmd = [
            sys.executable, "-m", "scripts.merge_summaries",
            "--inputs", *agg_csvs,
            "--outdir", str(merged_dir),
        ]
        rc = _run(cmd, merge_desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        results.append(("7b", merge_desc, STATUS_DRY if dry_run else (STATUS_DONE if rc == 0 else STATUS_FAIL)))
    else:
        print(f"[SKIP] {merge_desc}: no aggregated CSVs available")
        results.append(("7b", merge_desc, STATUS_SKIP))

    # 7c -- LaTeX tables (main profile)
    tables_dir = outdir / "tables"
    tables_desc = "LaTeX tables (main)"
    if dry_run or merged_csv.exists():
        cmd = [
            sys.executable, "-m", "scripts.make_latex_table",
            "--summary-csv", str(merged_csv),
            "--outdir", str(tables_dir),
            "--profile", "configs/paper_profiles.yaml:main",
        ]
        rc = _run(cmd, tables_desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        results.append(("7c", tables_desc, STATUS_DRY if dry_run else (STATUS_DONE if rc == 0 else STATUS_FAIL)))
    else:
        print(f"[SKIP] {tables_desc}: merged CSV not found")
        results.append(("7c", tables_desc, STATUS_SKIP))

    # 7d -- LaTeX tables (supplement profile)
    tables_supp_dir = outdir / "tables_supplement"
    tables_supp_desc = "LaTeX tables (supplement)"
    if dry_run or merged_csv.exists():
        cmd = [
            sys.executable, "-m", "scripts.make_latex_table",
            "--summary-csv", str(merged_csv),
            "--outdir", str(tables_supp_dir),
            "--profile", "configs/paper_profiles.yaml:supplement",
        ]
        rc = _run(cmd, tables_supp_desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        results.append(("7d", tables_supp_desc, STATUS_DRY if dry_run else (STATUS_DONE if rc == 0 else STATUS_FAIL)))
    else:
        print(f"[SKIP] {tables_supp_desc}: merged CSV not found")
        results.append(("7d", tables_supp_desc, STATUS_SKIP))

    # 7e -- Figures (main profile)
    figures_dir = outdir / "figures"
    figures_desc = "Figures (main)"
    if dry_run or merged_csv.exists():
        cmd = [
            sys.executable, "-m", "scripts.viz_make_figures",
            "--summary-csv", str(merged_csv),
            "--outdir", str(figures_dir),
            "--profile", "configs/paper_profiles.yaml:main",
        ]
        rc = _run(cmd, figures_desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        results.append(("7e", figures_desc, STATUS_DRY if dry_run else (STATUS_DONE if rc == 0 else STATUS_FAIL)))
    else:
        print(f"[SKIP] {figures_desc}: merged CSV not found")
        results.append(("7e", figures_desc, STATUS_SKIP))

    # 7f -- Figures (supplement profile)
    figures_supp_dir = outdir / "figures_supplement"
    figures_supp_desc = "Figures (supplement)"
    if dry_run or merged_csv.exists():
        cmd = [
            sys.executable, "-m", "scripts.viz_make_figures",
            "--summary-csv", str(merged_csv),
            "--outdir", str(figures_supp_dir),
            "--profile", "configs/paper_profiles.yaml:supplement",
        ]
        rc = _run(cmd, figures_supp_desc, dry_run=dry_run, cwd=str(PROJECT_ROOT))
        results.append(("7f", figures_supp_desc, STATUS_DRY if dry_run else (STATUS_DONE if rc == 0 else STATUS_FAIL)))
    else:
        print(f"[SKIP] {figures_supp_desc}: merged CSV not found")
        results.append(("7f", figures_supp_desc, STATUS_SKIP))

    return results


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary(
    records: List[Tuple[str, str, str]],
    elapsed_sec: float,
) -> None:
    """Print a formatted summary table of all stage results."""
    hours = int(elapsed_sec // 3600)
    minutes = int((elapsed_sec % 3600) // 60)

    print()
    print("=" * 64)
    print("  RUN ALL EXPERIMENTS -- SUMMARY")
    print("=" * 64)
    print(f"  {'Stage':<7}{'Description':<40}{'Status'}")
    print(f"  {'-----':<7}{'-----------------------------------':<40}{'-------'}")

    for sub_id, description, status in records:
        icon = _STATUS_ICON.get(status, "?")
        print(f"  {sub_id:<7}{description:<40}{icon} {status}")

    if hours > 0:
        print(f"\n  Total elapsed: {hours}h {minutes}m")
    else:
        print(f"\n  Total elapsed: {minutes}m")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ALL paper experiments in stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--outdir", type=str, default="results_all",
        help="Root output directory (default: results_all).",
    )
    ap.add_argument(
        "--stages", nargs="+", default=["1", "2", "3", "4", "5", "6", "7"],
        help="Which stages to run (e.g. --stages 1 2 7). Default: all.",
    )
    ap.add_argument(
        "--n-jobs-cpu", type=int, default=-1,
        help="Parallel workers for CPU manifests (default: -1 = all cores).",
    )
    ap.add_argument(
        "--n-jobs-gpu", type=int, default=1,
        help="Parallel workers for GPU manifests (default: 1).",
    )
    ap.add_argument(
        "--use-gpu", type=str, default="true", choices=["true", "false"],
        help="Auto GPU flag for stages that support it (default: true).",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="Skip completed experiments (pass --skip-existing to runners).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    ap.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-run timeout for HyperImpute in seconds (default: 1800).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_stages = set(args.stages)
    use_gpu = args.use_gpu
    dry_run = args.dry_run

    t_global_start = time.time()

    # ------------------------------------------------------------------
    # Environment snapshot
    # ------------------------------------------------------------------
    _write_environment_snapshot(outdir)

    # Track results for the final summary
    all_results: List[Tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Stages 1-3: Manifest-based runs
    # ------------------------------------------------------------------
    for stage_def in MANIFEST_STAGES:
        sub_id = stage_def[0]
        stage_number = sub_id[0]  # e.g. "1" from "1a"

        if stage_number not in selected_stages:
            continue

        manifest_rel = stage_def[1]
        description = stage_def[6]

        # Check manifest file existence
        manifest_path = PROJECT_ROOT / manifest_rel
        if not manifest_path.exists():
            print(f"\n[SKIP] Stage {sub_id} ({description}): manifest not found ({manifest_rel})")
            all_results.append((sub_id, description, STATUS_SKIP))
            continue

        cmd = _build_manifest_cmd(
            stage_def,
            outdir,
            use_gpu=use_gpu,
            n_jobs_cpu=args.n_jobs_cpu,
            n_jobs_gpu=args.n_jobs_gpu,
            skip_existing=args.skip_existing,
            timeout=args.timeout,
        )
        rc = _run(cmd, f"Stage {sub_id}: {description}", dry_run=dry_run, cwd=str(PROJECT_ROOT))

        if dry_run:
            all_results.append((sub_id, description, STATUS_DRY))
        elif rc == 0:
            all_results.append((sub_id, description, STATUS_DONE))
        else:
            all_results.append((sub_id, description, STATUS_FAIL))

    # ------------------------------------------------------------------
    # Stages 4-6: Script-based runs
    # ------------------------------------------------------------------
    script_stages = _build_script_stages(outdir, use_gpu)

    for sub_id, description, cmd, required_data in script_stages:
        stage_number = sub_id[0]

        if stage_number not in selected_stages:
            continue

        # Check required data files
        if required_data:
            missing = _check_data_exists(required_data)
            if missing is not None:
                print(f"\n[SKIP] Stage {sub_id} ({description}): required data not found ({missing})")
                all_results.append((sub_id, description, STATUS_SKIP))
                continue

        rc = _run(cmd, f"Stage {sub_id}: {description}", dry_run=dry_run, cwd=str(PROJECT_ROOT))

        if dry_run:
            all_results.append((sub_id, description, STATUS_DRY))
        elif rc == 0:
            all_results.append((sub_id, description, STATUS_DONE))
        else:
            all_results.append((sub_id, description, STATUS_FAIL))

    # ------------------------------------------------------------------
    # Stage 7: Aggregation + Tables + Figures
    # ------------------------------------------------------------------
    if "7" in selected_stages:
        stage7_results = _run_stage7(outdir, dry_run=dry_run)
        all_results.extend(stage7_results)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_global_start
    _print_summary(all_results, elapsed)

    # Exit with non-zero if any stage failed
    n_fail = sum(1 for _, _, s in all_results if s == STATUS_FAIL)
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
