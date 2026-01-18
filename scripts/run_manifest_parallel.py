#!/usr/bin/env python3
"""
Parallel version of run_manifest.py for SNI experiments.

This script runs multiple experiments in parallel using joblib, significantly
reducing total runtime on multi-core systems. Each experiment is fully independent
and can run on separate CPU cores, with optional GPU sharing.

Features:
    - Parallel execution with configurable number of workers
    - tqdm progress bar with ETA and speed estimation
    - Skip already completed experiments
    - Partial run support (row range filtering)

Usage:
    # Run with 8 parallel workers (recommended for 32-core system with 1 GPU)
    python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8

    # Run with all available cores (CPU-only mode)
    python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs -1 --default-use-gpu false

    # Run specific rows only (for debugging or partial runs)
    python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 4 --row-start 0 --row-end 50

    # Disable progress bar (use joblib verbose output instead)
    python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8 --no-progress --verbose 10

tqdm: # 默认启用进度条
python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8
# 禁用进度条，使用 joblib verbose
python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8 --no-progress --verbose 10   
"""
from __future__ import annotations

import argparse
import json
import os

# Conservative thread defaults to avoid BLAS/OMP oversubscription in parallel runs.
# Users can override by exporting these variables before launching.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import contextlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.
    
    This allows tqdm to track progress of joblib parallel execution.
    
    Usage:
        with tqdm_joblib(tqdm(desc="Processing", total=100)) as pbar:
            Parallel(n_jobs=4)(delayed(func)(i) for i in range(100))
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Add parent directory to path for SNI_v0_2 imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SNI_v0_2 import SNIImputer
from SNI_v0_2.dataio import cast_dataframe_to_schema, load_complete_and_missing
from SNI_v0_2.imputer import SNIConfig
from SNI_v0_2.metrics import evaluate_imputation

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message=r".*\[IterativeImputer\] Early stopping criterion not reached.*"
)

def _split_vars(s: Any) -> List[str]:
    """Parse a comma/space separated variable list from a manifest cell.

    Notes
    -----
    * When the manifest is loaded by ``pandas.read_csv``, empty cells are often
      parsed as ``NaN``. Casting that value to ``str`` yields the literal token
      ``'nan'``, which then propagates into the variable list and can trigger
      ``KeyError: ['nan'] not in index`` when selecting DataFrame columns.
    * We also defensively strip leading/trailing whitespace from variable names
      to avoid surprises such as ``'ConcreteCS '`` vs ``'ConcreteCS'``.
    """
    if s is None:
        return []
    # pandas may parse empty cells as NaN (float) in CSV manifests.
    try:
        if pd.isna(s):
            return []
    except Exception:
        pass

    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []

    raw: List[str] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk == "" or chunk.lower() == "nan":
            continue
        raw.extend([p.strip() for p in chunk.split() if p.strip()])

    # de-duplicate while preserving order
    out: List[str] = []
    seen = set()
    for v in raw:
        if v.lower() == "nan":
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _as_bool(x) -> bool:
    """Convert various truthy values to bool."""
    return str(x).lower() in {"1", "true", "yes", "y"}


def run_single_experiment(
    row: pd.Series,
    out_root: Path,
    default_use_gpu: bool,
    exp_index: int,
    total_exps: int,
    torch_num_threads: int = 1,
) -> Tuple[Optional[Dict[str, Any]], str, float]:
    """
    Run a single experiment from manifest row.
    
    Returns:
        Tuple of (summary_dict or None, status_message, runtime_seconds)
    """
    exp_id = str(row["exp_id"])
    outdir = out_root / exp_id
    t0 = time.time()
    
    try:
        outdir.mkdir(parents=True, exist_ok=True)

        # Per-worker CPU thread control (helps avoid oversubscription in parallel runs)
        try:
            import torch
            if torch_num_threads is not None and int(torch_num_threads) > 0:
                torch.set_num_threads(int(torch_num_threads))
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
        except Exception:
            pass


        cat_vars = _split_vars(row["categorical_vars"])
        cont_vars = _split_vars(row["continuous_vars"])
        all_vars = cat_vars + cont_vars

        # Load data with stable schema
        X_complete, X_missing, schema = load_complete_and_missing(
            input_complete=str(row["input_complete"]),
            input_missing=str(row["input_missing"]),
            categorical_vars=cat_vars,
            continuous_vars=cont_vars,
        )

        mask_df = X_missing.isna()
        if "mask_file" in row.index and pd.notna(row.get("mask_file", None)):
            mask_df = pd.read_csv(row["mask_file"])[all_vars].astype(int).astype(bool)

        # Build config from manifest row
        cfg = SNIConfig(
            alpha0=float(row.get("alpha0", 1.0)),
            gamma=float(row.get("gamma", 0.9)),
            max_iters=int(row.get("max_iters", 3)),
            tol=float(row.get("tol", 1e-4)),
            use_stat_refine=_as_bool(row.get("use_stat_refine", True)),
            mask_fraction=float(row.get("mask_fraction", 0.15)),
            hidden_dims=tuple(int(x) for x in str(row.get("hidden_dims", "64,32")).split(",") if str(x).strip()),
            emb_dim=int(row.get("emb_dim", 32)),
            num_heads=int(row.get("num_heads", 4)),
            lr=float(row.get("lr", 1e-3)),
            epochs=int(row.get("epochs", 80)),
            batch_size=int(row.get("batch_size", 64)),
            variant=str(row["variant"]),
            hard_prior_lambda=float(row.get("hard_prior_lambda", 10.0)),
            seed=int(row["seed"]),
            use_gpu=_as_bool(row.get("use_gpu", default_use_gpu)),
        )

        # Handle SNI+KNN variant
        train_variant = cfg.variant if cfg.variant != "SNI+KNN" else "SNI"
        cfg.variant = train_variant

        imputer = SNIImputer(cat_vars, cont_vars, config=cfg)

        t_train_start = time.time()
        X_imp = imputer.impute(X_missing=X_missing, X_complete=X_complete, mask_df=mask_df)
        runtime = float(time.time() - t_train_start)

        # Ensure clean output dtypes
        X_imp = cast_dataframe_to_schema(X_imp, schema)

        # Evaluate
        eval_res = evaluate_imputation(
            X_imputed=X_imp,
            X_complete=X_complete,
            X_missing=X_missing,
            categorical_vars=cat_vars,
            continuous_vars=cont_vars,
            mask_df=mask_df,
        )

        # Save results
        X_imp.to_csv(outdir / "imputed.csv", index=False)
        eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)

        summary = dict(eval_res.summary)
        summary.update({
            "exp_id": exp_id,
            "variant": str(row["variant"]),
            "seed": int(row["seed"]),
            "runtime_sec": runtime,
        })
        pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False)
        (outdir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", 
            encoding="utf-8"
        )

        # Dependency matrix
        D = imputer.compute_dependency_matrix()
        D.to_csv(outdir / "dependency_matrix.csv", index=True)

        total_time = time.time() - t0
        status = f"[OK] ({exp_index+1}/{total_exps}) {exp_id} done in {total_time:.1f}s -> {outdir}"
        return summary, status, total_time

    except Exception as e:
        total_time = time.time() - t0
        error_msg = f"[FAIL] ({exp_index+1}/{total_exps}) {exp_id} failed after {total_time:.1f}s: {str(e)}"
        
        # Save error log
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "error.log").write_text(
                f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}",
                encoding="utf-8"
            )
        except:
            pass
        
        return None, error_msg, total_time


def main():
    ap = argparse.ArgumentParser(
        description="Run SNI experiments in parallel from manifest CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 8 parallel workers (shows tqdm progress bar by default)
  python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8

  # Run with all CPU cores (no GPU)
  python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs -1 --default-use-gpu false

  # Run rows 100-199 only
  python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 4 --row-start 100 --row-end 200

  # Disable progress bar, use joblib verbose output instead
  python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8 --no-progress --verbose 10

  # Skip already completed experiments (useful for resuming interrupted runs)
  python scripts/run_manifest_parallel.py --manifest data/manifest.csv --outdir results --n-jobs 8 --skip-existing
        """
    )
    ap.add_argument("--manifest", type=str, required=True, help="CSV manifest (one experiment per row).")
    ap.add_argument("--outdir", type=str, required=True, help="Root output directory.")
    ap.add_argument("--default-use-gpu", type=str, default="true", choices=["true", "false"],
                    help="Default GPU usage if not specified in manifest (default: true).")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="Number of parallel workers. Use -1 for all cores. (default: 4)")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "multiprocessing", "threading"],
                    help="Joblib parallel backend (default: loky).")
    ap.add_argument("--torch-num-threads", type=int, default=1,
                    help="Set torch.set_num_threads(...) in each worker to reduce CPU oversubscription (default: 1).")
    ap.add_argument("--row-start", type=int, default=None,
                    help="Start row index (0-based, inclusive). For partial runs.")
    ap.add_argument("--row-end", type=int, default=None,
                    help="End row index (0-based, exclusive). For partial runs.")
    ap.add_argument("--verbose", type=int, default=5,
                    help="Joblib verbosity level (0=silent, 10=debug). Default: 5")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip experiments that already have metrics_summary.json")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable tqdm progress bar (use joblib verbose output instead)")
    ap.add_argument("--progress-interval", type=float, default=0.5,
                    help="Progress bar update interval in seconds (default: 0.5)")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    # Validate required columns
    required = {"exp_id", "input_complete", "input_missing", "variant", "seed", "categorical_vars", "continuous_vars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    # Apply row filtering
    if args.row_start is not None or args.row_end is not None:
        start = args.row_start or 0
        end = args.row_end or len(df)
        df = df.iloc[start:end].reset_index(drop=True)
        print(f"[INFO] Running rows {start} to {end} (total: {len(df)} experiments)")

    # Skip existing experiments if requested
    if args.skip_existing:
        skip_count = 0
        rows_to_run = []
        for i, row in df.iterrows():
            exp_id = str(row["exp_id"])
            summary_file = out_root / exp_id / "metrics_summary.json"
            if summary_file.exists():
                skip_count += 1
            else:
                rows_to_run.append(row)
        if skip_count > 0:
            print(f"[INFO] Skipping {skip_count} already completed experiments")
            df = pd.DataFrame(rows_to_run).reset_index(drop=True)

    total_exps = len(df)
    if total_exps == 0:
        print("[INFO] No experiments to run.")
        return

    default_use_gpu = args.default_use_gpu == "true"
    
    print(f"[INFO] Starting {total_exps} experiments with {args.n_jobs} parallel workers")
    print(f"[INFO] Backend: {args.backend}, GPU default: {default_use_gpu}")
    print(f"[INFO] Output directory: {out_root}")
    print("-" * 60)

    t_total_start = time.time()

    # Prepare the delayed tasks
    tasks = [
        delayed(run_single_experiment)(
            row=row,
            out_root=out_root,
            default_use_gpu=default_use_gpu,
            exp_index=i,
            total_exps=total_exps,
            torch_num_threads=args.torch_num_threads,
        )
        for i, row in df.iterrows()
    ]

    # Run experiments in parallel with optional progress bar
    if args.no_progress:
        # Use joblib's built-in verbose output
        results = Parallel(
            n_jobs=args.n_jobs,
            backend=args.backend,
            verbose=args.verbose,
        )(tasks)
    else:
        # Use tqdm progress bar (more user-friendly)
        with tqdm_joblib(tqdm(
            desc="🔬 Experiments",
            total=total_exps,
            unit="exp",
            dynamic_ncols=True,
            mininterval=args.progress_interval,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )):
            results = Parallel(
                n_jobs=args.n_jobs,
                backend=args.backend,
                verbose=0,  # Suppress joblib output when using tqdm
            )(tasks)

    # Collect results
    summaries = []
    success_count = 0
    fail_count = 0
    
    for summary, status, runtime in results:
        print(status)
        if summary is not None:
            summaries.append(summary)
            success_count += 1
        else:
            fail_count += 1

    total_time = time.time() - t_total_start

    print("-" * 60)
    print(f"[DONE] Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[DONE] Success: {success_count}, Failed: {fail_count}")

    # Save global summary
    if len(summaries) > 0:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(out_root / "summary_all_runs.csv", index=False)
        print(f"[DONE] Wrote {out_root / 'summary_all_runs.csv'}")

    # Save run metadata
    run_meta = {
        "manifest": args.manifest,
        "n_jobs": args.n_jobs,
        "backend": args.backend,
        "total_experiments": total_exps,
        "success": success_count,
        "failed": fail_count,
        "total_time_sec": total_time,
        "row_start": args.row_start,
        "row_end": args.row_end,
    }
    (out_root / "parallel_run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()