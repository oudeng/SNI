#!/usr/bin/env python3
"""Run baseline imputers from a CSV manifest.

This script mirrors :mod:`scripts/run_manifest_parallel.py`, but runs **baseline**
methods (MeanMode/KNN/MICE/MissForest/GAIN/MIWAE/HyperImpute/TabCSDI) instead of SNI.

Key goals
---------
- Same data loader / dtype casting logic as SNI (read *_complete.csv schema and
  cast *_missing.csv accordingly).
- Same seeds as provided in the manifest.
- Parallel execution via joblib.
- Store outputs in a per-experiment directory under ``results_baselines``.

Usage
-----
    python scripts/run_manifest_baselines.py \
        --manifest data/manifest_baselines.csv \
        --outdir results_baselines \
        --n-jobs 8 \
        --skip-existing

    # For HyperImpute / TabCSDI (heavier methods), consider lower parallelism:
    python scripts/run_manifest_baselines.py \
        --manifest data/manifest_baselines_new.csv \
        --outdir results_baselines_new \
        --n-jobs 1 \
        --default-timeout 1800 \
        --skip-existing

Notes
-----
- Deep baselines (GAIN/MIWAE/TabCSDI) can optionally use GPU. If you have a
  single GPU, running many of them in parallel may cause OOM. In that case,
  either set ``--n-jobs 1`` for those runs, or keep GPU off (default).
- TabCSDI is memory-intensive during training; ``--n-jobs 1`` is recommended.
- HyperImpute uses AutoML search internally and can be slow; the
  ``--default-timeout`` flag (default 1800s) controls the per-run timeout.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=r".*correlation coefficient is not defined.*")
warnings.filterwarnings("ignore", message=r".*A single label was found.*")

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Conservative thread defaults to avoid BLAS/OMP oversubscription in parallel runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""

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


# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import build_baseline_imputer
from SNI_v0_2.dataio import cast_dataframe_to_schema, load_complete_and_missing
from SNI_v0_2.metrics import evaluate_imputation

import warnings
from sklearn.exceptions import ConvergenceWarning

# Some baselines may internally rely on sklearn iterative estimators.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message=r".*Early stopping criterion not reached.*",
)

# pandas sometimes emits a RuntimeWarning when casting NaNs to strings when
# reading/writing manifests. It is harmless for our use case.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered in cast.*",
)


def _split_vars(s: Any) -> List[str]:
    """Parse a comma/space separated variable list from a manifest cell."""
    if s is None:
        return []
    try:
        if pd.isna(s):
            return []
    except Exception:
        pass

    tokens: List[str] = []
    for part in str(s).replace(";", ",").replace(" ", ",").split(","):
        t = part.strip()
        if not t:
            continue
        if t.lower() == "nan":
            continue
        tokens.append(t)
    return tokens


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _collect_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
    """Collect optional baseline hyperparameters from a manifest row.

    The baseline registry will ignore unknown keys and NaNs.
    """
    reserved = {
        "exp_id",
        "input_complete",
        "input_missing",
        "method",
        "seed",
        "categorical_vars",
        "continuous_vars",
        "use_gpu",
    }

    kwargs: Dict[str, Any] = {}
    for k, v in row.items():
        if k in reserved:
            continue
        kwargs[k] = v
    return kwargs


def _format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m}m{s}s"


def run_one_experiment(
    row: Dict[str, Any],
    outdir_root: Path,
    *,
    default_use_gpu: bool,
    default_timeout: int = 1800,
    skip_existing: bool,
    verbose: bool,
    show_progress: bool = False,  # New: show per-experiment progress
    exp_index: int = 0,           # New: experiment index
    total_exps: int = 1,          # New: total experiments
) -> Dict[str, Any]:
    """Run a single baseline experiment and write outputs."""

    exp_id = str(row.get("exp_id"))
    outdir = outdir_root / exp_id

    metrics_path = outdir / "metrics_summary.json"
    if skip_existing and metrics_path.exists():
        if show_progress:
            print(f"  [{exp_index+1}/{total_exps}] {exp_id}: skipped (exists)")
        return {"exp_id": exp_id, "status": "skipped"}

    outdir.mkdir(parents=True, exist_ok=True)

    # Always capture baseline stdout/stderr to a per-run log.
    log_path = outdir / "run.log"
    err_path = outdir / "error.log"

    t0 = time.time()

    try:
        method = str(row.get("method")).strip()
        seed = int(row.get("seed"))

        categorical_vars = _split_vars(row.get("categorical_vars"))
        continuous_vars = _split_vars(row.get("continuous_vars"))

        # Decide GPU usage.
        # - Only GAIN/MIWAE/TabCSDI support GPU in our wrappers.
        # - A per-row 'use_gpu' value overrides the script default.
        use_gpu_row = row.get("use_gpu")
        use_gpu = _as_bool(use_gpu_row, default=default_use_gpu)
        if method not in {"GAIN", "MIWAE", "TabCSDI"}:
            use_gpu = False

        # Show progress: experiment started
        if show_progress:
            gpu_str = " [GPU]" if use_gpu else ""
            print(f"  [{exp_index+1}/{total_exps}] {exp_id}: {method}{gpu_str} running...", end="", flush=True)

        # Load data (schema from complete; cast missing to match).
        # We reuse the same loader as SNI so that:
        #   (i) *_missing.csv gets dtype-cast to match *_complete.csv,
        #   (ii) if an "ID" column exists, rows are aligned deterministically.
        X_complete, X_missing, schema = load_complete_and_missing(
            input_complete=str(row["input_complete"]),
            input_missing=str(row["input_missing"]),
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
        )

        # Build baseline imputer.
        extra_kwargs = _collect_kwargs(row)

        # Inject default timeout for HyperImpute if not specified in manifest.
        if method == "HyperImpute":
            if "timeout" not in extra_kwargs or pd.isna(extra_kwargs.get("timeout")):
                extra_kwargs["timeout"] = default_timeout

        imputer = build_baseline_imputer(
            method,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            seed=seed,
            use_gpu=use_gpu,
            **extra_kwargs,
        )

        # Reduce torch CPU thread usage inside each worker.
        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        with open(log_path, "w", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            if verbose:
                print(f"[RUN] {exp_id} | method={method} | seed={seed} | use_gpu={use_gpu}")
            X_imputed = imputer.impute(X_complete, X_missing)

        # Enforce schema dtypes (Int64/category/etc) for clean CSV outputs.
        X_imputed = cast_dataframe_to_schema(X_imputed, schema)

        # Evaluate on the missing positions.
        eval_res = evaluate_imputation(
            X_imputed,
            X_complete,
            X_missing,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
        )

        runtime_sec = float(time.time() - t0)

        # Save outputs
        X_imputed.to_csv(outdir / "imputed.csv", index=True)
        eval_res.per_feature.to_csv(outdir / "metrics_per_feature.csv", index=False)

        summary = dict(eval_res.summary)
        summary.update(
            {
                "exp_id": exp_id,
                "method": method,
                "seed": seed,
                "use_gpu": use_gpu,
                "runtime_sec": runtime_sec,
            }
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        run_cfg = {
            "exp_id": exp_id,
            "method": method,
            "seed": seed,
            "use_gpu": use_gpu,
            "input_complete": row["input_complete"],
            "input_missing": row["input_missing"],
            "categorical_vars": categorical_vars,
            "continuous_vars": continuous_vars,
            "extra_kwargs": {k: v for k, v in extra_kwargs.items() if k is not None},
        }
        with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)

        # Show progress: experiment completed
        if show_progress:
            # Extract key metrics for display
            rmse = summary.get("RMSE_overall", summary.get("RMSE_cont", 0))
            acc = summary.get("Accuracy_overall", summary.get("Accuracy_cat", 0))
            if rmse and acc:
                print(f" done ({_format_time(runtime_sec)}) RMSE={rmse:.4f} Acc={acc:.3f}")
            elif rmse:
                print(f" done ({_format_time(runtime_sec)}) RMSE={rmse:.4f}")
            elif acc:
                print(f" done ({_format_time(runtime_sec)}) Acc={acc:.3f}")
            else:
                print(f" done ({_format_time(runtime_sec)})")

        return {"exp_id": exp_id, "status": "ok", "runtime_sec": runtime_sec}

    except Exception as e:
        runtime_sec = float(time.time() - t0)
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"[ERROR] {exp_id}: {str(e)}\n")
            f.write(traceback.format_exc())
        
        # Show progress: experiment failed
        if show_progress:
            print(f" FAILED ({_format_time(runtime_sec)}): {str(e)[:50]}")
        
        return {"exp_id": exp_id, "status": "fail", "runtime_sec": runtime_sec, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV manifest file")
    ap.add_argument("--outdir", default="results_baselines", help="Output directory")
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (joblib)")
    ap.add_argument("--row-start", type=int, default=0, help="Start row (inclusive)")
    ap.add_argument("--row-end", type=int, default=-1, help="End row (exclusive); -1 means all")
    ap.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing metrics_summary.json")
    ap.add_argument("--default-use-gpu", type=str, default="false", help="Default GPU usage (true/false) for deep baselines")
    ap.add_argument("--default-timeout", type=int, default=1800, help="Default per-run timeout in seconds for HyperImpute (default: 1800)")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging inside per-run log")

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    default_use_gpu = _as_bool(args.default_use_gpu, default=False)

    df = pd.read_csv(manifest_path)

    # Slice rows
    row_end = args.row_end if args.row_end >= 0 else len(df)
    df = df.iloc[args.row_start:row_end].reset_index(drop=True)

    required = {"exp_id", "input_complete", "input_missing", "method", "seed", "categorical_vars", "continuous_vars"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    rows = df.to_dict(orient="records")

    default_timeout = int(args.default_timeout)

    print(f"[INFO] Starting {len(rows)} baseline experiments with {args.n_jobs} workers")
    print(f"[INFO] Output directory: {outdir_root}")
    print(f"[INFO] Default GPU for deep baselines: {default_use_gpu}")
    print(f"[INFO] Default timeout for HyperImpute: {default_timeout}s")

    t0_all = time.time()

    # Determine execution mode based on n_jobs and GPU usage
    # For n_jobs=1 or GPU mode: use sequential execution with per-experiment progress
    # For n_jobs>1 CPU mode: use parallel execution with tqdm_joblib
    use_sequential = (args.n_jobs == 1) or (default_use_gpu and args.n_jobs <= 2)
    
    if use_sequential:
        # Sequential execution with real-time per-experiment progress
        print(f"[INFO] Running sequentially with per-experiment progress...")
        print("-" * 60)
        results = []
        for i, row in enumerate(rows):
            result = run_one_experiment(
                row,
                outdir_root,
                default_use_gpu=default_use_gpu,
                default_timeout=default_timeout,
                skip_existing=args.skip_existing,
                verbose=args.verbose,
                show_progress=True,
                exp_index=i,
                total_exps=len(rows),
            )
            results.append(result)
    else:
        # Parallel execution with tqdm progress bar
        delayed_jobs = [
            delayed(run_one_experiment)(
                r,
                outdir_root,
                default_use_gpu=default_use_gpu,
                default_timeout=default_timeout,
                skip_existing=args.skip_existing,
                verbose=args.verbose,
                show_progress=False,
                exp_index=i,
                total_exps=len(rows),
            )
            for i, r in enumerate(rows)
        ]

        if args.no_progress:
            results = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed_jobs)
        else:
            with tqdm_joblib(tqdm(total=len(rows), desc="Baselines", unit="exp")):
                results = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed_jobs)

    dt_all = float(time.time() - t0_all)

    # Write a summary CSV
    summary_rows: List[Dict[str, Any]] = []
    for r, info in zip(rows, results):
        rec = {
            "exp_id": r.get("exp_id"),
            "method": r.get("method"),
            "seed": r.get("seed"),
            "status": info.get("status"),
            "runtime_sec": info.get("runtime_sec"),
        }
        if info.get("status") == "fail":
            rec["error"] = info.get("error")
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir_root / "summary_all_runs.csv", index=False)

    ok = int((summary_df["status"] == "ok").sum())
    fail = int((summary_df["status"] == "fail").sum())
    skipped = int((summary_df["status"] == "skipped").sum())

    print("-" * 60)
    print(f"[DONE] Completed in {_format_time(dt_all)}")
    print(f"[DONE] Success: {ok}, Failed: {fail}, Skipped: {skipped}")
    print(f"[DONE] Wrote {outdir_root / 'summary_all_runs.csv'}")


if __name__ == "__main__":
    main()