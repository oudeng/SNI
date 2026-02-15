#!/usr/bin/env python3
"""
Generate shell scripts to run manifest experiments in parallel batches.

This script splits a manifest into N parts and generates shell commands
that can be run in separate terminals or submitted to a job scheduler.

Usage:
    # Generate 8 batch scripts
    python scripts/split_manifest_runner.py --manifest data/manifest.csv --outdir results --n-splits 8

    # Then run in separate terminals:
    bash results/_parallel_scripts/run_batch_0.sh
    bash results/_parallel_scripts/run_batch_1.sh
    ...

    # Or use GNU parallel:
    ls results/_parallel_scripts/run_batch_*.sh | parallel -j 8 bash {}
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message=r".*\[IterativeImputer\] Early stopping criterion not reached.*"
)

def main():
    ap = argparse.ArgumentParser(
        description="Split manifest into parallel batch scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=str, required=True, help="CSV manifest file.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--n-splits", type=int, default=8, help="Number of parallel batches (default: 8).")
    ap.add_argument("--use-gpu", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    total = len(df)
    batch_size = math.ceil(total / args.n_splits)

    out_root = Path(args.outdir)
    script_dir = out_root / "_parallel_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual manifest splits
    manifest_dir = script_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)

    all_scripts = []
    
    for i in range(args.n_splits):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        
        if start >= total:
            break
            
        # Save split manifest
        df_split = df.iloc[start:end]
        manifest_path = manifest_dir / f"manifest_batch_{i}.csv"
        df_split.to_csv(manifest_path, index=False)
        
        # Generate shell script
        script_content = f"""#!/bin/bash
# Batch {i}: rows {start}-{end-1} ({end-start} experiments)
# Generated from: {args.manifest}

cd "$(dirname "$0")/../.."

echo "[Batch {i}] Starting {end-start} experiments..."
python scripts/run_manifest_parallel.py \\
    --manifest {manifest_path.relative_to(out_root.parent) if out_root.parent.exists() else manifest_path} \\
    --outdir {args.outdir} \\
    --default-use-gpu {args.use_gpu}

echo "[Batch {i}] Done!"
"""
        script_path = script_dir / f"run_batch_{i}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        all_scripts.append(script_path.name)
        
        print(f"[INFO] Batch {i}: {end-start} experiments (rows {start}-{end-1}) -> {script_path.name}")

    # Generate master script
    master_content = f"""#!/bin/bash
# Master script to run all batches sequentially
# For parallel execution, use: ls run_batch_*.sh | parallel -j {args.n_splits} bash {{}}

cd "$(dirname "$0")"

"""
    for script in all_scripts:
        master_content += f"bash {script}\n"
    
    master_path = script_dir / "run_all_sequential.sh"
    master_path.write_text(master_content)
    master_path.chmod(0o755)

    # Generate GNU parallel command
    parallel_cmd = f"""#!/bin/bash
# Run all batches in parallel using GNU parallel
# Install: apt install parallel / brew install parallel

cd "$(dirname "$0")"
ls run_batch_*.sh | parallel -j {args.n_splits} bash {{}}
"""
    parallel_path = script_dir / "run_all_parallel.sh"
    parallel_path.write_text(parallel_cmd)
    parallel_path.chmod(0o755)

    print(f"\n[DONE] Generated {len(all_scripts)} batch scripts in {script_dir}")
    print(f"\nTo run in parallel, choose one of:")
    print(f"  1) Python joblib:  python scripts/run_manifest_parallel.py --manifest {args.manifest} --outdir {args.outdir} --n-jobs {args.n_splits}")
    print(f"  2) GNU parallel:   cd {script_dir} && bash run_all_parallel.sh")
    print(f"  3) Manual:         Open {args.n_splits} terminals and run run_batch_0.sh, run_batch_1.sh, ...")


if __name__ == "__main__":
    main()
