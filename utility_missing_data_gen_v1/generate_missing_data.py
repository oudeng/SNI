#!/usr/bin/env python3
"""
generate_missing_data.py
========================================
Main controller / CLI for generating missing-data CSVs.

Example:
    python generate_missing_data.py \
        --input /path/to/ComCri_selabl.csv \
        --output-dir /path/to/out \
        --dataset ComCri \
        --mechanisms MCAR MAR MNAR \
        --rates 0.1 0.3 0.5 \
        --seed 2025 \
        --save-mask --save-metadata

Output naming rule (as requested):
    {dataset}_{mechanism}_{rate}.csv
where rate is formatted as "10per/30per/50per".

We also recommend exporting:
    - mask (.npy): exact missing pattern for reproducibility
    - meta (.json): column typing, actual missing rate, seed, parameters
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from missing_data_generator import (
    format_rate,
    generate_missing_dataset,
    parse_comma_list,
    save_mask,
    save_metadata,
)


def _parse_float_list(xs: List[str], arg_name: str) -> List[float]:
    try:
        vals = [float(x) for x in xs]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{arg_name}: failed to parse float list: {xs}") from e
    for v in vals:
        if v < 0 or v > 1:
            raise argparse.ArgumentTypeError(f"{arg_name}: values must be in [0,1], got {v}")
    return vals


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_missing_data",
        description="Generate MCAR/MAR/MNAR missing-data CSVs for tabular datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=str, help="Path to the *complete* input CSV.")
    p.add_argument("--output-dir", required=True, type=str, help="Directory to save generated CSVs.")
    p.add_argument("--dataset", required=True, type=str, help="Dataset name used in output filenames.")
    p.add_argument(
        "--mechanisms",
        nargs="+",
        default=["MCAR", "MAR", "MNAR"],
        help="Missing mechanisms to generate. Choices: MCAR MAR MNAR",
    )
    p.add_argument(
        "--rates",
        nargs="+",
        default=["0.1", "0.3", "0.5"],
        help="Missing rates to generate, e.g. 0.1 0.3 0.5",
    )
    p.add_argument(
        "--rate-style",
        type=str,
        default="percent",
        choices=["percent", "float", "p"],
        help='How to encode missing rate in filenames: "percent"->30per, "float"->0.3, "p"->0p3.',
    )

    p.add_argument("--seed", type=int, default=2025, help="Random seed (reproducible).")

    # Column typing (recommended for integer-coded categorical features)
    p.add_argument(
        "--categorical-cols",
        type=str,
        default="",
        help="Comma-separated categorical columns (recommended).",
    )
    p.add_argument(
        "--continuous-cols",
        type=str,
        default="",
        help="Comma-separated continuous columns. If provided (or categorical-cols provided), overrides inference.",
    )
    p.add_argument(
        "--exclude-cols",
        type=str,
        default="",
        help="Comma-separated columns excluded from missingness generation (e.g., IDs).",
    )

    # MAR params
    p.add_argument(
        "--mar-driver-cols",
        type=str,
        default="",
        help="Comma-separated driver columns for MAR. If empty, use first up to 2 continuous cols.",
    )
    p.add_argument("--mar-logistic-scale", type=float, default=1.0, help="Scale factor in MAR sigmoid.")
    p.add_argument(
        "--strict-mar",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "If true, enforce *strict MAR*: missingness is driven only by --mar-driver-cols, "
            "and those driver columns are guaranteed to remain observed (auto-added to --exclude-cols)."
        ),
    )

    # Calibration / constraints
    p.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for overall missing rate deviation.")
    p.add_argument("--min-missing-per-col", type=int, default=1, help="Ensure each column has at least this many missing values.")

    # I/O
    p.add_argument("--sep", type=str, default=",", help="CSV delimiter.")
    p.add_argument("--allow-input-missing", action="store_true", help="Allow input CSV to already contain missing values (not recommended).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")

    # Optional artifacts
    p.add_argument("--save-mask", action="store_true", help="Save mask (.npy) for each generated dataset.")
    p.add_argument("--save-metadata", action="store_true", help="Save metadata (.json) for each generated dataset.")

    p.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mechanisms = [m.strip().upper() for m in args.mechanisms]
    for m in mechanisms:
        if m not in {"MCAR", "MAR", "MNAR"}:
            print(f"[ERROR] Unknown mechanism: {m}. Use MCAR/MAR/MNAR.", file=sys.stderr)
            return 2

    rates = _parse_float_list(args.rates, "--rates")

    cat_cols = parse_comma_list(args.categorical_cols)
    con_cols = parse_comma_list(args.continuous_cols)
    ex_cols = parse_comma_list(args.exclude_cols)
    mar_drivers = parse_comma_list(args.mar_driver_cols)

    strict_mar = str(args.strict_mar).strip().lower() == 'true'

    # Validate strict MAR configuration early (avoid silent non-strict MAR)
    if strict_mar and ('MAR' in mechanisms) and (not mar_drivers):
        print('[ERROR] strict-mar=true but --mar-driver-cols is empty. '
              'Provide an always-observed driver (e.g., ID) or set --strict-mar false.',
              file=sys.stderr)
        return 2

    if not args.quiet:
        print("=" * 72)
        print("Missing-data generator (refactored)")
        print("=" * 72)
        print(f"Input:      {in_path}")
        print(f"Output dir: {out_dir}")
        print(f"Dataset:    {args.dataset}")
        print(f"Mechanisms: {mechanisms}")
        print(f"Rates:      {rates}")
        print(f"Seed:       {args.seed}")
        if cat_cols:
            print(f"Categorical cols (user): {cat_cols}")
        if con_cols:
            print(f"Continuous cols (user):  {con_cols}")
        if ex_cols:
            print(f"Excluded cols:           {ex_cols}")
        if mar_drivers:
            print(f"MAR driver cols:         {mar_drivers}")
        print(f"Strict MAR:             {strict_mar}")
        print("-" * 72)

    # Read input
    try:
        df = pd.read_csv(in_path, sep=args.sep)
    except Exception as e:
        print(f"[ERROR] Failed to read input CSV: {e}", file=sys.stderr)
        return 2

    if (not args.allow_input_missing) and df.isna().any().any():
        na_rate = float(df.isna().to_numpy().mean())
        print(
            f"[ERROR] Input contains missing values (overall rate {na_rate:.2%}). "
            f"Provide a complete dataset or pass --allow-input-missing.",
            file=sys.stderr,
        )
        return 2

    # Auto-exclude ID column if present.
    # This prevents accidental masking of row identifiers, which would break
    # downstream alignment (complete vs. missing and masks).
    if "ID" in df.columns and "ID" not in ex_cols:
        ex_cols.append("ID")
        if not args.quiet:
            print("[INFO] Detected 'ID' column -> auto-added to --exclude-cols (kept always observed).")

    # Generate
    for rate in rates:
        rate_tag = format_rate(rate, style=args.rate_style)
        for mech in mechanisms:
            stem = f"{args.dataset}_{mech}_{rate_tag}"
            out_csv = out_dir / f"{stem}.csv"
            out_mask = out_dir / f"{stem}_mask.npy"
            out_meta = out_dir / f"{stem}_meta.json"

            if out_csv.exists() and (not args.overwrite):
                if not args.quiet:
                    print(f"[SKIP] {out_csv.name} exists (use --overwrite to replace).")
                continue

            if not args.quiet:
                print(f"[RUN] {stem}")

            res = generate_missing_dataset(
                df,
                mechanism=mech,
                rate=rate,
                seed=args.seed,
                dataset_name=args.dataset,
                categorical_cols=cat_cols if cat_cols else None,
                continuous_cols=con_cols if con_cols else None,
                exclude_cols=ex_cols if ex_cols else None,
                tolerance=args.tolerance,
                min_missing_per_col=args.min_missing_per_col,
                mar_driver_cols=mar_drivers if mar_drivers else None,
                mar_logistic_scale=args.mar_logistic_scale,
                strict_mar=strict_mar,
                allow_input_missing=args.allow_input_missing,
            )

            # Save CSV
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            res.data_missing.to_csv(out_csv, index=False)

            # Optional artifacts
            if args.save_mask:
                save_mask(res.mask, out_mask)
            if args.save_metadata:
                save_metadata(res.metadata, out_meta)

            if not args.quiet:
                all_rate = res.metadata.get("actual_rate_all", None)
                if all_rate is None:
                    print(f"      -> {out_csv.name} (missing {res.actual_rate:.2%})")
                else:
                    print(f"      -> {out_csv.name} (eligible missing {res.actual_rate:.2%}, all-cells missing {float(all_rate):.2%})")
                if args.save_mask:
                    print(f"      -> {out_mask.name}")
                if args.save_metadata:
                    print(f"      -> {out_meta.name}")

    if not args.quiet:
        print("-" * 72)
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())