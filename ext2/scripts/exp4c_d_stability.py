#!/usr/bin/env python3
"""exp4c_d_stability.py

Ext2 / Experiment 4C: Cross-seed stability of the dependency matrix D
---------------------------------------------------------------------

This script fills the Supplementary placeholder:

  - Table S7.C: "Stability of D across random seeds"

Motivation
----------
A reviewer might question whether the learned dependency matrix D is a stable,
reproducible artifact or merely noise that varies wildly across random seeds.
To address this, we:

1) Run SNI with ``K`` different random seeds on the *same* dataset and
   missingness pattern.
2) Collect the resulting D matrices.
3) For each target feature, compute pairwise Spearman correlation of D rows
   across all seed pairs.
4) Report the mean +/- std of these correlations as a stability index.

High correlation (e.g., > 0.8) demonstrates that the learned feature-reliance
structure is robust and not an artifact of random initialization.

Outputs
-------
<outdir>/
  d_matrices/seed<s>.csv          : one D matrix per seed
  pairwise_spearman.csv           : seed_a x seed_b x target x rho
  table_S7C_d_stability.csv       : target x mean_rho x std_rho (Table S7.C)

Example (MIMIC-IV, 5 seeds)
----------------------------
python ext2/scripts/exp4c_d_stability.py \\
  --input-complete data/MIMIC_complete.csv \\
  --dataset-name MIMIC \\
  --categorical-vars SpO2 ALARM \\
  --continuous-vars RESP ABP SBP DBP HR PULSE \\
  --mechanism MAR --missing-rate 0.30 \\
  --seeds 1 2 3 5 8 \\
  --targets ALARM SBP \\
  --outdir results_ext2/table_S7C_stability/MIMIC \\
  --use-gpu false
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so internal modules can be imported.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for _candidate in (str(_PROJECT_ROOT), os.getcwd()):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ---------------------------------------------------------------------------
# Missingness generator (project internal)
# ---------------------------------------------------------------------------
_HAS_MISSING_GEN = False
try:
    from utility_missing_data_gen_v1.missing_data_generator import generate_missing_dataset  # type: ignore
    _HAS_MISSING_GEN = True
except Exception as e:  # pragma: no cover
    print(f"[WARN] Could not import missing_data_generator: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    print("[ERROR] scipy not found. Install via: pip install scipy", file=sys.stderr)
    raise


# ===================================================================== #
#                               Helpers                                 #
# ===================================================================== #

def _align_mask_to_nan(mask: pd.DataFrame, df_missing: pd.DataFrame) -> pd.DataFrame:
    """Return boolean mask where True indicates missing positions."""
    mask_bool = mask.astype(bool)
    nan_bool = df_missing.isna()
    agree = (mask_bool == nan_bool).to_numpy().sum()
    agree_inv = ((~mask_bool) == nan_bool).to_numpy().sum()
    return ~mask_bool if agree_inv > agree else mask_bool


def _generate_missing_data(
    df: pd.DataFrame, mechanism: str, missing_rate: float, seed: int,
    mar_driver_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate missingness and return (df_missing, mask_missing_bool)."""
    if not _HAS_MISSING_GEN:
        raise RuntimeError("missing_data_generator not available in this environment.")

    kwargs: Dict[str, Any] = dict(
        mechanism=mechanism,
        rate=missing_rate,
        seed=seed,
        allow_input_missing=False,
    )
    if mar_driver_cols is not None:
        kwargs["mar_driver_cols"] = mar_driver_cols

    result = generate_missing_dataset(df, **kwargs)
    mask_df = pd.DataFrame(result.mask.astype(int), index=df.index, columns=df.columns)
    df_miss = result.data_missing.copy()

    for col in df_miss.columns:
        if isinstance(df_miss[col].dtype, pd.CategoricalDtype) or pd.api.types.is_extension_array_dtype(df_miss[col]):
            df_miss[col] = df_miss[col].astype(object)
    for col in df_miss.columns:
        if df_miss[col].dtype == object:
            numeric = pd.to_numeric(df_miss[col], errors="coerce")
            orig_notna = df_miss[col].notna().sum()
            if orig_notna > 0 and numeric.notna().sum() == orig_notna:
                df_miss[col] = numeric

    mask_missing = _align_mask_to_nan(mask_df, df_miss)
    return df_miss, mask_missing


def _run_sni_and_get_d(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    use_gpu: bool,
) -> Optional[pd.DataFrame]:
    """Run SNI imputation and return D matrix (or None)."""
    from SNI_v0_3 import SNIImputer  # type: ignore
    from SNI_v0_3.imputer import SNIConfig  # type: ignore

    cfg = SNIConfig(seed=seed, use_gpu=use_gpu)
    imputer = SNIImputer(
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
        config=cfg,
    )
    imputer.impute(
        X_missing=df_missing,
        X_complete=df_complete,
        mask_df=mask_missing.astype(int),
    )
    try:
        return imputer.compute_dependency_matrix()
    except Exception as e:
        print(f"[WARN] compute_dependency_matrix() failed: {e}", file=sys.stderr)
        return None


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ext2/Exp4C: Cross-seed stability of dependency matrix D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-complete", required=True, help="Path to complete CSV")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--categorical-vars", nargs="+", required=True)
    parser.add_argument("--continuous-vars", nargs="+", required=True)
    parser.add_argument("--mechanism", default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--missing-seed", type=int, default=2026,
                        help="Fixed seed for missingness generation (same pattern across runs).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 5, 8],
                        help="SNI model seeds (each produces a different D matrix).")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Targets to report (default: all features)")
    parser.add_argument("--outdir", default="results_ext2/table_S7C_stability")
    parser.add_argument("--use-gpu", default="false")
    parser.add_argument("--mar-driver-cols", nargs="+", default=None,
                        help="Fully-observed driver columns for strict MAR.")
    args = parser.parse_args()

    use_gpu = args.use_gpu.lower() in ("true", "1", "yes")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d_dir = outdir / "d_matrices"
    d_dir.mkdir(parents=True, exist_ok=True)

    df_complete = pd.read_csv(args.input_complete)

    # --- Column validation ---
    available_cols = set(df_complete.columns)
    cat_valid = [v for v in args.categorical_vars if v in available_cols]
    cont_valid = [v for v in args.continuous_vars if v in available_cols]
    cat_missing = [v for v in args.categorical_vars if v not in available_cols]
    cont_missing = [v for v in args.continuous_vars if v not in available_cols]

    if cat_missing or cont_missing:
        print(f"[Exp4C][WARN] Columns NOT in CSV (skipped): {cat_missing + cont_missing}")

    args.categorical_vars = cat_valid
    args.continuous_vars = cont_valid
    all_vars = args.categorical_vars + args.continuous_vars

    if not all_vars:
        print("[Exp4C][ERROR] No valid variables remain. Exiting.")
        sys.exit(1)

    # Decide targets (default: all features)
    if args.targets is not None:
        targets = [t for t in args.targets if t in all_vars]
    else:
        targets = list(all_vars)

    print(f"[Exp4C] Dataset: {args.dataset_name}")
    print(f"[Exp4C] Mechanism: {args.mechanism} @ rate={args.missing_rate}")
    print(f"[Exp4C] Missing seed: {args.missing_seed}")
    print(f"[Exp4C] Model seeds: {args.seeds}")
    print(f"[Exp4C] Targets: {targets}")
    print(f"[Exp4C] Output: {outdir}")

    # 1) Generate missing data with a FIXED seed (same pattern for all runs)
    df_missing, mask_missing = _generate_missing_data(
        df_complete[all_vars].copy(),
        mechanism=args.mechanism,
        missing_rate=args.missing_rate,
        seed=args.missing_seed,
        mar_driver_cols=args.mar_driver_cols,
    )

    # 2) Run SNI with different model seeds, collect D matrices
    d_matrices: Dict[int, pd.DataFrame] = {}
    for seed in args.seeds:
        print(f"[Exp4C] Running SNI with seed={seed}...")
        D = _run_sni_and_get_d(
            df_missing=df_missing,
            df_complete=df_complete[all_vars],
            mask_missing=mask_missing,
            categorical_vars=args.categorical_vars,
            continuous_vars=args.continuous_vars,
            seed=seed,
            use_gpu=use_gpu,
        )
        if D is not None:
            d_matrices[seed] = D
            D.to_csv(d_dir / f"seed{seed}.csv")
            print(f"  -> D saved ({D.shape[0]}x{D.shape[1]})")
        else:
            print(f"  -> D not available (skipped)")

    if len(d_matrices) < 2:
        print("[Exp4C][ERROR] Need at least 2 valid D matrices for pairwise comparison. Exiting.")
        sys.exit(1)

    seed_list = sorted(d_matrices.keys())
    print(f"[Exp4C] Collected {len(seed_list)} D matrices: seeds={seed_list}")

    # 3) Pairwise Spearman for each target
    pairwise_rows: List[Dict[str, Any]] = []
    target_stats: Dict[str, List[float]] = {t: [] for t in targets}

    for sa, sb in itertools.combinations(seed_list, 2):
        Da = d_matrices[sa]
        Db = d_matrices[sb]

        for target in targets:
            if target not in Da.index or target not in Db.index:
                continue

            # D row for this target, excluding self-dependency (diagonal = 0)
            src_features = [f for f in all_vars if f != target]
            common = [c for c in src_features if c in Da.columns and c in Db.columns]
            if len(common) < 3:
                continue

            row_a = Da.loc[target, common].to_numpy(dtype=float)
            row_b = Db.loc[target, common].to_numpy(dtype=float)

            rho, p = spearmanr(row_a, row_b)

            pairwise_rows.append({
                "dataset": args.dataset_name,
                "target": target,
                "seed_a": int(sa),
                "seed_b": int(sb),
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_sources": int(len(common)),
            })

            if np.isfinite(rho):
                target_stats[target].append(float(rho))

    # Save pairwise
    df_pairwise = pd.DataFrame(pairwise_rows)
    df_pairwise.to_csv(outdir / "pairwise_spearman.csv", index=False)
    print(f"[Exp4C] Saved: {outdir / 'pairwise_spearman.csv'} (n={len(df_pairwise)})")

    # 4) Aggregate: mean +/- std per target -> Table S7.C
    summary_rows: List[Dict[str, Any]] = []
    for target in targets:
        rhos = target_stats.get(target, [])
        if len(rhos) == 0:
            continue
        arr = np.array(rhos)
        summary_rows.append({
            "dataset": args.dataset_name,
            "target": target,
            "n_seed_pairs": int(len(rhos)),
            "n_seeds": int(len(seed_list)),
            "mean_rho": float(np.mean(arr)),
            "std_rho": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min_rho": float(np.min(arr)),
            "max_rho": float(np.max(arr)),
            "interpretation": (
                "highly stable" if np.mean(arr) >= 0.9 else
                "stable" if np.mean(arr) >= 0.7 else
                "moderately stable" if np.mean(arr) >= 0.5 else
                "unstable"
            ),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(outdir / "table_S7C_d_stability.csv", index=False)
    print(f"[Exp4C] Saved: {outdir / 'table_S7C_d_stability.csv'} (n={len(df_summary)})")

    # Print summary
    if len(df_summary) > 0:
        print("\n[Exp4C] === Stability Summary ===")
        for _, row in df_summary.iterrows():
            print(f"  {row['target']}: mean rho = {row['mean_rho']:.3f} "
                  f"+/- {row['std_rho']:.3f} "
                  f"[{row['min_rho']:.3f}, {row['max_rho']:.3f}] "
                  f"({row['interpretation']})")

    print("\n[Exp4C] Done.")


if __name__ == "__main__":
    main()
