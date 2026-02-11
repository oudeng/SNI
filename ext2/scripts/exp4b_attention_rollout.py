#!/usr/bin/env python3
"""exp4b_attention_rollout.py

Ext2 / Experiment 4B: D vs attention rollout / attention flow
-------------------------------------------------------------

This script fills the Supplementary placeholder:

  - Table S7.B: "Aggregation method comparison for attention-derived feature
    importance — D (head-mean) vs attention rollout vs attention flow"

Background (Abnar & Zuidema, 2020)
-----------------------------------
For a multi-layer, multi-head transformer the raw per-head attention is not
directly interpretable because each layer re-mixes the representation.
*Attention rollout* multiplies attention matrices across layers (adding
residual connections as identity), while *attention flow* computes
maximum-flow on the attention graph.

In SNI/CPFA, each target feature is imputed by a *single* attention layer
with ``num_heads`` heads.  Because there is only one layer:
  - Rollout degenerates to the raw attention (no layer product needed).
  - The residual-aware rollout adds the identity: A' = 0.5 * A + 0.5 * I,
    then row-normalizes.
  - Flow computes maximum-flow through the (heads x sources) bipartite graph.

What we actually compare (for each target)
-------------------------------------------
1. **D (head-mean)** — ``imputer.compute_dependency_matrix()``, the default.
2. **Rollout** — residual-aware: ``A' = 0.5 * mean_over_heads(A) + 0.5 * I``,
   then row-normalize.  For a single layer this adds uniform self-reliance.
3. **Flow (max-head)** — per-source, take ``max`` across heads instead of
   ``mean``, then row-normalize.  This approximates the max-flow idea: if
   *any* head attends strongly to a source, it is credited.

We rank features under each method, then report pairwise Spearman rho.

Outputs
-------
<outdir>/
  attention_per_head.csv          : target × head × source attention weight
  aggregation_comparison.csv      : target × method × source × weight
  table_S7B_aggregation_comparison.csv : top-k ranking comparison table
  spearman_aggregation.csv        : pairwise Spearman among the three methods

Example (MIMIC-IV)
------------------
python ext2/scripts/exp4b_attention_rollout.py \\
  --input-complete data/MIMIC_complete.csv \\
  --dataset-name MIMIC \\
  --categorical-vars SpO2 ALARM \\
  --continuous-vars RESP ABP SBP DBP HR PULSE \\
  --mechanism MAR --missing-rate 0.30 \\
  --seed 2026 \\
  --targets ALARM SBP \\
  --top-k 5 \\
  --outdir results_ext2/table_S7B_rollout/MIMIC \\
  --use-gpu false
"""

from __future__ import annotations

import argparse
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


def _run_sni(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask_missing: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
    seed: int,
    use_gpu: bool,
) -> Any:
    """Run SNI imputation and return the imputer object (for attention_maps access)."""
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
    return imputer


def _row_normalize(v: np.ndarray) -> np.ndarray:
    """Row-normalize a 1-D vector (or each row of 2-D) to sum to 1."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        s = v.sum()
        return v / s if s > 0 else v
    row_sums = v.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return v / row_sums


def _top_k_features(scores: pd.Series, k: int) -> List[str]:
    return [str(x) for x in scores.sort_values(ascending=False).head(k).index.tolist()]


# ===================================================================== #
#                            Aggregation methods                         #
# ===================================================================== #

def aggregate_head_mean(attn_heads: np.ndarray) -> np.ndarray:
    """D (default): mean over heads, then row-normalize.

    Parameters
    ----------
    attn_heads : (num_heads, d-1) attention weights for one target.

    Returns
    -------
    (d-1,) normalized importance vector.
    """
    avg = attn_heads.mean(axis=0)
    return _row_normalize(avg)


def aggregate_rollout(attn_heads: np.ndarray) -> np.ndarray:
    """Residual-aware rollout for a single attention layer.

    A' = 0.5 * mean(A_heads) + 0.5 * uniform
    Then row-normalize.
    """
    avg = attn_heads.mean(axis=0)
    d_minus_1 = avg.shape[0]
    uniform = np.ones(d_minus_1, dtype=float) / d_minus_1
    rollout = 0.5 * avg + 0.5 * uniform
    return _row_normalize(rollout)


def aggregate_flow_max(attn_heads: np.ndarray) -> np.ndarray:
    """Max-flow approximation: take max across heads per source.

    For each source feature, if *any* head attends strongly,
    it receives credit.
    """
    max_per_src = attn_heads.max(axis=0)
    return _row_normalize(max_per_src)


# ===================================================================== #
#                                  Main                                 #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ext2/Exp4B: D vs attention rollout vs attention flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-complete", required=True, help="Path to complete CSV")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--categorical-vars", nargs="+", required=True)
    parser.add_argument("--continuous-vars", nargs="+", required=True)
    parser.add_argument("--mechanism", default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Targets to report (default: ALARM SBP if present else all)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--outdir", default="results_ext2/table_S7B_rollout")
    parser.add_argument("--use-gpu", default="false")
    parser.add_argument("--mar-driver-cols", nargs="+", default=None,
                        help="Fully-observed driver columns for strict MAR.")
    args = parser.parse_args()

    use_gpu = args.use_gpu.lower() in ("true", "1", "yes")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_complete = pd.read_csv(args.input_complete)

    # --- Column validation ---
    available_cols = set(df_complete.columns)
    cat_valid = [v for v in args.categorical_vars if v in available_cols]
    cont_valid = [v for v in args.continuous_vars if v in available_cols]
    cat_missing = [v for v in args.categorical_vars if v not in available_cols]
    cont_missing = [v for v in args.continuous_vars if v not in available_cols]

    if cat_missing or cont_missing:
        print(f"[Exp4B][WARN] Columns NOT in CSV (skipped): {cat_missing + cont_missing}")

    args.categorical_vars = cat_valid
    args.continuous_vars = cont_valid
    all_vars = args.categorical_vars + args.continuous_vars

    if not all_vars:
        print("[Exp4B][ERROR] No valid variables remain. Exiting.")
        sys.exit(1)

    # Decide targets
    if args.targets is None:
        default_targets = [t for t in ("ALARM", "SBP") if t in all_vars]
        targets = default_targets if default_targets else all_vars
    else:
        targets = args.targets

    print(f"[Exp4B] Dataset: {args.dataset_name}")
    print(f"[Exp4B] Mechanism: {args.mechanism} @ rate={args.missing_rate}")
    print(f"[Exp4B] Seed: {args.seed}")
    print(f"[Exp4B] Targets: {targets}")
    print(f"[Exp4B] Output: {outdir}")

    # 1) Generate missing data
    df_missing, mask_missing = _generate_missing_data(
        df_complete[all_vars].copy(),
        mechanism=args.mechanism,
        missing_rate=args.missing_rate,
        seed=args.seed,
        mar_driver_cols=args.mar_driver_cols,
    )

    # 2) Run SNI — keep the imputer object to access attention_maps
    print("[Exp4B] Running SNI...")
    imputer = _run_sni(
        df_missing=df_missing,
        df_complete=df_complete[all_vars],
        mask_missing=mask_missing,
        categorical_vars=args.categorical_vars,
        continuous_vars=args.continuous_vars,
        seed=args.seed,
        use_gpu=use_gpu,
    )

    # 3) Extract per-head attention and compute three aggregation methods
    METHOD_NAMES = ["D (head-mean)", "Rollout (residual-aware)", "Flow (max-head)"]
    AGGREGATORS = [aggregate_head_mean, aggregate_rollout, aggregate_flow_max]

    perhead_rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    spearman_rows: List[Dict[str, Any]] = []

    for target in targets:
        if target not in all_vars:
            print(f"[Exp4B] Skip target={target} (not in vars)")
            continue

        if target not in imputer.attention_maps:
            print(f"[Exp4B] Skip target={target} (no attention map)")
            continue

        attn = imputer.attention_maps[target]  # (num_heads, d-1)
        src_features = [f for f in all_vars if f != target]

        if attn.shape[1] != len(src_features):
            print(f"[Exp4B][WARN] target={target}: attn shape {attn.shape} vs "
                  f"{len(src_features)} sources — skipping", file=sys.stderr)
            continue

        # Save per-head attention
        for h in range(attn.shape[0]):
            for j, src in enumerate(src_features):
                perhead_rows.append({
                    "dataset": args.dataset_name,
                    "target": target,
                    "head": int(h),
                    "source": src,
                    "attention_weight": float(attn[h, j]),
                })

        # Compute the three aggregation methods
        scores: Dict[str, pd.Series] = {}
        for method_name, aggregator in zip(METHOD_NAMES, AGGREGATORS):
            weights = aggregator(attn)
            s = pd.Series(weights, index=src_features, dtype=float)
            scores[method_name] = s

            for src, w in s.items():
                agg_rows.append({
                    "dataset": args.dataset_name,
                    "target": target,
                    "method": method_name,
                    "source": src,
                    "weight": float(w),
                })

            top_feats = _top_k_features(s, args.top_k)
            table_rows.append({
                "Target": target,
                "Method": method_name,
                "Top features": ", ".join(top_feats),
            })

        # Pairwise Spearman among the three methods
        for i in range(len(METHOD_NAMES)):
            for j in range(i + 1, len(METHOD_NAMES)):
                m1, m2 = METHOD_NAMES[i], METHOD_NAMES[j]
                rho, p = spearmanr(scores[m1].to_numpy(), scores[m2].to_numpy())
                spearman_rows.append({
                    "dataset": args.dataset_name,
                    "target": target,
                    "method_A": m1,
                    "method_B": m2,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n_sources": len(src_features),
                })
                print(f"[Exp4B] target={target}: {m1} vs {m2} -> rho={rho:.3f} (p={p:.3g})")

    # Save outputs
    pd.DataFrame(perhead_rows).to_csv(outdir / "attention_per_head.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(outdir / "aggregation_comparison.csv", index=False)
    pd.DataFrame(table_rows).to_csv(outdir / "table_S7B_aggregation_comparison.csv", index=False)
    pd.DataFrame(spearman_rows).to_csv(outdir / "spearman_aggregation.csv", index=False)

    print(f"[Exp4B] Saved: {outdir / 'attention_per_head.csv'}")
    print(f"[Exp4B] Saved: {outdir / 'aggregation_comparison.csv'}")
    print(f"[Exp4B] Saved: {outdir / 'table_S7B_aggregation_comparison.csv'}")
    print(f"[Exp4B] Saved: {outdir / 'spearman_aggregation.csv'}")
    print("[Exp4B] Done.")


if __name__ == "__main__":
    main()
