#!/usr/bin/env python3
"""
Synthetic data generator for SNI sanity-check (Supplementary Section S5).

This script creates:
  (1) a *complete* mixed-type tabular dataset from a known directed dependency graph (DAG),
  (2) a corresponding *missing* dataset under MCAR/MAR/MNAR,
  (3) the ground-truth dependency matrix G (row=target, col=source; G[f,j]=1 if j is a parent of f),
  (4) a JSON metadata file (variables, categorical vars, settings, etc.).

Design goal (reviewer-facing):
- include both linear and *interaction / non-linear* dependencies so that correlation-only
  dependency estimates are insufficient, while SNI's attention-based dependency matrix D
  can (in many cases) recover the true parent sets.

Example:

# 在 repo 根目录执行
mkdir -p data/synth_s5

# linear_gaussian setting
for s in 2025 2026 2027 2028 2029; do
  python scripts/synth_generate_s5.py \
    --outdir data/synth_s5 \
    --setting linear_gaussian \
    --seed $s \
    --n 2000 --n-cont 10 --n-cat 2 \
    --mechanism MAR --missing-rate 0.30 \
    --driver-cols x0
done

# nonlinear_mixed setting (包含 product interaction：对 Pearson correlation “不友好”，更能体现 SNI 的优势)
for s in 2025 2026 2027 2028 2029; do
  python scripts/synth_generate_s5.py \
    --outdir data/synth_s5 \
    --setting nonlinear_mixed \
    --seed $s \
    --n 2000 --n-cont 10 --n-cat 2 \
    --mechanism MAR --missing-rate 0.30 \
    --driver-cols x0
done  


  
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# utilities
# -----------------------------
def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bisection_for_intercept(z: np.ndarray, a: float, target_rate: float, lo: float = -12, hi: float = 12, iters: int = 60) -> float:
    """
    Find b such that mean(sigmoid(a*z + b)) ≈ target_rate.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = sigmoid(a * z + mid).mean()
        if p > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class SyntheticSpec:
    n: int = 2000
    n_cont: int = 10
    n_cat: int = 2
    edge_prob: float = 0.25
    max_parents: int = 3
    setting: str = "nonlinear_mixed"  # {"linear_gaussian", "nonlinear_mixed"}
    noise_std: float = 0.6
    seed: int = 2025
    cat_levels: int = 4  # categories per categorical feature
    # missingness
    mechanism: str = "MAR"  # {"MCAR","MAR","MNAR"}
    missing_rate: float = 0.30
    driver_cols: Tuple[str, ...] = ("x0",)  # for MAR only
    mar_strength: float = 1.8  # slope magnitude range for MAR
    mnar_strength: float = 1.8  # slope magnitude range for MNAR


def sample_parents(rng: np.random.Generator, i: int, edge_prob: float, max_parents: int, min_parents: int) -> List[int]:
    if i == 0:
        return []
    candidates = list(range(i))
    # sample edges independently, then cap
    sel = [j for j in candidates if rng.random() < edge_prob]
    if len(sel) > max_parents:
        sel = [int(x) for x in rng.choice(sel, size=max_parents, replace=False)]
    if len(sel) < min_parents:
        # ensure at least min_parents for non-root nodes when possible
        need = min_parents - len(sel)
        remaining = [j for j in candidates if j not in sel]
        if remaining:
            extra = [int(x) for x in rng.choice(remaining, size=min(need, len(remaining)), replace=False)]
            sel += extra
    return sorted(sel)


def generate_dag(rng: np.random.Generator, d: int, edge_prob: float, max_parents: int, n_roots: int = 2) -> List[List[int]]:
    """
    Build a random DAG using a fixed topological order 0..d-1.
    We keep the first n_roots nodes parent-free to provide 'sources'.
    """
    parents: List[List[int]] = []
    for i in range(d):
        if i < n_roots:
            parents.append([])
        else:
            ps = sample_parents(rng, i, edge_prob=edge_prob, max_parents=max_parents, min_parents=1)
            parents.append(ps)
    return parents


def generate_mixed_type_data(spec: SyntheticSpec) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Return:
      - X_complete (DataFrame)
      - G_true (dependency adjacency; row=target, col=source; binary)
      - metadata dict
    """
    rng = set_seed(spec.seed)

    d = spec.n_cont + spec.n_cat
    parents = generate_dag(rng, d=d, edge_prob=spec.edge_prob, max_parents=spec.max_parents, n_roots=2)

    # variable names: first continuous then categorical (categorical placed at the end by construction)
    cont_vars = [f"x{i}" for i in range(spec.n_cont)]
    cat_vars = [f"c{i}" for i in range(spec.n_cat)]
    all_vars = cont_vars + cat_vars

    # map index -> name
    idx_to_name = {i: all_vars[i] for i in range(d)}
    name_to_idx = {v: i for i, v in idx_to_name.items()}

    # ground-truth dependency matrix (target row, source col)
    G = np.zeros((d, d), dtype=int)
    for i in range(d):
        for p in parents[i]:
            G[i, p] = 1
    G_df = pd.DataFrame(G, index=all_vars, columns=all_vars)

    # generate continuous latent values in topo order
    Z = np.zeros((spec.n, d), dtype=float)

    # roots
    for i in range(min(2, d)):
        Z[:, i] = rng.normal(0.0, 1.0, size=spec.n)

    # generate non-root continuous (up to n_cont-1), then categorical latents
    for i in range(2, d):
        ps = parents[i]
        if len(ps) == 0:
            Z[:, i] = rng.normal(0.0, 1.0, size=spec.n)
            continue

        # choose functional form
        if spec.setting == "linear_gaussian":
            form = "linear"
        else:
            # nonlinear_mixed: mix of linear / tanh / product interaction
            # product edges create near-zero Pearson correlation with each parent for independent normals
            # => correlation-only dependency baselines struggle.
            r = rng.random()
            if r < 0.40 and len(ps) >= 2:
                form = "product"
            elif r < 0.70:
                form = "tanh"
            else:
                form = "linear"

        # weights
        w = rng.uniform(0.6, 1.4, size=len(ps)) * rng.choice([-1.0, 1.0], size=len(ps))
        noise = rng.normal(0.0, spec.noise_std, size=spec.n)

        if form == "linear":
            val = np.zeros(spec.n)
            for k, p in enumerate(ps):
                val += w[k] * Z[:, p]
            Z[:, i] = val + noise

        elif form == "tanh":
            val = np.zeros(spec.n)
            for k, p in enumerate(ps):
                val += w[k] * np.tanh(Z[:, p])
            # small linear residual (keeps correlation prior somewhat informative)
            val += 0.15 * np.sum([Z[:, p] for p in ps], axis=0)
            Z[:, i] = val + noise

        elif form == "product":
            # pick two parents for interaction term
            p1, p2 = rng.choice(ps, size=2, replace=False)
            # interaction coefficient (dominant)
            w_int = float(rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0]))
            val = w_int * (Z[:, p1] * Z[:, p2])
            # add small linear terms from all parents (keeps some signal)
            for k, p in enumerate(ps):
                val += 0.10 * w[k] * Z[:, p]
            Z[:, i] = val + noise

        else:
            raise ValueError(f"Unknown form: {form}")

    # build observed DataFrame
    X = pd.DataFrame(index=np.arange(spec.n))

    # continuous vars are standardized for nicer conditioning (not necessary, but helps training stability)
    for i, v in enumerate(cont_vars):
        x = Z[:, i].copy()
        x = (x - x.mean()) / (x.std() + 1e-8)
        X[v] = x.astype(float)

    # categorical vars: discretize their latent Z into balanced quantile bins
    # (categorical nodes are at the end: indices spec.n_cont .. d-1)
    for k, v in enumerate(cat_vars):
        idx = spec.n_cont + k
        z = Z[:, idx]
        # quantile thresholds -> balanced categories
        qs = np.linspace(0, 1, spec.cat_levels + 1)[1:-1]
        th = np.quantile(z, qs)
        cats = np.digitize(z, th, right=False)  # 0..(K-1)
        # cast to strings for robust CSV round-trip without floatification
        X[v] = pd.Series(cats).map(lambda t: f"{v}_{int(t)}").astype("object")

    metadata = {
        "n": spec.n,
        "n_cont": spec.n_cont,
        "n_cat": spec.n_cat,
        "d": d,
        "setting": spec.setting,
        "noise_std": spec.noise_std,
        "seed": spec.seed,
        "cat_levels": spec.cat_levels,
        "cont_vars": cont_vars,
        "cat_vars": cat_vars,
        "all_vars": all_vars,
        "parents_index": parents,  # list-of-lists in index space
        "parents_named": {idx_to_name[i]: [idx_to_name[p] for p in parents[i]] for i in range(d)},
    }
    return X, G_df, metadata


def apply_missingness(spec: SyntheticSpec, X_complete: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply MCAR/MAR/MNAR missingness at approx. spec.missing_rate per *non-driver* column.
    Returns:
      - X_missing
      - mask_df (True if missing)
      - missingness metadata
    """
    rng = set_seed(spec.seed + 999)  # decouple from data-generation seed

    X = X_complete.copy()
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)

    mech = spec.mechanism.upper()
    rate = float(spec.missing_rate)

    if mech == "MCAR":
        for col in X.columns:
            p = rate
            m = rng.random(size=len(X)) < p
            mask.loc[:, col] = m
            X.loc[m, col] = np.nan

    elif mech == "MAR":
        drivers = list(spec.driver_cols)
        for dcol in drivers:
            if dcol not in X.columns:
                raise ValueError(f"Driver column '{dcol}' not in X columns.")
        # z = standardized driver (use first driver only for simplicity)
        z = X[drivers[0]].astype(float).values
        z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-8)

        for col in X.columns:
            if col in drivers:
                continue  # keep driver fully observed
            # slope a, intercept b so mean prob ~= rate
            a = float(rng.uniform(-spec.mar_strength, spec.mar_strength))
            if abs(a) < 0.2:
                a = 0.2 * np.sign(a) if a != 0 else 0.2
            b = bisection_for_intercept(z, a=a, target_rate=rate)
            p = sigmoid(a * z + b)
            m = rng.random(size=len(X)) < p
            mask.loc[:, col] = m
            X.loc[m, col] = np.nan

    elif mech == "MNAR":
        # missingness depends on the feature's own (standardized) value -> not MAR
        for col in X.columns:
            # categorical: use a simple per-category missing bias
            if X[col].dtype == "object":
                # map each category to a logit offset
                cats = pd.Series(X[col].dropna().unique())
                offsets = {c: float(rng.uniform(-1.0, 1.0)) for c in cats}
                base = math.log(rate / (1.0 - rate))
                logits = np.full(len(X), base, dtype=float)
                for c, off in offsets.items():
                    logits[X[col].values == c] += off
                p = sigmoid(logits)
            else:
                x = X[col].astype(float).values
                x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
                a = float(rng.uniform(-spec.mnar_strength, spec.mnar_strength))
                if abs(a) < 0.2:
                    a = 0.2 * np.sign(a) if a != 0 else 0.2
                b = bisection_for_intercept(x, a=a, target_rate=rate)
                p = sigmoid(a * x + b)

            m = rng.random(size=len(X)) < p
            mask.loc[:, col] = m
            X.loc[m, col] = np.nan
    else:
        raise ValueError(f"Unknown mechanism: {spec.mechanism}")

    miss_meta = {
        "mechanism": mech,
        "missing_rate": rate,
        "driver_cols": list(spec.driver_cols) if mech == "MAR" else [],
        "seed_mask": spec.seed + 999,
    }
    return X, mask, miss_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--setting", type=str, default="nonlinear_mixed", choices=["linear_gaussian", "nonlinear_mixed"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-cont", type=int, default=10)
    ap.add_argument("--n-cat", type=int, default=2)
    ap.add_argument("--edge-prob", type=float, default=0.25)
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--noise-std", type=float, default=0.6)
    ap.add_argument("--cat-levels", type=int, default=4)
    ap.add_argument("--mechanism", type=str, default="MAR", choices=["MCAR", "MAR", "MNAR"])
    ap.add_argument("--missing-rate", type=float, default=0.30)
    ap.add_argument("--driver-cols", type=str, default="x0", help="Comma-separated drivers for MAR (kept fully observed).")
    args = ap.parse_args()

    spec = SyntheticSpec(
        n=args.n,
        n_cont=args.n_cont,
        n_cat=args.n_cat,
        edge_prob=args.edge_prob,
        max_parents=args.max_parents,
        setting=args.setting,
        noise_std=args.noise_std,
        seed=args.seed,
        cat_levels=args.cat_levels,
        mechanism=args.mechanism,
        missing_rate=args.missing_rate,
        driver_cols=tuple([s.strip() for s in args.driver_cols.split(",") if s.strip()]),
    )

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    X_complete, G_true, meta = generate_mixed_type_data(spec)
    X_missing, mask_df, miss_meta = apply_missingness(spec, X_complete)

    base = f"synth_{spec.setting}_n{spec.n}_d{spec.n_cont+spec.n_cat}_seed{spec.seed}"
    complete_path = outdir / f"{base}_complete.csv"
    missing_path = outdir / f"{base}_{spec.mechanism.upper()}_{int(spec.missing_rate*100)}per.csv"
    mask_path = outdir / f"{base}_{spec.mechanism.upper()}_{int(spec.missing_rate*100)}per_mask.csv"
    gt_path = outdir / f"{base}_ground_truth_G.csv"
    meta_path = outdir / f"{base}_metadata.json"

    X_complete.to_csv(complete_path, index=False)
    X_missing.to_csv(missing_path, index=False)
    mask_df.astype(int).to_csv(mask_path, index=False)
    G_true.to_csv(gt_path, index=True)

    meta_all = {"data": meta, "missingness": miss_meta}
    meta_path.write_text(json.dumps(meta_all, indent=2), encoding="utf-8")

    print(f"[OK] Wrote complete: {complete_path}")
    print(f"[OK] Wrote missing : {missing_path}")
    print(f"[OK] Wrote mask   : {mask_path}")
    print(f"[OK] Wrote GT G   : {gt_path}")
    print(f"[OK] Wrote meta   : {meta_path}")


if __name__ == "__main__":
    main()
