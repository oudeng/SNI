#!/usr/bin/env python3
""" 
Sanity check for SNI dependency matrix (Supplementary Section S5) - v2.

This script is based on the provided sanity_check_s5.py, with one additional
synthetic setting designed to explicitly test the failure mode of
*correlation-only* priors:

  - interaction_xor (NEW): interaction-only / XOR-like generators where each
    child depends on a *pair* of sources but has (approximately) zero marginal
    correlation/association with each source individually.

    Continuous (interaction-only):
      x8 = x0 * x2 + eps,  x9 = x1 * x3 + eps
      where x0..x4 are independent N(0,1) roots.

    Categorical (XOR-like):
      c1 = XOR(c0, 1[x1>0])
      with c0 ~ Bernoulli(0.5) independent of x1.

In such settings, a statistics-only dependency matrix derived from pairwise
correlations (PriorOnly) is expected to perform poorly, while SNI can still
recover dependencies via conditional modeling. This addresses reviewer requests
for a sanity check when "attention\u2192dependency" is used as an interpretability
proxy (without implying causality).

Outputs (same as v1):
  - metrics_per_run.csv
  - table_S21.csv
  - table_S21.tex
  - S5_sanity_check_snippet.tex

Example:
mkdir -p results_sanity_s5_v2
python scripts/sanity_check_v2_s5.py \
  --data-dir data/synth_s5 \
  --outdir results_sanity_s5_v2 \
  --settings linear_gaussian nonlinear_mixed interaction_xor \
  --seeds 2025 2026 2027 2028 2029 \
  --mechanism MAR --missing-rate 0.30 \
  --use-gpu true \
  --epochs 60 --num-heads 4 --emb-dim 32 --batch-size 64 --max-iters 3

"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Suppress sklearn warnings about division by zero during incremental mean/variance
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

from sklearn.metrics import average_precision_score, roc_auc_score

# SNI imports (repo root must be on PYTHONPATH; same as scripts/run_experiment.py)
from SNI_v0_2.imputer import SNIConfig, SNIImputer


# -----------------------------
# helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # ROC-AUC undefined if only one class or if y_score contains NaN/Inf
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if np.any(np.isnan(y_score)) or np.any(np.isinf(y_score)):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    if np.any(np.isnan(y_score)) or np.any(np.isinf(y_score)):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def per_target_metrics(D: np.ndarray, G: np.ndarray) -> Dict[str, float]:
    """Macro-average edge recovery metrics across targets (rows).

    D: score matrix (d,d), diag ignored.
    G: binary GT adjacency (d,d), diag ignored.
    """
    d = D.shape[0]
    aurocs, auprcs, precs, recs, f1s = [], [], [], [], []

    for i in range(d):
        y_true = G[i, :].copy()
        y_score = D[i, :].copy()
        y_true[i] = 0
        y_score[i] = -np.inf  # ignore diagonal

        pos = int(y_true.sum())
        neg = int((d - 1) - pos)
        if pos == 0 or neg == 0:
            # skip root targets (no parents) or degenerate
            continue

        # remove diagonal
        mask = np.ones(d, dtype=bool)
        mask[i] = False
        yt = y_true[mask]
        ys = y_score[mask]

        # Skip if y_score contains NaN (can happen if D matrix has NaN)
        if np.any(np.isnan(ys)):
            continue

        aurocs.append(safe_roc_auc(yt, ys))
        auprcs.append(safe_auprc(yt, ys))

        k = pos
        topk = np.argsort(-ys)[:k]
        tp = int(yt[topk].sum())
        prec = tp / max(1, k)
        rec = tp / max(1, pos)
        f1 = 2 * prec * rec / max(1e-12, (prec + rec))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    def nanmean(x: List[float]) -> float:
        x = np.array(x, dtype=float)
        return float(np.nanmean(x)) if x.size else float("nan")

    return {
        "AUROC_macro": nanmean(aurocs),
        "AUPRC_macro": nanmean(auprcs),
        "PrecAtK_macro": nanmean(precs),
        "RecAtK_macro": nanmean(recs),
        "F1AtK_macro": nanmean(f1s),
    }


def hub_recovery(D: np.ndarray, G: np.ndarray) -> float:
    """Compare column-sum hubness Σ_j = sum_i D_{i,j} to GT out-degree."""
    if np.any(np.isnan(D)):
        return float("nan")
    Sigma = D.sum(axis=0)
    outdeg = G.sum(axis=0)
    if np.allclose(Sigma, Sigma[0]) or np.allclose(outdeg, outdeg[0]):
        return float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a))
        return ranks

    r1 = rankdata(Sigma)
    r2 = rankdata(outdeg)
    r1 = (r1 - r1.mean()) / (r1.std() + 1e-12)
    r2 = (r2 - r2.mean()) / (r2.std() + 1e-12)
    return float(np.mean(r1 * r2))


def compute_all_metrics(
    D: pd.DataFrame,
    G: pd.DataFrame,
    nan_warnings: List[str] | None = None,
    tag: str = "",
) -> Dict[str, float]:
    # align
    G = G.loc[D.index, D.columns]
    Dm = D.values.astype(float)
    Gm = G.values.astype(int)

    nan_count = int(np.isnan(Dm).sum())
    if nan_count > 0:
        if nan_warnings is not None:
            nan_warnings.append(f"{tag}: {nan_count} NaN values")
        Dm = np.nan_to_num(Dm, nan=0.0)

    m = per_target_metrics(Dm, Gm)
    m["Hub_Spearman"] = hub_recovery(Dm, Gm)
    return m


def build_prior_only_D(imputer: SNIImputer, X_missing: pd.DataFrame) -> pd.DataFrame:
    """Build a dependency matrix using only the *correlation prior* construction used inside SNI."""
    X_stat = imputer._initial_stat_impute(X_missing.copy())
    Prior_matrix, cat_onehot_dims, corr_cols = imputer._compute_correlation_prior(X_stat)

    features = imputer.all_vars
    d = len(features)
    D = np.zeros((d, d), dtype=float)

    for i, target in enumerate(features):
        P_f = imputer._extract_feature_prior(Prior_matrix, target, corr_cols, cat_onehot_dims)
        P_f = imputer._normalize_prior(P_f)
        other = [f for f in features if f != target]
        for j, src in enumerate(features):
            if src == target:
                D[i, j] = 0.0
            else:
                D[i, j] = float(P_f[other.index(src)])

    rs = D.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    D = D / rs
    return pd.DataFrame(D, index=features, columns=features)


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "--"
    m = f"{mean:.{digits}f}"
    s = f"{std:.{digits}f}"
    return f"{m}$\\pm${s}"


def write_table_s21(df_agg: pd.DataFrame, out_tex: Path, caption: str, label: str) -> None:
    metrics = ["AUROC_macro", "AUPRC_macro", "PrecAtK_macro", "RecAtK_macro", "Hub_Spearman"]

    rows = []
    for _, r in df_agg.iterrows():
        row = {"Setting": r["setting"], "Method": r["method"]}
        for m in metrics:
            row[m] = fmt_pm(r[f"{m}_mean"], r[f"{m}_std"], digits=3)
        rows.append(row)
    T = pd.DataFrame(rows)

    # bold best per setting+metric
    best_mean: Dict[Tuple[str, str], float] = {}
    for metric in metrics:
        for setting in df_agg["setting"].unique():
            sub = df_agg[df_agg["setting"] == setting]
            best_mean[(setting, metric)] = float(sub[f"{metric}_mean"].max())

    for i, r in T.iterrows():
        setting = r["Setting"]
        method = r["Method"]
        sub = df_agg[(df_agg["setting"] == setting) & (df_agg["method"] == method)]
        if len(sub) != 1:
            continue
        sub = sub.iloc[0]
        for metric in metrics:
            if np.isfinite(sub[f"{metric}_mean"]) and abs(sub[f"{metric}_mean"] - best_mean[(setting, metric)]) < 1e-12:
                T.loc[i, metric] = "\\textbf{" + str(T.loc[i, metric]) + "}"

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Setting & Method & AUROC$\\uparrow$ & AUPRC$\\uparrow$ & Prec@K$\\uparrow$ & Rec@K$\\uparrow$ & Hub-$\\rho$$\\uparrow$\\\\")
    lines.append("\\midrule")

    for setting in T["Setting"].unique():
        subT = T[T["Setting"] == setting].copy()
        for k, (_, rr) in enumerate(subT.iterrows()):
            s = setting if k == 0 else ""
            lines.append(
                f"{s} & {rr['Method']} & {rr['AUROC_macro']} & {rr['AUPRC_macro']} & {rr['PrecAtK_macro']} & {rr['RecAtK_macro']} & {rr['Hub_Spearman']}\\\\"
            )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\begin{minipage}{0.95\\linewidth}")
    lines.append("\\footnotesize")
    lines.append(
        "\\textbf{Notes:} We evaluate recovery of the ground-truth dependency graph $G$ (row=target, column=source) using the induced dependency matrix $D$ (same orientation). "
        "AUROC/AUPRC/Prec@K/Rec@K are macro-averaged across targets (K equals the number of true parents for each target). "
        "Hub-$\\rho$ is Spearman correlation between column-sum hubness $\\Sigma_j=\\sum_i D_{ij}$ and the ground-truth out-degree. "
        "The \"interaction\\_xor\" setting is constructed so that marginal correlations can be near-zero even when conditional dependence is strong."
    )
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_s5_text(out_tex: Path) -> None:
    txt = r"""
\section{Sanity Check: Dependency Recovery on Synthetic Data}
\label{sec:supp_sanity}

To validate that the dependency matrix $D$ induced by CPFA provides a meaningful proxy for feature dependencies (while not implying causality), we conduct a sanity check on synthetic data with \emph{known} ground-truth structure.
We generate mixed-type tabular datasets from a directed acyclic graph (DAG) under three settings: (i) \emph{linear\_gaussian} (pairwise correlations are informative), (ii) \emph{nonlinear\_mixed} (non-linear interactions with mixed variable types), and (iii) \emph{interaction\_xor} where several children are driven primarily by interaction/XOR terms such that marginal correlations with each parent are near zero.
We then inject missingness (strict MAR at 30\% by default) and fit SNI on the incomplete table.
For identifiability in the interaction\_xor setting, we keep the exogenous roots (\texttt{x0}--\texttt{x4}) and \texttt{c0} fully observed (serving as always-measured covariates), and only apply missingness to downstream variables.
After training, we extract the dependency matrix $D$ (row: target; column: source) and evaluate edge recovery against the ground-truth dependency matrix $G$ of the data-generating process.

\paragraph{Metrics.}
For each target feature $f$, we rank candidate source features $j\neq f$ using $D_{f,j}$ and compute AUROC and AUPRC, as well as Precision@K and Recall@K where $K$ equals the number of true parents of $f$.
We report macro-averages across targets.
We also assess whether the induced hubness $\Sigma_j=\sum_f D_{f,j}$ aligns with the true out-degree of each source feature via Spearman correlation (Hub-$\rho$).

\paragraph{Baselines.}
We compare (i) SNI, (ii) a neural-only ablation (NoPrior) where the prior loss is disabled, and (iii) a statistics-only baseline (PriorOnly) that constructs a dependency matrix solely from the correlation-based prior used by SNI.
The interaction\_xor setting serves as an explicit stress test where PriorOnly is expected to underperform, because pairwise correlations are insufficient to capture interaction-only dependencies.
Table~\ref{tab:S21_dependency_recovery} summarizes the results.
"""
    out_tex.write_text(txt.strip() + "\n", encoding="utf-8")


# -----------------------------
# NEW: interaction_xor generator
# -----------------------------

def _generate_interaction_xor_complete(seed: int, n: int = 2000, noise: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Generate a mixed-type table with interaction-only / XOR-like dependencies.

    Continuous roots: x0..x4 ~ iid N(0,1)
    Linear children:   x5, x6, x7
    Interaction-only:  x8 = x0*x2 + eps, x9 = x1*x3 + eps
    Categorical:       c0 ~ Bern(0.5), c1 = XOR(c0, 1[x1>0])

    Returns:
        X_complete (DataFrame), G (DataFrame), meta (dict)
    """
    rng = np.random.default_rng(seed)

    # roots (keep a raw copy for XOR sign)
    x0 = rng.normal(0, 1, size=n)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    x4 = rng.normal(0, 1, size=n)

    # linear children (correlation-informative)
    x5 = 0.9 * x0 + 0.3 * x1 + rng.normal(0, noise, size=n)
    x6 = 0.8 * x2 - 0.4 * x3 + rng.normal(0, noise, size=n)
    x7 = 0.7 * x1 + 0.5 * x4 + rng.normal(0, noise, size=n)

    # interaction-only children (marginal correlation ~ 0)
    x8 = (x0 * x2) + rng.normal(0, noise, size=n)
    x9 = (x1 * x3) + rng.normal(0, noise, size=n)

    cont_vars = [f"x{i}" for i in range(10)]
    cont = {
        "x0": x0,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "x6": x6,
        "x7": x7,
        "x8": x8,
        "x9": x9,
    }
    # standardize each continuous column for stable training
    for k in cont:
        cont[k] = _zscore(cont[k])

    # categorical
    c0 = rng.integers(0, 2, size=n)
    b = (x1 > 0).astype(int)  # use raw x1 sign; balanced Bernoulli(0.5)
    c1 = np.bitwise_xor(c0, b)

    cat_vars = ["c0", "c1"]

    # assemble: categorical first (consistent with other scripts), then continuous
    X_complete = pd.DataFrame({"c0": c0, "c1": c1, **{k: cont[k] for k in cont_vars}})

    # ground-truth adjacency G (row=target, col=source)
    features = cat_vars + cont_vars
    d = len(features)
    G = np.zeros((d, d), dtype=int)

    def add_edge(t: str, s: str) -> None:
        G[features.index(t), features.index(s)] = 1

    # linear edges
    add_edge("x5", "x0")
    add_edge("x5", "x1")

    add_edge("x6", "x2")
    add_edge("x6", "x3")

    add_edge("x7", "x1")
    add_edge("x7", "x4")

    # interaction-only edges
    add_edge("x8", "x0")
    add_edge("x8", "x2")

    add_edge("x9", "x1")
    add_edge("x9", "x3")

    # XOR-like categorical
    add_edge("c1", "c0")
    add_edge("c1", "x1")

    G_df = pd.DataFrame(G, index=features, columns=features)

    meta = {
        "setting": "interaction_xor",
        "data": {
            "n": int(n),
            "d": int(d),
            "cat_vars": cat_vars,
            "cont_vars": cont_vars,
        },
        "generator": {
            "description": "Interaction-only / XOR-like setting where pairwise correlations can be near zero despite strong conditional dependence.",
            "linear_children": {"x5": ["x0", "x1"], "x6": ["x2", "x3"], "x7": ["x1", "x4"]},
            "interaction_children": {"x8": ["x0", "x2"], "x9": ["x1", "x3"]},
            "xor_child": {"c1": ["c0", "x1"]},
            "noise_std": float(noise),
        },
    }

    return X_complete, G_df, meta


def _inject_missingness(
    X: pd.DataFrame,
    mechanism: str,
    miss_rate: float,
    seed: int,
    driver_cols: List[str],
    protect_cols: List[str],
    mar_scale: float = 2.0,
    mnar_scale: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject missingness; protect_cols are never masked."""
    rng = np.random.default_rng(seed)
    mechanism = mechanism.upper()

    Xm = X.copy()
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)

    protect = set(protect_cols)
    protect.update(driver_cols)

    n = len(X)

    # precompute MAR probability as a function of drivers
    if mechanism == "MAR":
        score = np.zeros(n, dtype=float)
        for c in driver_cols:
            if c in X.columns:
                score += _zscore(pd.to_numeric(X[c], errors="coerce").fillna(0.0).values)
        score = _zscore(score)
        p_raw = _sigmoid(mar_scale * score)
        p = p_raw / (p_raw.mean() + 1e-12) * miss_rate
        p = np.clip(p, 0.0, 0.95)
    else:
        p = None

    for col in X.columns:
        if col in protect:
            continue

        if mechanism == "MCAR":
            miss = rng.random(n) < miss_rate

        elif mechanism == "MAR":
            assert p is not None
            # add tiny column-specific jitter to avoid identical masks for all cols
            jitter = rng.normal(0, 0.01, size=n)
            p_col = np.clip(p + jitter, 0.0, 0.95)
            miss = rng.random(n) < p_col

        elif mechanism == "MNAR":
            v = pd.to_numeric(X[col], errors="coerce").fillna(0.0).values.astype(float)
            sc = _zscore(v)
            p_raw = _sigmoid(mnar_scale * sc)
            p_col = p_raw / (p_raw.mean() + 1e-12) * miss_rate
            p_col = np.clip(p_col, 0.0, 0.95)
            miss = rng.random(n) < p_col

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        Xm.loc[miss, col] = np.nan
        mask.loc[miss, col] = True

    return Xm, mask


def _maybe_generate_interaction_xor_files(
    data_dir: Path,
    seed: int,
    mechanism: str,
    miss_rate: float,
    n: int = 2000,
    noise: float = 0.15,
) -> None:
    """Generate complete/missing/GT/meta files for interaction_xor if absent."""

    Xc, G, meta = _generate_interaction_xor_complete(seed=seed, n=n, noise=noise)

    features = meta["data"]["cat_vars"] + meta["data"]["cont_vars"]
    d = len(features)

    complete_path = data_dir / f"synth_interaction_xor_n{n}_d{d}_seed{seed}_complete.csv"
    gt_path = data_dir / f"synth_interaction_xor_n{n}_d{d}_seed{seed}_ground_truth_G.csv"
    meta_path = data_dir / f"synth_interaction_xor_n{n}_d{d}_seed{seed}_metadata.json"

    miss_tag = f"{mechanism.upper()}_{int(miss_rate*100)}per.csv"
    missing_path = Path(str(complete_path).replace("_complete.csv", f"_{miss_tag}"))
    mask_path = Path(str(complete_path).replace("_complete.csv", f"_{mechanism.upper()}_{int(miss_rate*100)}per_mask.csv"))

    if complete_path.exists() and gt_path.exists() and meta_path.exists() and missing_path.exists():
        return

    ensure_dir(data_dir)

    # strict MAR driver: x0 (always observed). also keep all roots and c0 observed for identifiability.
    roots = ["x0", "x1", "x2", "x3", "x4", "c0"]
    Xm, mask = _inject_missingness(
        Xc,
        mechanism=mechanism,
        miss_rate=miss_rate,
        seed=seed + 12345,
        driver_cols=["x0"],
        protect_cols=roots,
    )

    # write
    Xc.to_csv(complete_path, index=False)
    Xm.to_csv(missing_path, index=False)
    mask.to_csv(mask_path, index=False)
    G.to_csv(gt_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# I/O
# -----------------------------

def run_one(setting: str, seed: int, data_dir: Path, mechanism: str, miss_rate: float, auto_generate: bool, n_synth: int, noise: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    # find matching complete file
    candidates = list(data_dir.glob(f"synth_{setting}_n*_seed{seed}_complete.csv"))

    if not candidates and setting == "interaction_xor" and auto_generate:
        _maybe_generate_interaction_xor_files(data_dir=data_dir, seed=seed, mechanism=mechanism, miss_rate=miss_rate, n=n_synth, noise=noise)
        candidates = list(data_dir.glob(f"synth_{setting}_n*_seed{seed}_complete.csv"))

    if not candidates:
        raise FileNotFoundError(f"No complete CSV found for setting={setting}, seed={seed} under {data_dir}")

    complete_path = sorted(candidates)[0]
    meta_path = complete_path.with_name(complete_path.name.replace("_complete.csv", "_metadata.json"))
    gt_path = complete_path.with_name(complete_path.name.replace("_complete.csv", "_ground_truth_G.csv"))

    miss_tag = f"{mechanism.upper()}_{int(miss_rate*100)}per.csv"
    missing_path = complete_path.with_name(complete_path.name.replace("_complete.csv", f"_{miss_tag}"))

    Xc = pd.read_csv(complete_path)
    Xm = pd.read_csv(missing_path)
    G = pd.read_csv(gt_path, index_col=0)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Xc, Xm, G, meta


# -----------------------------
# main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument(
        "--settings",
        nargs="+",
        default=["linear_gaussian", "nonlinear_mixed", "interaction_xor"],
        help="Synthetic settings to evaluate. v2 adds 'interaction_xor'.",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027, 2028, 2029])
    ap.add_argument("--mechanism", type=str, default="MAR", choices=["MCAR", "MAR", "MNAR"])
    ap.add_argument("--missing-rate", type=float, default=0.30)

    # NEW: auto-generate interaction_xor if files are missing
    ap.add_argument("--auto-generate", type=str, default="true", help="If true, auto-generate interaction_xor files when absent.")
    ap.add_argument("--n-synth", type=int, default=2000, help="n for auto-generated interaction_xor data.")
    ap.add_argument("--noise", type=float, default=0.15, help="noise std for auto-generated interaction_xor data.")

    # SNI training knobs
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--use-gpu", type=str, default="false")

    args = ap.parse_args()

    args.use_gpu = args.use_gpu.lower() in ("true", "1", "yes")
    args.auto_generate = args.auto_generate.lower() in ("true", "1", "yes")

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    all_rows: List[Dict[str, float]] = []
    nan_warnings: List[str] = []

    total_runs = len(args.settings) * len(args.seeds)
    run_idx = 0

    for setting in args.settings:
        for seed in args.seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] setting={setting}, seed={seed} ...")

            Xc, Xm, G, meta = run_one(
                setting=setting,
                seed=seed,
                data_dir=data_dir,
                mechanism=args.mechanism,
                miss_rate=args.missing_rate,
                auto_generate=args.auto_generate,
                n_synth=args.n_synth,
                noise=args.noise,
            )

            cat_vars = meta["data"]["cat_vars"]
            cont_vars = meta["data"]["cont_vars"]

            # keep consistent column order
            Xm_use = Xm[cat_vars + cont_vars]

            # -------------------------
            # 1) PriorOnly
            # -------------------------
            cfg_prior = SNIConfig(
                seed=seed,
                use_gpu=False,
                epochs=args.epochs,
                num_heads=args.num_heads,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                max_iters=args.max_iters,
                variant="SNI",
            )
            imp_prior = SNIImputer(categorical_vars=cat_vars, continuous_vars=cont_vars, config=cfg_prior)
            D_prior = build_prior_only_D(imp_prior, Xm_use)

            m_prior = compute_all_metrics(D_prior, G, nan_warnings, f"{setting}/seed{seed}/PriorOnly")
            m_prior.update({"setting": setting, "seed": seed, "method": "PriorOnly"})
            all_rows.append(m_prior)

            # -------------------------
            # 2) NoPrior
            # -------------------------
            cfg_np = SNIConfig(
                seed=seed,
                use_gpu=args.use_gpu,
                epochs=args.epochs,
                num_heads=args.num_heads,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                max_iters=args.max_iters,
                variant="NoPrior",
            )
            imp_np = SNIImputer(categorical_vars=cat_vars, continuous_vars=cont_vars, config=cfg_np)
            _ = imp_np.impute(Xm_use, X_complete=None, mask_df=None, return_artifacts=True)
            D_np = imp_np.compute_dependency_matrix()

            m_np = compute_all_metrics(D_np, G, nan_warnings, f"{setting}/seed{seed}/NoPrior")
            m_np.update({"setting": setting, "seed": seed, "method": "NoPrior"})
            all_rows.append(m_np)

            # -------------------------
            # 3) SNI
            # -------------------------
            cfg_sni = SNIConfig(
                seed=seed,
                use_gpu=args.use_gpu,
                epochs=args.epochs,
                num_heads=args.num_heads,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                max_iters=args.max_iters,
                variant="SNI",
            )
            imp_sni = SNIImputer(categorical_vars=cat_vars, continuous_vars=cont_vars, config=cfg_sni)
            _ = imp_sni.impute(Xm_use, X_complete=None, mask_df=None, return_artifacts=True)
            D_sni = imp_sni.compute_dependency_matrix()

            m_sni = compute_all_metrics(D_sni, G, nan_warnings, f"{setting}/seed{seed}/SNI")
            m_sni.update({"setting": setting, "seed": seed, "method": "SNI"})
            all_rows.append(m_sni)

            # save D matrices (debug/transparency)
            save_tag = f"{setting}_seed{seed}_{args.mechanism.upper()}_{int(args.missing_rate*100)}per"
            D_sni.to_csv(outdir / f"D_SNI_{save_tag}.csv")
            D_np.to_csv(outdir / f"D_NoPrior_{save_tag}.csv")
            D_prior.to_csv(outdir / f"D_PriorOnly_{save_tag}.csv")

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "metrics_per_run.csv", index=False)

    # aggregate mean±std
    metrics = ["AUROC_macro", "AUPRC_macro", "PrecAtK_macro", "RecAtK_macro", "Hub_Spearman"]
    agg_rows = []

    for (setting, method), sub in df.groupby(["setting", "method"], sort=False):
        row = {"setting": setting, "method": method}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(sub[m].values.astype(float)))
            row[f"{m}_std"] = float(np.nanstd(sub[m].values.astype(float)))
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)

    # enforce human-readable ordering (settings as provided; methods in SNI/NoPrior/PriorOnly)
    df_agg["setting"] = pd.Categorical(df_agg["setting"], categories=args.settings, ordered=True)
    method_order = ["SNI", "NoPrior", "PriorOnly"]
    df_agg["method"] = pd.Categorical(df_agg["method"], categories=method_order, ordered=True)
    df_agg = df_agg.sort_values(["setting", "method"])

    df_agg.to_csv(outdir / "table_S21.csv", index=False)

    caption = (
        "Sanity check on synthetic data with known ground-truth dependencies: recovery of the dependency graph using the induced matrix $D$ "
        f"({args.mechanism.upper()}, {int(args.missing_rate*100)}\\% missingness). Results are mean$\\pm$SD over seeds."
    )
    label = "tab:S21_dependency_recovery"
    write_table_s21(df_agg, out_tex=outdir / "table_S21.tex", caption=caption, label=label)

    write_s5_text(out_tex=outdir / "S5_sanity_check_snippet.tex")

    print("------------------------------------------------------------")
    if nan_warnings:
        print(f"[WARN] {len(nan_warnings)} dependency matrices contained NaN values (replaced by 0 for metrics):")
        for w in nan_warnings[:5]:
            print(f"       - {w}")
        if len(nan_warnings) > 5:
            print(f"       ... and {len(nan_warnings) - 5} more")
        print("------------------------------------------------------------")

    print(f"[DONE] Wrote: {outdir/'metrics_per_run.csv'}")
    print(f"[DONE] Wrote: {outdir/'table_S21.csv'}")
    print(f"[DONE] Wrote: {outdir/'table_S21.tex'}")
    print(f"[DONE] Wrote: {outdir/'S5_sanity_check_snippet.tex'}")


if __name__ == "__main__":
    main()
