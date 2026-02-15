"""
missing_data_generator.py
========================================
Utility module for generating synthetic missingness (MCAR / MAR / MNAR)
for tabular datasets and exporting reproducible artifacts for downstream
imputation experiments (e.g., SNI).

Design goals (refactor of Generate_Missing_Data_v4.py):
1) Separate **core functions** (this file) from a **main controller / CLI**.
2) Reproducible generation via explicit RNG seed (no global RNG state).
3) Configurable column typing (continuous vs categorical) to avoid dtype pitfalls.
4) Optional mask + metadata export to support auditability & ablation experiments.

Notes on mechanisms
-------------------
- MCAR: Missing Completely At Random (cell-wise uniform).
- MAR: Missing At Random (row-wise propensity depends on observed driver columns).
- MNAR: Missing Not At Random (propensity depends on the feature's own values).

This module returns both the missing DataFrame and the boolean mask (True=missing)
so that downstream experiments can exactly reuse the same missing pattern.

Author: (refactored by GPT-5.2 Pro)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ColumnTypes:
    """Container for column-type information."""
    continuous: List[str]
    categorical: List[str]
    excluded: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {"continuous": self.continuous, "categorical": self.categorical, "excluded": self.excluded}


@dataclass
class GenerationResult:
    """Return object for missing-data generation."""
    data_missing: pd.DataFrame
    mask: np.ndarray  # bool, True indicates missing
    actual_rate: float
    column_types: ColumnTypes
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_rate(rate: float, style: str = "percent") -> str:
    """
    Convert a float missing rate (e.g., 0.3) to a stable filename token.

    Parameters
    ----------
    rate : float
        Missing rate in [0, 1].
    style : {"percent", "float", "p"}
        - "percent": 0.3 -> "30per"   (default, compact and filesystem-safe)
        - "float":   0.3 -> "0.3"     (human readable; includes a dot)
        - "p":       0.3 -> "0p3"     (dot replaced by "p")
    """
    if rate < 0 or rate > 1:
        raise ValueError(f"rate must be in [0,1], got {rate}")
    style = style.lower().strip()
    if style == "percent":
        return f"{int(round(rate * 100)):02d}per"
    if style == "float":
        # keep one decimal if it matches common experimental settings; otherwise use full string
        return f"{rate:g}"
    if style == "p":
        return f"{rate:g}".replace(".", "p")
    raise ValueError(f"Unknown style: {style}. Use percent/float/p.")

def parse_comma_list(text: Optional[str]) -> List[str]:
    """
    Parse comma-separated list from CLI args.
    Accepts: "A,B,C" or "A, B, C". Returns [] for None/empty.
    """
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def _safe_columns(df: pd.DataFrame, cols: Sequence[str], arg_name: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{arg_name}: columns not found in input CSV: {missing}")
    return list(cols)


def infer_column_types(
    df: pd.DataFrame,
    categorical_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    *,
    treat_bool_as_categorical: bool = True,
    int_as_categorical_unique_threshold: int = 20,
    int_as_categorical_unique_ratio: float = 0.05,
) -> ColumnTypes:
    """
    Infer continuous/categorical columns with optional user overrides.

    Why this is needed:
    Many real datasets store categorical features as integers (0/1, 1/2/3, 97/98/99/100).
    Pure dtype-based detection would misclassify them as numeric, which later affects
    both missingness generation and evaluation. This function allows:
      - explicit specification (recommended), or
      - heuristic detection for integer columns.

    Heuristic rule (only if user does not specify):
      - object/category -> categorical
      - bool -> categorical (optional)
      - integer -> categorical if n_unique <= threshold OR n_unique/n_rows <= ratio
      - float -> continuous
    """
    exclude_cols = list(exclude_cols or [])
    exclude_cols = _safe_columns(df, exclude_cols, "exclude_cols")

    user_cat = list(categorical_cols or [])
    user_con = list(continuous_cols or [])

    if user_cat:
        user_cat = _safe_columns(df, user_cat, "categorical_cols")
    if user_con:
        user_con = _safe_columns(df, user_con, "continuous_cols")

    # If user provides either list, trust them and derive the rest (minus excluded)
    if user_cat or user_con:
        all_cols = [c for c in df.columns if c not in exclude_cols]
        cat_set = set(user_cat)
        con_set = set(user_con)
        # If only one is specified, derive the other as complement
        if user_cat and not user_con:
            con_set = set(all_cols) - cat_set
        if user_con and not user_cat:
            cat_set = set(all_cols) - con_set
        # Remove excluded cols
        cat = [c for c in all_cols if c in cat_set and c not in exclude_cols]
        con = [c for c in all_cols if c in con_set and c not in exclude_cols]
        # Ensure disjoint
        overlap = set(cat) & set(con)
        if overlap:
            raise ValueError(f"categorical_cols and continuous_cols overlap: {sorted(overlap)}")
        return ColumnTypes(continuous=con, categorical=cat, excluded=exclude_cols)

    # Otherwise: heuristic inference
    cat: List[str] = []
    con: List[str] = []

    n_rows = len(df)
    for col in df.columns:
        if col in exclude_cols:
            continue
        s = df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            cat.append(col)
            continue
        if treat_bool_as_categorical and pd.api.types.is_bool_dtype(s):
            cat.append(col)
            continue
        if pd.api.types.is_integer_dtype(s):
            nunq = s.nunique(dropna=True)
            if nunq <= int_as_categorical_unique_threshold or (nunq / max(n_rows, 1)) <= int_as_categorical_unique_ratio:
                cat.append(col)
            else:
                con.append(col)
            continue
        # default numeric floats -> continuous
        if pd.api.types.is_numeric_dtype(s):
            con.append(col)
        else:
            # fallback: treat unknown as categorical
            cat.append(col)

    return ColumnTypes(continuous=con, categorical=cat, excluded=exclude_cols)


def apply_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Return a copy of df with mask positions set to NaN."""
    if mask.shape != df.shape:
        raise ValueError(f"mask shape {mask.shape} != df shape {df.shape}")
    # IMPORTANT: do NOT use df.values[...] assignment for mixed dtypes, it may not be a view.
    # DataFrame.mask handles mixed dtypes correctly.
    out = df.copy()
    return out.mask(mask)

def missing_rate(df: pd.DataFrame) -> float:
    """Overall missing rate (fraction of NaN cells)."""
    return float(df.isna().to_numpy().mean())


def per_column_missing_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Per-column missing rate."""
    return {c: float(df[c].isna().mean()) for c in df.columns}


def cast_dataframe_for_generation(df: pd.DataFrame, col_types: ColumnTypes) -> pd.DataFrame:
    """\
    Cast columns to stable dtypes *before* applying missing masks.

    Motivation:
      - If a column is int64 and we insert NaNs, pandas upcasts it to float64 and
        values like 82 become 82.0 in the exported CSV.
      - For downstream pipelines (e.g., SNI) and reviewer auditability, we prefer:
          * continuous -> float64
          * integer-coded categorical -> Int64 (nullable integer)
          * string categorical -> category
      - Excluded columns (e.g., ID) are kept observed; if numeric & integer-like,
        we cast them to Int64 as well.

    This function does *not* change the semantic values (it only normalises dtype).
    """

    out = df.copy()

    # continuous -> float64
    for c in col_types.continuous:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float64')

    # categorical -> Int64 if integer-like, else category
    for c in col_types.categorical:
        if c not in out.columns:
            continue
        s = out[c]
        num = pd.to_numeric(s, errors='coerce')
        non_na = num.dropna()
        if len(non_na) > 0:
            frac = (non_na - non_na.round()).abs()
            if bool((frac <= 1e-8).all()):
                out[c] = num.round().astype('Int64')
            else:
                out[c] = s.astype('category')
        else:
            out[c] = s.astype('category')

    # excluded -> keep, but if numeric & integer-like cast to Int64 (for clean CSVs)
    for c in col_types.excluded:
        if c not in out.columns:
            continue
        s = out[c]
        num = pd.to_numeric(s, errors='coerce')
        non_na = num.dropna()
        if len(non_na) == 0:
            continue
        frac = (non_na - non_na.round()).abs()
        if bool((frac <= 1e-8).all()):
            out[c] = num.round().astype('Int64')

    return out


def _enforce_min_missing_per_column(
    mask: np.ndarray,
    propensity: np.ndarray,
    *,
    min_missing_per_col: int,
    rng: np.random.Generator,
    excluded_cols_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """\
    Ensure each *non-excluded* column has at least ``min_missing_per_col`` missing entries.

    Notes
    -----
    - Excluded columns (e.g., an always-observed MAR driver like ``ID``) are left fully observed.
    - This is important for *strict MAR*: the driver column must not become missing.
    """
    n_rows, n_cols = mask.shape
    if min_missing_per_col <= 0:
        return mask

    if excluded_cols_mask is not None:
        excluded_cols_mask = np.asarray(excluded_cols_mask, dtype=bool)
        if excluded_cols_mask.shape != (n_cols,):
            raise ValueError(
                f"excluded_cols_mask shape {excluded_cols_mask.shape} != ({n_cols},)"
            )

    new_mask = mask.copy()
    for j in range(n_cols):
        if excluded_cols_mask is not None and bool(excluded_cols_mask[j]):
            continue

        cur = int(new_mask[:, j].sum())
        if cur >= min_missing_per_col:
            continue
        need = min_missing_per_col - cur

        # Candidate rows not missing currently
        cand = np.where(~new_mask[:, j])[0]
        if cand.size == 0:
            continue

        # Sort candidates by propensity descending
        p = propensity[cand, j]
        # In case of ties, add a little random noise for stability
        noise = rng.random(cand.size) * 1e-12
        order = np.argsort(-(p + noise))
        chosen = cand[order[: min(need, cand.size)]]
        new_mask[chosen, j] = True

    return new_mask


def calibrate_mask_to_rate(
    mask: np.ndarray,
    propensity: np.ndarray,
    target_rate: float,
    *,
    tolerance: float,
    rng: np.random.Generator,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """\
    Adjust a mask to achieve ``target_rate`` missingness **over eligible cells only**.

    Eligible cells are those with ``exclude_mask == False``.
    This matters when you introduce always-observed columns (e.g., ``ID``) that must
    never be masked, while still wanting the *reported* missing rate (0.1/0.3/0.5)
    to apply to the *imputation variables*.

    Strategy:
      - If missing is too low: add missing cells among currently observed eligible cells,
        biasing selection towards higher propensity.
      - If missing is too high: restore cells among currently missing eligible cells,
        biasing selection towards lower propensity.

    This preserves MAR/MNAR patterns better than random add/remove.
    """
    if exclude_mask is not None:
        if exclude_mask.shape != mask.shape:
            raise ValueError(f"exclude_mask shape {exclude_mask.shape} != mask shape {mask.shape}")
        eligible = ~exclude_mask
    else:
        eligible = np.ones_like(mask, dtype=bool)

    total = int(eligible.sum())
    if total == 0:
        # nothing to calibrate
        return mask

    target = int(round(float(target_rate) * total))

    new_mask = mask.copy()
    # guarantee excluded cells are never missing
    if exclude_mask is not None:
        new_mask[exclude_mask] = False

    cur = int(new_mask[eligible].sum())

    # Fast path
    if abs(cur / total - float(target_rate)) <= float(tolerance):
        return new_mask

    if cur < target:
        k = target - cur
        # pick from observed eligible positions
        cand_pos = np.where((~new_mask & eligible).ravel())[0]
        if cand_pos.size == 0:
            return new_mask
        p = propensity.ravel()[cand_pos].astype(float)
        p = np.clip(p, 0.0, 1.0)
        if float(p.sum()) <= 0.0:
            chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False)
        else:
            w = p / p.sum()
            chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False, p=w)
        new_mask.ravel()[chosen] = True
        if exclude_mask is not None:
            new_mask[exclude_mask] = False
        return new_mask

    # cur > target
    k = cur - target
    cand_pos = np.where((new_mask & eligible).ravel())[0]
    if cand_pos.size == 0:
        return new_mask
    p = propensity.ravel()[cand_pos].astype(float)
    p = np.clip(p, 0.0, 1.0)
    # Restore preferentially where propensity is LOW (least likely to be missing)
    score = 1.0 - p
    if float(score.sum()) <= 0.0:
        chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False)
    else:
        w = score / score.sum()
        chosen = rng.choice(cand_pos, size=min(k, cand_pos.size), replace=False, p=w)
    new_mask.ravel()[chosen] = False

    if exclude_mask is not None:
        new_mask[exclude_mask] = False

    return new_mask


# ---------------------------------------------------------------------
# Propensity (probability) builders
# ---------------------------------------------------------------------

def _mcar_propensity(df: pd.DataFrame, rate: float, *, exclude_mask: Optional[np.ndarray] = None) -> np.ndarray:
    p = np.full(df.shape, float(rate), dtype=float)
    if exclude_mask is not None:
        p[exclude_mask] = 0.0
    return p


def _standardize_series(x: pd.Series) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    sd = sd if sd > 1e-12 else 1.0
    return (arr - mu) / sd


def _mar_propensity(
    df: pd.DataFrame,
    rate: float,
    *,
    driver_cols: Sequence[str],
    exclude_mask: Optional[np.ndarray] = None,
    logistic_scale: float = 1.0,
) -> np.ndarray:
    """
    Row-wise MAR propensity:
      p_row = sigmoid(scale * sum(z(driver_cols)))
      then scaled to match target mean.

    All columns share the same p_row (row-level missingness).
    """
    if len(driver_cols) == 0:
        raise ValueError("MAR requires at least one driver column.")

    z_sum = None
    for c in driver_cols:
        z = _standardize_series(df[c])
        z_sum = z if z_sum is None else (z_sum + z)

    # base sigmoid
    lin = logistic_scale * z_sum
    p_row = 1.0 / (1.0 + np.exp(-lin))
    # scale to desired mean (avoid division by zero)
    mean_p = float(np.mean(p_row))
    if mean_p <= 1e-12:
        p_row = np.full_like(p_row, rate, dtype=float)
    else:
        p_row = p_row * (rate / mean_p)
    p_row = np.clip(p_row, 0.0, 1.0)

    p = np.repeat(p_row[:, None], df.shape[1], axis=1)
    if exclude_mask is not None:
        p[exclude_mask] = 0.0
    return p


def _mnar_propensity(
    df: pd.DataFrame,
    rate: float,
    *,
    col_types: ColumnTypes,
    rng: np.random.Generator,
    exclude_mask: Optional[np.ndarray] = None,
    # numeric strategy params
    q_low: float = 0.25,
    q_mid: float = 0.50,
    q_high: float = 0.75,
    p_low_mult: float = 0.4,
    p_mid_low_mult: float = 0.8,
    p_mid_high_mult: float = 1.2,
    p_high_mult: float = 1.8,
    # categorical strategy params
    cat_high_frac: float = 0.5,
    cat_high_mult: float = 1.5,
    cat_low_mult: float = 0.7,
) -> np.ndarray:
    """
    Column-wise MNAR propensity.

    Numeric columns: propensity increases for more "extreme" values, using
    quantile bins.

    Categorical columns: some categories are randomly designated as "high missing".
    """
    n_rows, n_cols = df.shape
    p = np.zeros((n_rows, n_cols), dtype=float)

    # numeric
    for col in col_types.continuous:
        j = df.columns.get_loc(col)
        values = df[col].to_numpy(dtype=float)

        # If NaNs exist in input, np.percentile will fail. Caller should ensure complete input,
        # but we still guard with nanpercentile.
        q1 = np.nanpercentile(values, q_low * 100.0)
        q2 = np.nanpercentile(values, q_mid * 100.0)
        q3 = np.nanpercentile(values, q_high * 100.0)

        probs = np.empty_like(values, dtype=float)
        # highest
        probs[values > q3] = rate * p_high_mult
        # mid-high
        probs[(values > q2) & (values <= q3)] = rate * p_mid_high_mult
        # mid-low
        probs[(values > q1) & (values <= q2)] = rate * p_mid_low_mult
        # low
        probs[values <= q1] = rate * p_low_mult

        probs = np.clip(probs, 0.0, 1.0)
        p[:, j] = probs

    # categorical
    for col in col_types.categorical:
        j = df.columns.get_loc(col)
        s = df[col]
        # keep as object for stable uniqueness
        uniques = pd.unique(s.astype("object"))
        uniques = [u for u in uniques if pd.notna(u)]
        if len(uniques) == 0:
            # degenerate column, fall back to uniform rate
            p[:, j] = rate
            continue

        n_high = max(1, int(round(len(uniques) * cat_high_frac)))
        high_cats = set(rng.choice(uniques, size=n_high, replace=False).tolist())

        probs = np.empty(n_rows, dtype=float)
        vals = s.astype("object").to_numpy()
        for i, v in enumerate(vals):
            if pd.isna(v):
                probs[i] = rate
            elif v in high_cats:
                probs[i] = rate * cat_high_mult
            else:
                probs[i] = rate * cat_low_mult
        probs = np.clip(probs, 0.0, 1.0)
        p[:, j] = probs

    # For columns not covered (e.g., excluded), keep zero propensity
    if exclude_mask is not None:
        p[exclude_mask] = 0.0

    return p


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def generate_missing_dataset(
    df: pd.DataFrame,
    *,
    mechanism: str,
    rate: float,
    seed: int = 2025,
    dataset_name: str = "DATA",
    col_types: Optional[ColumnTypes] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    # calibration / constraints
    tolerance: float = 0.01,
    min_missing_per_col: int = 1,
    # MAR params
    mar_driver_cols: Optional[Sequence[str]] = None,
    mar_logistic_scale: float = 1.0,
    strict_mar: bool = True,
    # MNAR params
    mnar_params: Optional[Dict[str, Any]] = None,
    # input checks
    allow_input_missing: bool = False,
) -> GenerationResult:
    """\
    Generate a missing-data dataset according to a specified mechanism.

    Important (Strict MAR)
    ----------------------
    If ``mechanism='MAR'`` and ``strict_mar=True`` (default), missingness is driven ONLY by
    user-specified ``mar_driver_cols`` which are guaranteed to remain observed (never masked).

    This is the most "review-safe" MAR simulation: the driver is always observed, so the
    generated mask is genuinely MAR (not accidentally MNAR due to the driver being missing).

    Rate definition with excluded columns
    -------------------------------------
    ``rate`` is calibrated over *eligible cells* (i.e., excluding ``exclude_cols`` / driver cols).
    We also report both:
      - ``actual_rate_eligible``: missing rate over eligible cells (should match ``rate``)
      - ``actual_rate_all``: missing rate over all cells (slightly smaller if you add an ID column)
    """
    if rate < 0 or rate > 1:
        raise ValueError(f"rate must be in [0,1], got {rate}")

    if (not allow_input_missing) and df.isna().any().any():
        raise ValueError(
            "Input dataframe already contains missing values. For ground-truth evaluation, "
            "please provide a complete dataset. If you really want to proceed, set "
            "allow_input_missing=True."
        )

    rng = np.random.default_rng(seed)

    mechanism_u = mechanism.strip().upper()
    if mechanism_u not in {"MCAR", "MAR", "MNAR"}:
        raise ValueError(f"Unknown mechanism: {mechanism}. Use MCAR/MAR/MNAR.")

    # ------------------------------------------------------------------
    # Engineering safety: keep row-identifier columns observed.
    # If a column named 'ID' exists, we treat it as excluded by default
    # (never masked). This avoids downstream alignment/metric corruption.
    # If you truly want to mask an identifier, rename it or pass a different
    # identifier column name.
    # ------------------------------------------------------------------
    if "ID" in df.columns:
        if exclude_cols is None:
            exclude_cols = ["ID"]
        else:
            ex_list = list(exclude_cols)
            if "ID" not in ex_list:
                exclude_cols = list(dict.fromkeys(ex_list + ["ID"]))

    # Strict MAR: require explicit drivers and force them to be excluded (never masked)
    drivers_used: List[str] = []
    if mechanism_u == "MAR":
        if strict_mar:
            if mar_driver_cols is None or len(list(mar_driver_cols)) == 0:
                raise ValueError(
                    "strict_mar=True requires explicit mar_driver_cols (e.g., --mar-driver-cols ID)."
                )
            drivers_used = _safe_columns(df, list(mar_driver_cols), "mar_driver_cols")
            # ensure drivers are excluded (always observed)
            exclude_cols = list(dict.fromkeys(list(exclude_cols or []) + drivers_used))
        else:
            # non-strict MAR: allow default drivers
            if mar_driver_cols is None or len(list(mar_driver_cols)) == 0:
                drivers_used = []
            else:
                drivers_used = _safe_columns(df, list(mar_driver_cols), "mar_driver_cols")

    if col_types is None:
        col_types = infer_column_types(
            df,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            exclude_cols=exclude_cols,
        )
    else:
        # If strict MAR, also ensure col_types reflects the forced exclusions
        if mechanism_u == "MAR" and strict_mar and len(drivers_used) > 0:
            ex = list(dict.fromkeys(list(col_types.excluded) + drivers_used))
            ex_set = set(ex)
            con = [c for c in col_types.continuous if c not in ex_set]
            cat = [c for c in col_types.categorical if c not in ex_set]
            col_types = ColumnTypes(continuous=con, categorical=cat, excluded=ex)

    # Cast for stable dtypes prior to masking (prevents 82 -> 82.0 artifacts)
    df_cast = cast_dataframe_for_generation(df, col_types)

    # Build exclude mask for propensity (excluded cols never masked)
    exclude_mask = np.zeros(df_cast.shape, dtype=bool)
    excluded_cols_mask = np.zeros(df_cast.shape[1], dtype=bool)
    if col_types.excluded:
        for col in col_types.excluded:
            j = df_cast.columns.get_loc(col)
            exclude_mask[:, j] = True
            excluded_cols_mask[j] = True

    if mechanism_u == "MCAR":
        propensity = _mcar_propensity(df_cast, rate, exclude_mask=exclude_mask)
    elif mechanism_u == "MAR":
        if strict_mar:
            drivers = drivers_used
        else:
            if drivers_used:
                drivers = drivers_used
            else:
                drivers = col_types.continuous[:2]

        if len(drivers) == 0:
            # fallback: MCAR if cannot find a driver
            propensity = _mcar_propensity(df_cast, rate, exclude_mask=exclude_mask)
        else:
            propensity = _mar_propensity(
                df_cast,
                rate,
                driver_cols=drivers,
                exclude_mask=exclude_mask,
                logistic_scale=mar_logistic_scale,
            )
    else:  # MNAR
        params = dict(mnar_params or {})
        propensity = _mnar_propensity(
            df_cast,
            rate,
            col_types=col_types,
            rng=rng,
            exclude_mask=exclude_mask,
            **params,
        )

    # Sample initial mask
    mask = rng.random(df_cast.shape) < propensity

    # Hard guarantee: excluded cells are never missing
    if exclude_mask is not None:
        mask[exclude_mask] = False

    # Enforce per-column coverage (skip excluded columns)
    mask = _enforce_min_missing_per_column(
        mask,
        propensity,
        min_missing_per_col=min_missing_per_col,
        rng=rng,
        excluded_cols_mask=excluded_cols_mask,
    )
    if exclude_mask is not None:
        mask[exclude_mask] = False

    # Calibrate to target rate over eligible cells
    mask = calibrate_mask_to_rate(mask, propensity, rate, tolerance=tolerance, rng=rng, exclude_mask=exclude_mask)
    if exclude_mask is not None:
        mask[exclude_mask] = False

    # Apply again min-missing constraint (calibration may break it slightly)
    mask = _enforce_min_missing_per_column(
        mask,
        propensity,
        min_missing_per_col=min_missing_per_col,
        rng=rng,
        excluded_cols_mask=excluded_cols_mask,
    )
    if exclude_mask is not None:
        mask[exclude_mask] = False

    df_missing = apply_mask(df_cast, mask)

    # Missing-rate bookkeeping
    actual_rate_all = missing_rate(df_missing)
    if exclude_mask is not None and int((~exclude_mask).sum()) > 0:
        eligible = ~exclude_mask
        actual_rate_eligible = float(mask[eligible].mean())
    else:
        actual_rate_eligible = float(actual_rate_all)

    meta: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "mechanism": mechanism_u,
        "target_rate": float(rate),
        "actual_rate_eligible": float(actual_rate_eligible),
        "actual_rate_all": float(actual_rate_all),
        "seed": int(seed),
        "n_rows": int(df_cast.shape[0]),
        "n_cols": int(df_cast.shape[1]),
        "columns": list(df_cast.columns),
        "column_types": col_types.as_dict(),
        "tolerance": float(tolerance),
        "min_missing_per_col": int(min_missing_per_col),
    }

    if mechanism_u == "MAR":
        meta["strict_mar"] = bool(strict_mar)
        meta["mar_driver_cols"] = list(drivers_used) if strict_mar else (list(mar_driver_cols) if mar_driver_cols else list(col_types.continuous[:2]))
        meta["mar_logistic_scale"] = float(mar_logistic_scale)

    if mechanism_u == "MNAR":
        meta["mnar_params"] = mnar_params or {}

    # Add per-column missing rates for audit
    meta["per_column_missing_rate"] = per_column_missing_rates(df_missing)

    return GenerationResult(
        data_missing=df_missing,
        mask=mask,
        actual_rate=float(actual_rate_eligible),
        column_types=col_types,
        metadata=meta,
    )



def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Save mask to .npy (recommended) or .csv depending on suffix.
    .npy preserves exact dtype/shape and is fast.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, mask.astype(np.uint8))
    elif suffix == ".csv":
        pd.DataFrame(mask.astype(int)).to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported mask format: {suffix}. Use .npy or .csv")


def save_metadata(metadata: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)