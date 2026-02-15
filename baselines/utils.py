from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def set_categories_from_complete(
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    categorical_vars: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure categorical columns in (complete, missing) share the same category set.

    Why this matters
    ---------------
    Some baseline implementations derive category sets from the *missing* table.
    Under missingness, some categories may be absent in the observed part,
    leading to inconsistent one-hot dimensions or silently reduced label space.

    We take the category set from *_complete.csv (ground truth) and apply it to
    both complete and missing frames.

    Notes
    -----
    - If a column is numeric (Int64/float), categories will be numeric.
    - We preserve the order of pandas Categorical categories when already present.
    """
    Xc = X_complete.copy()
    Xm = X_missing.copy()

    for col in categorical_vars:
        if col not in Xc.columns or col not in Xm.columns:
            continue

        # Determine category set from complete.
        if pd.api.types.is_categorical_dtype(Xc[col]):
            cats = list(Xc[col].cat.categories)
        else:
            cats = pd.Series(Xc[col].dropna().unique()).tolist()
            # deterministic ordering when possible
            try:
                cats = sorted(cats)
            except Exception:
                pass

        Xc[col] = pd.Categorical(Xc[col], categories=cats)
        Xm[col] = pd.Categorical(Xm[col], categories=cats)

    return Xc, Xm


def fallback_fillna(
    X_imputed: pd.DataFrame,
    X_complete: pd.DataFrame,
    categorical_vars: List[str],
    continuous_vars: List[str],
) -> pd.DataFrame:
    """Fill any remaining NaNs using mean/mode from the complete data.

    This is a safety net for baselines that may leave a small number of NaNs
    (e.g., if no valid neighbors exist for a KNN cell).

    Evaluation code assumes imputed values are finite at evaluation positions.
    """
    out = X_imputed.copy()

    # Continuous: mean
    for col in continuous_vars:
        if col not in out.columns or col not in X_complete.columns:
            continue
        if out[col].isna().any():
            mean_val = pd.to_numeric(X_complete[col], errors="coerce").mean(skipna=True)
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(mean_val)

    # Categorical: mode
    for col in categorical_vars:
        if col not in out.columns or col not in X_complete.columns:
            continue
        if out[col].isna().any():
            mode_series = X_complete[col].mode(dropna=True)
            if len(mode_series) > 0:
                mode_val = mode_series.iloc[0]
                out[col] = out[col].fillna(mode_val)

    return out
