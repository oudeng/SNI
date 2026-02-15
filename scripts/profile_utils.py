"""Shared profile parsing for aggregate/viz/latex scripts."""
from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any


def load_profile(profile_spec: Optional[str]) -> Dict[str, Any]:
    """Parse --profile 'path:name' and return filter dict.

    Returns empty dict if profile_spec is None (= no filtering).
    """
    if not profile_spec:
        return {}
    path_str, _, name = profile_spec.partition(":")
    if not name:
        name = "main"
    with open(path_str) as f:
        profiles = yaml.safe_load(f)
    if name not in profiles:
        raise ValueError(f"Profile '{name}' not found. Available: {list(profiles.keys())}")
    return profiles[name]


def _expand_rate_variants(rates: List[Any]) -> List[str]:
    """Expand profile rate values to all common string representations.

    Profile may specify rates as [10, 30, 50] (ints) or ["10per", "30per"]
    (strings).  The data column may contain "10per", "30per", "0.1", "0.3",
    10, 30, etc.  This helper returns a set of all plausible string forms so
    that ``isin()`` can match regardless of format.
    """
    expanded: List[str] = []
    for r in rates:
        s = str(r).strip()
        expanded.append(s)
        # If it looks like a bare integer (e.g. 30), also add "30per"
        if re.fullmatch(r"\d+", s):
            expanded.append(f"{s}per")
        # If it's "30per", also add "30"
        m = re.fullmatch(r"(\d+)\s*per", s, re.IGNORECASE)
        if m:
            expanded.append(m.group(1))
    return expanded


def filter_dataframe(df, profile: Dict[str, Any]):
    """Apply profile filters to a summary DataFrame.

    Expected columns: algo, dataset, mechanism, rate.
    """
    if not profile:
        return df
    if profile.get("include_algos"):
        df = df[df["algo"].isin(profile["include_algos"])]
    if profile.get("datasets"):
        df = df[df["dataset"].isin(profile["datasets"])]
    if profile.get("mechanisms"):
        df = df[df["mechanism"].isin(profile["mechanisms"])]
    if profile.get("rates"):
        expanded = _expand_rate_variants(profile["rates"])
        df = df[df["rate"].astype(str).isin(expanded)]
    return df
