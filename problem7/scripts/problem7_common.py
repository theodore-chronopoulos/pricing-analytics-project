"""Shared utilities for Problem 7.

Conventions kept consistent with Problem 1 (`scripts/mnl_utils.py`):
- continuous features are z-scored using full-data scaling stats,
- binary features (`prop_brand_bool`, `promotion_flag`) are NOT scaled.

The two functions in this module produce per-query MNL utilities and per-query
choice probabilities under those conventions; both are used by the held-out
predictor in `problem7c_predictive_eval.py`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


CONTINUOUS_FEATURES = [
    "prop_starrating",
    "prop_review_score",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
]

BINARY_FEATURES = [
    "prop_brand_bool",
    "promotion_flag",
]

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES


def compute_scaling_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return continuous means, continuous stds, continuous medians, binary medians.

    Sample standard deviation (ddof=1) is used. For the Expedia dataset this is
    numerically identical to Problem 1's population-std convention to four
    decimals because $N \approx 153{,}000$.
    """
    cont_means = df[CONTINUOUS_FEATURES].mean()
    cont_stds = df[CONTINUOUS_FEATURES].std().replace(0, 1)
    cont_medians = df[CONTINUOUS_FEATURES].median()
    bin_medians = df[BINARY_FEATURES].median()
    return cont_means, cont_stds, cont_medians, bin_medians


def prepare_features(
    group: pd.DataFrame,
    cont_means: pd.Series,
    cont_stds: pd.Series,
    cont_medians: pd.Series,
    bin_medians: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize continuous features, leave binaries at raw 0/1.

    Returns a (cleaned_group, design_matrix) tuple.
    """
    g = group.copy()
    g[CONTINUOUS_FEATURES] = (
        g[CONTINUOUS_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cont_medians)
    )
    g[BINARY_FEATURES] = (
        g[BINARY_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(bin_medians)
    )
    X = pd.DataFrame(index=g.index)
    X[CONTINUOUS_FEATURES] = (g[CONTINUOUS_FEATURES] - cont_means) / cont_stds
    X[BINARY_FEATURES] = g[BINARY_FEATURES]
    return g, X


def utilities(
    group: pd.DataFrame,
    beta: Dict[str, float],
    cont_means: pd.Series,
    cont_stds: pd.Series,
    cont_medians: pd.Series,
    bin_medians: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (cleaned_group, per-row utility) using `beta` keyed by feature."""
    g, X = prepare_features(group, cont_means, cont_stds, cont_medians, bin_medians)
    u = pd.Series(beta["intercept"], index=g.index, dtype=float)
    for f in ALL_FEATURES:
        u += beta[f] * X[f]
    return g, u


def probabilities(
    group: pd.DataFrame,
    beta: Dict[str, float],
    cont_means: pd.Series,
    cont_stds: pd.Series,
    cont_medians: pd.Series,
    bin_medians: pd.Series,
) -> Tuple[pd.Series, float]:
    """MNL choice probabilities, with outside option utility 0.

    Returns (per-hotel probabilities indexed like the input group, no-purchase probability).
    """
    _, u = utilities(group, beta, cont_means, cont_stds, cont_medians, bin_medians)
    max_u = max(0.0, float(u.max()))
    exp_u = np.exp(u - max_u)
    exp_outside = np.exp(0 - max_u)
    denom = exp_outside + exp_u.sum()
    hotel_probs = exp_u / denom
    no_purchase_prob = exp_outside / denom
    return hotel_probs, float(no_purchase_prob)


HUMAN_BETA_FROM_PROBLEM1 = {
    "intercept": -1.9815321907864278,
    "prop_starrating": 0.4081249536151655,
    "prop_review_score": 0.10876096623704055,
    "prop_brand_bool": 0.22992269948768013,
    "prop_location_score": 0.02202632301274303,
    "prop_accesibility_score": 0.04344412341249515,
    "prop_log_historical_price": -0.06686945209512846,
    "price_usd": -1.3311099651462353,
    "promotion_flag": 0.45402977040295234,
}
