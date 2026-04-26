#!/usr/bin/env python3
"""Problem 7b: ridge-regularized MNL re-estimation on the AI-labelled sample.

Reads the row-level AI sample CSV (default: 500 queries / 9 005 rows from
`inputs/ai_booking_sample500_rows.csv`), z-scores all eight features within
the sample, and fits the MNL log-likelihood with an L2 penalty on the
non-intercept coefficients.

The choice of `lambda = 1` is a stabilization device, not a tuned
hyper-parameter; AI choices are highly deterministic and an unregularized
MLE blows up. Coefficient signs are interpretable; magnitudes should not be
read as exact behavioural parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
]


def negative_log_likelihood(beta: np.ndarray, groups, choice_col: str) -> float:
    nll = 0.0
    for _, group in groups:
        X = group[FEATURES].values
        y = group[choice_col].values
        u = beta[0] + X @ beta[1:]
        max_u = max(0.0, float(np.max(u)))
        exp_u = np.exp(u - max_u)
        denom = np.exp(0.0 - max_u) + exp_u.sum()
        if y.sum() == 1:
            chosen = int(np.where(y == 1)[0][0])
            log_prob = (u[chosen] - max_u) - np.log(denom)
        elif y.sum() == 0:
            log_prob = (0.0 - max_u) - np.log(denom)
        else:
            raise ValueError("query has more than one chosen alternative")
        nll -= log_prob
    return nll


def penalized_nll(beta: np.ndarray, groups, choice_col: str, lam: float) -> float:
    return negative_log_likelihood(beta, groups, choice_col) + lam * float(
        np.sum(beta[1:] ** 2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit ridge MNL on AI-labelled sample.")
    parser.add_argument(
        "--ai-sample",
        default="problem7/inputs/ai_booking_sample500_rows.csv",
        help="Row-level AI sample CSV.",
    )
    parser.add_argument(
        "--choice-col",
        default="booking_ai_sample500",
        help="Column with the binary AI choice (default booking_ai_sample500).",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--output-json",
        default="problem7/results/problem7b_results.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.ai_sample)
    df = df[["srch_id", args.choice_col] + FEATURES].copy()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    feature_means = df[FEATURES].mean()
    feature_stds = df[FEATURES].std().replace(0, 1)

    df_scaled = df.copy()
    df_scaled[FEATURES] = (df_scaled[FEATURES] - feature_means) / feature_stds

    groups = list(df_scaled.groupby("srch_id"))

    initial_beta = np.zeros(1 + len(FEATURES))
    result = minimize(
        penalized_nll,
        initial_beta,
        args=(groups, args.choice_col, args.lam),
        method="L-BFGS-B",
        options={"maxiter": args.maxiter},
    )

    beta_hat = result.x
    coef_dict = {"intercept": float(beta_hat[0])}
    for i, name in enumerate(FEATURES, start=1):
        coef_dict[name] = float(beta_hat[i])

    payload = {
        "problem": "7b",
        "ai_sample_path": str(Path(args.ai_sample).resolve()),
        "choice_col": args.choice_col,
        "lambda": args.lam,
        "n_queries": int(df_scaled["srch_id"].nunique()),
        "n_rows": int(len(df_scaled)),
        "scaling_used": "z-score on the AI sample (continuous and binary alike)",
        "scaling_means": feature_means.to_dict(),
        "scaling_stds": feature_stds.to_dict(),
        "ai_regularized_coefficients": coef_dict,
        "fit": {
            "converged": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit) if hasattr(result, "nit") else None,
            "penalized_nll": float(result.fun),
            "unpenalized_nll": float(
                negative_log_likelihood(beta_hat, groups, args.choice_col)
            ),
        },
        "interpretation_note": (
            "Magnitudes are not directly comparable to the Problem 1 normalized "
            "coefficients because (i) lambda=1 is a stabilization device, not a "
            "tuned regularizer, and (ii) the AI fit z-scores all features on the "
            "AI sample whereas Problem 1 left binary features unscaled. "
            "Compare signs and broad relative magnitudes only."
        ),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
