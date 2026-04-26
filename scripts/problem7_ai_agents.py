#!/usr/bin/env python3
"""Solve Problem 7 (AI Agents as Customers) end-to-end.

Sub-parts 7a, 7b, 7c are run sequentially; each prints a header and a
summary block to stdout, and the final unified payload is optionally saved
to JSON.

Manual artefacts produced from in-chat ChatGPT prompting are read from
`problem7/inputs/` by default:

  - `ai_booking_sample500_rows.csv`   row-level AI sample for 7a/7b
  - `ai_heldout_predictions.csv`      AI's held-out predictions for 7c
  - `ai_generation_prompt_template.txt` (informational; used in 7a)

Reproducibility:
  - 7c uses ``numpy.random.default_rng(42)`` for the held-out split, matching
    the seed under which the AI's held-out predictions were collected.

Dependencies (Problem 7 only):
  - pandas, numpy, scipy, statsmodels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.stats.proportion import proportion_confint


# ---------------------------------------------------------------------------
# Conventions kept consistent with Problem 1 (`scripts/mnl_utils.py`):
# continuous features are z-scored using full-data scaling stats, binary
# features are kept at raw 0/1.
# ---------------------------------------------------------------------------

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


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# Problem 7a: build the AI booking dataset from manual prompt responses.
# =============================================================================

def run_7a(args: argparse.Namespace) -> Dict[str, object]:
    banner("Problem 7a: build AI-labelled booking dataset")

    df = pd.read_csv(args.data)
    df["alt_id"] = df.groupby("srch_id").cumcount() + 1

    sample = pd.read_csv(args.ai_sample_rows)
    decisions = (
        sample.groupby("srch_id")["ai_choice_sample500"]
        .first()
        .rename("ai_choice")
        .reset_index()
    )
    decisions["ai_choice"] = decisions["ai_choice"].astype(str).str.replace(
        r"^alt_id_", "", regex=True
    )

    sample_ids = set(decisions["srch_id"].unique())
    df["in_ai_sample"] = df["srch_id"].isin(sample_ids)
    df["ai_choice"] = df["srch_id"].map(decisions.set_index("srch_id")["ai_choice"])

    def booking_for_row(row: pd.Series) -> float:
        if not row["in_ai_sample"]:
            return np.nan
        choice = row["ai_choice"]
        if choice == "NO_PURCHASE":
            return 0
        try:
            return 1 if int(float(choice)) == int(row["alt_id"]) else 0
        except (TypeError, ValueError):
            return 0

    df["booking_ai"] = df.apply(booking_for_row, axis=1)

    sample_df = df[df["in_ai_sample"]].copy()
    n_buy = int(sample_df.groupby("srch_id")["booking_ai"].sum().eq(1).sum())
    n_no = int(sample_df.groupby("srch_id")["booking_ai"].sum().eq(0).sum())

    if args.merged_csv:
        Path(args.merged_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.merged_csv, index=False)
        print(f"Wrote merged AI-labelled dataset to {args.merged_csv}")

    print(f"Total rows in data.csv:        {len(df):>7}")
    print(f"Total queries:                 {df['srch_id'].nunique():>7}")
    print(f"AI sample queries:             {len(sample_ids):>7}")
    print(f"AI sample rows:                {len(sample_df):>7}")
    print(f"AI BUY queries:                {n_buy:>7}")
    print(f"AI NO_PURCHASE queries:        {n_no:>7}")

    return {
        "ai_sample_path": str(Path(args.ai_sample_rows).resolve()),
        "merged_csv": str(Path(args.merged_csv).resolve()) if args.merged_csv else None,
        "n_total_rows": int(len(df)),
        "n_total_queries": int(df["srch_id"].nunique()),
        "n_sample_queries": int(len(sample_ids)),
        "n_sample_rows": int(len(sample_df)),
        "n_ai_buy_queries": n_buy,
        "n_ai_no_purchase_queries": n_no,
    }


# =============================================================================
# Problem 7b: ridge-regularized MNL on the AI sample.
# =============================================================================

def _negative_log_likelihood(
    beta: np.ndarray,
    groups: Sequence[Tuple[int, pd.DataFrame]],
    choice_col: str,
    feature_cols: Sequence[str],
) -> float:
    nll = 0.0
    for _, group in groups:
        X = group[feature_cols].values
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


def _penalized_nll(
    beta: np.ndarray,
    groups: Sequence[Tuple[int, pd.DataFrame]],
    choice_col: str,
    feature_cols: Sequence[str],
    lam: float,
) -> float:
    return _negative_log_likelihood(beta, groups, choice_col, feature_cols) + lam * float(
        np.sum(beta[1:] ** 2)
    )


def run_7b(args: argparse.Namespace) -> Dict[str, object]:
    banner("Problem 7b: ridge-regularized MNL on AI sample")

    df = pd.read_csv(args.ai_sample_rows)
    df = df[["srch_id", args.choice_col] + ALL_FEATURES].copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

    feature_means = df[ALL_FEATURES].mean()
    feature_stds = df[ALL_FEATURES].std().replace(0, 1)

    df_scaled = df.copy()
    df_scaled[ALL_FEATURES] = (df_scaled[ALL_FEATURES] - feature_means) / feature_stds

    groups = list(df_scaled.groupby("srch_id"))

    initial_beta = np.zeros(1 + len(ALL_FEATURES))
    result = minimize(
        _penalized_nll,
        initial_beta,
        args=(groups, args.choice_col, ALL_FEATURES, args.lam),
        method="L-BFGS-B",
        options={"maxiter": args.maxiter},
    )

    beta_hat = result.x
    coef_dict: Dict[str, float] = {"intercept": float(beta_hat[0])}
    for i, name in enumerate(ALL_FEATURES, start=1):
        coef_dict[name] = float(beta_hat[i])

    print(f"AI sample queries:             {df_scaled['srch_id'].nunique()}")
    print(f"AI sample rows:                {len(df_scaled)}")
    print(f"Lambda (ridge):                {args.lam}")
    print(f"Converged:                     {bool(result.success)}")
    print(f"Penalized NLL:                 {float(result.fun):.4f}")
    unpen = _negative_log_likelihood(beta_hat, groups, args.choice_col, ALL_FEATURES)
    print(f"Unpenalized NLL:               {unpen:.4f}")
    print()
    print("Coefficients (AI ridge MNL on z-scored features):")
    print(f"  {'parameter':<28}{'AI (ridge)':>12}{'human (P1 norm.)':>20}")
    for name, value in coef_dict.items():
        human_value = HUMAN_BETA_FROM_PROBLEM1[name]
        print(f"  {name:<28}{value:>12.4f}{human_value:>20.4f}")
    print()
    print("Note: magnitudes are not directly comparable. Compare signs and broad")
    print("relative magnitudes only -- the AI fit z-scores everything on the AI")
    print("sample, while the human normalized coefs leave binaries at 0/1.")

    return {
        "ai_sample_path": str(Path(args.ai_sample_rows).resolve()),
        "choice_col": args.choice_col,
        "lambda": args.lam,
        "n_queries": int(df_scaled["srch_id"].nunique()),
        "n_rows": int(len(df_scaled)),
        "scaling_used": "z-score on the AI sample (continuous and binary alike)",
        "scaling_means": feature_means.to_dict(),
        "scaling_stds": feature_stds.to_dict(),
        "ai_regularized_coefficients": coef_dict,
        "human_normalized_coefficients": HUMAN_BETA_FROM_PROBLEM1,
        "fit": {
            "converged": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit) if hasattr(result, "nit") else None,
            "penalized_nll": float(result.fun),
            "unpenalized_nll": float(unpen),
        },
    }


# =============================================================================
# Problem 7c: held-out predictive comparison (AI vs MNL vs trivial baseline).
# =============================================================================

def _compute_scaling_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    cont_means = df[CONTINUOUS_FEATURES].mean()
    cont_stds = df[CONTINUOUS_FEATURES].std().replace(0, 1)
    cont_medians = df[CONTINUOUS_FEATURES].median()
    bin_medians = df[BINARY_FEATURES].median()
    return cont_means, cont_stds, cont_medians, bin_medians


def _prepare_features(
    group: pd.DataFrame,
    cont_means: pd.Series,
    cont_stds: pd.Series,
    cont_medians: pd.Series,
    bin_medians: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _utilities(group, beta, cm, cs, ccm, bm):
    g, X = _prepare_features(group, cm, cs, ccm, bm)
    u = pd.Series(beta["intercept"], index=g.index, dtype=float)
    for f in ALL_FEATURES:
        u += beta[f] * X[f]
    return g, u


def _probabilities(group, beta, cm, cs, ccm, bm):
    _, u = _utilities(group, beta, cm, cs, ccm, bm)
    max_u = max(0.0, float(u.max()))
    exp_u = np.exp(u - max_u)
    exp_outside = np.exp(0 - max_u)
    denom = exp_outside + exp_u.sum()
    return exp_u / denom, float(exp_outside / denom)


def _hard_mnl_predict(group, beta, cm, cs, ccm, bm) -> str:
    g, u = _utilities(group, beta, cm, cs, ccm, bm)
    max_idx = u.idxmax()
    if u.loc[max_idx] <= 0:
        return "NO_PURCHASE"
    return str(int(g.loc[max_idx, "alt_id"]))


def _get_true_choice(group, choice_col):
    booked = group[group[choice_col] == 1]
    if len(booked) == 1:
        return int(booked.iloc[0]["alt_id"])
    if len(booked) == 0:
        return "NO_PURCHASE"
    raise ValueError(f"query {group['srch_id'].iloc[0]} has multiple bookings")


def _summarize_query(group, choice_col, include_answer):
    srch_id = group["srch_id"].iloc[0]
    context_cols = [
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
    ]
    context_cols = [c for c in context_cols if c in group.columns]
    hotel_cols = ["alt_id"] + ALL_FEATURES
    hotel_cols = [c for c in hotel_cols if c in group.columns]
    context = group.iloc[0][context_cols].to_dict()
    table = group[hotel_cols].to_string(index=False)
    text = (
        f"Search query {srch_id}\n"
        f"Customer/search context:\n{context}\n\n"
        f"Hotel options:\n{table}"
    )
    if include_answer:
        text += f"\nObserved customer choice: {_get_true_choice(group, choice_col)}"
    return text


def _build_prompt(real_df, context_ids, heldout_ids, choice_col):
    context = []
    for _, g in real_df[real_df["srch_id"].isin(context_ids)].groupby("srch_id"):
        context.append(_summarize_query(g, choice_col, include_answer=True))
    heldout = []
    for _, g in real_df[real_df["srch_id"].isin(heldout_ids)].groupby("srch_id"):
        heldout.append(_summarize_query(g, choice_col, include_answer=False))
    return (
        "You are predicting hotel booking choices.\n\n"
        "Below are examples from real hotel search queries. Each example "
        "includes the customer/search context, the hotel alternatives shown, "
        "and the actual observed customer choice. The observed choice is "
        "either one hotel alt_id or NO_PURCHASE.\n\n"
        f"Context examples:\n{chr(10).join([chr(10).join([e, '']) for e in context])}\n\n"
        "Now predict the customer choice for each held-out search query "
        "below. For each query, return exactly one prediction:\n"
        "- one listed alt_id, or\n"
        "- NO_PURCHASE\n\n"
        "Return your answer as a CSV with columns:\n"
        "srch_id,predicted_choice\n\n"
        f"Held-out queries:\n{chr(10).join([chr(10).join([h, '']) for h in heldout])}"
    )


def _behaviour_summary(pred_df, pred_col, label, heldout_df):
    chosen_attr_cols = [
        "price_usd",
        "prop_starrating",
        "prop_review_score",
        "prop_location_score",
        "prop_brand_bool",
        "promotion_flag",
    ]
    rows = []
    for _, row in pred_df.iterrows():
        sid = row["srch_id"]
        choice = str(row[pred_col])
        if choice == "NO_PURCHASE":
            rows.append(
                {"srch_id": sid, "model": label, "purchased": 0,
                 **{c: np.nan for c in chosen_attr_cols}}
            )
            continue
        alt_id = int(float(choice))
        match = heldout_df[(heldout_df["srch_id"] == sid) & (heldout_df["alt_id"] == alt_id)]
        if len(match) != 1:
            raise ValueError(f"could not match srch_id={sid} alt_id={alt_id}")
        m = match.iloc[0]
        rows.append(
            {"srch_id": sid, "model": label, "purchased": 1,
             **{c: m[c] for c in chosen_attr_cols}}
        )
    return pd.DataFrame(rows)


def run_7c(args: argparse.Namespace) -> Dict[str, object]:
    banner("Problem 7c: held-out AI vs MNL comparison (n=50)")

    real_df = pd.read_csv(args.data)
    real_df["alt_id"] = real_df.groupby("srch_id").cumcount() + 1

    rng = np.random.default_rng(args.seed)
    all_ids = real_df["srch_id"].unique()
    context_ids = rng.choice(all_ids, size=args.n_context, replace=False)
    remaining = np.setdiff1d(all_ids, context_ids)
    heldout_ids = rng.choice(remaining, size=args.n_heldout, replace=False)

    heldout_df = real_df[real_df["srch_id"].isin(heldout_ids)].copy()
    train_df = real_df[~real_df["srch_id"].isin(heldout_ids)].copy()

    # Persist the prompt so it can be sent to ChatGPT manually if needed.
    if args.prompt_out:
        Path(args.prompt_out).parent.mkdir(parents=True, exist_ok=True)
        prompt_text = _build_prompt(real_df, context_ids, heldout_ids, "booking_bool")
        Path(args.prompt_out).write_text(prompt_text, encoding="utf-8")

    # Truth, AI, MNL, baseline.
    true_choices = (
        heldout_df.groupby("srch_id")
        .apply(lambda g: _get_true_choice(g, "booking_bool"))
        .reset_index(name="true_choice")
    )
    true_choices["true_choice"] = true_choices["true_choice"].astype(str)

    ai_predictions = pd.read_csv(args.ai_predictions)
    ai_predictions["predicted_choice"] = ai_predictions["predicted_choice"].astype(str)

    cm, cs, ccm, bm = _compute_scaling_stats(train_df)

    mnl_rows = []
    for sid, g in heldout_df.groupby("srch_id"):
        pred = _hard_mnl_predict(g, HUMAN_BETA_FROM_PROBLEM1, cm, cs, ccm, bm)
        mnl_rows.append({"srch_id": sid, "mnl_predicted_choice": pred})
    mnl_predictions = pd.DataFrame(mnl_rows)

    baseline_predictions = pd.DataFrame(
        {"srch_id": true_choices["srch_id"], "baseline_predicted_choice": "NO_PURCHASE"}
    )

    eval_df = (
        true_choices.merge(ai_predictions, on="srch_id")
        .merge(mnl_predictions, on="srch_id")
        .merge(baseline_predictions, on="srch_id")
    )
    eval_df["ai_correct"] = eval_df["true_choice"] == eval_df["predicted_choice"].astype(str)
    eval_df["mnl_correct"] = eval_df["true_choice"] == eval_df["mnl_predicted_choice"].astype(str)
    eval_df["baseline_correct"] = eval_df["true_choice"] == eval_df["baseline_predicted_choice"].astype(str)

    n = int(len(eval_df))
    accuracy_rows = []
    print(f"Held-out queries: {n} | context queries: {args.n_context} | seed: {args.seed}")
    print()
    print(f"  {'model':<32}{'acc':>6}{'  correct':>10}{'  wilson 95% CI':>22}")
    for label, col in [("AI agent", "ai_correct"),
                       ("MNL hard rule", "mnl_correct"),
                       ("Always NO_PURCHASE baseline", "baseline_correct")]:
        correct = int(eval_df[col].sum())
        lo, hi = proportion_confint(count=correct, nobs=n, alpha=0.05, method="wilson")
        accuracy_rows.append({
            "model": label,
            "accuracy": correct / n,
            "correct": correct,
            "n": n,
            "wilson_95_ci": [float(lo), float(hi)],
        })
        print(f"  {label:<32}{correct/n:>6.2f}{correct:>5d}/{n:<3d}    [{float(lo):.3f}, {float(hi):.3f}]")

    obs = _behaviour_summary(true_choices.rename(columns={"true_choice": "predicted_choice"}),
                             "predicted_choice", "Observed human data", heldout_df)
    ai_b = _behaviour_summary(ai_predictions, "predicted_choice", "AI agent", heldout_df)
    mnl_b = _behaviour_summary(mnl_predictions, "mnl_predicted_choice", "MNL hard-choice", heldout_df)
    base_b = _behaviour_summary(baseline_predictions, "baseline_predicted_choice",
                                "Always NO_PURCHASE baseline", heldout_df)
    behaviour = pd.concat([obs, ai_b, mnl_b, base_b], ignore_index=True)
    behaviour_summary_table = (
        behaviour.groupby("model")
        .agg(
            purchase_rate=("purchased", "mean"),
            num_predicted_purchases=("purchased", "sum"),
            avg_chosen_price=("price_usd", "mean"),
            avg_chosen_star_rating=("prop_starrating", "mean"),
            avg_chosen_review_score=("prop_review_score", "mean"),
            avg_chosen_location_score=("prop_location_score", "mean"),
            share_chosen_brand=("prop_brand_bool", "mean"),
            share_chosen_promotion=("promotion_flag", "mean"),
        )
        .reset_index()
    )

    # MNL probability-implied summary.
    mnl_prob_rows = []
    for sid, g in heldout_df.groupby("srch_id"):
        hp, np_prob = _probabilities(g, HUMAN_BETA_FROM_PROBLEM1, cm, cs, ccm, bm)
        tmp = g.copy()
        tmp["mnl_prob"] = hp
        tmp["no_purchase_prob"] = np_prob
        mnl_prob_rows.append(tmp)
    mnl_prob_df = pd.concat(mnl_prob_rows, ignore_index=True)
    total_purchase_prob = float(mnl_prob_df["mnl_prob"].sum())
    num_queries = heldout_df["srch_id"].nunique()
    prob_implied = {
        "model": "MNL probability-implied",
        "purchase_rate": total_purchase_prob / num_queries,
        "avg_chosen_price": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["price_usd"]).sum() / total_purchase_prob),
        "avg_chosen_star_rating": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["prop_starrating"]).sum() / total_purchase_prob),
        "avg_chosen_review_score": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["prop_review_score"]).sum() / total_purchase_prob),
        "avg_chosen_location_score": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["prop_location_score"]).sum() / total_purchase_prob),
        "share_chosen_brand": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["prop_brand_bool"]).sum() / total_purchase_prob),
        "share_chosen_promotion": float((mnl_prob_df["mnl_prob"] * mnl_prob_df["promotion_flag"]).sum() / total_purchase_prob),
    }
    print()
    print(f"MNL probability-implied purchase rate: {prob_implied['purchase_rate']:.3f}")
    print(f"  (observed human purchase rate:       {behaviour_summary_table.loc[behaviour_summary_table['model']=='Observed human data', 'purchase_rate'].iloc[0]:.3f})")

    return {
        "data_path": str(Path(args.data).resolve()),
        "ai_predictions_path": str(Path(args.ai_predictions).resolve()),
        "prompt_path": str(Path(args.prompt_out).resolve()) if args.prompt_out else None,
        "seed": args.seed,
        "n_context": args.n_context,
        "n_heldout": args.n_heldout,
        "scaling_convention": (
            "Continuous features z-scored using train-set sample std; "
            "binary features (prop_brand_bool, promotion_flag) left at raw 0/1, "
            "matching Problem 1's normalized-coefficient convention."
        ),
        "accuracy_table": accuracy_rows,
        "behaviour_summary": behaviour_summary_table.to_dict(orient="records"),
        "mnl_probability_implied": prob_implied,
    }


# =============================================================================
# Driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 7 (AI agents) end-to-end.")
    parser.add_argument(
        "--data",
        default="data.csv",
        help="Path to the full Expedia dataset (defaults to project root data.csv).",
    )
    parser.add_argument(
        "--ai-sample-rows",
        default="problem7/inputs/ai_booking_sample500_rows.csv",
        help="Row-level AI sample CSV produced by manual prompting (used by 7a/7b).",
    )
    parser.add_argument(
        "--ai-predictions",
        default="problem7/inputs/ai_heldout_predictions.csv",
        help="AI's held-out predictions CSV produced by manual prompting (used by 7c).",
    )
    parser.add_argument(
        "--merged-csv",
        default="problem7/inputs/data_with_ai_bookings.csv",
        help="Where 7a writes the merged data + AI booking column. Pass empty to skip.",
    )
    parser.add_argument(
        "--prompt-out",
        default="problem7/inputs/ai_heldout_prompt.txt",
        help="Where 7c writes the held-out prompt. Pass empty to skip.",
    )
    parser.add_argument("--choice-col", default="booking_ai_sample500")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-context", type=int, default=10)
    parser.add_argument("--n-heldout", type=int, default=50)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional unified JSON output path (e.g. results/problem7_results.json).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["7a", "7b", "7c"],
        default=[],
        help="Optional list of sub-parts to skip (defaults to running all three).",
    )
    args = parser.parse_args()

    payload: Dict[str, object] = {"problem": 7, "seed": args.seed}

    if "7a" not in args.skip:
        payload["7a"] = run_7a(args)
    if "7b" not in args.skip:
        payload["7b"] = run_7b(args)
    if "7c" not in args.skip:
        payload["7c"] = run_7c(args)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
        print(f"\nWrote unified payload to {args.output_json}")


if __name__ == "__main__":
    main()
