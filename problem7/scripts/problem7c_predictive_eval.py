#!/usr/bin/env python3
"""Problem 7c: held-out predictive comparison between the AI agent and the MNL.

Pipeline:
  1. Reproducibly split the full dataset at the query level
     (10 in-context examples + 50 held-out queries; numpy default_rng(42)).
  2. Build the held-out prompt and write it to disk for the manual LLM step.
  3. Load the manually-collected AI predictions (`inputs/ai_heldout_predictions.csv`).
  4. Compute the MNL hard prediction using the Problem-1 normalized
     coefficients with continuous features z-scored and binary features at
     raw 0/1 (i.e. the convention Problem 1 was estimated under).
  5. Add an always-NO_PURCHASE baseline.
  6. Report exact-choice accuracy with Wilson 95 % confidence intervals,
     hard-prediction behavioural summary, and the MNL probability-implied
     summary.
  7. Save everything to `results/problem7c_results.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# Allow running the script either from the project root or from problem7/.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from problem7_common import (  # noqa: E402
    ALL_FEATURES,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    HUMAN_BETA_FROM_PROBLEM1,
    compute_scaling_stats,
    probabilities,
    utilities,
)


def get_true_choice(group: pd.DataFrame, choice_col: str):
    booked = group[group[choice_col] == 1]
    if len(booked) == 1:
        return int(booked.iloc[0]["alt_id"])
    if len(booked) == 0:
        return "NO_PURCHASE"
    raise ValueError(f"query {group['srch_id'].iloc[0]} has multiple bookings")


def summarize_query(group: pd.DataFrame, choice_col: str, include_answer: bool) -> str:
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
        text += f"\nObserved customer choice: {get_true_choice(group, choice_col)}"
    return text


def build_prompt(
    real_df: pd.DataFrame,
    context_ids,
    heldout_ids,
    choice_col: str,
) -> str:
    context = []
    for sid, g in real_df[real_df["srch_id"].isin(context_ids)].groupby("srch_id"):
        context.append(summarize_query(g, choice_col, include_answer=True))
    heldout = []
    for sid, g in real_df[real_df["srch_id"].isin(heldout_ids)].groupby("srch_id"):
        heldout.append(summarize_query(g, choice_col, include_answer=False))

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


def hard_mnl_predict(group, beta, cm, cs, ccm, bm) -> str:
    g, u = utilities(group, beta, cm, cs, ccm, bm)
    max_idx = u.idxmax()
    if u.loc[max_idx] <= 0:
        return "NO_PURCHASE"
    return str(int(g.loc[max_idx, "alt_id"]))


def behaviour_summary(
    pred_df: pd.DataFrame,
    pred_col: str,
    label: str,
    heldout_df: pd.DataFrame,
) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out AI vs MNL comparison.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument(
        "--ai-predictions",
        default="problem7/inputs/ai_heldout_predictions.csv",
    )
    parser.add_argument(
        "--prompt-out",
        default="problem7/inputs/ai_heldout_prompt.txt",
        help="Where to write the held-out prompt for the manual LLM step.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-context", type=int, default=10)
    parser.add_argument("--n-heldout", type=int, default=50)
    parser.add_argument(
        "--output-json",
        default="problem7/results/problem7c_results.json",
    )
    args = parser.parse_args()

    real_df = pd.read_csv(args.data)
    real_df["alt_id"] = real_df.groupby("srch_id").cumcount() + 1
    rng = np.random.default_rng(args.seed)
    all_ids = real_df["srch_id"].unique()
    context_ids = rng.choice(all_ids, size=args.n_context, replace=False)
    remaining = np.setdiff1d(all_ids, context_ids)
    heldout_ids = rng.choice(remaining, size=args.n_heldout, replace=False)

    heldout_df = real_df[real_df["srch_id"].isin(heldout_ids)].copy()
    train_df = real_df[~real_df["srch_id"].isin(heldout_ids)].copy()

    # Persist the prompt so it can be sent to ChatGPT manually.
    Path(args.prompt_out).parent.mkdir(parents=True, exist_ok=True)
    prompt_text = build_prompt(real_df, context_ids, heldout_ids, "booking_bool")
    Path(args.prompt_out).write_text(prompt_text, encoding="utf-8")

    # Truth, AI, MNL, baseline.
    true_choices = (
        heldout_df.groupby("srch_id")
        .apply(lambda g: get_true_choice(g, "booking_bool"))
        .reset_index(name="true_choice")
    )
    true_choices["true_choice"] = true_choices["true_choice"].astype(str)
    ai_predictions = pd.read_csv(args.ai_predictions)
    ai_predictions["predicted_choice"] = ai_predictions["predicted_choice"].astype(str)

    cm, cs, ccm, bm = compute_scaling_stats(train_df)

    mnl_rows = []
    for sid, g in heldout_df.groupby("srch_id"):
        pred = hard_mnl_predict(g, HUMAN_BETA_FROM_PROBLEM1, cm, cs, ccm, bm)
        mnl_rows.append({"srch_id": sid, "mnl_predicted_choice": pred})
    mnl_predictions = pd.DataFrame(mnl_rows)

    baseline_predictions = pd.DataFrame(
        {"srch_id": true_choices["srch_id"], "baseline_predicted_choice": "NO_PURCHASE"}
    )

    # Accuracy + Wilson 95 % CIs.
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
    for label, col in [("AI agent", "ai_correct"), ("MNL hard rule", "mnl_correct"),
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

    # Behavioural summary on hard predictions.
    obs = behaviour_summary(true_choices.rename(columns={"true_choice": "predicted_choice"}),
                            "predicted_choice", "Observed human data", heldout_df)
    ai_b = behaviour_summary(ai_predictions, "predicted_choice", "AI agent", heldout_df)
    mnl_b = behaviour_summary(mnl_predictions, "mnl_predicted_choice", "MNL hard-choice", heldout_df)
    base_b = behaviour_summary(baseline_predictions, "baseline_predicted_choice",
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
        hp, np_prob = probabilities(g, HUMAN_BETA_FROM_PROBLEM1, cm, cs, ccm, bm)
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

    payload = {
        "problem": "7c",
        "data_path": str(Path(args.data).resolve()),
        "ai_predictions_path": str(Path(args.ai_predictions).resolve()),
        "prompt_path": str(Path(args.prompt_out).resolve()),
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

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)

    print(json.dumps({
        "accuracy_table": accuracy_rows,
        "mnl_probability_implied_purchase_rate": prob_implied["purchase_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
