#!/usr/bin/env python3
"""Solve Problem 4: early-vs-late mixture of MNL models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

from mnl_utils import (
    DEFAULT_SEED,
    DEFAULT_SCALE_FEATURES,
    FEATURE_NAMES,
    compute_scaling_stats,
    estimate_mnl_from_raw_queries,
    load_query_booking_windows,
    load_raw_queries,
    save_json,
    set_reproducibility_seed,
)


def split_queries_by_booking_window(
    data_path: str,
    threshold: float,
) -> Tuple[List, List, Dict[str, float]]:
    raw_queries = load_raw_queries(data_path)
    booking_windows = load_query_booking_windows(data_path)

    early_queries = []
    late_queries = []
    for query in raw_queries:
        booking_window = booking_windows[query.srch_id]
        if booking_window < threshold:
            late_queries.append(query)
        else:
            early_queries.append(query)

    total = len(raw_queries)
    thetas = {
        "theta_early": len(early_queries) / total,
        "theta_late": len(late_queries) / total,
        "threshold_days": threshold,
    }
    return early_queries, late_queries, thetas


def coefficient_differences(
    early: Dict[str, float],
    late: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    keys = ["intercept"] + FEATURE_NAMES
    for key in keys:
        result[key] = {
            "early": early[key],
            "late": late[key],
            "early_minus_late": early[key] - late[key],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 4: mixture of MNL models.")
    parser.add_argument("--data", default="data.csv", help="Path to the Expedia dataset.")
    parser.add_argument(
        "--late-threshold",
        type=float,
        default=7.0,
        help="Booking window threshold; strictly below this is late, otherwise early.",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum Newton iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Gradient norm stopping tolerance.")
    parser.add_argument("--initial-damping", type=float, default=1e-6)
    parser.add_argument("--min-step-size", type=float, default=1e-6)
    parser.add_argument(
        "--scale-binary",
        action="store_true",
        help="Scale binary variables too. By default, only continuous features are z-scored.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path for a JSON result dump.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Reproducibility seed. No randomness is used, but the seed is recorded explicitly.",
    )
    args = parser.parse_args()
    set_reproducibility_seed(args.seed)

    print("Problem 4: fitting early/late mixture of MNL models")
    print(f"Reproducibility seed: {args.seed}")
    print(f"Late customer threshold: booking window < {args.late_threshold} days")

    scale_features = set(FEATURE_NAMES) if args.scale_binary else DEFAULT_SCALE_FEATURES
    all_queries = load_raw_queries(args.data)
    scaling_stats = compute_scaling_stats(all_queries, scale_features)
    early_queries, late_queries, thetas = split_queries_by_booking_window(
        args.data,
        threshold=args.late_threshold,
    )
    print(f"Loaded {len(all_queries)} total queries: {len(early_queries)} early, {len(late_queries)} late")
    print("Fitting early-customer MNL")

    early_result = estimate_mnl_from_raw_queries(
        early_queries,
        scaling_stats,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
    )
    print("Fitting late-customer MNL")
    late_result = estimate_mnl_from_raw_queries(
        late_queries,
        scaling_stats,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
    )

    comparison = {
        "normalized_coefficients": coefficient_differences(
            early_result["normalized_coefficients"],
            late_result["normalized_coefficients"],
        ),
        "raw_scale_coefficients": coefficient_differences(
            early_result["raw_scale_coefficients"],
            late_result["raw_scale_coefficients"],
        ),
    }

    payload = {
        "problem": 4,
        "data_path": os.path.abspath(args.data),
        "seed": args.seed,
        "customer_type_definition": {
            "type_1": "early",
            "type_2": "late",
            "late_if_booking_window_lt_days": args.late_threshold,
        },
        "theta_estimates": thetas,
        "shared_scaling_stats": scaling_stats,
        "early_model": early_result,
        "late_model": late_result,
        "coefficient_comparison": comparison,
    }

    print("Problem 4 mixture estimation complete.")
    print("Theta estimates:")
    print(json.dumps(thetas, indent=2))
    print("Early model fit summary:")
    print(json.dumps(early_result["fit_summary"], indent=2))
    print("Late model fit summary:")
    print(json.dumps(late_result["fit_summary"], indent=2))
    print("Raw-scale coefficient differences (early - late):")
    print(
        json.dumps(
            {
                key: value["early_minus_late"]
                for key, value in comparison["raw_scale_coefficients"].items()
            },
            indent=2,
        )
    )

    if args.output_json:
        save_json(payload, args.output_json)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
