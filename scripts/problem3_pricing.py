#!/usr/bin/env python3
"""Solve Problem 3: pricing optimization under the fitted MNL model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from mnl_utils import (
    DEFAULT_SEED,
    FEATURE_NAMES,
    PRICE_FEATURE,
    bracket_unimodal_maximum,
    estimate_problem1_model,
    expected_revenue_from_components,
    golden_section_maximize,
    load_json,
    load_small_dataset,
    price_free_utility,
    save_json,
    set_reproducibility_seed,
)


def fit_or_load_model(args) -> Dict[str, object]:
    if args.problem1_json:
        return load_json(args.problem1_json)
    return estimate_problem1_model(
        data_path=args.data,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
        scale_binary=args.scale_binary,
    )


def solve_dataset(data_path: str, raw_coeffs: Dict[str, float]) -> Dict[str, object]:
    beta_price = raw_coeffs[PRICE_FEATURE]
    if beta_price >= 0.0:
        raise ValueError(
            "Problem 3 requires a negative price coefficient for a well-posed interior optimum."
        )

    rows = load_small_dataset(data_path)
    price_free_utilities = [price_free_utility(row, raw_coeffs) for row in rows]
    current_prices = [row[PRICE_FEATURE] for row in rows]

    def objective(common_price: float) -> float:
        prices = [common_price] * len(rows)
        return expected_revenue_from_components(prices, price_free_utilities, beta_price)

    initial_high = max(max(current_prices), 1.0 / abs(beta_price))
    left, right = bracket_unimodal_maximum(objective, initial_high=initial_high)
    optimal_price, optimal_revenue = golden_section_maximize(objective, left, right)
    current_revenue = expected_revenue_from_components(current_prices, price_free_utilities, beta_price)

    optimized_rows = []
    for idx, row in enumerate(rows, start=1):
        updated = {name: row[name] for name in FEATURE_NAMES}
        updated[PRICE_FEATURE] = optimal_price
        optimized_rows.append(
            {
                "index_1_based": idx,
                "original_price": row[PRICE_FEATURE],
                "optimal_price": optimal_price,
            }
        )

    return {
        "dataset_path": os.path.abspath(data_path),
        "beta_price": beta_price,
        "current_expected_revenue": current_revenue,
        "optimal_expected_revenue": optimal_revenue,
        "common_optimal_price": optimal_price,
        "optimal_prices_1_based": [optimal_price] * len(rows),
        "rows": optimized_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 3 under the fitted MNL model.")
    parser.add_argument("--data", default="data.csv", help="Full Expedia dataset for Problem 1 estimation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data1.csv", "data2.csv", "data3.csv", "data4.csv"],
        help="Small datasets for pricing optimization.",
    )
    parser.add_argument(
        "--problem1-json",
        default=None,
        help="Optional cached Problem 1 JSON output. If omitted, the model is refit.",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--initial-damping", type=float, default=1e-6)
    parser.add_argument("--min-step-size", type=float, default=1e-6)
    parser.add_argument("--scale-binary", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Reproducibility seed. No randomness is used, but the seed is recorded explicitly.",
    )
    args = parser.parse_args()
    set_reproducibility_seed(args.seed)

    print("Problem 3: pricing optimization under fitted MNL model")
    print(f"Reproducibility seed: {args.seed}")
    print(
        "Model note: with one shared linear price coefficient, the unconstrained optimum is a common price per dataset."
    )
    if args.problem1_json:
        print(f"Loading cached Problem 1 fit from {args.problem1_json}")
    else:
        print(f"Refitting Problem 1 model from {args.data}")

    model = fit_or_load_model(args)
    raw_coeffs = model["raw_scale_coefficients"]
    results = {}
    for dataset in args.datasets:
        print(f"Solving pricing problem for {dataset}")
        results[dataset] = solve_dataset(dataset, raw_coeffs)

    payload = {
        "problem": 3,
        "model_source": args.problem1_json or "refit-from-data",
        "seed": args.seed,
        "raw_scale_coefficients": raw_coeffs,
        "results": results,
    }

    print("Problem 3 pricing optimization complete.")
    for dataset, result in results.items():
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "common_optimal_price": result["common_optimal_price"],
                    "current_expected_revenue": result["current_expected_revenue"],
                    "optimal_expected_revenue": result["optimal_expected_revenue"],
                },
                indent=2,
            )
        )

    if args.output_json:
        save_json(payload, args.output_json)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
