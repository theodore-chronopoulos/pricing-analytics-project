#!/usr/bin/env python3
"""Solve Problem 2: assortment optimization under the fitted MNL model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from mnl_utils import (
    DEFAULT_SEED,
    PRICE_FEATURE,
    assortment_revenue,
    estimate_problem1_model,
    load_json,
    load_small_dataset,
    raw_utility,
    save_json,
    safe_exp,
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
    rows = load_small_dataset(data_path)
    ranked = []
    for idx, row in enumerate(rows, start=1):
        utility = raw_utility(row, raw_coeffs)
        weight = safe_exp(utility)
        ranked.append(
            {
                "index": idx,
                "row": row,
                "price": row[PRICE_FEATURE],
                "utility": utility,
                "weight": weight,
            }
        )

    ranked.sort(key=lambda item: item["price"], reverse=True)

    best = {
        "selected_indices_1_based": [],
        "expected_revenue": 0.0,
        "selected_count": 0,
    }

    prefix_rows: List[Dict[str, float]] = []
    prefix_indices: List[int] = []
    for item in ranked:
        prefix_rows.append(item["row"])
        prefix_indices.append(item["index"])
        revenue = assortment_revenue(prefix_rows, raw_coeffs)
        if revenue > best["expected_revenue"]:
            best = {
                "selected_indices_1_based": prefix_indices[:],
                "expected_revenue": revenue,
                "selected_count": len(prefix_indices),
            }

    best["price_ranked_hotels"] = [
        {
            "index_1_based": item["index"],
            "price": item["price"],
            "utility": item["utility"],
            "weight": item["weight"],
        }
        for item in ranked
    ]
    best["dataset_path"] = os.path.abspath(data_path)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 2 under the fitted MNL model.")
    parser.add_argument("--data", default="data.csv", help="Full Expedia dataset for Problem 1 estimation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data1.csv", "data2.csv", "data3.csv", "data4.csv"],
        help="Small datasets for assortment optimization.",
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

    print("Problem 2: assortment optimization under fixed fitted MNL weights")
    print(f"Reproducibility seed: {args.seed}")
    if args.problem1_json:
        print(f"Loading cached Problem 1 fit from {args.problem1_json}")
    else:
        print(f"Refitting Problem 1 model from {args.data}")

    model = fit_or_load_model(args)
    raw_coeffs = model["raw_scale_coefficients"]
    results = {}
    for dataset in args.datasets:
        print(f"Solving assortment for {dataset}")
        results[dataset] = solve_dataset(dataset, raw_coeffs)

    payload = {
        "problem": 2,
        "model_source": args.problem1_json or "refit-from-data",
        "seed": args.seed,
        "raw_scale_coefficients": raw_coeffs,
        "results": results,
    }

    print("Problem 2 assortment optimization complete.")
    for dataset, result in results.items():
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "selected_indices_1_based": result["selected_indices_1_based"],
                    "selected_count": result["selected_count"],
                    "expected_revenue": result["expected_revenue"],
                },
                indent=2,
            )
        )

    if args.output_json:
        save_json(payload, args.output_json)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
