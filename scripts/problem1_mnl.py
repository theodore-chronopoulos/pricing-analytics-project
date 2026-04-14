#!/usr/bin/env python3
"""Estimate the Problem 1 MNL model from the Expedia query-choice data."""

from __future__ import annotations

import argparse
import json

from mnl_utils import DEFAULT_SEED, estimate_problem1_model, save_json, set_reproducibility_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the Problem 1 MNL model.")
    parser.add_argument("--data", default="data.csv", help="Path to the Expedia dataset.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum Newton iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Gradient norm stopping tolerance.")
    parser.add_argument(
        "--initial-damping",
        type=float,
        default=1e-6,
        help="Initial diagonal damping added to the Newton system.",
    )
    parser.add_argument(
        "--min-step-size",
        type=float,
        default=1e-6,
        help="Smallest backtracking step size before giving up on an iteration.",
    )
    parser.add_argument(
        "--scale-binary",
        action="store_true",
        help="Scale binary variables too. By default, only continuous features are z-scored.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a JSON result dump.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Reproducibility seed. The solver is deterministic, but the seed is recorded explicitly.",
    )
    args = parser.parse_args()
    set_reproducibility_seed(args.seed)

    print(f"Problem 1: fitting deterministic MNL model from {args.data}")
    print(f"Reproducibility seed: {args.seed}")

    result = estimate_problem1_model(
        data_path=args.data,
        max_queries=args.max_queries,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
        scale_binary=args.scale_binary,
    )

    print("Problem 1 MNL estimation complete.")
    print(json.dumps(result["dataset_summary"], indent=2))
    print("Normalized coefficients:")
    print(json.dumps(result["normalized_coefficients"], indent=2))
    print("Raw-scale coefficients:")
    print(json.dumps(result["raw_scale_coefficients"], indent=2))
    print("Fit summary:")
    print(json.dumps(result["fit_summary"], indent=2))

    if args.output_json:
        result["seed"] = args.seed
        save_json(result, args.output_json)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
