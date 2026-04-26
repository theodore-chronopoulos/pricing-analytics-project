#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from mnl_utils import (
    DEFAULT_SCALE_FEATURES,
    DEFAULT_SEED,
    FEATURE_NAMES,
    PRICE_FEATURE,
    compute_scaling_stats,
    estimate_mnl_from_raw_queries,
    load_raw_queries,
    load_small_dataset,
    raw_utility,
    safe_exp,
    save_json,
    segment_revenue_from_sums,
    set_reproducibility_seed,
    solve_revenue_ordered_assortment,
)


def load_query_children_counts(data_path: str) -> Dict[str, int]:
    children_counts: Dict[str, int] = {}
    with open(data_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            srch_id = row["srch_id"]
            count = int(float(row["srch_children_count"]))
            previous = children_counts.get(srch_id)
            if previous is not None and previous != count:
                raise ValueError(f"Inconsistent children count for query {srch_id}.")
            children_counts[srch_id] = count
    return children_counts


def split_queries_by_family(
    data_path: str,
) -> Tuple[List, List, Dict[str, float]]:
    raw_queries = load_raw_queries(data_path)
    children_counts = load_query_children_counts(data_path)

    family_queries = []
    non_family_queries = []
    for query in raw_queries:
        if children_counts[query.srch_id] > 0:
            family_queries.append(query)
        else:
            non_family_queries.append(query)

    total = len(raw_queries)
    thetas = {
        "theta_family": len(family_queries) / total,
        "theta_non_family": len(non_family_queries) / total,
    }
    return family_queries, non_family_queries, thetas


def coefficient_differences(
    family: Dict[str, float],
    non_family: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    keys = ["intercept"] + FEATURE_NAMES
    for key in keys:
        result[key] = {
            "family": family[key],
            "non_family": non_family[key],
            "family_minus_non_family": family[key] - non_family[key],
        }
    return result


@dataclass
class Item:
    index_1_based: int
    price: float
    w_family: float
    w_non_family: float


def build_items(
    data_path: str,
    family_coeffs: Dict[str, float],
    non_family_coeffs: Dict[str, float],
) -> List[Item]:
    rows = load_small_dataset(data_path)
    items: List[Item] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            Item(
                index_1_based=idx,
                price=row[PRICE_FEATURE],
                w_family=safe_exp(raw_utility(row, family_coeffs)),
                w_non_family=safe_exp(raw_utility(row, non_family_coeffs)),
            )
        )
    return items


def segment_subset_revenue(indices: Set[int], items: Sequence[Item], weight_attr: str) -> float:
    weighted_price_sum = 0.0
    weight_sum = 0.0
    for item in items:
        if item.index_1_based in indices:
            weight = getattr(item, weight_attr)
            weighted_price_sum += item.price * weight
            weight_sum += weight
    return segment_revenue_from_sums(weighted_price_sum, weight_sum)


def solve_known_type(items: Sequence[Item], weight_attr: str) -> Dict[str, object]:
    prices = [item.price for item in items]
    weights = [getattr(item, weight_attr) for item in items]
    return solve_revenue_ordered_assortment(prices, weights)


def best_completion_segment(
    base_weighted_price_sum: float,
    base_weight_sum: float,
    remaining_items: Sequence[Item],
    weight_attr: str,
) -> float:
    best = segment_revenue_from_sums(base_weighted_price_sum, base_weight_sum)
    cum_weighted_price = base_weighted_price_sum
    cum_weight = base_weight_sum

    for item in sorted(remaining_items, key=lambda current: current.price, reverse=True):
        weight = getattr(item, weight_attr)
        cum_weighted_price += item.price * weight
        cum_weight += weight
        revenue = segment_revenue_from_sums(cum_weighted_price, cum_weight)
        if revenue > best:
            best = revenue

    return best


class MixtureBranchAndBound:
    def __init__(self, items: Sequence[Item], theta_family: float, theta_non_family: float) -> None:
        self.theta_family = theta_family
        self.theta_non_family = theta_non_family
        self.branch_items = sorted(
            items,
            key=lambda item: item.price * (theta_family * item.w_family + theta_non_family * item.w_non_family),
            reverse=True,
        )
        self.best_value = 0.0
        self.best_indices: List[int] = []
        self.nodes_visited = 0

    def objective(self, a_family: float, b_family: float, a_non_family: float, b_non_family: float) -> float:
        return (
            self.theta_family * segment_revenue_from_sums(a_family, b_family)
            + self.theta_non_family * segment_revenue_from_sums(a_non_family, b_non_family)
        )

    def upper_bound(
        self,
        pos: int,
        a_family: float,
        b_family: float,
        a_non_family: float,
        b_non_family: float,
    ) -> float:
        remaining = self.branch_items[pos:]
        family_bound = best_completion_segment(a_family, b_family, remaining, "w_family")
        non_family_bound = best_completion_segment(a_non_family, b_non_family, remaining, "w_non_family")
        return self.theta_family * family_bound + self.theta_non_family * non_family_bound

    def search(
        self,
        pos: int,
        a_family: float,
        b_family: float,
        a_non_family: float,
        b_non_family: float,
        selected_indices: List[int],
    ) -> None:
        self.nodes_visited += 1

        current_value = self.objective(a_family, b_family, a_non_family, b_non_family)
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_indices = selected_indices[:]

        if pos >= len(self.branch_items):
            return

        bound = self.upper_bound(pos, a_family, b_family, a_non_family, b_non_family)
        if bound <= self.best_value + 1e-12:
            return

        item = self.branch_items[pos]

        selected_indices.append(item.index_1_based)
        self.search(
            pos + 1,
            a_family + item.price * item.w_family,
            b_family + item.w_family,
            a_non_family + item.price * item.w_non_family,
            b_non_family + item.w_non_family,
            selected_indices,
        )
        selected_indices.pop()

        self.search(pos + 1, a_family, b_family, a_non_family, b_non_family, selected_indices)

    def solve(self) -> Dict[str, object]:
        self.search(0, 0.0, 0.0, 0.0, 0.0, [])
        self.best_indices.sort()
        return {
            "selected_indices_1_based": self.best_indices,
            "selected_count": len(self.best_indices),
            "expected_mixture_revenue": self.best_value,
            "nodes_visited": self.nodes_visited,
        }


def solve_dataset(
    data_path: str,
    family_coeffs: Dict[str, float],
    non_family_coeffs: Dict[str, float],
    theta_family: float,
    theta_non_family: float,
) -> Dict[str, object]:
    items = build_items(data_path, family_coeffs, non_family_coeffs)

    s1 = solve_known_type(items, "w_family")
    s2 = solve_known_type(items, "w_non_family")

    mixture_solver = MixtureBranchAndBound(items, theta_family, theta_non_family)
    s = mixture_solver.solve()

    s_set = set(s["selected_indices_1_based"])
    s1_set = set(s1["selected_indices_1_based"])
    s2_set = set(s2["selected_indices_1_based"])

    s["expected_revenue_under_family_model"] = segment_subset_revenue(s_set, items, "w_family")
    s["expected_revenue_under_non_family_model"] = segment_subset_revenue(s_set, items, "w_non_family")

    s1["expected_revenue_under_family_model"] = segment_subset_revenue(s1_set, items, "w_family")
    s1["expected_revenue_under_non_family_model"] = segment_subset_revenue(s1_set, items, "w_non_family")

    s2["expected_revenue_under_family_model"] = segment_subset_revenue(s2_set, items, "w_family")
    s2["expected_revenue_under_non_family_model"] = segment_subset_revenue(s2_set, items, "w_non_family")

    comparisons = {
        "type_1_value_of_knowing_customer_type": (
            s1["expected_revenue_under_family_model"] - s["expected_revenue_under_family_model"]
        ),
        "type_2_value_of_knowing_customer_type": (
            s2["expected_revenue_under_non_family_model"] - s["expected_revenue_under_non_family_model"]
        ),
    }

    return {
        "dataset_path": os.path.abspath(data_path),
        "unknown_type_optimal_assortment": s,
        "known_family_optimal_assortment": s1,
        "known_non_family_optimal_assortment": s2,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 6: family vs non-family mixture of MNL.")
    parser.add_argument("--data", default="data.csv", help="Path to the Expedia dataset.")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--initial-damping", type=float, default=1e-6)
    parser.add_argument("--min-step-size", type=float, default=1e-6)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data1.csv", "data2.csv", "data3.csv", "data4.csv"],
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_reproducibility_seed(args.seed)

    all_queries = load_raw_queries(args.data)
    scaling_stats = compute_scaling_stats(all_queries, DEFAULT_SCALE_FEATURES)

    family_queries, non_family_queries, thetas = split_queries_by_family(args.data)

    family_result = estimate_mnl_from_raw_queries(
        family_queries,
        scaling_stats,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
    )

    non_family_result = estimate_mnl_from_raw_queries(
        non_family_queries,
        scaling_stats,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        initial_damping=args.initial_damping,
        min_step_size=args.min_step_size,
    )

    comparison = {
        "normalized_coefficients": coefficient_differences(
            family_result["normalized_coefficients"],
            non_family_result["normalized_coefficients"],
        ),
        "raw_scale_coefficients": coefficient_differences(
            family_result["raw_scale_coefficients"],
            non_family_result["raw_scale_coefficients"],
        ),
    }

    family_coeffs = family_result["raw_scale_coefficients"]
    non_family_coeffs = non_family_result["raw_scale_coefficients"]
    theta_family = thetas["theta_family"]
    theta_non_family = thetas["theta_non_family"]

    assortment_results = {}
    for dataset in args.datasets:
        assortment_results[dataset] = solve_dataset(
            dataset,
            family_coeffs,
            non_family_coeffs,
            theta_family,
            theta_non_family,
        )

    payload = {
        "problem": 6,
        "data_path": os.path.abspath(args.data),
        "seed": args.seed,
        "customer_type_definition": {
            "type_1": "family",
            "type_2": "non_family",
            "family_if_srch_children_count_gt": 0,
        },
        "theta_estimates": thetas,
        "shared_scaling_stats": scaling_stats,
        "family_model": family_result,
        "non_family_model": non_family_result,
        "coefficient_comparison": comparison,
        "assortment_results": assortment_results,
    }

    if args.output_json:
        save_json(payload, args.output_json)

    print("Finished runtime")


if __name__ == "__main__":
    main()
