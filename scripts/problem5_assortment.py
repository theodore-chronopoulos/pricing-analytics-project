#!/usr/bin/env python3
"""Solve Problem 5: type-aware and type-unaware assortment optimization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from mnl_utils import (
    DEFAULT_SEED,
    PRICE_FEATURE,
    load_json,
    load_small_dataset,
    raw_utility,
    safe_exp,
    save_json,
    segment_revenue_from_sums,
    set_reproducibility_seed,
    solve_revenue_ordered_assortment,
)


@dataclass
class Item:
    index_1_based: int
    price: float
    w_early: float
    w_late: float


def build_items(
    data_path: str,
    early_coeffs: Dict[str, float],
    late_coeffs: Dict[str, float],
) -> List[Item]:
    rows = load_small_dataset(data_path)
    items: List[Item] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            Item(
                index_1_based=idx,
                price=row[PRICE_FEATURE],
                w_early=safe_exp(raw_utility(row, early_coeffs)),
                w_late=safe_exp(raw_utility(row, late_coeffs)),
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
    def __init__(self, items: Sequence[Item], theta_early: float, theta_late: float) -> None:
        self.theta_early = theta_early
        self.theta_late = theta_late
        self.branch_items = sorted(
            items,
            key=lambda item: item.price * (theta_early * item.w_early + theta_late * item.w_late),
            reverse=True,
        )
        self.best_value = 0.0
        self.best_indices: List[int] = []
        self.nodes_visited = 0

    def objective(self, a_early: float, b_early: float, a_late: float, b_late: float) -> float:
        return (
            self.theta_early * segment_revenue_from_sums(a_early, b_early)
            + self.theta_late * segment_revenue_from_sums(a_late, b_late)
        )

    def upper_bound(
        self,
        pos: int,
        a_early: float,
        b_early: float,
        a_late: float,
        b_late: float,
    ) -> float:
        remaining = self.branch_items[pos:]
        early_bound = best_completion_segment(a_early, b_early, remaining, "w_early")
        late_bound = best_completion_segment(a_late, b_late, remaining, "w_late")
        return self.theta_early * early_bound + self.theta_late * late_bound

    def search(
        self,
        pos: int,
        a_early: float,
        b_early: float,
        a_late: float,
        b_late: float,
        selected_indices: List[int],
    ) -> None:
        self.nodes_visited += 1

        current_value = self.objective(a_early, b_early, a_late, b_late)
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_indices = selected_indices[:]

        if pos >= len(self.branch_items):
            return

        bound = self.upper_bound(pos, a_early, b_early, a_late, b_late)
        if bound <= self.best_value + 1e-12:
            return

        item = self.branch_items[pos]

        selected_indices.append(item.index_1_based)
        self.search(
            pos + 1,
            a_early + item.price * item.w_early,
            b_early + item.w_early,
            a_late + item.price * item.w_late,
            b_late + item.w_late,
            selected_indices,
        )
        selected_indices.pop()

        self.search(pos + 1, a_early, b_early, a_late, b_late, selected_indices)

    def solve(self) -> Dict[str, object]:
        self.search(0, 0.0, 0.0, 0.0, 0.0, [])
        self.best_indices.sort()
        return {
            "selected_indices_1_based": self.best_indices,
            "selected_count": len(self.best_indices),
            "expected_mixture_revenue": self.best_value,
            "nodes_visited": self.nodes_visited,
            "solver_backend": "branch_and_bound",
        }


def write_problem5_milp_lp(
    items: Sequence[Item],
    theta_early: float,
    theta_late: float,
    lp_path: str,
) -> None:
    lines: List[str] = []
    lines.append("\\ Problem 5 unknown-type assortment optimization")
    lines.append("Maximize")
    objective_terms: List[str] = []
    for item in items:
        objective_terms.append(f"{theta_early * item.price * item.w_early:.16f} se_{item.index_1_based}")
        objective_terms.append(f"{theta_late * item.price * item.w_late:.16f} sl_{item.index_1_based}")
    lines.append(" obj: " + " + ".join(objective_terms))
    lines.append("Subject To")

    early_terms = " + ".join(f"{item.w_early:.16f} se_{item.index_1_based}" for item in items)
    late_terms = " + ".join(f"{item.w_late:.16f} sl_{item.index_1_based}" for item in items)
    lines.append(f" early_norm: te + {early_terms} = 1")
    lines.append(f" late_norm: tl + {late_terms} = 1")

    for item in items:
        j = item.index_1_based
        lines.append(f" se_up_t_{j}: se_{j} - te <= 0")
        lines.append(f" se_up_x_{j}: se_{j} - x_{j} <= 0")
        lines.append(f" se_low_{j}: se_{j} - te - x_{j} >= -1")
        lines.append(f" sl_up_t_{j}: sl_{j} - tl <= 0")
        lines.append(f" sl_up_x_{j}: sl_{j} - x_{j} <= 0")
        lines.append(f" sl_low_{j}: sl_{j} - tl - x_{j} >= -1")

    lines.append("Bounds")
    lines.append(" 0 <= te <= 1")
    lines.append(" 0 <= tl <= 1")
    for item in items:
        j = item.index_1_based
        lines.append(f" 0 <= se_{j} <= 1")
        lines.append(f" 0 <= sl_{j} <= 1")

    lines.append("Binaries")
    for item in items:
        lines.append(f" x_{item.index_1_based}")

    lines.append("End")

    with open(lp_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_gurobi_sol(sol_path: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with open(sol_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            values[parts[0]] = float(parts[1])
    return values


def solve_unknown_type_with_gurobi_cl(
    items: Sequence[Item],
    theta_early: float,
    theta_late: float,
    seed: int,
) -> Dict[str, object]:
    artifacts_root = os.path.join(os.getcwd(), "results", "gurobi_artifacts")
    os.makedirs(artifacts_root, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="problem5_gurobi_", dir=artifacts_root)
    lp_path = os.path.join(workdir, "problem5_unknown.lp")
    sol_path = os.path.join(workdir, "problem5_unknown.sol")
    log_path = os.path.join(workdir, "problem5_unknown.log")
    write_problem5_milp_lp(items, theta_early, theta_late, lp_path)

    cmd = [
        "gurobi_cl",
        f"Threads=1",
        f"Seed={seed}",
        f"LogFile={log_path}",
        f"ResultFile={sol_path}",
        lp_path,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "gurobi_cl failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not os.path.exists(sol_path):
        raise RuntimeError(f"gurobi_cl finished without producing a solution file at {sol_path}")

    solution = parse_gurobi_sol(sol_path)
    selected = sorted(
        item.index_1_based for item in items if solution.get(f"x_{item.index_1_based}", 0.0) > 0.5
    )
    weighted_early = sum(item.price * item.w_early for item in items if item.index_1_based in selected)
    denom_early = sum(item.w_early for item in items if item.index_1_based in selected)
    weighted_late = sum(item.price * item.w_late for item in items if item.index_1_based in selected)
    denom_late = sum(item.w_late for item in items if item.index_1_based in selected)
    objective = theta_early * segment_revenue_from_sums(weighted_early, denom_early) + theta_late * segment_revenue_from_sums(weighted_late, denom_late)

    return {
        "selected_indices_1_based": selected,
        "selected_count": len(selected),
        "expected_mixture_revenue": objective,
        "solver_backend": "gurobi_cl_milp",
        "gurobi_artifacts": {
            "lp_path": lp_path,
            "sol_path": sol_path,
            "log_path": log_path,
        },
    }


def solve_dataset(
    data_path: str,
    early_coeffs: Dict[str, float],
    late_coeffs: Dict[str, float],
    theta_early: float,
    theta_late: float,
    seed: int,
    use_gurobi: bool,
) -> Dict[str, object]:
    items = build_items(data_path, early_coeffs, late_coeffs)

    s1 = solve_known_type(items, "w_early")
    s2 = solve_known_type(items, "w_late")
    if use_gurobi:
        s = solve_unknown_type_with_gurobi_cl(items, theta_early, theta_late, seed)
    else:
        mixture_solver = MixtureBranchAndBound(items, theta_early, theta_late)
        s = mixture_solver.solve()

    s_set = set(s["selected_indices_1_based"])
    s1_set = set(s1["selected_indices_1_based"])
    s2_set = set(s2["selected_indices_1_based"])

    s["expected_revenue_under_early_model"] = segment_subset_revenue(s_set, items, "w_early")
    s["expected_revenue_under_late_model"] = segment_subset_revenue(s_set, items, "w_late")

    s1["expected_revenue_under_early_model"] = segment_subset_revenue(s1_set, items, "w_early")
    s1["expected_revenue_under_late_model"] = segment_subset_revenue(s1_set, items, "w_late")

    s2["expected_revenue_under_early_model"] = segment_subset_revenue(s2_set, items, "w_early")
    s2["expected_revenue_under_late_model"] = segment_subset_revenue(s2_set, items, "w_late")

    comparisons = {
        "type_1_value_of_knowing_customer_type": (
            s1["expected_revenue_under_early_model"] - s["expected_revenue_under_early_model"]
        ),
        "type_2_value_of_knowing_customer_type": (
            s2["expected_revenue_under_late_model"] - s["expected_revenue_under_late_model"]
        ),
    }

    return {
        "dataset_path": os.path.abspath(data_path),
        "unknown_type_optimal_assortment": s,
        "known_early_optimal_assortment": s1,
        "known_late_optimal_assortment": s2,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 5 under the fitted Problem 4 models.")
    parser.add_argument(
        "--problem4-json",
        default="/tmp/problem4_results.json",
        help="Path to cached Problem 4 results JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data1.csv", "data2.csv", "data3.csv", "data4.csv"],
        help="Small datasets for assortment optimization.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path for a JSON result dump.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Reproducibility seed. No randomness is used, but the seed is recorded explicitly.",
    )
    parser.add_argument(
        "--use-gurobi",
        action="store_true",
        help="Solve the unknown-type assortment S using an explicit MILP with gurobi_cl.",
    )
    args = parser.parse_args()
    set_reproducibility_seed(args.seed)

    print("Problem 5: type-aware and type-unaware assortment optimization")
    print(f"Reproducibility seed: {args.seed}")
    print(f"Loading Problem 4 results from {args.problem4_json}")
    print(
        "Unknown-type solver backend: "
        + ("gurobi_cl MILP" if args.use_gurobi else "exact branch-and-bound fallback")
    )

    problem4 = load_json(args.problem4_json)
    theta_early = problem4["theta_estimates"]["theta_early"]
    theta_late = problem4["theta_estimates"]["theta_late"]
    early_coeffs = problem4["early_model"]["raw_scale_coefficients"]
    late_coeffs = problem4["late_model"]["raw_scale_coefficients"]

    print(f"Theta estimates: early={theta_early:.6f}, late={theta_late:.6f}")
    results = {}
    for dataset in args.datasets:
        print(f"Solving Problem 5 for {dataset}")
        results[dataset] = solve_dataset(
            dataset,
            early_coeffs,
            late_coeffs,
            theta_early,
            theta_late,
            args.seed,
            args.use_gurobi,
        )

    payload = {
        "problem": 5,
        "problem4_json": os.path.abspath(args.problem4_json),
        "seed": args.seed,
        "theta_estimates": {
            "theta_early": theta_early,
            "theta_late": theta_late,
        },
        "results": results,
    }

    print("Problem 5 assortment optimization complete.")
    for dataset, result in results.items():
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "S": result["unknown_type_optimal_assortment"]["selected_indices_1_based"],
                    "unknown_type_solver_backend": result["unknown_type_optimal_assortment"]["solver_backend"],
                    "nodes_visited_for_S": result["unknown_type_optimal_assortment"].get("nodes_visited"),
                    "S1": result["known_early_optimal_assortment"]["selected_indices_1_based"],
                    "S2": result["known_late_optimal_assortment"]["selected_indices_1_based"],
                    "type_1_value_of_knowing_customer_type": result["comparisons"][
                        "type_1_value_of_knowing_customer_type"
                    ],
                    "type_2_value_of_knowing_customer_type": result["comparisons"][
                        "type_2_value_of_knowing_customer_type"
                    ],
                },
                indent=2,
            )
        )

    if args.output_json:
        save_json(payload, args.output_json)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
