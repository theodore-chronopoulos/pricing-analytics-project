#!/usr/bin/env python3
"""Shared utilities for the Pricing Analytics project."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FEATURE_NAMES = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
]

PRICE_FEATURE = "price_usd"

DEFAULT_SCALE_FEATURES = {
    "prop_starrating",
    "prop_review_score",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
}

DEFAULT_SEED = 5132


@dataclass
class RawQuery:
    srch_id: str
    alternatives: List[List[float]]
    chosen_index: Optional[int]


@dataclass
class QueryData:
    srch_id: str
    alternatives: List[List[float]]
    chosen_index: Optional[int]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_add_in_place(dst: List[float], src: Sequence[float], scale: float = 1.0) -> None:
    for i, value in enumerate(src):
        dst[i] += scale * value


def outer_add_in_place(dst: List[List[float]], vec: Sequence[float], scale: float = 1.0) -> None:
    for i, value_i in enumerate(vec):
        row = dst[i]
        scaled_i = scale * value_i
        for j, value_j in enumerate(vec):
            row[j] += scaled_i * value_j


def matrix_subtract_rank_one_in_place(
    dst: List[List[float]],
    vec_a: Sequence[float],
    vec_b: Sequence[float],
    scale: float = 1.0,
) -> None:
    for i, value_i in enumerate(vec_a):
        row = dst[i]
        scaled_i = scale * value_i
        for j, value_j in enumerate(vec_b):
            row[j] -= scaled_i * value_j


def solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        pivot_value = a[pivot_row][col]
        if abs(pivot_value) < 1e-15:
            raise ValueError("Singular linear system encountered.")

        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]

        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        value = b[row]
        for col in range(row + 1, n):
            value -= a[row][col] * solution[col]
        solution[row] = value / a[row][row]

    return solution


def gradient_norm(gradient: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in gradient))


def set_reproducibility_seed(seed: int) -> int:
    """Set the Python RNG seed for reproducibility bookkeeping.

    The current project code paths are deterministic and do not rely on randomness,
    but we still seed the standard library RNG explicitly so the scripts expose a
    consistent reproducibility contract.
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def load_raw_queries(data_path: str) -> List[RawQuery]:
    grouped: Dict[str, RawQuery] = {}

    with open(data_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            srch_id = row["srch_id"]
            features = [float(row[name]) for name in FEATURE_NAMES]

            query = grouped.get(srch_id)
            if query is None:
                query = RawQuery(srch_id=srch_id, alternatives=[], chosen_index=None)
                grouped[srch_id] = query

            query.alternatives.append(features)
            current_index = len(query.alternatives) - 1

            booked = int(float(row["booking_bool"]))
            if booked:
                if query.chosen_index is not None:
                    raise ValueError(f"Multiple booked hotels found for query {srch_id}.")
                query.chosen_index = current_index

    return [grouped[key] for key in sorted(grouped)]


def select_queries(raw_queries: List[RawQuery], max_queries: Optional[int]) -> List[RawQuery]:
    if max_queries is None:
        return raw_queries
    return raw_queries[:max_queries]


def compute_scaling_stats(
    raw_queries: Sequence[RawQuery],
    scale_features: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    scale_set = set(scale_features)
    counts = {name: 0 for name in FEATURE_NAMES}
    sums = {name: 0.0 for name in FEATURE_NAMES}
    sums_sq = {name: 0.0 for name in FEATURE_NAMES}

    for query in raw_queries:
        for alt in query.alternatives:
            for idx, name in enumerate(FEATURE_NAMES):
                value = alt[idx]
                counts[name] += 1
                sums[name] += value
                sums_sq[name] += value * value

    stats: Dict[str, Dict[str, float]] = {}
    for name in FEATURE_NAMES:
        count = counts[name]
        mean = sums[name] / count
        variance = max((sums_sq[name] / count) - mean * mean, 0.0)
        std = math.sqrt(variance)
        should_scale = name in scale_set
        stats[name] = {
            "mean": mean,
            "std": std if should_scale and std > 1e-12 else 1.0,
            "scaled": should_scale,
        }
    return stats


def normalize_queries(
    raw_queries: Sequence[RawQuery],
    scaling_stats: Dict[str, Dict[str, float]],
) -> List[QueryData]:
    normalized_queries: List[QueryData] = []
    for query in raw_queries:
        normalized_alternatives: List[List[float]] = []
        for alt in query.alternatives:
            transformed = [1.0]
            for idx, name in enumerate(FEATURE_NAMES):
                stats = scaling_stats[name]
                value = alt[idx]
                if stats["scaled"]:
                    value = (value - stats["mean"]) / stats["std"]
                transformed.append(value)
            normalized_alternatives.append(transformed)

        normalized_queries.append(
            QueryData(
                srch_id=query.srch_id,
                alternatives=normalized_alternatives,
                chosen_index=query.chosen_index,
            )
        )
    return normalized_queries


def log_denom_and_probabilities(utilities: Sequence[float]) -> Tuple[float, List[float]]:
    pivot = max([0.0] + list(utilities))
    exp_outside = math.exp(-pivot)
    exp_terms = [math.exp(u - pivot) for u in utilities]
    scaled_denom = exp_outside + sum(exp_terms)
    log_denom = pivot + math.log(scaled_denom)
    probabilities = [term / scaled_denom for term in exp_terms]
    return log_denom, probabilities


def objective_gradient_hessian(
    queries: Sequence[QueryData],
    beta: Sequence[float],
) -> Tuple[float, List[float], List[List[float]]]:
    dimension = len(beta)
    log_likelihood = 0.0
    gradient = [0.0] * dimension
    neg_hessian = [[0.0] * dimension for _ in range(dimension)]

    for query in queries:
        utilities = [dot(beta, alt) for alt in query.alternatives]
        log_denom, probabilities = log_denom_and_probabilities(utilities)

        expected_x = [0.0] * dimension
        for alt, prob in zip(query.alternatives, probabilities):
            vector_add_in_place(expected_x, alt, prob)
            outer_add_in_place(neg_hessian, alt, prob)

        matrix_subtract_rank_one_in_place(neg_hessian, expected_x, expected_x, 1.0)

        if query.chosen_index is None:
            log_likelihood -= log_denom
            vector_add_in_place(gradient, expected_x, -1.0)
        else:
            chosen = query.alternatives[query.chosen_index]
            log_likelihood += utilities[query.chosen_index] - log_denom
            vector_add_in_place(gradient, chosen, 1.0)
            vector_add_in_place(gradient, expected_x, -1.0)

    return log_likelihood, gradient, neg_hessian


def fit_mnl(
    queries: Sequence[QueryData],
    max_iter: int,
    tolerance: float,
    initial_damping: float,
    min_step_size: float,
) -> Tuple[List[float], List[Dict[str, float]]]:
    beta = [0.0] * (len(FEATURE_NAMES) + 1)
    history: List[Dict[str, float]] = []

    current_ll, current_grad, current_neg_hess = objective_gradient_hessian(queries, beta)

    for iteration in range(1, max_iter + 1):
        grad_norm = gradient_norm(current_grad)
        history.append(
            {
                "iteration": iteration,
                "log_likelihood": current_ll,
                "gradient_norm": grad_norm,
            }
        )

        if grad_norm < tolerance:
            break

        step: Optional[List[float]] = None
        damping = initial_damping
        for _ in range(12):
            system = [row[:] for row in current_neg_hess]
            for i in range(len(system)):
                system[i][i] += damping
            try:
                step = solve_linear_system(system, current_grad)
                break
            except ValueError:
                damping *= 10.0

        if step is None:
            raise RuntimeError("Failed to compute a stable Newton step.")

        step_size = 1.0
        accepted = False
        while step_size >= min_step_size:
            candidate = [b + step_size * delta for b, delta in zip(beta, step)]
            candidate_ll, candidate_grad, candidate_neg_hess = objective_gradient_hessian(queries, candidate)
            if candidate_ll > current_ll:
                beta = candidate
                current_ll = candidate_ll
                current_grad = candidate_grad
                current_neg_hess = candidate_neg_hess
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            break

    return beta, history


def build_raw_scale_coefficients(
    beta_normalized: Sequence[float],
    scaling_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    raw_coeffs = {"intercept": beta_normalized[0]}
    intercept_adjustment = 0.0

    for idx, name in enumerate(FEATURE_NAMES, start=1):
        coeff = beta_normalized[idx]
        stats = scaling_stats[name]
        if stats["scaled"]:
            raw_coeff = coeff / stats["std"]
            intercept_adjustment += coeff * stats["mean"] / stats["std"]
        else:
            raw_coeff = coeff
        raw_coeffs[name] = raw_coeff

    raw_coeffs["intercept"] -= intercept_adjustment
    return raw_coeffs


def summarize_dataset(raw_queries: Sequence[RawQuery]) -> Dict[str, float]:
    total_alternatives = sum(len(query.alternatives) for query in raw_queries)
    chosen_queries = sum(query.chosen_index is not None for query in raw_queries)
    no_purchase_queries = len(raw_queries) - chosen_queries
    return {
        "queries": len(raw_queries),
        "alternatives": total_alternatives,
        "avg_alternatives_per_query": total_alternatives / len(raw_queries),
        "chosen_queries": chosen_queries,
        "no_purchase_queries": no_purchase_queries,
    }


def estimate_mnl_from_raw_queries(
    raw_queries: Sequence[RawQuery],
    scaling_stats: Dict[str, Dict[str, float]],
    max_iter: int = 50,
    tolerance: float = 1e-8,
    initial_damping: float = 1e-6,
    min_step_size: float = 1e-6,
) -> Dict[str, object]:
    normalized_queries = normalize_queries(raw_queries, scaling_stats)
    beta, history = fit_mnl(
        normalized_queries,
        max_iter=max_iter,
        tolerance=tolerance,
        initial_damping=initial_damping,
        min_step_size=min_step_size,
    )

    final_ll, final_grad, _ = objective_gradient_hessian(normalized_queries, beta)
    avg_ll = final_ll / len(normalized_queries)
    raw_coeffs = build_raw_scale_coefficients(beta, scaling_stats)

    return {
        "dataset_summary": summarize_dataset(raw_queries),
        "normalized_coefficients": {
            "intercept": beta[0],
            **{name: beta[idx] for idx, name in enumerate(FEATURE_NAMES, start=1)},
        },
        "raw_scale_coefficients": raw_coeffs,
        "fit_summary": {
            "final_log_likelihood": final_ll,
            "average_log_likelihood_per_query": avg_ll,
            "final_gradient_norm": gradient_norm(final_grad),
            "iterations_recorded": len(history),
        },
        "history": history,
    }


def estimate_problem1_model(
    data_path: str,
    max_queries: Optional[int] = None,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    initial_damping: float = 1e-6,
    min_step_size: float = 1e-6,
    scale_binary: bool = False,
) -> Dict[str, object]:
    scale_features = set(FEATURE_NAMES) if scale_binary else DEFAULT_SCALE_FEATURES

    raw_queries = load_raw_queries(data_path)
    raw_queries = select_queries(raw_queries, max_queries)
    scaling_stats = compute_scaling_stats(raw_queries, scale_features)
    result = estimate_mnl_from_raw_queries(
        raw_queries,
        scaling_stats,
        max_iter=max_iter,
        tolerance=tolerance,
        initial_damping=initial_damping,
        min_step_size=min_step_size,
    )

    return {
        "data_path": os.path.abspath(data_path),
        "scaling_stats": scaling_stats,
        **result,
    }


def load_query_booking_windows(data_path: str) -> Dict[str, float]:
    booking_windows: Dict[str, float] = {}
    with open(data_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            srch_id = row["srch_id"]
            booking_window = float(row["srch_booking_window"])
            previous = booking_windows.get(srch_id)
            if previous is not None and abs(previous - booking_window) > 1e-12:
                raise ValueError(f"Inconsistent booking window for query {srch_id}.")
            booking_windows[srch_id] = booking_window
    return booking_windows


def save_json(payload: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_small_dataset(data_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(data_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({name: float(row[name]) for name in FEATURE_NAMES})
    return rows


def raw_utility(row: Dict[str, float], raw_coeffs: Dict[str, float]) -> float:
    utility = raw_coeffs["intercept"]
    for name in FEATURE_NAMES:
        utility += raw_coeffs[name] * row[name]
    return utility


def price_free_utility(row: Dict[str, float], raw_coeffs: Dict[str, float]) -> float:
    utility = raw_coeffs["intercept"]
    for name in FEATURE_NAMES:
        if name == PRICE_FEATURE:
            continue
        utility += raw_coeffs[name] * row[name]
    return utility


def safe_exp(value: float) -> float:
    if value > 700:
        return math.exp(700)
    if value < -700:
        return math.exp(-700)
    return math.exp(value)


def assortment_revenue(rows: Sequence[Dict[str, float]], raw_coeffs: Dict[str, float]) -> float:
    weights = [safe_exp(raw_utility(row, raw_coeffs)) for row in rows]
    weighted_prices = sum(row[PRICE_FEATURE] * weight for row, weight in zip(rows, weights))
    return weighted_prices / (1.0 + sum(weights))


def expected_revenue_from_components(
    prices: Sequence[float],
    price_free_utilities: Sequence[float],
    beta_price: float,
) -> float:
    weights = [safe_exp(base + beta_price * price) for base, price in zip(price_free_utilities, prices)]
    weighted_prices = sum(price * weight for price, weight in zip(prices, weights))
    return weighted_prices / (1.0 + sum(weights))


def segment_revenue_from_sums(weighted_price_sum: float, weight_sum: float) -> float:
    return weighted_price_sum / (1.0 + weight_sum)


def solve_revenue_ordered_assortment(
    prices: Sequence[float],
    weights: Sequence[float],
) -> Dict[str, object]:
    ranked = list(enumerate(zip(prices, weights), start=1))
    ranked.sort(key=lambda item: item[1][0], reverse=True)

    best_revenue = 0.0
    best_indices: List[int] = []
    cum_weight = 0.0
    cum_weighted_price = 0.0
    current_indices: List[int] = []

    for index_1_based, (price, weight) in ranked:
        cum_weight += weight
        cum_weighted_price += price * weight
        current_indices.append(index_1_based)
        revenue = segment_revenue_from_sums(cum_weighted_price, cum_weight)
        if revenue > best_revenue:
            best_revenue = revenue
            best_indices = current_indices[:]

    return {
        "selected_indices_1_based": best_indices,
        "selected_count": len(best_indices),
        "expected_revenue": best_revenue,
        "price_ranked_indices_1_based": [index for index, _ in ranked],
    }


def bracket_unimodal_maximum(
    objective,
    initial_high: float,
    max_high: float = 1_000_000.0,
) -> Tuple[float, float]:
    left = 0.0
    right = max(initial_high, 1.0)
    prev_right = 0.0
    prev_value = objective(prev_right)
    right_value = objective(right)

    while right < max_high and right_value > prev_value:
        left = prev_right
        prev_right = right
        prev_value = right_value
        right *= 2.0
        right_value = objective(right)

    return left, right


def golden_section_maximize(
    objective,
    left: float,
    right: float,
    tolerance: float = 1e-8,
    max_iter: int = 200,
) -> Tuple[float, float]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = right - (right - left) * inv_phi
    d = left + (right - left) * inv_phi
    fc = objective(c)
    fd = objective(d)

    for _ in range(max_iter):
        if abs(right - left) < tolerance:
            break
        if fc > fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * inv_phi
            fc = objective(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * inv_phi
            fd = objective(d)

    point = (left + right) / 2.0
    return point, objective(point)
