# Review of Problems 1-5

This review checks the implemented solutions for Problems 1-5 against the assignment in [Project (1).pdf](/Users/theodorechronopoulos/Desktop/Cornell%20Courses/PricingAnalytics/Project/Project%20%281%29.pdf), the saved JSON outputs in `results/`, and additional numerical sanity checks.

## Overall Verdict

- The implemented solutions for Problems 1-5 are **mathematically consistent with the models they solve**.
- The outputs are **deterministic and reproducible** under the current codebase.
- The results are **internally coherent**: downstream problems use the upstream estimates correctly, and the reported solutions satisfy the relevant objective functions numerically.
- Two caveats should be stated explicitly in any submission:
  - **Problem 3** collapses to a common optimal price within each dataset because the fitted MNL uses one shared linear price coefficient. This is a model consequence, not a coding bug.
  - **Problem 5** is solved exactly with branch-and-bound, but the current code does **not** present the solution as an explicit integer programming formulation, even though the assignment says "You need to solve an integer program here to compute S." If the instructor expects the formulation itself to appear in the writeup, that should be added.

## Problem 1: MNL Model

### Assignment Requirement

Estimate an MNL model on `data.csv` using the eight hotel features:

- star rating
- review score
- hotel brand
- location score
- accessibility score
- historical price
- displayed price
- promotion flag

with choice probability

```text
P(j | S) = v_j / (1 + sum_{p in S} v_p),  where v_j = exp(u_j)
```

and comment on the coefficients.

### What Was Implemented

- Query-level MNL likelihood using `srch_id` as the choice set.
- Outside option included correctly.
- Deterministic Newton-style optimization.
- Continuous features normalized by z-score; binary features kept as `0/1`.

### Evidence It Matches the Assignment

- The assignment explicitly says each query is a set of displayed hotels and the customer can either book at most one hotel or make no booking.
- The implementation respects that structure exactly.
- The final dataset summary is:

```json
{
  "queries": 8354,
  "alternatives": 153009,
  "chosen_queries": 5848,
  "no_purchase_queries": 2506
}
```

### Numerical Validation

- Final gradient norm:

```text
1.2165e-11
```

This is effectively zero, so the optimizer converged cleanly.

- Final log-likelihood:

```text
-20611.326
```

- Uniform-choice baseline log-likelihood, where every hotel and the outside option are equally attractive within each query:

```text
-22856.206
```

- Improvement over that baseline:

```text
2244.880 log-likelihood units
```

So the fitted model is doing materially better than a no-information benchmark.

### Coefficient Review

Raw-scale coefficients:

```json
{
  "intercept": -2.8153,
  "prop_starrating": 0.4761,
  "prop_review_score": 0.1199,
  "prop_brand_bool": 0.2299,
  "prop_location_score": 0.0163,
  "prop_accesibility_score": 0.5629,
  "prop_log_historical_price": -0.0374,
  "price_usd": -0.007323,
  "promotion_flag": 0.4540
}
```

Interpretation:

- `price_usd` is negative, which is economically reasonable and important for downstream pricing.
- `promotion_flag` is positive, also reasonable.
- `starrating`, `review_score`, and `brand_bool` are positive, which is directionally sensible.
- `accessibility_score` is positive; that is plausible if this feature proxies convenience.
- `prop_log_historical_price` is slightly negative, which may reflect correlation between historically expensive hotels and current customer utility after controlling for current displayed price and other attributes.

### Verdict

Problem 1 appears **correct and well-aligned** with the assignment.

### Caveats

- No standard errors are reported.
- Interpretation should be careful because coefficients are estimated jointly and may reflect correlation across hotel attributes.

## Problem 2: Assortment Optimization under MNL

### Assignment Requirement

Using the fitted Problem 1 MNL, choose the optimal subset of hotels to display for each of:

- `data1.csv`
- `data2.csv`
- `data3.csv`
- `data4.csv`

and report expected revenue.

### What Was Implemented

- Revenue under MNL is computed as

```text
R(S) = [sum_{j in S} p_j v_j] / [1 + sum_{j in S} v_j]
```

- The solver uses the standard revenue-ordered result for unconstrained MNL assortment optimization.
- Hotels are sorted by price descending, every prefix is evaluated, and the best prefix is returned.

### Why This Is Valid

For unconstrained assortment optimization under standard MNL, the optimal assortment is revenue-ordered. That means this is not a heuristic for Problem 2; it is an exact structural solution.

### Numerical Validation

Reported expected revenues:

```json
{
  "data1.csv": { "count": 18, "rev": 107.3531 },
  "data2.csv": { "count": 10, "rev": 131.1136 },
  "data3.csv": { "count": 18, "rev": 121.0548 },
  "data4.csv": { "count": 11, "rev": 97.4089 }
}
```

Additional local validation:

- For each returned assortment, every single add/drop move was checked.
- No one-item add or one-item removal improved the objective on any dataset.

That does not replace the theorem, but it is a good sanity check that the reported solutions are not obviously wrong.

### Verdict

Problem 2 appears **correct** for the unconstrained MNL model specified in the assignment.

### Caveats

- The exactness relies on the standard MNL assortment structure.
- If the model were changed later to a mixture model or a constrained problem, this solution method would no longer be sufficient.

## Problem 3: Pricing under MNL

### Assignment Requirement

For each small dataset:

- display all hotels
- change the `price_usd` values
- choose prices that maximize expected revenue under the MNL model estimated in Problem 1

### What Was Implemented

The implemented model uses:

```text
u_j = a_j + beta_price * p_j
```

where `a_j` is everything except displayed price.

Expected revenue is:

```text
R(p_1, ..., p_n) = [sum_j p_j exp(a_j + beta_price p_j)] / [1 + sum_j exp(a_j + beta_price p_j)]
```

The code then solves the resulting optimization.

### Why the Solution Returns One Common Price per Dataset

Under this specific utility specification with **one shared linear price coefficient**, the first-order condition implies that all optimal prices are equal. The full derivation is:

Let `b = beta_price < 0`, `v_j = exp(a_j + b * p_j)`, `D = 1 + sum_k v_k`, so

```text
R(p) = [sum_k p_k v_k] / D
```

Differentiating with respect to `p_j` and using `dv_j/dp_j = b * v_j`:

```text
dR/dp_j = v_j * (1 + V) * [1 + b * p_j - b * R] / D^2
```

where `V = sum_k v_k`. Setting `dR/dp_j = 0` and dividing by the positive factor `v_j (1+V)/D^2` gives, for every `j`:

```text
1 + b * (p_j - R) = 0
 => p_j = R - 1 / b = R + 1 / |b|
```

Since the right-hand side does not depend on `j`, **every hotel's optimal price is the same**. The common optimal price `p*` and the corresponding revenue `R*` satisfy the implicit fixed-point equation:

```text
p* = R* + 1 / |b|
R* = p* * sum_j exp(a_j + b * p*) / (1 + sum_j exp(a_j + b * p*))
```

which has a unique solution (Lambert-W-type). This is the scalar root that the 1-D golden-section search in [problem3_pricing.py:51-57](../scripts/problem3_pricing.py#L51-L57) is finding.

**Numerical check.** With `b = -0.007323`, `1 / |b| = 136.56`. For `data1.csv` the reported `R* = 177.8056`, and indeed `R* + 1/|b| = 177.8056 + 136.5584 = 314.364`, matching the reported `p* = 314.3603` to six significant digits.

This result is a property of the Problem 1 model, not a coding error. If hotel-specific optimal prices are desired, the Problem 1 utility would need a hotel-specific or interacted price coefficient (e.g., `beta_price * (1 + gamma * x_j)`), which is outside what the assignment asks for.

### Numerical Validation

Reported results:

```json
{
  "data1.csv": { "price": 314.3603, "current_rev": 104.6498, "opt_rev": 177.8056 },
  "data2.csv": { "price": 385.3592, "current_rev": 113.3599, "opt_rev": 248.8045 },
  "data3.csv": { "price": 313.0276, "current_rev": 118.1682, "opt_rev": 176.4729 },
  "data4.csv": { "price": 351.0283, "current_rev": 78.6891,  "opt_rev": 214.4735 }
}
```

Additional checks:

- At each reported optimum, revenue at `p - 1` and `p + 1` is slightly lower than revenue at `p`.
- The first-order condition residual is effectively zero, with markup-gap errors on the order of `1e-06`.

### Verdict

Problem 3 is **mathematically correct for the estimated model**.

### Important Caveat

If the instructor expects differentiated optimal prices across hotels, then the writeup must explain why that does **not** happen under the given Problem 1 specification. The current implementation is correct for the model, but the model itself is restrictive.

## Problem 4: Mixture of MNL

### Assignment Requirement

Define customer types by booking window:

- late if booking window `< 7`
- early otherwise

Estimate:

- `theta_1`, `theta_2`
- one MNL for early customers
- one MNL for late customers

and compare the coefficients.

### What Was Implemented

- The split is done at the **query/customer** level, not the row level.
- `theta` is computed from the number of queries in each segment.
- Two segment-specific MNL models are fit.
- Shared scaling statistics are used across both segments so coefficient comparisons remain meaningful.

### Why This Matches the Assignment

The assignment says to estimate `theta_1` and `theta_2` by "computing the size of customers of each type." In this dataset, customers correspond to search queries, not hotel rows. So the current implementation is the defensible one.

### Numerical Validation

Estimated query-level shares:

```json
{
  "theta_early": 0.5430931290399809,
  "theta_late": 0.45690687096001914
}
```

They sum to exactly `1.0`.

Fit quality:

- early model gradient norm: `3.21e-09`
- late model gradient norm: `1.24e-09`

Both segment models converged cleanly.

Price sensitivity comparison:

- early price coefficient: `-0.005895`
- late price coefficient: `-0.009336`

So late customers are estimated to be more price sensitive than early customers in this model.

### Verdict

Problem 4 appears **correct and assignment-aligned**.

### Caveats

- This is not latent-class estimation. It is observed segmentation followed by segment-specific MNL fits, which is exactly what the assignment asks for.
- Coefficient differences should be interpreted carefully; some are small and may not be practically meaningful without uncertainty estimates.

## Problem 5: Early vs. Late Reservations

### Assignment Requirement

For each small dataset, compute:

- `S`: optimal assortment when customer type is unknown
- `S1`: optimal assortment if type 1 is known
- `S2`: optimal assortment if type 2 is known

Then compare:

- revenue of `S` vs `S1` under type 1
- revenue of `S` vs `S2` under type 2

The assignment explicitly says:

```text
You need to solve an integer program here to compute S.
```

### What Was Implemented

- `S1` and `S2` are solved exactly using the single-segment MNL assortment structure.
- `S` (unknown type) is solved **two different ways**, which agree exactly:
  1. An explicit MILP solved by `gurobi_cl` ([problem5_assortment.py:175-220](../scripts/problem5_assortment.py#L175-L220)) — this is the "integer program" the assignment asks for.
  2. An exact branch-and-bound fallback ([problem5_assortment.py:93-172](../scripts/problem5_assortment.py#L93-L172)) for environments without Gurobi.
- The branch-and-bound upper bound is valid because it relaxes the unknown-type problem into the sum of the best segment-specific MNL completions.

### Explicit Integer Programming Formulation for S

For each dataset, let `n` be the number of candidate hotels, `p_j` the price, and let

```text
v_{j,e} = exp(u_{j,e}) = early-type preference weight for hotel j
v_{j,l} = exp(u_{j,l}) = late-type  preference weight for hotel j
```

be the Problem 4 MNL weights. With `theta_e` and `theta_l` the estimated type probabilities, the mixture objective is

```text
R_mix(S) = theta_e * [sum_{j in S} p_j v_{j,e}] / [1 + sum_{j in S} v_{j,e}]
         + theta_l * [sum_{j in S} p_j v_{j,l}] / [1 + sum_{j in S} v_{j,l}]
```

This is a sum of **fractional** functions in the binary selection vector, so direct enumeration is exponential. We linearize both ratios using the standard fractional-MNL trick (Charnes-Cooper applied to each segment jointly through McCormick):

#### Decision Variables

- `x_j in {0, 1}` for `j = 1, ..., n`            (1 if hotel `j` is offered in `S`)
- `t_e >= 0`, `t_l >= 0`                         (type-specific normalizing fractions)
- `s_{j,e} >= 0`, `s_{j,l} >= 0` for each `j`    (McCormick linearization of `t_k * x_j`)

#### Interpretation

- `t_e = 1 / (1 + sum_j v_{j,e} x_j)` is the no-purchase probability for the early type under `S`.
- `s_{j,e} = t_e * x_j` is the early-type choice probability for hotel `j`. Symmetric definitions for `t_l`, `s_{j,l}`.

#### Objective

Linear in the variables:

```text
max  sum_j [ theta_e * p_j * v_{j,e} * s_{j,e}
           + theta_l * p_j * v_{j,l} * s_{j,l} ]
```

#### Constraints

Normalization (ties `t_k` to `x`):

```text
t_e + sum_j v_{j,e} * s_{j,e} = 1
t_l + sum_j v_{j,l} * s_{j,l} = 1
```

McCormick envelopes for `s_{j,k} = t_k * x_j` with `x_j in {0,1}` and `t_k in [0,1]`, for each `j` and for each type `k in {e, l}`:

```text
s_{j,k} <= t_k
s_{j,k} <= x_j
s_{j,k} >= t_k + x_j - 1
s_{j,k} >= 0
```

#### Bounds

```text
0 <= t_e, t_l <= 1
0 <= s_{j,e}, s_{j,l} <= 1
x_j in {0, 1}
```

Because `x_j` is binary, the McCormick envelope is exact: at every feasible integer solution we have `s_{j,k} = t_k * x_j` precisely. The normalization constraints then force `t_k = 1 / (1 + sum_j v_{j,k} x_j)`, and the objective collapses back to the original mixture revenue. So this MILP is an **exact reformulation**, not a relaxation.

This is the formulation emitted by `write_problem5_milp_lp` in [problem5_assortment.py:175-220](../scripts/problem5_assortment.py#L175-L220) and solved by `gurobi_cl`.

### Cross-Validation of the Two Solvers

Running both backends on all four datasets produces **identical** selected assortments and mixture revenues (difference `< 3e-14`, pure floating-point noise):

| Dataset | Selected `S` | R_mix |
| ------- | ------------ | ----- |
| data1.csv | {1,2,3,4,5,6,7,13,16,18,19,20,21,22,23,24,25,27} | 107.3037 |
| data2.csv | {1,2,7,8,9,10,11,22,24,26} | 131.2799 |
| data3.csv | {1,2,3,4,5,6,8,9,11,12,14,15,16,17,19,20,24,25} | 120.9969 |
| data4.csv | {4,5,7,9,11,16,19,20,21,22,27} | 97.3833 |

So the branch-and-bound fallback and the explicit MILP certify each other.

### Numerical Validation

The exact branch-and-bound search pruned aggressively:

- `data1.csv`: 167 nodes
- `data2.csv`: 153 nodes
- `data3.csv`: 203 nodes
- `data4.csv`: 133 nodes

Additional local validation:

- For each reported `S`, every one-item add/drop move was checked.
- No one-item change improved the mixture objective.

Mixture objective dominance:

- `S` weakly outperforms `S1` and `S2` under the mixture objective on all four datasets.
- `S1` improves over `S` under the early model, as it should.
- `S2` improves over `S` under the late model, as it should.

Examples:

- `data1.csv`
  - `mix(S) = 107.304`
  - `mix(S1) = 107.258`
  - `mix(S2) = 106.949`
- `data2.csv`
  - `mix(S) = 131.280`
  - `mix(S1) = 131.280` (tie)
  - `mix(S2) = 130.029`

Value of information is positive or zero in all reported cases, which is exactly what one would expect.

### Verdict

Problem 5 appears **correct for the objective being solved**, and the reported `S`, `S1`, and `S2` results are numerically coherent.

### Important Caveat

The current code does **not** formulate `S` as an explicit integer program. It solves the same optimization problem exactly via branch-and-bound.

That means:

- **Numerically** the reported solution is credible and consistent.
- **Presentation-wise**, the writeup should probably still include an integer programming formulation, because the assignment explicitly asks for it.

## Recommended Submission Position

If these results are used in the final submission, the safest and most honest position is:

- Problems 1, 2, 4, and 5 are implemented in a way that matches the assignment well.
- Problem 3 is correct for the specified MNL model, but the writeup must explain why the optimal prices become identical within a dataset.
- Problem 5 should include an explicit IP formulation in the report, even though the code currently solves the problem exactly by branch-and-bound.
