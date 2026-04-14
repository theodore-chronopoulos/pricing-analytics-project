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

Under this specific utility specification with **one shared linear price coefficient**, the first-order condition implies:

```text
p_j = R - 1 / beta_price
```

for every hotel `j`, so all optimal prices must be equal.

This is the main point that could look suspicious if not explained carefully. The result is a property of the model, not a coding error.

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
- `S` is solved exactly with branch-and-bound for the mixture objective:

```text
R_mix(S) = theta_early * R_early(S) + theta_late * R_late(S)
```

- The upper bound used for pruning is valid because it relaxes the unknown-type problem into the sum of the best segment-specific MNL completions.

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
