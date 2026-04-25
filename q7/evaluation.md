# Candid evaluation of `q7/`

This implementation follows the assignment structure (4 sub-parts, real held-out evaluation, MNL re-estimation, discussion), and the probability-implied MNL summary in 7c is a genuine improvement over a naive argmax comparison. Granting that the AI prompting in Part 7a was done manually as claimed, two real bugs remain in 7c and one auditability concern stands. Going from worst to least-worst.

## Critical issues

### 1. Bug in the MNL utility computation in 7c — the "MNL hard prediction" results are unreliable

[q7/7c/P7c.ipynb](7c/P7c.ipynb), cell with `mnl_predict_query`, z-scores **all eight features** before applying `human_beta` (the normalized coefficients from Problem 1). But the Problem 1 model only z-scored continuous features — `prop_brand_bool` and `promotion_flag` were left at $\{0,1\}$. The normalized coefficient $0.23$ for brand and $0.45$ for promotion was estimated with binary inputs, not z-scored ones.

In the held-out sample, brand has mean $\approx 0.74$, std $\approx 0.44$, and promotion has mean $\approx 0.14$, std $\approx 0.35$. The 7c code therefore computes utility contributions like:

| feature flip | should be | 7c code computes |
| --- | --- | --- |
| brand from 0 to 1 | $+0.23$ | $+0.23 \times (1{-}0.74)/0.44 = +0.136$ |
| brand at 0 | $0$ | $-0.39$ |
| promo from 0 to 1 | $+0.45$ | $+1.11$ |
| promo at 0 | $0$ | $-0.18$ |

Effect: most non-branded, non-promoted hotels are pushed below $u=0$ that they shouldn't be, and promoted hotels get $\sim\!2.5\times$ too much weight. This explains why the MNL hard rule predicts NO_PURCHASE on $46/50$ queries — that's a feature-scaling bug, not a property of the calibrated MNL. The **22 % MNL accuracy and the 8 % MNL purchase-rate numbers should be considered wrong**. The probability-implied summary at the end of 7c uses the same buggy code, but it integrates over more outcomes so the bias is partly washed out — that's why its $65.6\%$ purchase rate looks more reasonable.

### 2. Both models underperform the trivial NO_PURCHASE baseline — and this isn't acknowledged

On the 50 held-out queries, $26\%$ are no-purchase (13 of 50). A model that always predicts `NO_PURCHASE` therefore gets $26\%$ exact-choice accuracy. Reported numbers: AI $18\%$, MNL hard $22\%$. **Both models score below the trivial baseline**, yet the discussion in 7c and 7d describes the MNL as "slightly outperforming the AI agent." The honest read is that on 50 queries, neither model beats "always say no" by exact-choice accuracy, and the AI–MNL gap of $11{-}9 = 2$ correct predictions has a standard error roughly $\pm 4$ — i.e. it is statistical noise. The conclusion "MNL slightly outperforms" is overstated.

## Moderate issues

### 3. Part 7a has no audit trail for the manual prompting

[q7/7a/P7a.ipynb](7a/P7a.ipynb) contains only markdown describing the prompt and what was done. Even granting that the prompting was done manually as claimed, there is no chat transcript, no per-query log, and no script in the repo that produced the AI booking column in [q7/7b/Input/1_data_with_ai_bookings_sample100_corrected.csv](7b/Input/1_data_with_ai_bookings_sample100_corrected.csv). One pattern in that file is worth flagging in advance: across all 80 booked queries, the booked alternative is exactly the argmax of `ai_score_corrected` (80/80 agreement), and the BUY/NO_PURCHASE split is a clean threshold at zero (BUY queries have min score $0.066$, NO_PURCHASE queries have max score $-0.059$). That pattern is consistent with deterministic post-processing of LLM-generated scores; it does not by itself disprove that ChatGPT was used upstream, but a grader who notices it will ask. **Add a chat export or per-query log to the repo before submission.**

### 4. Part 7b: ridge $\lambda = 1$ is arbitrary, and the comparison table mixes scales

The 7b regression solves
$$
\min_\beta\; -\mathcal{L}(\beta) + 1.0 \cdot \sum_{i \ge 1} \beta_i^2,
$$
i.e. MAP with a $\mathcal{N}(0, 0.5)$ prior. There is no cross-validation, no justification for $\lambda = 1$, and no sensitivity analysis. The notebook honestly says "the signs and relative magnitudes are interpreted rather than the coefficients as exact behavioral parameters" — that's correct, but it means the reported magnitudes (e.g.\ AI prop\_review\_score = $+3.08$ vs.\ human $+0.11$) are essentially "whatever ridge-with-lambda-1 returned on 100 deterministic queries". Don't give that ratio quantitative weight in the writeup.

The comparison table also mixes scaling conventions: the "human value" column is the Problem 1 normalized coefficients (binary features unscaled, continuous z-scored on the full $153\,009$ rows), while the "ai value" column is from a regression that z-scored everything on the 100-query sample. **They are not on the same axis**. Again, only signs are comparable.

### 5. Part 7c: 50 held-out queries, 10 in-context examples — too small to discriminate

A 50-query held-out evaluation gives standard errors around $\pm 6$ percentage points on accuracy. The AI–MNL gap is well within that. The 10-example context window is also small; the assignment doesn't require a specific size, but the whole exercise is essentially a single Bernoulli trial per query at $n=50$, so any conclusions should be hedged much more strongly than 7c/7d hedge them.

### 6. No record of which AI model produced `ai_heldout_predictions.csv`

The notebook saves `ai_heldout_prompt.txt` and then loads `ai_heldout_predictions.csv` without showing any pipeline that connects the two. If a real LLM was used, the model name and version need to be recorded; the assignment explicitly asks for this ("clearly state which AI agent you used and which version"). The 7a markdown says "ChatGPT (GPT-5.4 Thinking)" but this isn't carried into 7c, and we have no way to verify whether 7c's predictions came from the same model, a different model, a hand prediction, or a script.

## Minor issues

- 7c uses `data (1).csv` (filename has a space and parenthesis) — fragile.
- 7c emits a Pandas `DeprecationWarning` from `groupby().apply(...)` — cosmetic.
- 7a mentions "random seed 42" but no code is shown that sets it (since the notebook has no code at all).
- 7c's prompt format puts the customer context as a Python `dict` repr (`{'srch_booking_window': 0, ...}`). It works, but a clean key:value listing would be more LLM-friendly and reproducible.
- The notebook would benefit from a single-cell "regenerate everything from scratch" path so the downstream user can rerun the LLM-prompting step end-to-end.

## What's actually good

- Methodology is right at the high level: query-level random train/test split (seed 42), in-context examples, hold-out exact-choice + behavioral comparison.
- **Part 7b's MNL log-likelihood is correctly specified.** The outside option is included with $u_0 = 0$, log-sum-exp stabilization is right, the chosen / no-purchase branches are right, and the per-query loop respects the choice-set structure. The MLE on the 100-query AI sample is fit correctly. The remaining objection (issue #4) is about the comparison axis and the $\lambda=1$ choice, not about the MNL model itself.
- The probability-implied MNL summary in 7c is genuinely the right thing to do and addresses a real flaw I would have flagged otherwise.
- 7d's discussion is balanced and avoids overclaiming AI capability.
- Per-query treatment is consistent: AI booking on 80/100 sample queries; held-out 50; trained on the rest. No row-level / query-level confusion.

## Recommended fixes before submitting

1. **Fix the MNL prediction in 7c**: do not z-score `prop_brand_bool` and `promotion_flag`; for the continuous features, use the full-data scaling stats from `results/problem1_results.json` (or refit on the train split with the same `compute_scaling_stats` rule). Re-report the MNL accuracy and purchase rate after the fix.
2. **Add the trivial NO_PURCHASE baseline** ($26\%$) to the 7c accuracy table and acknowledge that both models fall below it on this sample. This is honest and matches how 7d's tone treats exact-choice accuracy as "a high bar."
3. **Add a chat transcript or per-query log** for the Part-7a manual prompting step. This is the audit trail a grader will look for.
4. **Quote 95 % CIs** on the 50-query accuracies (binomial CI is wide), so the AI/MNL comparison reads as "indistinguishable" rather than "MNL slightly better."
5. In 7b, justify $\lambda=1$ (or use 5-fold CV at the query level), and either re-do the AI regression with the **same scaling stats** as Problem 1, or re-do Problem 1's normalized comparison with z-scored binaries — pick one convention and stick to it. Otherwise the coefficient comparison is misleading.
6. Record the AI model name/version inside the 7c notebook (and ideally save the raw response text alongside the CSV).

Items 1 and 2 are the ones a careful grader will actually catch.
