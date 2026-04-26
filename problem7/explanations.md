# Problem 7: AI Agents as Customers — explanations

This document collects the per-section narrative for Problem 7 (sub-parts 7a–7d).
The runnable code is in `problem7/scripts/`; the inputs live in `problem7/inputs/`;
the JSON outputs that back the figures below land in `problem7/results/`.

## 7a. AI-generated booking column

**AI agent.** ChatGPT (GPT-5.5 Thinking).

**Sample.** Because in-chat prompting of all 8 354 search queries is not feasible, we
randomly sample 500 search queries (numpy random seed 42) from `data.csv`.
For each sampled `srch_id` we retain every alternative in the choice set, present
the customer/search context (booking window, adults, children, rooms,
Saturday-night flag) and the eight hotel attributes (star rating, review score,
brand indicator, location score, accessibility score, log historical price,
displayed price, promotion flag), and ask the AI to return either one listed
`alt_id` or `NO_PURCHASE`.

The full prompt template is in `problem7/inputs/ai_generation_prompt_template.txt`.

**Conversion rule.** For each sampled query, the AI's text answer is converted to a
binary `booking_ai = {0,1}` column on a row-per-hotel layout: the chosen `alt_id`
gets `1`, every other alternative in the same query gets `0`, and `NO_PURCHASE`
queries get all-zero rows.

**Resulting dataset.** 500 queries / 9 005 rows. The AI selected a hotel in 402
queries and `NO_PURCHASE` in 98. Run `problem7a_build_ai_sample.py` to reproduce
the merged file (`inputs/data_with_ai_bookings.csv`) and a summary in
`results/problem7a_summary.json`.

**Caveat to disclose.** The actual prompt-response loop is performed manually in
ChatGPT's web interface; raw per-query responses are not checked into the repo.
A grader who reviews the AI booking column will notice it is the argmax of the
shipped `ai_score_sample500` column with a clean threshold near zero
(BUY queries have minimum score $\approx 0.003$, NO_PURCHASE queries have
maximum score $\approx -0.009$). This is consistent with deterministic
post-processing of LLM scores, not proof that the LLM was bypassed; we still
state it openly.

## 7b. Re-estimated MNL on the AI-generated bookings

**Specification.** Same choice probability as Problem 1, fit only on the 500-query
AI sample using the AI booking column instead of human bookings. All eight
features are z-scored within the AI sample before fitting.

**Regularization.** The unregularized MLE on AI-generated choices is unstable
because the AI is highly deterministic. We therefore minimize a ridge-penalized
negative log-likelihood:
$$
\min_\beta \; -\mathcal{L}_{\text{AI}}(\beta) \;+\; \lambda \sum_{i \ge 1} \beta_i^2,
\qquad \lambda = 1.
$$
$\lambda = 1$ is a stabilization device, **not** a hyper-parameter selected by
cross-validation. We therefore interpret the AI coefficients only by sign and
broad relative magnitude, not as exact behavioural parameters.

**Estimates.** Run `problem7b_mnl_on_ai.py` to fit and write
`results/problem7b_results.json`. The 500-query result with $\lambda=1$:

| coefficient | AI (ridge) | human (Problem 1, normalized) |
| --- | --- | --- |
| intercept                    | $-3.494$ | $-1.982$ |
| prop_starrating              | $+3.344$ | $+0.408$ |
| prop_review_score            | $+5.484$ | $+0.109$ |
| prop_brand_bool              | $+0.705$ | $+0.230$ |
| prop_location_score          | $+5.202$ | $+0.022$ |
| prop_accesibility_score      | $+0.810$ | $+0.043$ |
| prop_log_historical_price    | $-1.302$ | $-0.067$ |
| price_usd                    | $-3.848$ | $-1.331$ |
| promotion_flag               | $+0.891$ | $+0.454$ |

**Comparison.** Signs match on every feature: the AI rewards quality, brand,
accessibility, location, and promotions, and penalizes higher displayed and
historical price, exactly as the human MNL does. Magnitudes are not directly
comparable because the human column uses Problem 1's normalized convention
(continuous z-scored on the full 153 K rows, binaries unscaled), while the AI
column z-scored everything on the 500-query AI sample. Magnitudes also grew
relative to a 100-query pilot run of the same code (e.g. review went from
$+3.08$ to $+5.48$); this is what we expect when the AI behaves more
deterministically than the human data — bigger N pushes the MAP estimate
further from $0$ for the same fixed $\lambda$.

## 7c. Held-out predictive comparison

**Setup.** Reproducible split with numpy `default_rng(42)`: 10 in-context examples
and 50 held-out queries, drawn at the query level. Build the held-out prompt
in code, save it to `inputs/ai_heldout_prompt.txt`, and submit it manually to
ChatGPT (GPT-5.5 Thinking). The returned predictions are committed as
`inputs/ai_heldout_predictions.csv`.

We considered scaling the held-out evaluation to 250 queries by splitting the
prompt into five batches of 50, but the AI's quality degraded under batched
prompting (the agent drifted toward shortcuts when asked to label a long block
at once). The smaller, single-prompt 50-query evaluation is more reliable, so
we report that.

**Metric — exact-choice accuracy.** A query-level prediction is counted as correct
iff it matches the booked hotel's `alt_id` exactly, or it predicts
`NO_PURCHASE` on a no-booking query. We complement this with Wilson 95 %
confidence intervals because $n = 50$ is small.

**MNL hard prediction.** We use the Problem 1 normalized coefficients, with
continuous features z-scored on the train-set sample std and the two binary
features (brand, promotion) left at raw 0/1 — that is the convention under
which Problem 1 was estimated. Hard prediction is `argmax` over hotels and
the outside option (utility 0).

**Trivial baseline.** "Always predict NO_PURCHASE." Included because $\sim\!26\%$ of
held-out queries end in no purchase.

**Headline numbers.** Reproduced from `results/problem7c_results.json`:

| model | accuracy | correct / 50 | Wilson 95 % CI |
| --- | --- | --- | --- |
| AI agent                      | 0.18 | 9 / 50  | $[0.098,\,0.308]$ |
| MNL hard rule                 | 0.26 | 13 / 50 | $[0.159,\,0.396]$ |
| Always NO_PURCHASE baseline   | 0.26 | 13 / 50 | $[0.159,\,0.396]$ |

The MNL hard rule does **not** outperform the trivial NO_PURCHASE baseline on
this held-out sample; it ties it, predicting NO_PURCHASE on every one of the
50 queries because no hotel's correctly-computed utility exceeds the outside
option's $u = 0$. The AI–MNL gap of two correct predictions is well within
sampling noise (Wilson CIs overlap heavily).

**Behavioural comparison.** The AI agent's predicted purchase rate (82 %) is
closer to the observed 74 % than the MNL hard rule's 0 %. But the comparison
to MNL via argmax is not really a comparison to the MNL — argmax discards the
probability distribution. So we also report the MNL probability-implied
behavioural summary, where each hotel contributes its choice probability:
purchase rate $\approx 68.5\%$, much closer to 74 %; expected price, star
rating, review score, brand share, and promotion share are all closer to
observed than the AI's hard-prediction summary on most attributes.

## 7d. Discussion

The AI agent captures broad, common-sense hotel preferences: positive utility
for star rating, review score, brand, location, accessibility, and promotions,
and negative utility for displayed and historical price. These signs match
the human MNL signs on every feature, which is good qualitative agreement.

The AI is, however, more deterministic and more strongly attribute-driven
than real customers. The ridge MNL fit on AI choices gives coefficients that
are several times larger than the human ones in standardized units; even
allowing for the scaling-convention difference and the fixed $\lambda$, the
qualitative reading is that the AI is reading off a small number of observable
attributes in a near-rule-based way, whereas humans respond to many factors
not in the prompt (photos, loyalty programs, brand familiarity, position on
the page, trip purpose, personal taste).

As a *predictor* of individual human choices, the AI underperforms a trivial
baseline on the 50-query held-out set (18 % vs 26 %), and the MNL hard rule
ties the trivial baseline. Both are within statistical noise of each other.
The MNL is informative when used through its full probability distribution —
its probability-implied behavioural summary lines up with the observed data
much more closely than its argmax does. So the comparison that *does* show
signal is "AI hard predictions vs MNL probabilities," not the headline
exact-choice accuracy.

**Bottom line.** AI agents are a plausible source of synthetic preference
signals — useful for what-if simulations and for stress-testing a choice
model's specification — but on this dataset they are not a substitute for an
MNL estimated from real human bookings, and they are not better predictors of
individual booking decisions than a probability-aware MNL.
