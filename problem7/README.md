# Problem 7: AI Agents as Customers

This folder contains the runnable solution to Problem 7 of the ORIE 5132
project, packaged so it integrates with the Problems 1–6 pipeline at the
project root. The narrative explanations are in
[explanations.md](explanations.md); this file is the run guide.

## Folder layout

```text
problem7/
├── README.md                               this file
├── explanations.md                         per-section narrative for 7a, 7b, 7c, 7d
├── inputs/
│   ├── ai_booking_sample500_rows.csv       row-level AI sample produced from the manual prompting
│   ├── ai_heldout_predictions.csv          AI's held-out predictions (manual ChatGPT step)
│   ├── ai_generation_prompt_template.txt   prompt template used in 7a
│   └── ai_heldout_prompt.txt               held-out prompt produced by 7c (regenerated on each run)
├── scripts/
│   ├── problem7_common.py                  shared MNL utility (binary-aware scaling)
│   ├── problem7a_build_ai_sample.py        7a: build the AI booking dataset
│   ├── problem7b_mnl_on_ai.py              7b: ridge-regularized MNL on the AI sample
│   └── problem7c_predictive_eval.py        7c: held-out AI vs MNL comparison
└── results/
    ├── problem7a_summary.json              counts written by 7a
    ├── problem7b_results.json              ridge MNL fit written by 7b
    └── problem7c_results.json              accuracy + behavioural tables written by 7c
```

## Dependencies

Unlike Problems 1–6 (standard library only), Problem 7 uses third-party
packages because the AI-labelled dataset is large and we use SciPy's
L-BFGS-B for the ridge MNL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels
```

The AI prompting itself is performed manually in ChatGPT's web interface
(GPT-5.5 Thinking). The scripts cover everything that happens before and
after the manual step.

## How to run

All commands assume the project root as the working directory.

### Problem 7a — build the AI booking dataset

```bash
python problem7/scripts/problem7a_build_ai_sample.py \
    --from-sample-rows problem7/inputs/ai_booking_sample500_rows.csv
```

This recovers the per-query AI decisions from the row-level sample CSV that
was produced from the manual prompting step, joins them with `data.csv`,
and writes:

- `problem7/inputs/data_with_ai_bookings.csv` (full dataset + AI booking column)
- `problem7/results/problem7a_summary.json` (counts: queries, rows, buy/no-purchase)

If you regenerate the AI labels in the future, you can instead pass a
two-column `ai_decisions.csv` (`srch_id, ai_choice`) via `--decisions`.

### Problem 7b — ridge MNL on the AI sample

```bash
python problem7/scripts/problem7b_mnl_on_ai.py
```

Writes `problem7/results/problem7b_results.json` with the regularized
coefficients, fit summary, and an interpretation note. Default
`--lambda 1.0` is the stabilization choice documented in the explanations;
override at the command line if you want to test sensitivity.

### Problem 7c — held-out predictive comparison

```bash
python problem7/scripts/problem7c_predictive_eval.py
```

This script:

1. Reproduces the 10-context / 50-held-out split with `np.random.default_rng(42)`.
2. Writes the held-out prompt to `problem7/inputs/ai_heldout_prompt.txt`
   (so it can be sent to ChatGPT manually if you want to refresh the predictions).
3. Loads `problem7/inputs/ai_heldout_predictions.csv` (the manual predictions)
   and computes:
    - exact-choice accuracy + Wilson 95 % CIs for AI, MNL, and an
      always-NO_PURCHASE baseline,
    - hard-prediction behavioural summaries (purchase rate, avg price/star/review/...),
    - the MNL probability-implied behavioural summary.
4. Saves everything to `problem7/results/problem7c_results.json`.

### Reproducing the discussion (7d)

There is no script for 7d. The discussion text is in `explanations.md` under
"7d. Discussion" and references the numbers produced by 7b and 7c.

## Notes on integration with the rest of the project

- The MNL convention used inside `problem7_common.py` matches Problem 1
  (`scripts/mnl_utils.py`): continuous features z-scored, binary features
  (`prop_brand_bool`, `promotion_flag`) at raw 0/1.
- `HUMAN_BETA_FROM_PROBLEM1` in `problem7_common.py` is the
  `normalized_coefficients` block of `results/problem1_results.json` —
  hard-coded so this folder is self-contained, but you can equivalently
  replace it with `json.load("results/problem1_results.json")` if you
  prefer a single source of truth.
- All three scripts default to writing under `problem7/results/`, so they
  do not collide with the Problems 1–6 outputs in the project's existing
  `results/` folder. If you want them in the shared `results/` folder, pass
  e.g. `--output-json results/problem7c_results.json`.
- The seed for the held-out split is `42` (matching the held-out predictions
  that were collected manually). The seed for the AI sampling step in 7a is
  also `42`, fixed inside the manual prompting step rather than in the
  scripts.
