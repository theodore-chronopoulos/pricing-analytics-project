# Pricing Analytics Project (ORIE 5132)

This repository contains the code, data, and write-up for the
ORIE 5132 course project on choice modeling, assortment optimization,
and pricing.

## Repository Layout

```text
.
├── README.md                       # this file (run instructions, repo overview)
├── report.tex                      # main report (LaTeX source)
├── Project.pdf                     # original assignment PDF
├── data.csv                        # full Expedia dataset (153,009 rows / 8,354 queries)
├── data1.csv … data4.csv           # small datasets for assortment / pricing
├── scripts/                        # all Python source code
│   ├── mnl_utils.py                #   shared MNL / assortment / pricing utilities
│   ├── problem1_mnl.py             #   Problem 1: MNL estimation
│   ├── problem2_assortment.py      #   Problem 2: assortment optimization
│   ├── problem3_pricing.py         #   Problem 3: pricing optimization
│   ├── problem4_mixture.py         #   Problem 4: early/late MMNL
│   ├── problem5_assortment.py      #   Problem 5: type-aware/unaware assortment
│   ├── problem6_mmnl.py            #   Problem 6: family/non-family MMNL + Problem 5 repeat
│   └── problem7_ai_agents.py       #   Problem 7: AI agents as customers (7a + 7b + 7c)
├── problem7/
│   ├── explanations.md             #   per-section narrative for 7a/7b/7c/7d
│   └── inputs/                     #   manual-prompt artefacts (AI sample, predictions, prompt template)
└── results/                        # JSON outputs produced by the scripts
    ├── problem1_results.json
    ├── problem2_results.json
    ├── problem3_results.json
    ├── problem4_results.json
    ├── problem5_results.json           # branch-and-bound backend
    ├── problem5_results_gurobi.json    # Gurobi MILP backend (matches BnB)
    ├── problem6_results.json
    └── problem7_results.json
```

## What Is Implemented

| Problem | Topic | Script | Output |
| --- | --- | --- | --- |
| 1 | MNL estimation on `data.csv` | `scripts/problem1_mnl.py` | `results/problem1_results.json` |
| 2 | Assortment optimization on `data1..4.csv` | `scripts/problem2_assortment.py` | `results/problem2_results.json` |
| 3 | Pricing optimization on `data1..4.csv` | `scripts/problem3_pricing.py` | `results/problem3_results.json` |
| 4 | Early-vs-late mixture of MNL | `scripts/problem4_mixture.py` | `results/problem4_results.json` |
| 5 | Type-aware and type-unaware assortment (MILP) | `scripts/problem5_assortment.py` | `results/problem5_results*.json` |
| 6 | Family/non-family MMNL + Problem-5 repeat | `scripts/problem6_mmnl.py` | `results/problem6_results.json` |
| 7 | AI Agents as Customers (7a + 7b + 7c) | `scripts/problem7_ai_agents.py` | `results/problem7_results.json` |

## Setup

Python 3 is required. The Problem 1–6 scripts use only the Python standard
library, so a plain virtualenv is enough:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No `pip install` is required for Problems 1–6.

For Problem 5 with the explicit MILP backend, `gurobi_cl` (the Gurobi
command-line optimizer) must be on `PATH`. The branch-and-bound fallback
in the same script does not need Gurobi.

Problem 7 needs `pandas`, `numpy`, `scipy`, and `statsmodels`:

```bash
pip install pandas numpy scipy statsmodels
```

## How to Run the Code

All commands are run from the repository root.

### One-shot full pipeline (no Gurobi needed)

```bash
python scripts/problem1_mnl.py        --output-json results/problem1_results.json
python scripts/problem2_assortment.py --problem1-json results/problem1_results.json --output-json results/problem2_results.json
python scripts/problem3_pricing.py    --problem1-json results/problem1_results.json --output-json results/problem3_results.json
python scripts/problem4_mixture.py    --output-json results/problem4_results.json
python scripts/problem5_assortment.py --problem4-json results/problem4_results.json --output-json results/problem5_results.json
python scripts/problem6_mmnl.py --output-json results/problem6_results.json
python scripts/problem7_ai_agents.py --output-json results/problem7_results.json
```

### Problem 5 with the Gurobi MILP backend

```bash
python scripts/problem5_assortment.py \
    --problem4-json results/problem4_results.json \
    --use-gurobi \
    --output-json results/problem5_results_gurobi.json
```

This solves the explicit integer program for the unknown-type assortment
$S$ requested by Problem 5. The branch-and-bound and MILP backends
return identical $S$ on all four datasets (floating-point difference
$<3\times10^{-14}$).

### Per-problem commands

```bash
# Problem 1
python scripts/problem1_mnl.py --output-json results/problem1_results.json

# Problem 2
python scripts/problem2_assortment.py \
    --problem1-json results/problem1_results.json \
    --output-json results/problem2_results.json

# Problem 3
python scripts/problem3_pricing.py \
    --problem1-json results/problem1_results.json \
    --output-json results/problem3_results.json

# Problem 4
python scripts/problem4_mixture.py --output-json results/problem4_results.json

# Problem 5 (branch-and-bound)
python scripts/problem5_assortment.py \
    --problem4-json results/problem4_results.json \
    --output-json results/problem5_results.json

# Problem 5 (Gurobi MILP)
python scripts/problem5_assortment.py \
    --problem4-json results/problem4_results.json \
    --use-gurobi \
    --output-json results/problem5_results_gurobi.json

# Problem 6
python scripts/problem6_mmnl.py --output-json results/problem6_results.json

# Problem 7 (runs 7a, 7b, 7c sequentially; reads manual artefacts from problem7/inputs/)
python scripts/problem7_ai_agents.py --output-json results/problem7_results.json
```

## Reproducibility

- Problems 1–6 default to seed `5132`. Pass `--seed <int>` to override.
- Problem 7 defaults to seed `42` (matching the seed under which the
  manual AI samples and held-out predictions were collected).
- The Problem 1–6 code paths are deterministic; the seed is recorded
  in every JSON output for traceability. Problem 7 is deterministic
  given the manual AI artefacts in `problem7/inputs/` (the LLM step
  itself is the only non-deterministic component, and it is performed
  outside the scripts).

## Notes on the Reported Numbers

- **Problem 3** returns one common optimal price per dataset. This is a
  consequence of the Problem 1 utility specification using a single
  shared linear price coefficient: the first-order condition becomes
  $p_j^* = R^* + 1/|\beta_{\text{price}}|$ for every $j$, so all
  optimal prices must be identical (see Section 3 of `report.tex`).
- **Problem 5** solved with `--use-gurobi` is the explicit integer
  programming formulation requested by the assignment. Without
  `--use-gurobi`, the same problem is solved exactly via branch-and-bound;
  the two backends agree on every dataset.
- **Problem 6** uses the family-vs-non-family segmentation
  (`srch_children_count > 0`) as the alternative customer-type definition
  required by the assignment.

## Problem 7

Problem 7 (AI Agents as Customers) is implemented in
`scripts/problem7_ai_agents.py`, which runs the three sub-parts in
sequence:

- **7a** — converts the manually-collected AI choices in
  `problem7/inputs/ai_booking_sample500_rows.csv` into a row-per-hotel
  dataset with a `booking_ai` column,
- **7b** — fits a ridge-regularized MNL ($\lambda = 1$) on that 500-query
  AI sample and prints the coefficients next to the Problem 1
  normalized coefficients,
- **7c** — uses `np.random.default_rng(42)` to draw 10 in-context
  examples and 50 held-out queries, builds the held-out prompt
  (saved to `problem7/inputs/ai_heldout_prompt.txt`), loads the
  manually-collected AI predictions in
  `problem7/inputs/ai_heldout_predictions.csv`, and reports
  exact-choice accuracy with Wilson 95 % CIs against the MNL hard rule
  and an always-`NO_PURCHASE` baseline, plus a behavioural and
  probability-implied summary.

The 7d discussion text lives in `problem7/explanations.md`. The full
LLM prompting step is performed manually in ChatGPT (GPT-5.5 Thinking);
the prompt template is documented in
`problem7/inputs/ai_generation_prompt_template.txt`.

## Building the Report

The report is in plain LaTeX (no exotic packages):

```bash
pdflatex report.tex
pdflatex report.tex   # second pass for cross-references
```
