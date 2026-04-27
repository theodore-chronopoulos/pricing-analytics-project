# Pricing Analytics Project

This repository contains code, results, report materials, and Problem 7 AI-agent materials for the ORIE 5132 pricing analytics project.

## Repository Structure

```text
pricing-analytics-project/
├── README.md
├── Project.pdf
├── report.tex
├── data.csv
├── data1.csv
├── data2.csv
├── data3.csv
├── data4.csv
├── scripts/
├── results/
├── tasks/
└── problem7/
    ├── 7a/
    ├── 7b/
    ├── 7c/
    │   └── InputOutput/
    └── 7d/
```

Problems 1-6 are organized through the shared `scripts/` and `results/` folders because they use the main modeling pipeline. Problem 7 is organized separately in `problem7/` because it includes AI prompt files, generated AI choice datasets, held-out prediction files, and separate notebooks for Parts 7a-7d.

## What Is Implemented

- Problem 1: MNL estimation on `data.csv`
- Problem 2: MNL assortment optimization on `data1.csv` to `data4.csv`
- Problem 3: MNL pricing optimization on `data1.csv` to `data4.csv`
- Problem 4: early-vs-late mixture of MNL models
- Problem 5: type-aware and type-unaware assortment optimization
- Problem 6: additional analysis and outputs in `scripts/` and `results/`
- Problem 7: AI agents as customers, organized in `problem7/`

## Problem 7 Organization

Problem 7 is separated from the main script pipeline because it uses AI-generated choices and prompt/response artifacts in addition to notebooks.

Expected contents:

```text
problem7/
├── 7a/
│   ├── P7a.ipynb
│   ├── ai_generation_prompt_template_7a_corrected.txt
│   ├── data_with_ai_bookings_sample500.csv
│   ├── ai_booking_sample500_rows.csv
│   └── ai_booking_sample500_decisions.csv
├── 7b/
│   ├── P7b.ipynb
│   └── mnl_ai_sample500_regularized_results.csv
├── 7c/
│   ├── P7c.ipynb
│   └── InputOutput/
│       ├── ai_heldout_prompt.txt
│       ├── ai_heldout_predictions.csv
│       └── any supporting held-out evaluation files
└── 7d/
    └── P7d.ipynb
```

For 7a/7b, the AI-generated booking-column exercise uses a random sample of 500 search queries. For 7c, the held-out predictive evaluation uses 10 context/example queries and 50 held-out queries. These sample sizes are different because 7a/7b creates an AI-imputed dataset for MNL re-estimation, while 7c evaluates direct AI predictions on held-out real outcomes.

## Solver Choice for Problem 5

There are two ways to solve the unknown-type assortment `S` in Problem 5:

1. Exact branch-and-bound fallback
   - Implemented directly in Python
   - Deterministic
   - Does not require Gurobi
   - Good default if you only want the project results

2. Gurobi MILP backend
   - Uses an explicit integer optimization formulation
   - Solved through the installed `gurobi_cl` command-line optimizer
   - Triggered with `--use-gurobi`
   - This is the closest implementation to the assignment language that says to solve an integer program

Important environment note:

- The current codebase does **not** use `gurobipy` directly because it is not available for the active Python installation in this workspace.
- Problem 5 uses `gurobi_cl` instead, which is installed locally and works.

## Dependencies

### Required

- Python 3
- Standard library only for Problems 1-5 scripts

### Optional

- Gurobi Optimizer CLI (`gurobi_cl`) for the MILP version of Problem 5
- Jupyter Notebook for the Problem 7 notebooks

## Current Scripts

- [scripts/mnl_utils.py](scripts/mnl_utils.py)
- [scripts/problem1_mnl.py](scripts/problem1_mnl.py)
- [scripts/problem2_assortment.py](scripts/problem2_assortment.py)
- [scripts/problem3_pricing.py](scripts/problem3_pricing.py)
- [scripts/problem4_mixture.py](scripts/problem4_mixture.py)
- [scripts/problem5_assortment.py](scripts/problem5_assortment.py)

## Setup

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

No pip installs are required for Problems 1-5.

## Reproducibility

- All scripts default to seed `5132`
- The current code paths are deterministic
- Each script accepts `--seed` explicitly
- Problem 7 sampling uses random seed `42` for the AI-generated samples described in the notebooks

Example:

```bash
python scripts/problem1_mnl.py --seed 5132 --output-json results/problem1_results.json
```

## Output Location

Most result files for Problems 1-6 are written to the project `results/` folder:

- [results](results)

Problem 7 outputs are stored in `problem7/` with their corresponding notebooks and prompt files.

## Exact Commands

Run these from the project root.

### Problem 1

```bash
python scripts/problem1_mnl.py --output-json results/problem1_results.json
```

### Problem 2

```bash
python scripts/problem2_assortment.py --problem1-json results/problem1_results.json --output-json results/problem2_results.json
```

### Problem 3

```bash
python scripts/problem3_pricing.py --problem1-json results/problem1_results.json --output-json results/problem3_results.json
```

### Problem 4

```bash
python scripts/problem4_mixture.py --output-json results/problem4_results.json
```

### Problem 5 with Exact Python Fallback

```bash
python scripts/problem5_assortment.py --problem4-json results/problem4_results.json --output-json results/problem5_results.json
```

### Problem 5 with Gurobi MILP

```bash
python scripts/problem5_assortment.py --problem4-json results/problem4_results.json --use-gurobi --output-json results/problem5_results_gurobi.json
```

## Full Run Sequence

### Full Run with Python Fallback for Problem 5

```bash
python scripts/problem1_mnl.py --output-json results/problem1_results.json
python scripts/problem2_assortment.py --problem1-json results/problem1_results.json --output-json results/problem2_results.json
python scripts/problem3_pricing.py --problem1-json results/problem1_results.json --output-json results/problem3_results.json
python scripts/problem4_mixture.py --output-json results/problem4_results.json
python scripts/problem5_assortment.py --problem4-json results/problem4_results.json --output-json results/problem5_results.json
```

### Full Run with Gurobi MILP for Problem 5

```bash
python scripts/problem1_mnl.py --output-json results/problem1_results.json
python scripts/problem2_assortment.py --problem1-json results/problem1_results.json --output-json results/problem2_results.json
python scripts/problem3_pricing.py --problem1-json results/problem1_results.json --output-json results/problem3_results.json
python scripts/problem4_mixture.py --output-json results/problem4_results.json
python scripts/problem5_assortment.py --problem4-json results/problem4_results.json --use-gurobi --output-json results/problem5_results_gurobi.json
```

## Existing Result Files

Current generated outputs include:

- `results/problem1_results.json`
- `results/problem2_results.json`
- `results/problem3_results.json`
- `results/problem4_results.json`
- `results/problem5_results.json`
- `results/problem5_results_gurobi.json`

Problem 7 generated files are stored in `problem7/`.

## Notes on Interpretation

- Problem 3 returns one common optimal price per dataset because the Problem 1 utility specification uses one shared linear price coefficient.
- Problem 5 solved with `--use-gurobi` is the explicit integer optimization version requested by the assignment.
- Problem 5 without `--use-gurobi` still solves the same unknown-type assortment problem exactly, but through an internal branch-and-bound algorithm.
- Problem 7 coefficient comparisons should be interpreted qualitatively because the AI-MNL estimates are ridge-regularized and are not directly comparable in magnitude to the human normalized coefficients.
