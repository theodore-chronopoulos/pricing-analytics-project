# Pricing Analytics Project

This repository contains code for Problems 1-5 of the ORIE 5132 pricing analytics project.

## What Is Implemented

- Problem 1: MNL estimation on `data.csv`
- Problem 2: MNL assortment optimization on `data1.csv` to `data4.csv`
- Problem 3: MNL pricing optimization on `data1.csv` to `data4.csv`
- Problem 4: early-vs-late mixture of MNL models
- Problem 5: type-aware and type-unaware assortment optimization

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

### Current Scripts

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

Example:

```bash
python scripts/problem1_mnl.py --seed 5132 --output-json results/problem1_results.json
```

## Output Location

All result files should be written to the project `results/` folder:

- [results](results)

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

## Notes on Interpretation

- Problem 3 returns one common optimal price per dataset because the Problem 1 utility specification uses one shared linear price coefficient.
- Problem 5 solved with `--use-gurobi` is the explicit integer optimization version requested by the assignment.
- Problem 5 without `--use-gurobi` still solves the same unknown-type assortment problem exactly, but through an internal branch-and-bound algorithm.
