# Task Plan

Use this file for non-trivial tasks before implementation starts.

## Active Task

Task: Reformulate Problem 5 unknown-type assortment as an explicit integer optimization model and solve it with Gurobi when available.

### Plan

- [x] Define the target behavior and constraints
- [x] Inspect the current Problem 5 implementation and local Gurobi availability
- [x] Implement a Gurobi-compatible integer optimization formulation for unknown-type assortment
- [x] Wire the Problem 5 script to use Gurobi when possible and keep an exact fallback
- [x] Verify the new backend and update the run instructions

### Review

- Status: Complete
- Verification: Added an explicit MILP formulation for the unknown-type assortment, solved it successfully with `gurobi_cl`, and confirmed it reproduces the previous exact solutions on all four small datasets.
- Notes: `gurobipy` is not available for the current Python installation, so the script uses the installed Gurobi optimizer via `gurobi_cl` when `--use-gurobi` is passed, while preserving the exact branch-and-bound fallback.
