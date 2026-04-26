#!/usr/bin/env python3
"""Problem 7a: build the AI booking dataset from manual prompt responses.

The actual prompting of ChatGPT is performed manually (one prompt per query,
template documented in `inputs/ai_generation_prompt_template.txt`). This
script does the *deterministic* part of 7a: load the AI's choices, assemble
a row-per-hotel dataset where the AI's pick gets `booking_ai = 1`, every
other alternative in the same query gets `0`, and `NO_PURCHASE` queries get
all-zeros.

Inputs:
- `data.csv`               (full Expedia dataset; provides choice sets and customer context)
- `inputs/ai_decisions.csv`  with columns `srch_id, ai_choice` where `ai_choice`
                             is either an alt_id (1-based, within-query) or the
                             literal string `NO_PURCHASE`.

If `inputs/ai_decisions.csv` is missing, the script can be run with
`--from-sample-rows` to recover the AI decisions from the prebuilt
`inputs/ai_booking_sample500_rows.csv` shipped with the project.

Output:
- `inputs/data_with_ai_bookings.csv`  -- the full dataset plus
   `alt_id, in_ai_sample, booking_ai, ai_choice`
- `results/problem7a_summary.json`   -- counts of buy / no_purchase / sample size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_ai_decisions_from_sample_rows(sample_rows_path: Path) -> pd.DataFrame:
    """Reverse-engineer per-query AI decisions from the row-level sample CSV.

    The shipped `ai_booking_sample500_rows.csv` already encodes the AI choice
    in the `ai_choice_sample500` column (one value repeated per query, either
    `alt_id_<n>` or `NO_PURCHASE`). We extract one decision per query.
    """
    sample = pd.read_csv(sample_rows_path)
    decisions = (
        sample.groupby("srch_id")["ai_choice_sample500"]
        .first()
        .rename("ai_choice")
        .reset_index()
    )
    decisions["ai_choice"] = decisions["ai_choice"].astype(str).str.replace(
        r"^alt_id_", "", regex=True
    )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AI-labelled booking dataset.")
    parser.add_argument(
        "--data",
        default="data.csv",
        help="Path to the full Expedia dataset (defaults to project root).",
    )
    parser.add_argument(
        "--decisions",
        default="problem7/inputs/ai_decisions.csv",
        help="CSV with columns srch_id,ai_choice (alt_id or NO_PURCHASE).",
    )
    parser.add_argument(
        "--from-sample-rows",
        default=None,
        help="Optional path to the row-level sample CSV to recover decisions from (e.g. inputs/ai_booking_sample500_rows.csv).",
    )
    parser.add_argument(
        "--output-csv",
        default="problem7/inputs/data_with_ai_bookings.csv",
        help="Where to write the merged AI-labelled dataset.",
    )
    parser.add_argument(
        "--output-json",
        default="problem7/results/problem7a_summary.json",
        help="Where to write the summary JSON.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["alt_id"] = df.groupby("srch_id").cumcount() + 1

    if args.from_sample_rows:
        decisions = load_ai_decisions_from_sample_rows(Path(args.from_sample_rows))
    else:
        decisions = pd.read_csv(args.decisions)

    decisions["ai_choice"] = decisions["ai_choice"].astype(str)
    sample_ids = set(decisions["srch_id"].unique())
    df["in_ai_sample"] = df["srch_id"].isin(sample_ids)
    df["ai_choice"] = df["srch_id"].map(decisions.set_index("srch_id")["ai_choice"])

    def booking_for_row(row: pd.Series) -> float:
        if not row["in_ai_sample"]:
            return np.nan
        choice = row["ai_choice"]
        if choice == "NO_PURCHASE":
            return 0
        try:
            return 1 if int(float(choice)) == int(row["alt_id"]) else 0
        except (TypeError, ValueError):
            return 0

    df["booking_ai"] = df.apply(booking_for_row, axis=1)

    sample_df = df[df["in_ai_sample"]].copy()
    n_buy = int(sample_df.groupby("srch_id")["booking_ai"].sum().eq(1).sum())
    n_no = int(sample_df.groupby("srch_id")["booking_ai"].sum().eq(0).sum())

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "problem": "7a",
        "data_path": str(Path(args.data).resolve()),
        "decisions_path": (
            str(Path(args.from_sample_rows).resolve())
            if args.from_sample_rows
            else str(Path(args.decisions).resolve())
        ),
        "output_csv": str(Path(args.output_csv).resolve()),
        "n_total_rows": int(len(df)),
        "n_total_queries": int(df["srch_id"].nunique()),
        "n_sample_queries": int(len(sample_ids)),
        "n_sample_rows": int(len(sample_df)),
        "n_ai_buy_queries": n_buy,
        "n_ai_no_purchase_queries": n_no,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
