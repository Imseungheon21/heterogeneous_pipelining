"""Recompute aggregate MPE error from an archived experiment CSV.

The raw CSV is intentionally not committed to the public-safe repository. Pass
its local path when publication permission and data-handling rules allow it.
"""

from __future__ import annotations

import argparse
import ast
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--expect-rows", type=int)
    args = parser.parse_args()

    with args.csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise SystemExit(f"expected {args.expect_rows} rows, found {len(rows)}")

    grouped: dict[str, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
    for row in rows:
        actual = float(row["Actual_Net_ms"])
        predicted = float(row["MPE_Ideal_ms"])
        if actual <= 0:
            raise ValueError("Actual_Net_ms must be positive")
        error = abs(actual - predicted) / actual * 100.0
        split = tuple(ast.literal_eval(row["Layer_Dist"]))
        if len(split) != 2:
            raise ValueError(f"expected a two-stage layer split, got {split}")
        grouped[row["Config_Name"]].append((error, split))

    all_errors = [error for values in grouped.values() for error, _ in values]
    print(f"rows={len(rows)} configurations={len(grouped)}")
    print(f"overall_mean_error_percent={statistics.mean(all_errors):.2f}")
    for config_name in sorted(grouped):
        values = grouped[config_name]
        minimum = min(values)
        maximum = max(values)
        print(
            f"{config_name}: mean={statistics.mean(error for error, _ in values):.2f}% "
            f"min={minimum[0]:.2f}%@{minimum[1]} max={maximum[0]:.2f}%@{maximum[1]}"
        )


if __name__ == "__main__":
    main()
