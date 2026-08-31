"""Run the dynamic-programming search on a small synthetic cost table."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heteropipe.search import CostEntry, search_pipeline_plan  # noqa: E402


def main() -> None:
    entries = [
        CostEntry(0, 2, "TP2", 2, 6.0),
        CostEntry(2, 4, "DP2", 2, 6.0),
        CostEntry(0, 1, "DP2", 2, 4.0),
        CostEntry(1, 4, "TP2", 2, 11.0),
    ]
    result = search_pipeline_plan(
        entries, total_layers=4, total_devices=4, microbatches=4
    )
    print(f"latency_ms={result.latency_ms:.2f}")
    for index, stage in enumerate(result.stages):
        print(
            f"stage={index} layers=[{stage.start_layer},{stage.end_layer}) "
            f"strategy={stage.strategy} devices={stage.devices} cost_ms={stage.cost_ms:.2f}"
        )


if __name__ == "__main__":
    main()
