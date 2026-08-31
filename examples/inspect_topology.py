"""Print the process-group and receive-fan-out plan for a JSON configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heteropipe.config import ExperimentConfig  # noqa: E402
from heteropipe.topology import build_stage_topologies, hierarchical_receive_plan  # noqa: E402


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        "heterogeneous_topology.json"
    )
    config = ExperimentConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

    print(f"{config.model}: {config.total_layers} layers on {config.total_devices} devices")
    for topology in build_stage_topologies(config.stages):
        print(f"\nStage {topology.stage_id}: ranks={topology.ranks}, leader={topology.leader}")
        print(f"  TP groups: {topology.tp_groups}")
        print(f"  DP groups: {topology.dp_groups}")
        for operation in hierarchical_receive_plan(topology):
            print(f"  {operation.kind}: src={operation.source} -> {operation.members}")


if __name__ == "__main__":
    main()
