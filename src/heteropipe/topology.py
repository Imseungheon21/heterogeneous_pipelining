"""Pure-Python planning for stage-specific TP and DP process groups.

The module deliberately stops at a process-group *plan*. A production runtime
would materialize the returned rank tuples with ``torch.distributed.new_group``
on every global rank in a deterministic order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import StageConfig


@dataclass(frozen=True)
class StageTopology:
    stage_id: int
    ranks: tuple[int, ...]
    leader: int
    tp_groups: tuple[tuple[int, ...], ...]
    dp_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BroadcastOperation:
    kind: str
    source: int
    members: tuple[int, ...]


def build_stage_topologies(stages: Iterable[StageConfig]) -> tuple[StageTopology, ...]:
    """Allocate contiguous global ranks and construct TP-row/DP-column groups."""

    topologies: list[StageTopology] = []
    offset = 0
    for expected_id, stage in enumerate(stages):
        if stage.stage_id != expected_id:
            raise ValueError("stages must be ordered by contiguous stage_id")

        tp_size = stage.parallelism.tensor
        dp_size = stage.parallelism.data
        ranks = tuple(range(offset, offset + stage.devices))
        grid = tuple(
            tuple(ranks[dp_index * tp_size + tp_index] for tp_index in range(tp_size))
            for dp_index in range(dp_size)
        )
        tp_groups = grid
        dp_groups = tuple(
            tuple(grid[dp_index][tp_index] for dp_index in range(dp_size))
            for tp_index in range(tp_size)
        )
        topologies.append(
            StageTopology(
                stage_id=stage.stage_id,
                ranks=ranks,
                leader=ranks[0],
                tp_groups=tp_groups,
                dp_groups=dp_groups,
            )
        )
        offset += stage.devices
    return tuple(topologies)


def hierarchical_receive_plan(topology: StageTopology) -> tuple[BroadcastOperation, ...]:
    """Plan leader receive fan-out for one pipeline-stage boundary.

    The stage leader first shares the activation with the first TP row. Each TP
    position then acts as the source for its corresponding DP column. This plan
    keeps every collective source inside the process group that it broadcasts to.
    """

    operations: list[BroadcastOperation] = []
    first_tp_row = topology.tp_groups[0]
    if len(first_tp_row) > 1:
        operations.append(
            BroadcastOperation("tp_broadcast", topology.leader, first_tp_row)
        )

    for dp_group in topology.dp_groups:
        if len(dp_group) > 1:
            operations.append(BroadcastOperation("dp_broadcast", dp_group[0], dp_group))
    return tuple(operations)
