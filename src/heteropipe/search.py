"""Dynamic-programming search over layer partitions and stage strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CostEntry:
    start_layer: int
    end_layer: int
    strategy: str
    devices: int
    cost_ms: float

    def __post_init__(self) -> None:
        if self.start_layer < 0 or self.end_layer <= self.start_layer:
            raise ValueError("layer ranges use non-empty [start, end) intervals")
        if self.devices < 1 or self.cost_ms <= 0:
            raise ValueError("devices and cost must be positive")


@dataclass(frozen=True)
class SearchResult:
    latency_ms: float
    stages: tuple[CostEntry, ...]


def _group_by_end(entries: Iterable[CostEntry]) -> Mapping[int, tuple[CostEntry, ...]]:
    grouped: dict[int, list[CostEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.end_layer, []).append(entry)
    return {end: tuple(values) for end, values in grouped.items()}


def search_pipeline_plan(
    entries: Iterable[CostEntry],
    *,
    total_layers: int,
    total_devices: int,
    microbatches: int,
) -> SearchResult:
    """Find the minimum ``sum(stage) + (B-1) * max(stage)`` pipeline plan.

    For each candidate bottleneck threshold, the inner dynamic program minimizes
    fill/drain cost while assigning every layer and every device exactly once.
    This is the same decomposition used by pipeline configuration searches where
    steady-state throughput is governed by the slowest stage.
    """

    if total_layers < 1 or total_devices < 1 or microbatches < 1:
        raise ValueError("search dimensions must be positive")

    candidates = tuple(entries)
    if not candidates:
        raise ValueError("at least one cost entry is required")
    by_end = _group_by_end(candidates)
    thresholds = sorted({entry.cost_ms for entry in candidates})

    best_result: SearchResult | None = None
    for threshold in thresholds:
        # state -> (sum of stage costs, chosen stages)
        dp: dict[tuple[int, int], tuple[float, tuple[CostEntry, ...]]] = {
            (0, 0): (0.0, ())
        }
        for end_layer in range(1, total_layers + 1):
            for entry in by_end.get(end_layer, ()):
                if entry.cost_ms > threshold:
                    continue
                for used_devices in range(entry.devices, total_devices + 1):
                    previous = dp.get((entry.start_layer, used_devices - entry.devices))
                    if previous is None:
                        continue
                    new_sum = previous[0] + entry.cost_ms
                    state = (end_layer, used_devices)
                    if new_sum < dp.get(state, (inf, ()))[0]:
                        dp[state] = (new_sum, previous[1] + (entry,))

        terminal = dp.get((total_layers, total_devices))
        if terminal is None:
            continue
        stage_sum, plan = terminal
        bottleneck = max(stage.cost_ms for stage in plan)
        latency = stage_sum + (microbatches - 1) * bottleneck
        result = SearchResult(latency_ms=latency, stages=plan)
        if best_result is None or result.latency_ms < best_result.latency_ms:
            best_result = result

    if best_result is None:
        raise ValueError("no feasible plan covers all layers and devices")
    return best_result
