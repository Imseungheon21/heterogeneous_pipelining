"""Validated configuration objects for a disjoint-stage pipeline topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Parallelism:
    tensor: int
    data: int

    def __post_init__(self) -> None:
        if self.tensor < 1 or self.data < 1:
            raise ValueError("tensor and data parallel sizes must be positive")

    @property
    def devices(self) -> int:
        return self.tensor * self.data


@dataclass(frozen=True)
class StageConfig:
    stage_id: int
    layers: int
    parallelism: Parallelism

    def __post_init__(self) -> None:
        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if self.layers < 1:
            raise ValueError("each pipeline stage must own at least one layer")

    @property
    def devices(self) -> int:
        return self.parallelism.devices


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    total_layers: int
    total_devices: int
    microbatches: int
    stages: tuple[StageConfig, ...]

    def __post_init__(self) -> None:
        if self.total_layers < 1 or self.total_devices < 1 or self.microbatches < 1:
            raise ValueError("layers, devices, and microbatches must be positive")
        if not self.stages:
            raise ValueError("at least one stage is required")

        stage_ids = [stage.stage_id for stage in self.stages]
        if stage_ids != list(range(len(self.stages))):
            raise ValueError("stage IDs must be ordered, contiguous, and start at zero")

        assigned_layers = sum(stage.layers for stage in self.stages)
        if assigned_layers != self.total_layers:
            raise ValueError(
                f"stage layer counts sum to {assigned_layers}, expected {self.total_layers}"
            )

        assigned_devices = sum(stage.devices for stage in self.stages)
        if assigned_devices != self.total_devices:
            raise ValueError(
                f"stage device counts sum to {assigned_devices}, expected {self.total_devices}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        stages = tuple(
            StageConfig(
                stage_id=int(stage["stage_id"]),
                layers=int(stage["layers"]),
                parallelism=Parallelism(
                    tensor=int(stage["parallelism"]["tensor"]),
                    data=int(stage["parallelism"]["data"]),
                ),
            )
            for stage in value["stages"]
        )
        return cls(
            model=str(value["model"]),
            total_layers=int(value["total_layers"]),
            total_devices=int(value["total_devices"]),
            microbatches=int(value["microbatches"]),
            stages=stages,
        )
