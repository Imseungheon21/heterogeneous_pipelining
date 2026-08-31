"""Small, explicit performance estimators used by the public reconstruction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkovianBlockEstimator:
    """Estimate repeated-block cost from one-block and adjacent-two-block profiles.

    ``one_block_ms`` contains fixed stage work plus the first block. The marginal
    cost of another adjacent block is inferred from the two-block measurement.
    Communication and other stage-boundary costs are supplied separately because
    they depend on the selected topology rather than only on the block count.
    """

    one_block_ms: float
    two_block_ms: float

    def __post_init__(self) -> None:
        if self.one_block_ms <= 0 or self.two_block_ms <= 0:
            raise ValueError("profile measurements must be positive")
        if self.two_block_ms < self.one_block_ms:
            raise ValueError("two-block time cannot be lower than one-block time")

    @property
    def marginal_block_ms(self) -> float:
        return self.two_block_ms - self.one_block_ms

    def predict_stage_ms(
        self,
        num_blocks: int,
        *,
        communication_ms: float = 0.0,
        boundary_ms: float = 0.0,
    ) -> float:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if communication_ms < 0 or boundary_ms < 0:
            raise ValueError("additional costs cannot be negative")
        repeated_cost = self.one_block_ms + (num_blocks - 1) * self.marginal_block_ms
        return repeated_cost + communication_ms + boundary_ms


def project_total_time_ms(step_time_ms: float, steps: int) -> float:
    if step_time_ms < 0 or steps < 0:
        raise ValueError("step time and step count cannot be negative")
    return step_time_ms * steps


def relative_error_percent(actual: float, predicted: float) -> float:
    """Return absolute percentage error using measured time as the denominator."""

    if actual <= 0:
        raise ValueError("actual time must be positive")
    return abs(actual - predicted) / actual * 100.0
