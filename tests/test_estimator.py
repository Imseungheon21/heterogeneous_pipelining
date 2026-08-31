import unittest

from heteropipe.estimator import (
    MarkovianBlockEstimator,
    project_total_time_ms,
    relative_error_percent,
)


class EstimatorTest(unittest.TestCase):
    def test_infers_marginal_cost_from_adjacent_blocks(self) -> None:
        estimator = MarkovianBlockEstimator(one_block_ms=12.0, two_block_ms=21.0)
        self.assertEqual(estimator.marginal_block_ms, 9.0)
        self.assertEqual(estimator.predict_stage_ms(4), 39.0)
        self.assertEqual(
            estimator.predict_stage_ms(4, communication_ms=3.0, boundary_ms=2.0),
            44.0,
        )

    def test_projects_total_and_uses_actual_denominator(self) -> None:
        self.assertEqual(project_total_time_ms(25.0, 8), 200.0)
        self.assertAlmostEqual(relative_error_percent(110.0, 100.0), 1000 / 110)


if __name__ == "__main__":
    unittest.main()
