import unittest

from heteropipe.search import CostEntry, search_pipeline_plan


class SearchTest(unittest.TestCase):
    def test_selects_balanced_pipeline(self) -> None:
        entries = [
            CostEntry(0, 2, "TP2", 2, 6.0),
            CostEntry(2, 4, "DP2", 2, 6.0),
            CostEntry(0, 1, "DP2", 2, 4.0),
            CostEntry(1, 4, "TP2", 2, 11.0),
        ]
        result = search_pipeline_plan(
            entries, total_layers=4, total_devices=4, microbatches=4
        )

        self.assertEqual([stage.strategy for stage in result.stages], ["TP2", "DP2"])
        self.assertEqual(result.latency_ms, 30.0)  # 6 + 6 + 3 * max(6, 6)

    def test_rejects_infeasible_search_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "no feasible plan"):
            search_pipeline_plan(
                [CostEntry(0, 1, "TP2", 2, 1.0)],
                total_layers=2,
                total_devices=4,
                microbatches=1,
            )


if __name__ == "__main__":
    unittest.main()
