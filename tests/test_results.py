import csv
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublishedResultsTest(unittest.TestCase):
    def test_search_space_counts_are_internally_consistent(self) -> None:
        summary = json.loads(
            (REPO_ROOT / "results" / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        search_space = summary["search_space"]

        self.assertEqual(search_space["strategy_pairs"], 3**2)
        self.assertEqual(search_space["layer_partitions"], 22 - 1)
        self.assertEqual(
            search_space["evaluated_configurations"],
            search_space["strategy_pairs"] * search_space["layer_partitions"],
        )

    def test_aggregate_table_covers_every_strategy_pair(self) -> None:
        with (REPO_ROOT / "results" / "mpe_error_by_configuration.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 9)
        self.assertEqual(
            {(row["stage_0"], row["stage_1"]) for row in rows},
            {
                (stage_0, stage_1)
                for stage_0 in ("TP1_DP4", "TP2_DP2", "TP4_DP1")
                for stage_1 in ("TP1_DP4", "TP2_DP2", "TP4_DP1")
            },
        )

    def test_selected_plan_is_reported_as_homogeneous(self) -> None:
        summary = json.loads(
            (REPO_ROOT / "results" / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        plan = summary["selected_plan"]

        self.assertFalse(plan["heterogeneous"])
        self.assertEqual(plan["stage_0"]["strategy"], plan["stage_1"]["strategy"])
        self.assertEqual(
            plan["stage_0"]["layer_range"][1],
            plan["stage_1"]["layer_range"][0],
        )


if __name__ == "__main__":
    unittest.main()
