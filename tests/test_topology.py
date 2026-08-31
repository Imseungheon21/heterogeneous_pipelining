import unittest

from heteropipe.config import Parallelism, StageConfig
from heteropipe.topology import build_stage_topologies, hierarchical_receive_plan


class TopologyTest(unittest.TestCase):
    def test_builds_tp_rows_and_dp_columns(self) -> None:
        stages = (
            StageConfig(0, 2, Parallelism(tensor=2, data=1)),
            StageConfig(1, 4, Parallelism(tensor=2, data=2)),
        )
        first, second = build_stage_topologies(stages)

        self.assertEqual(first.ranks, (0, 1))
        self.assertEqual(second.ranks, (2, 3, 4, 5))
        self.assertEqual(second.tp_groups, ((2, 3), (4, 5)))
        self.assertEqual(second.dp_groups, ((2, 4), (3, 5)))

    def test_hierarchical_sources_are_group_members(self) -> None:
        topology = build_stage_topologies(
            (StageConfig(0, 4, Parallelism(tensor=2, data=2)),)
        )[0]
        operations = hierarchical_receive_plan(topology)

        self.assertEqual(
            [(op.kind, op.source, op.members) for op in operations],
            [
                ("tp_broadcast", 0, (0, 1)),
                ("dp_broadcast", 0, (0, 2)),
                ("dp_broadcast", 1, (1, 3)),
            ],
        )
        self.assertTrue(all(op.source in op.members for op in operations))


if __name__ == "__main__":
    unittest.main()
