import unittest

from heteropipe.config import ExperimentConfig


class ExperimentConfigTest(unittest.TestCase):
    def test_validates_heterogeneous_configuration(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "model": "test-model",
                "total_layers": 6,
                "total_devices": 6,
                "microbatches": 4,
                "stages": [
                    {
                        "stage_id": 0,
                        "layers": 2,
                        "parallelism": {"tensor": 2, "data": 1},
                    },
                    {
                        "stage_id": 1,
                        "layers": 4,
                        "parallelism": {"tensor": 2, "data": 2},
                    },
                ],
            }
        )
        self.assertEqual(config.stages[1].devices, 4)

    def test_rejects_incomplete_layer_assignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer counts"):
            ExperimentConfig.from_dict(
                {
                    "model": "test-model",
                    "total_layers": 7,
                    "total_devices": 2,
                    "microbatches": 1,
                    "stages": [
                        {
                            "stage_id": 0,
                            "layers": 6,
                            "parallelism": {"tensor": 2, "data": 1},
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
