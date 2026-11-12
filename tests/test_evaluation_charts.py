import tempfile
import unittest
from pathlib import Path

from evaluator import generate_custom_evaluation_charts


class EvaluationChartTests(unittest.TestCase):
    def test_generates_curve_and_matrix_pngs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = generate_custom_evaluation_charts(
                "0.1, 0.4, 0.8",
                "8,2; 1,9",
                temporary_dir,
                "car, bus",
            )
            self.assertEqual(set(paths), {"curve", "matrix"})
            for path in paths.values():
                image_path = Path(path)
                self.assertTrue(image_path.is_file())
                self.assertGreater(image_path.stat().st_size, 0)

    def test_rejects_non_square_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                generate_custom_evaluation_charts(
                    "0.1, 0.4",
                    "8,2,1; 1,9,0",
                    temporary_dir,
                )


if __name__ == "__main__":
    unittest.main()
