import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluation_dataset import prepare_validation_dataset


class EvaluationDatasetTests(unittest.TestCase):
    def test_creates_portable_validation_manifest_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images = root / "valid" / "images"
            labels = root / "valid" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (12, 12), "white").save(images / "sample.jpg")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            manifest = root / "data.yaml"
            manifest.write_text(
                "path: .\ntrain: valid/images\nval: valid/images\nnc: 1\nnames: ['car']\n",
                encoding="utf-8",
            )

            generated = Path(prepare_validation_dataset(str(manifest), str(root / "portable")))

            self.assertTrue(generated.is_file())
            self.assertTrue((root / "portable" / "valid" / "images" / "sample.jpg").is_file())
            self.assertTrue((root / "portable" / "valid" / "labels" / "sample.txt").is_file())
            self.assertIn("val: valid/images", generated.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
