from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from rummikub.annotations import normalize_box, save_yolo_sample
from rummikub.classes import canonicalize_external_name
from import_roboflow import to_detection_line


class AnnotationTest(unittest.TestCase):
    def test_external_class_mapping(self):
        self.assertEqual(canonicalize_external_name("Black10"), "black_10")
        self.assertEqual(canonicalize_external_name("orange-3"), "orange_3")

    def test_polygon_is_converted_to_detection_box(self):
        line = to_detection_line(
            ["7", "0.1", "0.2", "0.3", "0.2", "0.3", "0.6", "0.1", "0.6"],
            4,
        )
        self.assertEqual(line, "4 0.200000 0.400000 0.200000 0.400000")

    def test_normalize_box(self):
        line = normalize_box({"class_id": 3, "x1": 10, "y1": 20, "x2": 30, "y2": 60}, 100, 100)
        self.assertEqual(line, "3 0.200000 0.400000 0.200000 0.400000")

    def test_save_image_and_label(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image_path, label_path = save_yolo_sample(
                image,
                [{"class_id": 0, "x1": 10, "y1": 10, "x2": 50, "y2": 80}],
                "train",
                root,
            )
            self.assertTrue(image_path.is_file())
            self.assertTrue(label_path.is_file())
            self.assertTrue(label_path.read_text(encoding="utf-8").startswith("0 "))


if __name__ == "__main__":
    unittest.main()
