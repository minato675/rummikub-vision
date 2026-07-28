import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from rummikub import dataset
from rummikub.classes import canonicalize_external_name
from rummikub.evaluation import iou, match_image
from rummikub.rules import Tile, validate_layout, validate_meld


def tile(name: str, x: float = 0, y: float = 0) -> Tile:
    return Tile(name, x1=x, y1=y, x2=x + 40, y2=y + 70)


class StudioCoreTest(unittest.TestCase):
    def test_external_class_mapping(self):
        self.assertEqual(canonicalize_external_name("Black12"), "black_12")
        self.assertEqual(canonicalize_external_name("Orange-6"), "orange_6")

    def test_group_and_run_rules(self):
        self.assertTrue(validate_meld([tile("red_7"), tile("blue_7"), tile("black_7")]).valid)
        self.assertTrue(validate_meld([tile("blue_3"), tile("blue_4"), tile("blue_5")]).valid)
        self.assertFalse(validate_meld([tile("red_3"), tile("red_5"), tile("red_4")]).valid)

    def test_layout_clustering(self):
        tiles = [tile("red_1", 0), tile("blue_1", 45), tile("black_1", 90)]
        result = validate_layout(tiles)
        self.assertTrue(result["valid"])
        self.assertEqual(result["invalid_tile_indices"], [])

    def test_layout_identifies_single_offending_tile(self):
        tiles = [tile("red_3", 0), tile("red_4", 45), tile("red_5", 90), tile("blue_9", 135)]
        result = validate_layout(tiles)
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_tile_indices"], [3])
        self.assertEqual(result["melds"][0]["offending_tile_indices"], [3])

    def test_short_invalid_meld_highlights_whole_meld(self):
        tiles = [tile("red_3", 0), tile("red_4", 45)]
        result = validate_layout(tiles)
        self.assertEqual(result["invalid_tile_indices"], [0, 1])

    def test_rotated_layout_clustering(self):
        tiles = [
            tile("black_9", 0, 120), tile("black_10", 40, 80),
            tile("black_11", 80, 40), tile("black_12", 120, 0),
            tile("red_7", 180, 120), tile("blue_7", 220, 80), tile("orange_7", 260, 40),
        ]
        result = validate_layout(tiles)
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(result["melds"]), 2)

    def test_iou_matching(self):
        truth = [{"class_id": 0, "x1": 0, "y1": 0, "x2": 100, "y2": 100}]
        predicted = [{"class_id": 0, "x1": 5, "y1": 5, "x2": 95, "y2": 95}]
        self.assertGreater(iou(truth[0], predicted[0]), 0.8)
        self.assertEqual(match_image(truth, predicted), (1, 0, 0))

    def test_save_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels, exports = root / "images", root / "labels", root / "exports"
            with patch.multiple(dataset, IMAGE_DIR=images, LABEL_DIR=labels, EXPORT_DIR=exports):
                image = np.zeros((100, 120, 3), dtype=np.uint8)
                box = {"class_id": 0, "x1": 10, "y1": 10, "x2": 50, "y2": 80}
                image_path, label_path = dataset.save_sample(image, [box])
                self.assertTrue(image_path.is_file())
                self.assertTrue(label_path.is_file())
                exported = dataset.export_roboflow_zip()
                self.assertTrue(exported.is_file())


if __name__ == "__main__":
    unittest.main()
