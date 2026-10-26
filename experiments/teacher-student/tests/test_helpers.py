import os
import unittest
from unittest.mock import patch

import numpy as np

from rummikub_inference import DEFAULT_MODEL_ID, draw_predictions, load_settings
from rummikub.visuals import class_color, draw_invalid_highlights


class HelperTest(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(settings.server_url, "http://127.0.0.1:9001")

    def test_draw_prediction_preserves_shape(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        result = {
            "predictions": [
                {"x": 60, "y": 50, "width": 30, "height": 40, "class": "Black1", "confidence": 0.9}
            ]
        }
        rendered = draw_predictions(image, result)
        self.assertEqual(rendered.shape, image.shape)
        self.assertTrue(np.any(rendered != image))

    def test_label_colors_match_tile_families(self):
        self.assertEqual(class_color("black_10"), (20, 20, 20))
        self.assertEqual(class_color("red_10"), (235, 45, 55))
        self.assertEqual(class_color("orange_10"), (255, 145, 0))
        self.assertEqual(class_color("blue_10"), (25, 105, 235))
        self.assertEqual(class_color("red_10", "BGR"), (55, 45, 235))
        self.assertEqual(class_color("Black10"), (20, 20, 20))
        self.assertEqual(class_color("Orange7", "BGR"), (0, 145, 255))

    def test_invalid_highlight_changes_box_region(self):
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        boxes = [{"x1": 40, "y1": 30, "x2": 120, "y2": 140}]
        rendered = draw_invalid_highlights(image, boxes, {0: 2})
        self.assertEqual(rendered.shape, image.shape)
        self.assertTrue(np.any(rendered[30:141, 40:121] != 0))


if __name__ == "__main__":
    unittest.main()
