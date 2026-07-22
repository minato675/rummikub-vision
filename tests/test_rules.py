from collections import Counter
import unittest

from rummikub.rules import Tile, cluster_melds, validate_meld, validate_turn


def tile(name: str, x: float = 0, y: float = 0) -> Tile:
    return Tile(name, x1=x, y1=y, x2=x + 40, y2=y + 70)


class MeldRulesTest(unittest.TestCase):
    def test_group(self):
        result = validate_meld([tile("red_7"), tile("blue_7"), tile("black_7")])
        self.assertTrue(result.valid)
        self.assertEqual(result.kind, "group")
        self.assertEqual(result.score, 21)

    def test_run_and_reverse_run(self):
        self.assertTrue(validate_meld([tile("blue_3"), tile("blue_4"), tile("blue_5")]).valid)
        self.assertTrue(validate_meld([tile("blue_5"), tile("blue_4"), tile("blue_3")]).valid)

    def test_joker_fills_run_gap(self):
        result = validate_meld([tile("red_6"), tile("joker"), tile("red_8")])
        self.assertTrue(result.valid)
        self.assertEqual(result.score, 21)

    def test_out_of_order_run_is_invalid(self):
        self.assertFalse(validate_meld([tile("red_6"), tile("red_8"), tile("red_7")]).valid)

    def test_duplicate_color_group_is_invalid(self):
        self.assertFalse(validate_meld([tile("red_5"), tile("red_5"), tile("blue_5")]).valid)

    def test_one_and_thirteen_do_not_wrap(self):
        self.assertFalse(validate_meld([tile("orange_12"), tile("orange_13"), tile("orange_1")]).valid)


class TurnRulesTest(unittest.TestCase):
    def test_cluster_splits_large_gap(self):
        tiles = [tile("red_1", 0), tile("blue_1", 45), tile("black_1", 90), tile("red_4", 220), tile("red_5", 265), tile("red_6", 310)]
        self.assertEqual(len(cluster_melds(tiles)), 2)

    def test_valid_initial_meld_of_thirty(self):
        current = [tile("red_10", 0), tile("blue_10", 45), tile("black_10", 90)]
        result = validate_turn([], current, initial_meld_done=False)
        self.assertTrue(result.valid, result.reasons)
        self.assertEqual(result.initial_score, 30)

    def test_initial_meld_below_thirty(self):
        current = [tile("red_7", 0), tile("blue_7", 45), tile("black_7", 90)]
        result = validate_turn([], current, initial_meld_done=False)
        self.assertFalse(result.valid)
        self.assertTrue(any("30점 미만" in reason for reason in result.reasons))

    def test_initial_joker_uses_represented_value(self):
        current = [tile("red_9", 0), tile("joker", 45), tile("red_11", 90)]
        result = validate_turn([], current, initial_meld_done=False)
        self.assertTrue(result.valid, result.reasons)
        self.assertEqual(result.initial_score, 30)
        self.assertTrue(result.warnings)

    def test_removed_table_tile_is_invalid(self):
        previous = [tile("red_10", 0), tile("blue_10", 45), tile("black_10", 90)]
        current = [tile("red_10", 0), tile("blue_10", 45), tile("orange_10", 90)]
        result = validate_turn(previous, current, initial_meld_done=True)
        self.assertFalse(result.valid)
        self.assertTrue(result.added == Counter({"orange_10": 1}))


if __name__ == "__main__":
    unittest.main()
