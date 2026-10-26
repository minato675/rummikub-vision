from collections import Counter
from pathlib import Path

from rummikub.cross_validation import SampleGroup, assign_groups, source_group


def test_source_group_keeps_augmented_and_sequence_frames_together():
    assert source_group("rf1_train_board.rf.abc") == "board"
    assert source_group("game01__frame001") == "game01"


def test_five_folds_are_complete_and_disjoint():
    groups = [SampleGroup(str(i), [Path(f"{i}.jpg")], Counter({i % 3: 1})) for i in range(25)]
    buckets = assign_groups(groups, folds=5, seed=42)
    keys = [g.key for bucket in buckets for g in bucket]
    assert len(buckets) == 5
    assert sorted(keys, key=int) == [str(i) for i in range(25)]
    assert all(3 <= len(bucket) <= 7 for bucket in buckets)
