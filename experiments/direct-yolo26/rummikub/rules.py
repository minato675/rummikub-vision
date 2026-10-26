"""Deterministic Rummikub meld and turn validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable

from .classes import parse_class_name


@dataclass(frozen=True)
class Tile:
    class_name: str
    confidence: float = 1.0
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def parts(self) -> tuple[str | None, int | None, bool]:
        return parse_class_name(self.class_name)


@dataclass
class MeldValidation:
    valid: bool
    kind: str | None = None
    score: int = 0
    reason: str = ""


@dataclass
class TurnValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    melds: list[list[Tile]] = field(default_factory=list)
    added: Counter[str] = field(default_factory=Counter)
    initial_score: int = 0


def _valid_run_interval(tiles: list[Tile]) -> tuple[int, int] | None:
    total = len(tiles)
    natural_numbers = [int(tile.parts[1]) for tile in tiles if not tile.parts[2]]
    if len(set(natural_numbers)) != len(natural_numbers):
        return None
    for start in range(1, 15 - total):
        ascending = list(range(start, start + total))
        for expected in (ascending, list(reversed(ascending))):
            if all(tile.parts[2] or tile.parts[1] == value for tile, value in zip(tiles, expected)):
                return start, start + total - 1
    return None


def validate_meld(tiles: Iterable[Tile]) -> MeldValidation:
    tiles = list(tiles)
    if len(tiles) < 3:
        return MeldValidation(False, reason="세트는 타일 3개 이상이어야 합니다.")
    numbered = [tile for tile in tiles if not tile.parts[2]]
    jokers = len(tiles) - len(numbered)
    colors = [tile.parts[0] for tile in numbered]
    numbers = [int(tile.parts[1]) for tile in numbered]

    if len(tiles) <= 4 and numbered:
        same_number = len(set(numbers)) == 1
        unique_colors = len(set(colors)) == len(colors)
        if same_number and unique_colors and len(numbered) + jokers <= 4:
            number = numbers[0]
            return MeldValidation(True, "group", number * len(tiles), "유효한 그룹")

    if numbered and len(set(colors)) == 1:
        interval = _valid_run_interval(tiles)
        if interval:
            start, end = interval
            return MeldValidation(True, "run", sum(range(start, end + 1)), "유효한 연속")

    return MeldValidation(
        False,
        reason="같은 숫자의 서로 다른 색 3~4개 또는 같은 색의 연속 숫자 3개 이상이 아닙니다.",
    )


def _joker_values_for_meld(tiles: list[Tile], validation: MeldValidation) -> list[int]:
    if not validation.valid:
        return []
    if validation.kind == "group":
        natural = next((tile.parts[1] for tile in tiles if not tile.parts[2]), 0)
        return [int(natural or 0) for tile in tiles if tile.parts[2]]
    interval = _valid_run_interval(tiles)
    if not interval:
        return []
    start, end = interval
    ascending = list(range(start, end + 1))
    sequences = (ascending, list(reversed(ascending)))
    expected = next(
        sequence
        for sequence in sequences
        if all(tile.parts[2] or tile.parts[1] == value for tile, value in zip(tiles, sequence))
    )
    return [value for tile, value in zip(tiles, expected) if tile.parts[2]]


def cluster_melds(tiles: Iterable[Tile], gap_factor: float = 0.75) -> list[list[Tile]]:
    """Cluster top-down detections into rows, then split rows at large horizontal gaps."""
    tiles = list(tiles)
    if not tiles:
        return []
    median_height = max(1.0, median(tile.height for tile in tiles))
    median_width = max(1.0, median(tile.width for tile in tiles))
    rows: list[list[Tile]] = []
    for tile in sorted(tiles, key=lambda item: item.center[1]):
        for row in rows:
            row_center = median(item.center[1] for item in row)
            if abs(tile.center[1] - row_center) <= median_height * 0.55:
                row.append(tile)
                break
        else:
            rows.append([tile])

    melds: list[list[Tile]] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.center[0])
        current = [ordered[0]]
        for tile in ordered[1:]:
            horizontal_gap = tile.x1 - current[-1].x2
            if horizontal_gap > median_width * gap_factor:
                melds.append(current)
                current = [tile]
            else:
                current.append(tile)
        melds.append(current)
    return melds


def _existing_tiles_moved(previous: list[Tile], current: list[Tile]) -> bool:
    if not previous:
        return False
    tolerance = max(8.0, median(tile.width for tile in previous) * 0.4)
    by_class: dict[str, list[Tile]] = {}
    for tile in current:
        by_class.setdefault(tile.class_name, []).append(tile)
    for old in previous:
        candidates = by_class.get(old.class_name, [])
        if not candidates:
            return True
        nearest = min(candidates, key=lambda item: (item.center[0] - old.center[0]) ** 2 + (item.center[1] - old.center[1]) ** 2)
        distance = ((nearest.center[0] - old.center[0]) ** 2 + (nearest.center[1] - old.center[1]) ** 2) ** 0.5
        if distance > tolerance:
            return True
        candidates.remove(nearest)
    return False


def validate_turn(
    previous: Iterable[Tile],
    current: Iterable[Tile],
    initial_meld_done: bool,
    gap_factor: float = 0.75,
) -> TurnValidation:
    previous, current = list(previous), list(current)
    result = TurnValidation(valid=True)
    previous_counts = Counter(tile.class_name for tile in previous)
    current_counts = Counter(tile.class_name for tile in current)
    result.added = current_counts - previous_counts

    missing = previous_counts - current_counts
    if missing:
        result.reasons.append(f"테이블에서 사라진 타일이 있습니다: {dict(missing)}")
    if not result.added:
        result.reasons.append("새로 내려놓은 타일이 탐지되지 않았습니다.")

    result.melds = cluster_melds(current, gap_factor)
    meld_results = [validate_meld(meld) for meld in result.melds]
    for index, meld_result in enumerate(meld_results, 1):
        if not meld_result.valid:
            names = ", ".join(tile.class_name for tile in result.melds[index - 1])
            result.reasons.append(f"세트 {index} 위반 ({names}): {meld_result.reason}")

    if not initial_meld_done:
        if _existing_tiles_moved(previous, current):
            result.reasons.append("최초 등록 전에는 기존 테이블 타일을 이동하거나 조합할 수 없습니다.")
        added_natural_score = sum(
            (parse_class_name(name)[1] or 0) * count
            for name, count in result.added.items()
        )
        added_jokers = result.added.get("joker", 0)
        joker_values: list[int] = []
        for meld, meld_result in zip(result.melds, meld_results):
            if meld_result.valid:
                joker_values.extend(_joker_values_for_meld(meld, meld_result))
        joker_score = sum(joker_values[:added_jokers])
        result.initial_score = added_natural_score + joker_score
        if added_jokers:
            result.warnings.append(
                "조커 점수는 화면에서 속한 세트의 대체 숫자로 계산했습니다. 동일 타일이 여러 개면 배치 확인이 필요합니다."
            )
        if result.initial_score < 30:
            result.reasons.append(f"최초 등록 합계가 30점 미만입니다 (자동 계산: {result.initial_score}점).")

    result.valid = not result.reasons
    return result
