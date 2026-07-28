"""Deterministic validation of the currently visible Rummikub layout."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin
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


@dataclass(frozen=True)
class MeldResult:
    valid: bool
    kind: str | None = None
    score: int = 0
    reason: str = ""


def _run_interval(tiles: list[Tile]) -> tuple[int, int] | None:
    natural_numbers = [int(tile.parts[1]) for tile in tiles if not tile.parts[2]]
    if len(set(natural_numbers)) != len(natural_numbers):
        return None
    for start in range(1, 15 - len(tiles)):
        ascending = list(range(start, start + len(tiles)))
        for sequence in (ascending, list(reversed(ascending))):
            if all(tile.parts[2] or tile.parts[1] == number for tile, number in zip(tiles, sequence)):
                return start, start + len(tiles) - 1
    return None


def validate_meld(tiles: Iterable[Tile]) -> MeldResult:
    tiles = list(tiles)
    if len(tiles) < 3:
        return MeldResult(False, reason="세트는 최소 3개의 타일이어야 합니다.")
    numbered = [tile for tile in tiles if not tile.parts[2]]
    jokers = len(tiles) - len(numbered)
    colors = [tile.parts[0] for tile in numbered]
    numbers = [int(tile.parts[1]) for tile in numbered]

    if len(tiles) <= 4 and numbered:
        if len(set(numbers)) == 1 and len(set(colors)) == len(colors) and len(numbered) + jokers <= 4:
            return MeldResult(True, "group", numbers[0] * len(tiles), "유효한 그룹")

    if numbered and len(set(colors)) == 1:
        interval = _run_interval(tiles)
        if interval:
            start, end = interval
            return MeldResult(True, "run", sum(range(start, end + 1)), "유효한 연속")

    return MeldResult(
        False,
        reason="같은 숫자의 서로 다른 색 3~4개 또는 같은 색의 연속 숫자 3개 이상이 아닙니다.",
    )


def cluster_melds(tiles: Iterable[Tile], gap_factor: float = 0.75) -> list[list[Tile]]:
    tiles = list(tiles)
    if not tiles:
        return []

    # Estimate the common direction of tile rows from each tile's nearest
    # neighbour. The doubled-angle average treats 0° and 180° as equivalent.
    nearest_vectors: list[tuple[float, float, float]] = []
    for index, tile in enumerate(tiles):
        candidates = []
        for other_index, other in enumerate(tiles):
            if index == other_index:
                continue
            dx, dy = other.center[0] - tile.center[0], other.center[1] - tile.center[1]
            candidates.append((hypot(dx, dy), dx, dy))
        if candidates:
            nearest_vectors.append(min(candidates))
    if nearest_vectors:
        angles = [atan2(dy, dx) for _, dx, dy in nearest_vectors]
        angle = 0.5 * atan2(sum(sin(2 * value) for value in angles), sum(cos(2 * value) for value in angles))
        neighbor_spacing = max(1.0, median(distance for distance, _, _ in nearest_vectors))
    else:
        angle, neighbor_spacing = 0.0, 1.0

    direction = (cos(angle), sin(angle))
    normal = (-sin(angle), cos(angle))

    def projected_center(tile: Tile) -> tuple[float, float]:
        x, y = tile.center
        return x * direction[0] + y * direction[1], x * normal[0] + y * normal[1]

    row_tolerance = max(1.0, median(min(tile.width, tile.height) for tile in tiles) * 0.65)
    rows: list[list[Tile]] = []
    for tile in sorted(tiles, key=lambda item: projected_center(item)[1]):
        for row in rows:
            row_center = median(projected_center(item)[1] for item in row)
            if abs(projected_center(tile)[1] - row_center) <= row_tolerance:
                row.append(tile)
                break
        else:
            rows.append([tile])

    melds: list[list[Tile]] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: projected_center(item)[0])
        current = [ordered[0]]
        for tile in ordered[1:]:
            center_gap = projected_center(tile)[0] - projected_center(current[-1])[0]
            if center_gap > neighbor_spacing * (1.0 + gap_factor):
                melds.append(current)
                current = [tile]
            else:
                current.append(tile)
        melds.append(current)
    return melds


def validate_layout(tiles: Iterable[Tile], gap_factor: float = 0.75) -> dict:
    tiles = list(tiles)
    melds = cluster_melds(tiles, gap_factor)
    results = [validate_meld(meld) for meld in melds]
    source_indices = {id(tile): index for index, tile in enumerate(tiles)}

    def offending_positions(meld: list[Tile], result: MeldResult) -> list[int]:
        if result.valid:
            return []
        removable = [
            index
            for index in range(len(meld))
            if len(meld) - 1 >= 3 and validate_meld(meld[:index] + meld[index + 1 :]).valid
        ]
        # When one or more single removals repair the meld, those are the likely
        # offending detections. Otherwise the entire ambiguous meld is unsafe.
        return removable or list(range(len(meld)))

    invalid_tile_indices: set[int] = set()
    meld_items = []
    for index, (meld, result) in enumerate(zip(melds, results), 1):
        tile_indices = [source_indices[id(tile)] for tile in meld]
        offending = [tile_indices[position] for position in offending_positions(meld, result)]
        invalid_tile_indices.update(offending)
        meld_items.append({
            "index": index,
            "tiles": [tile.class_name for tile in meld],
            "tile_indices": tile_indices,
            "offending_tile_indices": offending,
            "valid": result.valid,
            "kind": result.kind,
            "score": result.score,
            "reason": result.reason,
        })
    return {
        "valid": bool(tiles) and all(result.valid for result in results),
        "tile_count": len(tiles),
        "invalid_tile_indices": sorted(invalid_tile_indices),
        "melds": meld_items,
        "warning": "카메라 이미지 기반 자동 판정이므로 가림, 오탐 및 세트 간격에 따라 사람 확인이 필요합니다.",
    }
