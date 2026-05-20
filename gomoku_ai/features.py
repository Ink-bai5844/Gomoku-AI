from __future__ import annotations

from dataclasses import dataclass

from .game import Board, EMPTY, opponent


FEATURE_NAMES = [
    "bias",
    "center",
    "neighbor_density",
    "own_max_line",
    "opp_max_line",
    "own_open_two",
    "own_open_three",
    "own_open_four",
    "own_closed_four",
    "opp_open_two",
    "opp_open_three",
    "opp_open_four",
    "opp_closed_four",
    "own_win",
    "block_win",
]


@dataclass(frozen=True)
class LineInfo:
    length: int
    open_ends: int


def _count_side(board: Board, row: int, col: int, player: int, dr: int, dc: int) -> tuple[int, bool]:
    count = 0
    row += dr
    col += dc
    while board.in_bounds(row, col) and board.get(row, col) == player:
        count += 1
        row += dr
        col += dc
    is_open = board.in_bounds(row, col) and board.get(row, col) == EMPTY
    return count, is_open


def _line_info_for_move(board: Board, row: int, col: int, player: int, dr: int, dc: int) -> LineInfo:
    left, left_open = _count_side(board, row, col, player, -dr, -dc)
    right, right_open = _count_side(board, row, col, player, dr, dc)
    return LineInfo(length=1 + left + right, open_ends=int(left_open) + int(right_open))


def _line_summaries(board: Board, row: int, col: int, player: int) -> list[LineInfo]:
    return [
        _line_info_for_move(board, row, col, player, dr, dc)
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1))
    ]


def _neighbor_density(board: Board, row: int, col: int) -> float:
    neighbors = 0
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if board.in_bounds(nr, nc):
                total += 1
                if board.get(nr, nc) != EMPTY:
                    neighbors += 1
    return neighbors / max(total, 1)


def _pattern_counts(lines: list[LineInfo]) -> tuple[float, float, float, float]:
    open_two = sum(1 for line in lines if line.length == 2 and line.open_ends == 2)
    open_three = sum(1 for line in lines if line.length == 3 and line.open_ends == 2)
    open_four = sum(1 for line in lines if line.length == 4 and line.open_ends == 2)
    closed_four = sum(1 for line in lines if line.length == 4 and line.open_ends == 1)
    return open_two / 4, open_three / 4, open_four / 4, closed_four / 4


def move_features(board: Board, move: tuple[int, int], player: int) -> list[float]:
    row, col = move
    if not board.in_bounds(row, col) or not board.is_empty(row, col):
        raise ValueError("features can only be computed for an empty legal move")

    center = (board.size - 1) / 2
    max_dist = max(center, 1)
    center_score = 1 - (abs(row - center) + abs(col - center)) / (2 * max_dist)

    own_lines = _line_summaries(board, row, col, player)
    opp_lines = _line_summaries(board, row, col, opponent(player))
    own_open_two, own_open_three, own_open_four, own_closed_four = _pattern_counts(own_lines)
    opp_open_two, opp_open_three, opp_open_four, opp_closed_four = _pattern_counts(opp_lines)

    own_max = max(line.length for line in own_lines) / board.n_in_row
    opp_max = max(line.length for line in opp_lines) / board.n_in_row
    own_win = float(any(line.length >= board.n_in_row for line in own_lines))
    block_win = float(any(line.length >= board.n_in_row for line in opp_lines))

    return [
        1.0,
        center_score,
        _neighbor_density(board, row, col),
        own_max,
        opp_max,
        own_open_two,
        own_open_three,
        own_open_four,
        own_closed_four,
        opp_open_two,
        opp_open_three,
        opp_open_four,
        opp_closed_four,
        own_win,
        block_win,
    ]

