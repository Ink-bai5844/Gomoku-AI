from __future__ import annotations

from dataclasses import dataclass, field


EMPTY = 0
BLACK = 1
WHITE = -1


def opponent(player: int) -> int:
    return -player


def player_name(player: int) -> str:
    if player == BLACK:
        return "Black"
    if player == WHITE:
        return "White"
    return "Empty"


@dataclass
class Board:
    size: int = 15
    n_in_row: int = 5
    grid: list[list[int]] = field(init=False)
    last_move: tuple[int, int] | None = None
    move_count: int = 0

    def __post_init__(self) -> None:
        if self.size < self.n_in_row:
            raise ValueError("board size must be >= n_in_row")
        self.grid = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def reset(self) -> None:
        self.grid = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.last_move = None
        self.move_count = 0

    def copy(self) -> "Board":
        board = Board(size=self.size, n_in_row=self.n_in_row)
        board.grid = [row[:] for row in self.grid]
        board.last_move = self.last_move
        board.move_count = self.move_count
        return board

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def is_empty(self, row: int, col: int) -> bool:
        return self.grid[row][col] == EMPTY

    def place(self, row: int, col: int, player: int) -> bool:
        if player not in (BLACK, WHITE):
            raise ValueError("player must be BLACK or WHITE")
        if not self.in_bounds(row, col) or not self.is_empty(row, col):
            return False
        self.grid[row][col] = player
        self.last_move = (row, col)
        self.move_count += 1
        return True

    def is_full(self) -> bool:
        return self.move_count >= self.size * self.size

    def legal_moves(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] == EMPTY
        ]

    def candidate_moves(self, radius: int = 2) -> list[tuple[int, int]]:
        """Return empty cells close to existing stones.

        Searching the whole board is still possible, but self-play is far
        faster when it only considers plausible local moves after the opener.
        """
        if self.move_count == 0:
            center = self.size // 2
            return [(center, center)]

        seen: set[tuple[int, int]] = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == EMPTY:
                    continue
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if self.in_bounds(nr, nc) and self.grid[nr][nc] == EMPTY:
                            seen.add((nr, nc))
        return sorted(seen) if seen else self.legal_moves()

    def count_direction(self, row: int, col: int, player: int, dr: int, dc: int) -> int:
        count = 0
        row += dr
        col += dc
        while self.in_bounds(row, col) and self.grid[row][col] == player:
            count += 1
            row += dr
            col += dc
        return count

    def check_winner(self, row: int, col: int) -> int:
        player = self.grid[row][col]
        if player == EMPTY:
            return EMPTY

        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            total = (
                1
                + self.count_direction(row, col, player, dr, dc)
                + self.count_direction(row, col, player, -dr, -dc)
            )
            if total >= self.n_in_row:
                return player
        return EMPTY

    def winner(self) -> int:
        if self.last_move is None:
            return EMPTY
        row, col = self.last_move
        return self.check_winner(row, col)
