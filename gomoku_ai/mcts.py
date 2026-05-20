from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .game import Board, opponent


class PolicyValueLike(Protocol):
    def predict(self, board: Board, player: int) -> tuple[np.ndarray, float]:
        ...


@dataclass
class MCTSConfig:
    simulations: int = 80
    cpuct: float = 1.5


class MCTS:
    def __init__(self, policy: PolicyValueLike, config: MCTSConfig) -> None:
        self.policy = policy
        self.config = config
        self.qsa: dict[tuple[bytes, int], float] = {}
        self.nsa: dict[tuple[bytes, int], int] = {}
        self.ns: dict[bytes, int] = {}
        self.ps: dict[bytes, np.ndarray] = {}

    def action_probs(self, board: Board, player: int, temp: float = 1.0) -> np.ndarray:
        for _ in range(self.config.simulations):
            self.search(board.copy(), player)

        key = state_key(board, player)
        counts = np.array(
            [self.nsa.get((key, action), 0) for action in range(board.size * board.size)],
            dtype=np.float32,
        )
        legal = legal_mask(board)
        counts *= legal
        if counts.sum() <= 0:
            counts = legal

        if temp <= 0:
            probs = np.zeros_like(counts)
            probs[int(np.argmax(counts))] = 1.0
            return probs

        counts = np.power(counts, 1.0 / temp)
        total = counts.sum()
        if total <= 0:
            return legal / max(legal.sum(), 1)
        return counts / total

    def search(self, board: Board, player: int) -> float:
        winner = board.winner()
        if winner:
            return 1.0 if winner == player else -1.0
        if board.is_full():
            return 0.0

        key = state_key(board, player)
        if key not in self.ps:
            policy, value = self.policy.predict(board, player)
            valid = legal_mask(board)
            policy = policy.astype(np.float32) * valid
            if policy.sum() <= 0:
                policy = valid
            policy = policy / max(policy.sum(), 1e-8)
            self.ps[key] = policy
            self.ns[key] = 0
            return value

        valid = legal_mask(board)
        best_score = -float("inf")
        best_action = -1
        sqrt_ns = math.sqrt(max(self.ns[key], 1))
        for action in np.flatnonzero(valid):
            edge = (key, int(action))
            q = self.qsa.get(edge, 0.0)
            n = self.nsa.get(edge, 0)
            u = q + self.config.cpuct * self.ps[key][action] * sqrt_ns / (1 + n)
            if u > best_score:
                best_score = u
                best_action = int(action)

        next_board = board.copy()
        row, col = divmod(best_action, board.size)
        next_board.place(row, col, player)
        value = -self.search(next_board, opponent(player))

        edge = (key, best_action)
        old_n = self.nsa.get(edge, 0)
        old_q = self.qsa.get(edge, 0.0)
        self.qsa[edge] = (old_n * old_q + value) / (old_n + 1)
        self.nsa[edge] = old_n + 1
        self.ns[key] += 1
        return value


def state_key(board: Board, player: int) -> bytes:
    arr = np.array(board.grid, dtype=np.int8) * player
    return arr.tobytes()


def legal_mask(board: Board) -> np.ndarray:
    mask = np.zeros(board.size * board.size, dtype=np.float32)
    for row, col in board.legal_moves():
        mask[row * board.size + col] = 1.0
    return mask
