from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from .features import FEATURE_NAMES, move_features
from .game import Board


DEFAULT_WEIGHTS = [
    0.0,   # bias
    0.25,  # center
    0.35,  # neighbor_density
    1.40,  # own_max_line
    1.20,  # opp_max_line
    0.45,  # own_open_two
    1.30,  # own_open_three
    5.00,  # own_open_four
    3.20,  # own_closed_four
    0.35,  # opp_open_two
    1.25,  # opp_open_three
    4.60,  # opp_open_four
    3.80,  # opp_closed_four
    100.0,  # own_win
    90.0,   # block_win
]


@dataclass
class LinearPolicy:
    weights: list[float] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} weights, got {len(self.weights)}")

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = data.get("weights")
        names = data.get("feature_names", FEATURE_NAMES)
        if names != FEATURE_NAMES:
            raise ValueError("model feature names do not match this code version")
        return cls(weights=[float(value) for value in weights])

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "meta": meta or {},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def score_features(self, features: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))

    def score_move(self, board: Board, move: tuple[int, int], player: int) -> float:
        return self.score_features(move_features(board, move, player))

    def ranked_moves(
        self,
        board: Board,
        player: int,
        candidate_radius: int = 2,
    ) -> list[tuple[float, tuple[int, int], list[float]]]:
        moves = board.candidate_moves(radius=candidate_radius)
        ranked = []
        for move in moves:
            features = move_features(board, move, player)
            ranked.append((self.score_features(features), move, features))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def choose_move(
        self,
        board: Board,
        player: int,
        *,
        epsilon: float = 0.0,
        temperature: float = 0.15,
        candidate_radius: int = 2,
        rng: random.Random | None = None,
    ) -> tuple[tuple[int, int], list[float]]:
        rng = rng or random
        ranked = self.ranked_moves(board, player, candidate_radius=candidate_radius)
        if not ranked:
            raise RuntimeError("no legal moves available")

        if epsilon > 0 and rng.random() < epsilon:
            _, move, features = rng.choice(ranked)
            return move, features

        if temperature <= 0:
            _, move, features = ranked[0]
            return move, features

        max_score = ranked[0][0]
        scaled = [math.exp((score - max_score) / max(temperature, 1e-6)) for score, _, _ in ranked]
        total = sum(scaled)
        pick = rng.random() * total
        cumulative = 0.0
        for weight, (_, move, features) in zip(scaled, ranked):
            cumulative += weight
            if cumulative >= pick:
                return move, features
        _, move, features = ranked[-1]
        return move, features

    def update(self, features: list[float], reward: float, lr: float, l2: float = 0.0001) -> None:
        for index, value in enumerate(features):
            regularization = l2 * self.weights[index]
            self.weights[index] += lr * (reward * value - regularization)
        self._clip_weights()

    def _clip_weights(self) -> None:
        for index, value in enumerate(self.weights):
            limit = 120.0 if FEATURE_NAMES[index] in ("own_win", "block_win") else 20.0
            self.weights[index] = max(-limit, min(limit, value))

