from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .game import BLACK, WHITE, Board, opponent
from .model import LinearPolicy


@dataclass
class TrainConfig:
    episodes: int = 2000
    size: int = 15
    n_in_row: int = 5
    lr: float = 0.01
    gamma: float = 0.99
    epsilon: float = 0.15
    temperature: float = 0.25
    candidate_radius: int = 2
    seed: int | None = 7
    save_every: int = 100
    model_path: Path = Path("models/gomoku_policy.json")


def self_play_episode(policy: LinearPolicy, config: TrainConfig, rng: random.Random) -> tuple[int, int]:
    board = Board(size=config.size, n_in_row=config.n_in_row)
    player = BLACK
    trajectory: list[tuple[list[float], int]] = []

    while True:
        move, features = policy.choose_move(
            board,
            player,
            epsilon=config.epsilon,
            temperature=config.temperature,
            candidate_radius=config.candidate_radius,
            rng=rng,
        )
        board.place(move[0], move[1], player)
        trajectory.append((features, player))

        winner = board.winner()
        if winner or board.is_full():
            break
        player = opponent(player)

    # REINFORCE-style final reward. The same policy controls both colors, so
    # every move is judged from the color that made it.
    last_index = len(trajectory) - 1
    for index, (features, move_player) in enumerate(trajectory):
        if winner == 0:
            reward = 0.0
        else:
            reward = 1.0 if move_player == winner else -1.0
        discounted = reward * (config.gamma ** (last_index - index))
        policy.update(features, discounted, lr=config.lr)

    return winner, len(trajectory)


def train(
    policy: LinearPolicy,
    config: TrainConfig,
    verbose: bool = True,
    metrics_callback=None,
) -> dict[str, int | float]:
    rng = random.Random(config.seed)
    stats = {
        "black_wins": 0,
        "white_wins": 0,
        "draws": 0,
        "total_moves": 0,
    }

    for episode in range(1, config.episodes + 1):
        winner, moves = self_play_episode(policy, config, rng)
        stats["total_moves"] += moves
        if winner == BLACK:
            stats["black_wins"] += 1
        elif winner == WHITE:
            stats["white_wins"] += 1
        else:
            stats["draws"] += 1

        if config.save_every > 0 and episode % config.save_every == 0:
            avg_moves = stats["total_moves"] / episode
            policy.save(
                config.model_path,
                meta={
                    "episodes": episode,
                    "board_size": config.size,
                    "n_in_row": config.n_in_row,
                    "avg_moves": avg_moves,
                },
            )
            if verbose:
                print(
                    f"episode {episode:>6}/{config.episodes} | "
                    f"black {stats['black_wins']:>5} white {stats['white_wins']:>5} "
                    f"draw {stats['draws']:>4} | avg moves {avg_moves:5.1f}"
                )
            if metrics_callback:
                metrics_callback(
                    episode=episode,
                    black_wins=stats["black_wins"],
                    white_wins=stats["white_wins"],
                    draws=stats["draws"],
                    avg_moves=avg_moves,
                )

    policy.save(
        config.model_path,
        meta={
            "episodes": config.episodes,
            "board_size": config.size,
            "n_in_row": config.n_in_row,
            "avg_moves": stats["total_moves"] / max(config.episodes, 1),
        },
    )
    return stats
