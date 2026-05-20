from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .features import FEATURE_NAMES, move_features
from .game import BLACK, Board, opponent
from .game_visualizer import TrainingGameVisualizer
from .mcts import MCTS, MCTSConfig, PolicyValueLike
from .neural import (
    GomokuZeroPolicy,
    LegacyAlphaZeroPolicy,
    NeuralPolicy,
    canonical_batch_to_tensor,
    infer_gomokuzero_checkpoint_board_size,
    infer_legacy_checkpoint_board_size,
)
from .training_visualizer import TrainingVisualizer


@dataclass
class NeuralTrainConfig:
    size: int = 15
    n_in_row: int = 5
    iterations: int = 5
    self_play_games: int = 8
    opponent_games: int = 0
    epochs: int = 2
    batch_size: int = 64
    replay_size: int = 20000
    mcts_sims: int = 80
    opponent_mcts_sims: int = 80
    cpuct: float = 1.5
    temp_threshold: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    reward_weight: float = 0.0
    learn_after_step: int | None = None
    learn_opponent_wins: bool = True
    seed: int = 7
    model_path: Path = Path("models/gomoku_resnet.pth.tar")
    opponent_checkpoint: Path | None = None
    gomokuzero_checkpoint: Path | None = None
    device: str | None = None


def canonical_array(board: Board, player: int) -> np.ndarray:
    return (np.array(board.grid, dtype=np.int8) * player).astype(np.float32)


def augment_example(board: np.ndarray, pi: np.ndarray, value: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    size = board.shape[0]
    pi_board = pi.reshape(size, size)
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for k in range(4):
        b = np.rot90(board, k)
        p = np.rot90(pi_board, k)
        examples.append((b.copy(), p.reshape(-1).copy(), value))
        examples.append((np.fliplr(b).copy(), np.fliplr(p).reshape(-1).copy(), value))
    return examples


def shaped_reward(board: Board, move: tuple[int, int], player: int) -> float:
    features = move_features(board, move, player)
    values = dict(zip(FEATURE_NAMES, features))
    score = 0.0
    score += 0.02 * values["own_open_two"]
    score += 0.04 * values["opp_open_two"]
    score += 0.08 * values["own_open_three"]
    score += 0.10 * values["opp_open_three"]
    score += 0.30 * values["own_open_four"]
    score += 0.26 * values["opp_open_four"]
    score += 0.20 * values["own_closed_four"]
    score += 0.22 * values["opp_closed_four"]
    score += 0.75 * values["own_win"]
    score += 0.55 * values["block_win"]
    double_three = values["own_open_three"] >= 0.5
    double_four = values["own_closed_four"] + values["own_open_four"] >= 0.5
    four_three = (values["own_closed_four"] + values["own_open_four"]) > 0 and values["own_open_three"] > 0
    if double_three:
        score += 0.20
    if four_three:
        score += 0.35
    if double_four:
        score += 0.45
    return score


def sample_action(probs: np.ndarray, rng: random.Random) -> int:
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("empty action distribution")
    pick = rng.random() * total
    acc = 0.0
    for index, prob in enumerate(probs):
        acc += float(prob)
        if acc >= pick:
            return index
    return int(np.argmax(probs))


def play_training_game(
    learner: NeuralPolicy,
    config: NeuralTrainConfig,
    rng: random.Random,
    opponent_policy: PolicyValueLike | None = None,
    learner_player: int = BLACK,
    game_recorder: TrainingGameVisualizer | None = None,
    recorder_context: dict | None = None,
    use_temperature: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], int, int]:
    board = Board(size=config.size, n_in_row=config.n_in_row)
    player = BLACK
    step = 0
    records: list[tuple[int, np.ndarray, int, np.ndarray, float, str]] = []
    learner_mcts = MCTS(learner, MCTSConfig(simulations=config.mcts_sims, cpuct=config.cpuct))
    opponent_mcts = (
        MCTS(opponent_policy, MCTSConfig(simulations=config.opponent_mcts_sims, cpuct=config.cpuct))
        if opponent_policy
        else None
    )
    frames: list[dict] = []

    while True:
        step += 1
        is_learner_turn = opponent_policy is None or player == learner_player
        if is_learner_turn:
            temp = 1.0 if use_temperature and step <= config.temp_threshold else 0.0
            pi = learner_mcts.action_probs(board, player, temp=temp)
            action = sample_action(pi, rng) if temp > 0 else int(np.argmax(pi))
            move = divmod(action, board.size)
            reward = shaped_reward(board, move, player) if config.reward_weight > 0 else 0.0
            records.append((step, canonical_array(board, player), player, pi, reward, "learner"))
        else:
            assert opponent_mcts is not None
            temp = 1.0 if use_temperature and step <= config.temp_threshold else 0.0
            pi = opponent_mcts.action_probs(board, player, temp=temp)
            action = sample_action(pi, rng) if temp > 0 else int(np.argmax(pi))
            move = divmod(action, board.size)
            records.append((step, canonical_array(board, player), player, pi.copy(), 0.0, "opponent"))

        board.place(move[0], move[1], player)
        if game_recorder:
            frames.append(
                {
                    "board": [row[:] for row in board.grid],
                    "move": move,
                    "player": player,
                    "step": step,
                }
            )
        winner = board.winner()
        if winner or board.is_full():
            break
        player = opponent(player)

    if game_recorder:
        context = recorder_context or {}
        game_recorder.render(
            frames,
            iteration=int(context.get("iteration", 0)),
            phase=str(context.get("phase", "game")),
            game_index=int(context.get("game_index", 0)),
            winner=winner,
        )

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    learn_after_step = (
        config.temp_threshold
        if config.learn_after_step is None and use_temperature
        else 0
        if config.learn_after_step is None
        else config.learn_after_step
    )
    opponent_won = opponent_policy is not None and winner and winner != learner_player
    for record_step, canonical, record_player, pi, reward, source in records:
        if record_step <= learn_after_step:
            continue
        if source == "opponent" and not (config.learn_opponent_wins and opponent_won):
            continue
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if record_player == winner else -1.0
        value = float(np.clip(value + config.reward_weight * reward, -1.0, 1.0))
        examples.extend(augment_example(canonical, pi, value))
    return examples, winner, step


def train_batches(
    learner: NeuralPolicy,
    examples: list[tuple[np.ndarray, np.ndarray, float]],
    config: NeuralTrainConfig,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0
    learner.net.train()
    policy_losses: list[float] = []
    value_losses: list[float] = []

    for _ in range(config.epochs):
        random.shuffle(examples)
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            boards, target_pis, target_values = zip(*batch)
            board_tensor = canonical_batch_to_tensor(np.array(boards, dtype=np.float32), learner.device)
            target_pi_tensor = torch.from_numpy(np.array(target_pis, dtype=np.float32)).to(learner.device)
            target_value_tensor = torch.from_numpy(np.array(target_values, dtype=np.float32)).to(learner.device)

            logits, values = learner.net(board_tensor)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi_tensor * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values, target_value_tensor)
            loss = policy_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.net.parameters(), 1.0)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))

    learner.net.eval()
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def train_neural(
    learner: NeuralPolicy,
    config: NeuralTrainConfig,
    visualizer: TrainingVisualizer | None = None,
    game_visualizer: TrainingGameVisualizer | None = None,
) -> None:
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    replay: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=config.replay_size)
    optimizer = optim.AdamW(
        learner.net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    opponents: list[tuple[str, PolicyValueLike]] = []
    if config.opponent_checkpoint:
        opponent_size = infer_legacy_checkpoint_board_size(config.opponent_checkpoint)
        if opponent_size != config.size:
            raise ValueError(
                f"{config.opponent_checkpoint} is {opponent_size}x{opponent_size}; "
                f"run with --size {opponent_size} to adversarial-train against it."
            )
        opponents.append(
            (
                "alphazero-gomoku",
                LegacyAlphaZeroPolicy.load(
                    config.opponent_checkpoint,
                    board_size=config.size,
                    device=str(learner.device),
                ),
            )
        )

    if config.gomokuzero_checkpoint:
        gomokuzero_size = infer_gomokuzero_checkpoint_board_size(config.gomokuzero_checkpoint)
        if gomokuzero_size != config.size:
            raise ValueError(
                f"{config.gomokuzero_checkpoint} is {gomokuzero_size}x{gomokuzero_size}; "
                f"run with --size {gomokuzero_size} to adversarial-train against it."
            )
        opponents.append(
            (
                "GomokuZeroAI",
                GomokuZeroPolicy.load(
                    config.gomokuzero_checkpoint,
                    board_size=config.size,
                    device=str(learner.device),
                ),
            )
        )

    for iteration in range(1, config.iterations + 1):
        new_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for game_index in range(config.self_play_games):
            recorder = (
                game_visualizer
                if game_visualizer and game_visualizer.should_capture(iteration, game_index)
                else None
            )
            examples, _, _ = play_training_game(
                learner,
                config,
                rng,
                game_recorder=recorder,
                recorder_context={
                    "iteration": iteration,
                    "phase": "self_play",
                    "game_index": game_index,
                },
                use_temperature=True,
            )
            new_examples.extend(examples)
            print(f"iter {iteration} self-play {game_index + 1}/{config.self_play_games}", flush=True)

        opponent_start = len(new_examples)
        if opponents and config.opponent_games > 0:
            for game_index in range(config.opponent_games):
                learner_player = BLACK if game_index % 2 == 0 else -BLACK
                opponent_name, opponent_policy = opponents[game_index % len(opponents)]
                recorder_index = config.self_play_games + game_index
                recorder = (
                    game_visualizer
                    if game_visualizer and game_visualizer.should_capture(iteration, recorder_index)
                    else None
                )
                examples, winner, steps = play_training_game(
                    learner,
                    config,
                    rng,
                    opponent_policy=opponent_policy,
                    learner_player=learner_player,
                    game_recorder=recorder,
                    recorder_context={
                        "iteration": iteration,
                        "phase": f"vs_{opponent_name}",
                        "game_index": game_index,
                    },
                    use_temperature=True,
                )
                new_examples.extend(examples)
                result = "draw"
                if winner:
                    result = "learner win" if winner == learner_player else "learner loss"
                print(
                    f"iter {iteration} vs-{opponent_name} {game_index + 1}/{config.opponent_games} "
                    f"| {result} | steps {steps}",
                    flush=True,
                )
        opponent_example_count = len(new_examples) - opponent_start

        replay.extend(new_examples)
        train_examples = list(replay)
        policy_loss, value_loss = train_batches(learner, train_examples, config, optimizer)
        produced_examples = len(new_examples)
        learner.save(
            config.model_path,
            meta={
                "iteration": iteration,
                "examples": len(train_examples),
                "new_examples": produced_examples,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "opponent_checkpoint": str(config.opponent_checkpoint) if config.opponent_checkpoint else None,
                "gomokuzero_checkpoint": str(config.gomokuzero_checkpoint) if config.gomokuzero_checkpoint else None,
            },
        )
        if visualizer:
            visualizer.record(
                iteration=iteration,
                examples=len(train_examples),
                new_examples=produced_examples,
                opponent_examples=opponent_example_count,
                policy_loss=policy_loss,
                value_loss=value_loss,
                self_play_games=config.self_play_games,
                opponent_games=config.opponent_games,
                mcts_sims=config.mcts_sims,
                reward_weight=config.reward_weight,
                learn_after_step=config.learn_after_step if config.learn_after_step is not None else config.temp_threshold,
                opponent_learn_after_step=config.learn_after_step if config.learn_after_step is not None else 0,
                learn_opponent_wins=int(config.learn_opponent_wins),
            )
        print(
            f"iter {iteration}/{config.iterations} | examples {len(train_examples)} "
            f"| policy_loss {policy_loss:.4f} value_loss {value_loss:.4f} | saved {config.model_path}",
            flush=True,
        )
