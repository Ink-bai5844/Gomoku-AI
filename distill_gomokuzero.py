from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import optim

from gomoku_ai.game import BLACK, Board, opponent
from gomoku_ai.game_visualizer import TrainingGameVisualizer
from gomoku_ai.mcts import MCTS, MCTSConfig
from gomoku_ai.neural import (
    GomokuZeroPolicy,
    NeuralPolicy,
    infer_gomokuzero_checkpoint_board_size,
)
from gomoku_ai.neural_train import augment_example, canonical_array, sample_action, train_batches
from gomoku_ai.training_visualizer import TrainingVisualizer


DEFAULT_TEACHER = Path("GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a GomokuZeroAI teacher self-play match into this project's policy-value model."
    )
    parser.add_argument("--teacher-a", type=Path, default=DEFAULT_TEACHER, help="black-side GomokuZeroAI checkpoint")
    parser.add_argument("--teacher-b", type=Path, default=None, help="white-side GomokuZeroAI checkpoint; defaults to teacher-a")
    parser.add_argument("--student", type=Path, default=Path("models/gomoku_transformer.pth.tar"), help="student model path")
    parser.add_argument("--fresh", action="store_true", help="start a new student instead of loading --student")
    parser.add_argument("--architecture", choices=["resnet", "transformer"], default="transformer")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=8, help="teacher-vs-teacher games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=80, help="teacher MCTS simulations per move")
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--temp-threshold", type=int, default=8)
    parser.add_argument(
        "--learn-after-step",
        type=int,
        default=None,
        help="only distill moves after this step; defaults to --temp-threshold",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualize", action="store_true", help="write CSV/PNG training curves")
    parser.add_argument("--visual-dir", type=Path, default=Path("runs"))
    parser.add_argument("--visual-name", type=str, default="gomokuzero_distill")
    parser.add_argument("--visualize-games", action="store_true", help="save teacher game GIFs/final boards")
    parser.add_argument("--visual-games-dir", type=Path, default=Path("runs/distill_games"))
    parser.add_argument("--visual-games-every", type=int, default=1)
    parser.add_argument("--visual-games-max", type=int, default=1)
    parser.add_argument("--student-eval-games", type=int, default=0, help="student self-play games after each iteration; observation only")
    parser.add_argument("--student-eval-mcts-sims", type=int, default=40, help="MCTS simulations for student observation games")
    return parser.parse_args()


def teacher_game(
    teacher_a: GomokuZeroPolicy,
    teacher_b: GomokuZeroPolicy,
    board_size: int,
    args: argparse.Namespace,
    rng: random.Random,
    game_visualizer: TrainingGameVisualizer | None = None,
    iteration: int = 0,
    game_index: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    board = Board(size=board_size)
    player = BLACK
    step = 0
    records: list[tuple[int, np.ndarray, int, np.ndarray]] = []
    frames: list[dict] = []
    teachers = {
        BLACK: MCTS(teacher_a, MCTSConfig(simulations=args.mcts_sims, cpuct=args.cpuct)),
        -BLACK: MCTS(teacher_b, MCTSConfig(simulations=args.mcts_sims, cpuct=args.cpuct)),
    }

    while True:
        step += 1
        temp = 1.0 if step <= args.temp_threshold else 0.0
        pi = teachers[player].action_probs(board, player, temp=temp)
        action = sample_action(pi, rng) if temp > 0 else int(np.argmax(pi))
        move = divmod(action, board.size)
        records.append((step, canonical_array(board, player), player, pi.copy()))
        board.place(move[0], move[1], player)

        if game_visualizer:
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

    if game_visualizer:
        game_visualizer.render(
            frames,
            iteration=iteration,
            phase="gomokuzero_teacher",
            game_index=game_index,
            winner=winner,
        )

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    learn_after_step = args.temp_threshold if args.learn_after_step is None else args.learn_after_step
    for record_step, canonical, record_player, pi in records:
        if record_step <= learn_after_step:
            continue
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if record_player == winner else -1.0
        examples.extend(augment_example(canonical, pi, value))
    return examples


def student_observation_game(
    student: NeuralPolicy,
    board_size: int,
    args: argparse.Namespace,
    *,
    game_visualizer: TrainingGameVisualizer | None = None,
    iteration: int = 0,
    game_index: int = 0,
) -> tuple[int, int]:
    board = Board(size=board_size)
    player = BLACK
    step = 0
    frames: list[dict] = []
    mcts = {
        BLACK: MCTS(student, MCTSConfig(simulations=args.student_eval_mcts_sims, cpuct=args.cpuct)),
        -BLACK: MCTS(student, MCTSConfig(simulations=args.student_eval_mcts_sims, cpuct=args.cpuct)),
    }

    while True:
        step += 1
        pi = mcts[player].action_probs(board, player, temp=0.0)
        action = int(np.argmax(pi))
        move = divmod(action, board.size)
        board.place(move[0], move[1], player)
        if game_visualizer:
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

    if game_visualizer:
        game_visualizer.render(
            frames,
            iteration=iteration,
            phase="student_self_eval",
            game_index=game_index,
            winner=winner,
        )
    return winner, step


def load_student(args: argparse.Namespace, board_size: int) -> NeuralPolicy:
    if args.student.exists() and not args.fresh:
        student = NeuralPolicy.load(args.student, device=args.device)
        if student.board_size != board_size:
            raise ValueError(f"student is {student.board_size}x{student.board_size}, teacher is {board_size}x{board_size}")
        print(f"Loaded student: {args.student}")
        return student

    student = NeuralPolicy.create(
        board_size=board_size,
        channels=args.channels,
        blocks=args.blocks,
        architecture=args.architecture,
        device=args.device,
    )
    print(f"Started new {args.architecture} student on {student.device}: {args.student}")
    return student


def main() -> None:
    args = parse_args()
    args.teacher_b = args.teacher_b or args.teacher_a
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    board_size_a = infer_gomokuzero_checkpoint_board_size(args.teacher_a)
    board_size_b = infer_gomokuzero_checkpoint_board_size(args.teacher_b)
    if board_size_a != board_size_b:
        raise ValueError(f"teacher sizes differ: {board_size_a} vs {board_size_b}")

    student = load_student(args, board_size_a)
    teacher_a = GomokuZeroPolicy.load(args.teacher_a, board_size_a, device=str(student.device))
    teacher_b = GomokuZeroPolicy.load(args.teacher_b, board_size_a, device=str(student.device))
    optimizer = optim.AdamW(student.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=args.replay_size)
    visualizer = TrainingVisualizer(args.visual_dir, args.visual_name) if args.visualize else None
    game_visualizer = (
        TrainingGameVisualizer(
            args.visual_games_dir,
            every=args.visual_games_every,
            max_games_per_iteration=args.visual_games_max,
        )
        if args.visualize_games
        else None
    )

    for iteration in range(1, args.iterations + 1):
        new_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        learn_after_step = args.temp_threshold if args.learn_after_step is None else args.learn_after_step
        for game_index in range(args.games):
            recorder = (
                game_visualizer
                if game_visualizer and game_visualizer.should_capture(iteration, game_index)
                else None
            )
            examples = teacher_game(
                teacher_a,
                teacher_b,
                board_size_a,
                args,
                rng,
                game_visualizer=recorder,
                iteration=iteration,
                game_index=game_index,
            )
            new_examples.extend(examples)
            print(
                f"iter {iteration} teacher-game {game_index + 1}/{args.games} "
                f"| distill after step {learn_after_step} | examples {len(examples)}",
                flush=True,
            )

        replay.extend(new_examples)
        policy_loss, value_loss = train_batches(student, list(replay), args, optimizer)
        student.save(
            args.student,
            meta={
                "distilled_from": [str(args.teacher_a), str(args.teacher_b)],
                "iteration": iteration,
                "examples": len(replay),
                "new_examples": len(new_examples),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "mcts_sims": args.mcts_sims,
            },
        )
        if visualizer:
            visualizer.record(
                iteration=iteration,
                examples=len(replay),
                new_examples=len(new_examples),
                policy_loss=policy_loss,
                value_loss=value_loss,
                games=args.games,
                mcts_sims=args.mcts_sims,
                learn_after_step=learn_after_step,
            )

        eval_results: list[tuple[int, int]] = []
        for eval_index in range(args.student_eval_games):
            recorder = (
                game_visualizer
                if game_visualizer and game_visualizer.should_capture(iteration, eval_index)
                else None
            )
            winner, steps = student_observation_game(
                student,
                board_size_a,
                args,
                game_visualizer=recorder,
                iteration=iteration,
                game_index=eval_index,
            )
            eval_results.append((winner, steps))
            winner_text = "draw" if winner == 0 else "black" if winner == BLACK else "white"
            print(
                f"iter {iteration} student-self-eval {eval_index + 1}/{args.student_eval_games} "
                f"| winner {winner_text} | steps {steps}",
                flush=True,
            )

        print(
            f"iter {iteration}/{args.iterations} | examples {len(replay)} "
            f"| policy_loss {policy_loss:.4f} value_loss {value_loss:.4f} | saved {args.student}",
            flush=True,
        )


if __name__ == "__main__":
    main()
