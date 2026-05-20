from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.model import LinearPolicy
from gomoku_ai.game_visualizer import TrainingGameVisualizer
from gomoku_ai.neural import NeuralPolicy, infer_gomokuzero_checkpoint_board_size, infer_legacy_checkpoint_board_size
from gomoku_ai.neural_train import NeuralTrainConfig, train_neural
from gomoku_ai.self_play import TrainConfig as LinearTrainConfig
from gomoku_ai.self_play import train as train_linear
from gomoku_ai.training_visualizer import TrainingVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gomoku AI with neural self-play reinforcement learning.")
    parser.add_argument("--legacy-linear", action="store_true", help="use the old lightweight linear trainer")

    parser.add_argument("--size", type=int, default=15, help="board size")
    parser.add_argument("--n-in-row", type=int, default=5, help="stones in a row required to win")
    parser.add_argument("--model", type=Path, default=Path("models/gomoku_resnet.pth.tar"), help="model path")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing model and start from scratch")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto when omitted")

    parser.add_argument("--architecture", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--channels", type=int, default=96, help="network channels")
    parser.add_argument("--blocks", type=int, default=6, help="number of residual blocks")
    parser.add_argument("--iterations", type=int, default=5, help="training iterations")
    parser.add_argument("--self-play-games", type=int, default=8, help="self-play games per iteration")
    parser.add_argument("--opponent-games", type=int, default=0, help="games per iteration against frozen opponent checkpoints")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--mcts-sims", type=int, default=80)
    parser.add_argument("--opponent-mcts-sims", type=int, default=80)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--temp-threshold", type=int, default=4)
    parser.add_argument(
        "--learn-after-step",
        type=int,
        default=None,
        help="only learn moves after this step; defaults to --temp-threshold",
    )
    parser.add_argument(
        "--learn-opponent-wins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in opponent-play, also learn the opponent's moves from games the learner lost",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--reward-weight",
        type=float,
        default=0.0,
        help="optional tactical shape reward weight; try 0.05 to 0.2",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualize", action="store_true", help="write training CSV and PNG curves")
    parser.add_argument("--visual-dir", type=Path, default=Path("runs"), help="visualization output directory")
    parser.add_argument("--visual-name", type=str, default=None, help="visualization file stem")
    parser.add_argument("--visualize-games", action="store_true", help="save training game GIFs and final-board PNGs")
    parser.add_argument("--visual-games-dir", type=Path, default=Path("runs/games"), help="training game visualization output directory")
    parser.add_argument("--visual-games-every", type=int, default=1, help="capture games every N iterations")
    parser.add_argument("--visual-games-max", type=int, default=1, help="max games captured per selected iteration")
    parser.add_argument(
        "--alphazero-opponent",
        type=Path,
        default=None,
        help="path to alphazero-gomoku checkpoint, e.g. alphazero-gomoku/temp/best.pth.tar",
    )
    parser.add_argument(
        "--gomokuzero-opponent",
        type=Path,
        default=None,
        help="path to GomokuZeroAI checkpoint, e.g. GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt",
    )
    parser.add_argument(
        "--match-opponent-size",
        action="store_true",
        help="set --size automatically from the provided opponent checkpoint action head",
    )

    # Old trainer compatibility flags.
    parser.add_argument("--episodes", type=int, default=2000, help=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=0.99, help=argparse.SUPPRESS)
    parser.add_argument("--epsilon", type=float, default=0.15, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-radius", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--save-every", type=int, default=100, help=argparse.SUPPRESS)
    return parser.parse_args()


def make_visualizer(args: argparse.Namespace, run_name: str) -> TrainingVisualizer | None:
    if not args.visualize:
        return None
    return TrainingVisualizer(output_dir=args.visual_dir, run_name=args.visual_name or run_name)


def make_game_visualizer(args: argparse.Namespace) -> TrainingGameVisualizer | None:
    if not args.visualize_games:
        return None
    return TrainingGameVisualizer(
        output_dir=args.visual_games_dir,
        every=args.visual_games_every,
        max_games_per_iteration=args.visual_games_max,
    )


def run_linear(args: argparse.Namespace) -> None:
    model_path = args.model
    if model_path.suffix != ".json":
        model_path = Path("models/gomoku_policy.json")
    if model_path.exists() and not args.fresh:
        policy = LinearPolicy.load(model_path)
        print(f"Loaded existing linear model: {model_path}")
    else:
        policy = LinearPolicy()
        print("Started from built-in tactical linear weights.")

    config = LinearTrainConfig(
        episodes=args.episodes,
        size=args.size,
        n_in_row=args.n_in_row,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        temperature=args.temperature,
        candidate_radius=args.candidate_radius,
        seed=args.seed,
        save_every=args.save_every,
        model_path=model_path,
    )
    visualizer = make_visualizer(args, "linear_training")
    stats = train_linear(
        policy,
        config,
        metrics_callback=visualizer.record if visualizer else None,
    )
    print("Linear training complete.")
    print(f"Saved: {model_path} | black={stats['black_wins']} white={stats['white_wins']} draw={stats['draws']}")


def run_neural(args: argparse.Namespace) -> None:
    if args.match_opponent_size:
        if args.gomokuzero_opponent:
            args.size = infer_gomokuzero_checkpoint_board_size(args.gomokuzero_opponent)
            print(f"Matched board size to GomokuZeroAI opponent checkpoint: {args.size}x{args.size}")
        elif args.alphazero_opponent:
            args.size = infer_legacy_checkpoint_board_size(args.alphazero_opponent)
            print(f"Matched board size to AlphaZero opponent checkpoint: {args.size}x{args.size}")
    if args.model.exists() and not args.fresh:
        learner = NeuralPolicy.load(args.model, device=args.device)
        if learner.board_size != args.size:
            raise ValueError(f"loaded model is {learner.board_size}x{learner.board_size}, but --size is {args.size}")
        print(f"Loaded neural model: {args.model}")
    else:
        learner = NeuralPolicy.create(
            board_size=args.size,
            channels=args.channels,
            blocks=args.blocks,
            architecture=args.architecture,
            device=args.device,
        )
        print(f"Started new {args.architecture} policy-value network on {learner.device}.")

    config = NeuralTrainConfig(
        size=args.size,
        n_in_row=args.n_in_row,
        iterations=args.iterations,
        self_play_games=args.self_play_games,
        opponent_games=args.opponent_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        mcts_sims=args.mcts_sims,
        opponent_mcts_sims=args.opponent_mcts_sims,
        cpuct=args.cpuct,
        temp_threshold=args.temp_threshold,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reward_weight=args.reward_weight,
        learn_after_step=args.learn_after_step,
        learn_opponent_wins=args.learn_opponent_wins,
        seed=args.seed,
        model_path=args.model,
        opponent_checkpoint=args.alphazero_opponent,
        gomokuzero_checkpoint=args.gomokuzero_opponent,
        device=args.device,
    )
    visualizer = make_visualizer(args, "neural_training")
    game_visualizer = make_game_visualizer(args)
    train_neural(learner, config, visualizer=visualizer, game_visualizer=game_visualizer)
    print(f"Neural training complete. Saved: {args.model}")


def main() -> None:
    args = parse_args()
    if args.legacy_linear:
        run_linear(args)
    else:
        run_neural(args)


if __name__ == "__main__":
    main()
