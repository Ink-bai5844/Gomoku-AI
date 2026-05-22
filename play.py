import argparse
from pathlib import Path

from gomoku_ai.gui import find_neural_model_paths, main

DEFAULT_TEMPERATURE = 0.25


def preferred_model_path() -> Path | None:
    neural_models = find_neural_model_paths()
    return neural_models[0] if neural_models else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Gomoku against the local AI.")
    parser.add_argument(
        "--temperature",
        "--temprature",
        dest="temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "AI move sampling temperature. Use 0 for greedy/best move; "
            "higher values make play more varied. Default: %(default)s"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(preferred_model_path(), temperature=args.temperature)
