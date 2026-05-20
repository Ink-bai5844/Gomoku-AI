from pathlib import Path

from gomoku_ai.gui import find_neural_model_paths, main


def preferred_model_path() -> Path | None:
    neural_models = find_neural_model_paths()
    return neural_models[0] if neural_models else None


if __name__ == "__main__":
    main(preferred_model_path())
