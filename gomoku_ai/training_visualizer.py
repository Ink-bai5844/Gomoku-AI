from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingVisualizer:
    output_dir: Path
    run_name: str = "training"
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **metrics: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        row = {key: value for key, value in metrics.items() if value is not None}
        self.rows.append(row)
        self._write_csv()
        self._write_plot()

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrainingVisualizer":
        csv_path = Path(path)
        visualizer = cls(output_dir=csv_path.parent, run_name=csv_path.stem)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            visualizer.rows = [dict(row) for row in reader]
        return visualizer

    def replot(self) -> None:
        self._write_plot()

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.csv"

    @property
    def plot_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.png"

    @property
    def counts_plot_path(self) -> Path:
        return self.output_dir / f"{self.run_name}_counts.png"

    def _write_csv(self) -> None:
        if not self.rows:
            return
        keys: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_plot(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        numeric_keys = [
            key
            for key in self.rows[-1]
            if key not in {"iteration", "episode", "phase", "model"}
            and all(_is_number(row.get(key)) for row in self.rows if key in row)
        ]
        if not numeric_keys:
            return

        x_key = "iteration" if "iteration" in self.rows[-1] else "episode" if "episode" in self.rows[-1] else None
        x_values = [row.get(x_key, index + 1) if x_key else index + 1 for index, row in enumerate(self.rows)]

        count_keys = [key for key in numeric_keys if _is_count_metric(key)]
        main_keys = [key for key in numeric_keys if key not in count_keys]
        self._plot_keys(plt, x_values, x_key or "step", main_keys, self.plot_path, self.run_name)
        if count_keys:
            self._plot_keys(
                plt,
                x_values,
                x_key or "step",
                count_keys,
                self.counts_plot_path,
                f"{self.run_name} counts",
            )

    def _plot_keys(self, plt, x_values: list[Any], x_label: str, keys: list[str], path: Path, title: str) -> None:
        if not keys:
            return
        plt.figure(figsize=(10, 6))
        for key in keys:
            y_values = [float(row[key]) if key in row and _is_number(row[key]) else float("nan") for row in self.rows]
            plt.plot(x_values, y_values, marker="o", linewidth=1.4, label=key)
        plt.xlabel(x_label)
        plt.ylabel("value")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_count_metric(key: str) -> bool:
    return key in {
        "examples",
        "new_examples",
        "opponent_examples",
        "self_play_games",
        "opponent_games",
        "games",
        "mcts_sims",
    }
