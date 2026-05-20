from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TrainingGameVisualizer:
    output_dir: Path
    every: int = 1
    max_games_per_iteration: int = 1
    cell: int = 34
    margin: int = 34
    bottom: int = 54

    def should_capture(self, iteration: int, game_index: int) -> bool:
        if self.every <= 0:
            return False
        if iteration % self.every != 0:
            return False
        return game_index < self.max_games_per_iteration

    def render(
        self,
        frames: list[dict[str, Any]],
        *,
        iteration: int,
        phase: str,
        game_index: int,
        winner: int,
    ) -> None:
        if not frames:
            return
        target_dir = self.output_dir / f"iter_{iteration:04d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{phase}_game_{game_index + 1:03d}"
        images = [self._draw_frame(frame, winner=winner, phase=phase) for frame in frames]
        images[-1].save(target_dir / f"{stem}_final.png")
        images[0].save(
            target_dir / f"{stem}.gif",
            save_all=True,
            append_images=images[1:],
            duration=360,
            loop=0,
        )

    def _draw_frame(self, frame: dict[str, Any], *, winner: int, phase: str) -> Image.Image:
        board = frame["board"]
        size = len(board)
        canvas = self.margin * 2 + self.cell * (size - 1)
        width = canvas
        height = canvas + self.bottom
        image = Image.new("RGB", (width, height), "#d8ae63")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        line_color = "#3d2a12"
        for index in range(size):
            offset = self.margin + index * self.cell
            draw.line((self.margin, offset, self.margin + self.cell * (size - 1), offset), fill=line_color, width=1)
            draw.line((offset, self.margin, offset, self.margin + self.cell * (size - 1)), fill=line_color, width=1)

        for row, col in self._star_points(size):
            x, y = self._point(row, col)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=line_color)

        for row in range(size):
            for col in range(size):
                stone = board[row][col]
                if stone == 0:
                    continue
                x, y = self._point(row, col)
                radius = int(self.cell * 0.42)
                if stone == 1:
                    fill, outline = "#1e1e1e", "#050505"
                else:
                    fill, outline = "#f5f1e7", "#a79c89"
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)

        move = frame.get("move")
        if move:
            row, col = move
            x, y = self._point(row, col)
            draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline="#c4372f", width=2)

        step = frame.get("step", 0)
        player = frame.get("player", 0)
        winner_text = "draw" if winner == 0 else "black wins" if winner == 1 else "white wins"
        player_text = "black" if player == 1 else "white"
        label = f"{phase} | step {step} | {player_text} moved | {winner_text}"
        draw.rectangle((0, canvas, width, height), fill="#20242a")
        draw.text((12, canvas + 18), label, fill="#f3f4f6", font=font)
        return image

    def _point(self, row: int, col: int) -> tuple[int, int]:
        return self.margin + col * self.cell, self.margin + row * self.cell

    @staticmethod
    def _star_points(size: int) -> list[tuple[int, int]]:
        if size == 15:
            return [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        if size == 9:
            return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        center = size // 2
        return [(center, center)]
