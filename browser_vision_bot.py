from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import pyautogui
except ImportError as exc:  # pragma: no cover - user environment message
    raise SystemExit("Missing dependency: pyautogui. Install it with: pip install pyautogui pillow") from exc

from PIL import Image, ImageDraw

from gomoku_ai.game import BLACK, WHITE, Board
from gomoku_ai.model import LinearPolicy
from gomoku_ai.neural import NeuralPolicy


DEFAULT_BOARD_SIZE = 15
AI_PLAYER = BLACK
OPPONENT_PLAYER = WHITE
DEFAULT_NEURAL_MODEL = Path("models/gomoku_transformer.pth.tar")
DEFAULT_LINEAR_MODEL = Path("models/gomoku_policy.json")


@dataclass(frozen=True)
class GridGeometry:
    xs: list[int]
    ys: list[int]
    board_size: int

    @property
    def cell(self) -> float:
        return (self.xs[-1] - self.xs[0] + self.ys[-1] - self.ys[0]) / (2 * (self.board_size - 1))

    def point(self, row: int, col: int) -> tuple[int, int]:
        return self.xs[col], self.ys[row]


@dataclass
class DetectedPosition:
    board: Board
    red_count: int
    green_count: int
    geometry: GridGeometry


def is_grid_pixel(r: int, g: int, b: int) -> bool:
    if is_red_stone(r, g, b) or is_green_stone(r, g, b):
        return False
    return (
        18 <= r <= 105
        and 24 <= g <= 115
        and 30 <= b <= 135
        and g >= r - 8
        and b >= r - 6
        and max(r, g, b) - min(r, g, b) <= 82
    )


def is_red_stone(r: int, g: int, b: int) -> bool:
    return r >= 190 and 55 <= g <= 145 and 70 <= b <= 165 and r > g + 55 and r > b + 35


def is_green_stone(r: int, g: int, b: int) -> bool:
    return r <= 85 and g >= 140 and b >= 110 and g > r + 55 and b > r + 35


def cluster_peaks(counts: list[int], threshold: int) -> list[int]:
    peaks: list[int] = []
    start: int | None = None
    weighted_sum = 0
    weight_total = 0

    for index, count in enumerate(counts):
        if count >= threshold:
            if start is None:
                start = index
                weighted_sum = 0
                weight_total = 0
            weighted_sum += index * count
            weight_total += count
        elif start is not None:
            peaks.append(round(weighted_sum / max(weight_total, 1)))
            start = None

    if start is not None:
        peaks.append(round(weighted_sum / max(weight_total, 1)))
    return peaks


def adaptive_line_threshold(counts: list[int], minimum: int) -> int:
    non_zero = sorted(count for count in counts if count > 0)
    if not non_zero:
        return minimum
    # The board contributes about 17 peaks. Using roughly the 30th strongest
    # line avoids being dominated by browser borders or top-bar separators.
    anchor_index = max(0, len(non_zero) - 30)
    anchor = non_zero[anchor_index]
    return max(minimum, int(anchor * 0.42))


def best_grid_sequence(peaks: list[int], expected: int) -> list[int] | None:
    if len(peaks) < expected:
        return None

    best: tuple[float, list[int]] | None = None
    for start in range(0, len(peaks) - expected + 1):
        window = peaks[start : start + expected]
        gaps = [b - a for a, b in zip(window, window[1:])]
        mean_gap = sum(gaps) / len(gaps)
        if not 20 <= mean_gap <= 70:
            continue
        if window[-1] - window[0] < 320:
            continue
        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        score = variance + abs(mean_gap - round(mean_gap)) * 0.25
        if best is None or score < best[0]:
            best = (score, window)
    return best[1] if best else None


def count_grid_lines(image: Image.Image, *, stride: int = 1) -> tuple[list[int], list[int]]:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    col_counts = [0] * width
    row_counts = [0] * height

    for y in range(0, height, stride):
        row_hit = 0
        for x in range(0, width, stride):
            if is_grid_pixel(*pixels[x, y]):
                col_counts[x] += 1
                row_hit += 1
        row_counts[y] = row_hit

    if stride > 1:
        col_counts = [count * stride for count in col_counts]
        row_counts = [count * stride for count in row_counts]
    return col_counts, row_counts


def locate_board_from_image(image: Image.Image, board_size: int) -> GridGeometry:
    col_counts, row_counts = count_grid_lines(image)
    vertical_threshold = adaptive_line_threshold(col_counts, minimum=28)
    horizontal_threshold = adaptive_line_threshold(row_counts, minimum=28)
    vertical_peaks = cluster_peaks(col_counts, threshold=vertical_threshold)
    horizontal_peaks = cluster_peaks(row_counts, threshold=horizontal_threshold)
    xs = best_grid_sequence(vertical_peaks, expected=board_size)
    ys = best_grid_sequence(horizontal_peaks, expected=board_size)
    if xs is None or ys is None:
        raise RuntimeError(
            f"没有自动定位到 {board_size}x{board_size} 棋盘，请使用 --board-left --board-top --cell 手动指定；"
            f"debug: x_peaks={len(vertical_peaks)} y_peaks={len(horizontal_peaks)} "
            f"x_threshold={vertical_threshold} y_threshold={horizontal_threshold}"
        )
    return GridGeometry(xs=xs, ys=ys, board_size=board_size)


def manual_geometry(left: int, top: int, cell: float, board_size: int) -> GridGeometry:
    xs = [round(left + col * cell) for col in range(board_size)]
    ys = [round(top + row * cell) for row in range(board_size)]
    return GridGeometry(xs=xs, ys=ys, board_size=board_size)


def classify_cell(pixels, width: int, height: int, x: int, y: int, radius: int) -> int:
    red_hits = 0
    green_hits = 0
    total = 0
    r2 = radius * radius

    for py in range(max(0, y - radius), min(height, y + radius + 1)):
        for px in range(max(0, x - radius), min(width, x + radius + 1)):
            if (px - x) * (px - x) + (py - y) * (py - y) > r2:
                continue
            color = pixels[px, py]
            total += 1
            if is_red_stone(*color):
                red_hits += 1
            elif is_green_stone(*color):
                green_hits += 1

    min_hits = max(16, int(total * 0.08))
    if red_hits >= min_hits and red_hits > green_hits * 1.4:
        return OPPONENT_PLAYER
    if green_hits >= min_hits and green_hits > red_hits * 1.4:
        return AI_PLAYER
    return 0


def detect_position(image: Image.Image, geometry: GridGeometry) -> DetectedPosition:
    rgb_image = image.convert("RGB")
    pixels = rgb_image.load()
    width, height = rgb_image.size
    board = Board(size=geometry.board_size)
    radius = max(7, min(16, round(geometry.cell * 0.36)))
    red_count = 0
    green_count = 0

    for row in range(geometry.board_size):
        for col in range(geometry.board_size):
            x, y = geometry.point(row, col)
            value = classify_cell(pixels, width, height, x, y, radius)
            if value == OPPONENT_PLAYER:
                board.place(row, col, OPPONENT_PLAYER)
                red_count += 1
            elif value == AI_PLAYER:
                board.place(row, col, AI_PLAYER)
                green_count += 1
    return DetectedPosition(board=board, red_count=red_count, green_count=green_count, geometry=geometry)


def board_has_winner(board: Board) -> bool:
    for row in range(board.size):
        for col in range(board.size):
            if board.get(row, col) and board.check_winner(row, col):
                return True
    return False


def choose_ai_move(policy, board: Board) -> tuple[int, int] | None:
    if board_has_winner(board) or board.is_full():
        return None
    if isinstance(policy, LinearPolicy):
        move, _ = policy.choose_move(board, AI_PLAYER, temperature=0.0, candidate_radius=2)
    else:
        move, _ = policy.choose_move(board, AI_PLAYER)
    return move


def save_debug_image(image: Image.Image, position: DetectedPosition, path: Path) -> None:
    debug = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)
    radius = max(5, round(position.geometry.cell * 0.18))
    for row in range(position.geometry.board_size):
        for col in range(position.geometry.board_size):
            x, y = position.geometry.point(row, col)
            value = position.board.get(row, col)
            outline = "#ef5f78" if value == OPPONENT_PLAYER else "#18c3a7" if value == AI_PLAYER else "#64748b"
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=outline, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize a browser Gomoku board and click AI moves after red stones appear."
    )
    parser.add_argument("--model", type=Path, default=None, help="model path; defaults to neural .pth.tar then JSON fallback")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE, help="number of intersections per side")
    parser.add_argument("--interval", type=float, default=0.8, help="screenshot polling interval in seconds")
    parser.add_argument("--move-delay", type=float, default=0.25, help="delay before clicking an AI move")
    parser.add_argument("--board-left", type=int, default=None, help="manual x coordinate of the first grid line")
    parser.add_argument("--board-top", type=int, default=None, help="manual y coordinate of the first grid line")
    parser.add_argument("--cell", type=float, default=None, help="manual grid cell size in pixels")
    parser.add_argument("--dry-run", action="store_true", help="print the chosen move without clicking")
    parser.add_argument("--once", action="store_true", help="recognize once, optionally click once, then exit")
    parser.add_argument("--debug-image", type=Path, default=None, help="save a screenshot with detected cells marked")
    parser.add_argument(
        "--fail-screenshot",
        type=Path,
        default=Path("debug/last_screenshot.png"),
        help="save the raw screenshot here if board location fails",
    )
    parser.add_argument("--click-offset-x", type=int, default=0, help="extra x offset added to click position")
    parser.add_argument("--click-offset-y", type=int, default=0, help="extra y offset added to click position")
    return parser.parse_args()


def get_geometry(args: argparse.Namespace, image: Image.Image) -> GridGeometry:
    manual_values = (args.board_left, args.board_top, args.cell)
    if any(value is not None for value in manual_values):
        if not all(value is not None for value in manual_values):
            raise SystemExit("--board-left, --board-top and --cell must be provided together")
        return manual_geometry(args.board_left, args.board_top, args.cell, args.board_size)
    return locate_board_from_image(image, args.board_size)


def main() -> None:
    pyautogui.FAILSAFE = True
    args = parse_args()
    model_path = args.model or (DEFAULT_NEURAL_MODEL if DEFAULT_NEURAL_MODEL.exists() else DEFAULT_LINEAR_MODEL)
    if model_path.exists() and model_path.suffix == ".json":
        policy = LinearPolicy.load(model_path)
    elif model_path.exists():
        policy = NeuralPolicy.load(model_path)
        if policy.board_size != args.board_size:
            raise SystemExit(f"Model is {policy.board_size}x{policy.board_size}, but --board-size is {args.board_size}")
    else:
        policy = LinearPolicy()
    print(f"Model: {model_path if model_path.exists() else 'built-in linear weights'}")
    print("Move the mouse to the top-left screen corner to abort. Press Ctrl+C in this terminal to stop.")

    geometry: GridGeometry | None = None
    last_red_count: int | None = None
    last_green_count: int | None = None

    while True:
        screenshot = pyautogui.screenshot()
        if not isinstance(screenshot, Image.Image):
            screenshot = Image.frombytes("RGB", screenshot.size, screenshot.tobytes())

        if geometry is None:
            try:
                geometry = get_geometry(args, screenshot)
            except Exception:
                if args.fail_screenshot:
                    args.fail_screenshot.parent.mkdir(parents=True, exist_ok=True)
                    screenshot.save(args.fail_screenshot)
                    print(f"Saved failed screenshot: {args.fail_screenshot}")
                raise
            print(
                f"Board detected: left={geometry.xs[0]} top={geometry.ys[0]} "
                f"right={geometry.xs[-1]} bottom={geometry.ys[-1]} cell={geometry.cell:.2f}"
            )

        position = detect_position(screenshot, geometry)
        if args.debug_image:
            save_debug_image(screenshot, position, args.debug_image)

        red_changed = last_red_count is not None and position.red_count > last_red_count
        counts_changed = (
            last_red_count != position.red_count
            or last_green_count != position.green_count
            or args.once
        )
        if counts_changed:
            print(f"Detected stones: red={position.red_count} green={position.green_count}")

        should_move = args.once or red_changed
        if should_move:
            move = choose_ai_move(policy, position.board)
            if move is None:
                print("Game is finished or board is full; no move clicked.")
            else:
                row, col = move
                x, y = position.geometry.point(row, col)
                x += args.click_offset_x
                y += args.click_offset_y
                print(f"AI move: row={row + 1} col={col + 1} screen=({x}, {y})")
                if not args.dry_run:
                    time.sleep(args.move_delay)
                    pyautogui.click(x=x, y=y, button="left")
                    last_green_count = position.green_count + 1

        last_red_count = position.red_count
        if last_green_count is None:
            last_green_count = position.green_count
        elif not should_move:
            last_green_count = position.green_count

        if args.once:
            break
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()
