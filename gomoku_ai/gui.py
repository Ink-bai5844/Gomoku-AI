from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .game import BLACK, WHITE, Board, opponent, player_name
from .model import LinearPolicy
from .neural import NeuralPolicy


BOARD_SIZE = 15
N_IN_ROW = 5
MODEL_DIR = Path("models")
NEURAL_MODEL_PATH = Path("models/gomoku_resnet.pth.tar")
LINEAR_MODEL_PATH = Path("models/gomoku_policy.json")


def find_neural_model_paths() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*.pth.tar"), key=lambda path: path.stat().st_mtime, reverse=True)


class GomokuApp(tk.Tk):
    def __init__(self, preferred_model_path: str | Path | None = None) -> None:
        super().__init__()
        self.title("Gomoku AI - Self Play RL")
        self.resizable(False, False)

        self.cell = 36
        self.padding = 32
        self.canvas_size = self.padding * 2 + self.cell * (BOARD_SIZE - 1)

        self.board = Board(size=BOARD_SIZE, n_in_row=N_IN_ROW)
        self.model_path: Path | None = None
        self.policy = self._load_default_policy(preferred_model_path)
        self.human_color = BLACK
        self.current_player = BLACK
        self.game_over = False

        self.human_color_var = tk.StringVar(value="black")
        self.status_var = tk.StringVar(value="黑棋先行")
        self.model_var = tk.StringVar(value=self._model_label())

        self._build_ui()
        self.draw_board()

    def _default_model_candidates(self, preferred_model_path: str | Path | None) -> list[Path]:
        candidates: list[Path] = []
        if preferred_model_path:
            candidates.append(Path(preferred_model_path))
        candidates.extend(find_neural_model_paths())
        candidates.append(LINEAR_MODEL_PATH)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            normalized = path.resolve() if path.exists() else path.absolute()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(path)
        return deduped

    def _load_default_policy(self, preferred_model_path: str | Path | None = None) -> LinearPolicy:
        for path in self._default_model_candidates(preferred_model_path):
            if not path.exists():
                continue
            try:
                if path.suffix == ".json":
                    policy = LinearPolicy.load(path)
                    self.model_path = path
                    return policy
                policy = NeuralPolicy.load(path)
                if policy.board_size != BOARD_SIZE:
                    continue
                self.model_path = path
                return policy
            except Exception:
                continue
        self.model_path = None
        return LinearPolicy()

    def _model_label(self) -> str:
        if self.model_path:
            return f"模型: {self.model_path}"
        return "模型: 内置初始权重"

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#d8ae63",
            highlightthickness=1,
            highlightbackground="#7a5528",
        )
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_click)

        panel = ttk.Frame(root, padding=(14, 0, 0, 0), width=220)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        ttk.Label(panel, text="人类执子").grid(row=0, column=0, sticky="w", pady=(2, 6))
        ttk.Radiobutton(
            panel,
            text="黑棋",
            variable=self.human_color_var,
            value="black",
            command=self.new_game,
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            panel,
            text="白棋",
            variable=self.human_color_var,
            value="white",
            command=self.new_game,
        ).grid(row=2, column=0, sticky="w", pady=(0, 12))

        ttk.Button(panel, text="新对局", command=self.new_game).grid(row=3, column=0, sticky="ew", pady=4)
        ttk.Button(panel, text="加载模型", command=self.load_model).grid(row=4, column=0, sticky="ew", pady=4)

        ttk.Separator(panel).grid(row=5, column=0, sticky="ew", pady=14)
        ttk.Label(panel, textvariable=self.status_var, wraplength=200).grid(row=6, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.model_var, wraplength=200).grid(row=7, column=0, sticky="w", pady=(16, 0))

    def new_game(self) -> None:
        self.board.reset()
        self.human_color = BLACK if self.human_color_var.get() == "black" else WHITE
        self.current_player = BLACK
        self.game_over = False
        self.status_var.set("黑棋先行")
        self.draw_board()
        if self.current_player != self.human_color:
            self.status_var.set("AI 思考中...")
            self.after(350, self.ai_move)

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择训练好的模型",
            filetypes=[("Gomoku model", "*.pth.tar *.json"), ("PyTorch model", "*.pth.tar"), ("JSON model", "*.json"), ("All files", "*.*")],
            initialdir=str(Path("models").resolve()) if Path("models").exists() else str(Path.cwd()),
        )
        if not path:
            return
        try:
            if path.endswith(".json"):
                self.policy = LinearPolicy.load(path)
            else:
                policy = NeuralPolicy.load(path)
                if policy.board_size != BOARD_SIZE:
                    raise ValueError(f"模型是 {policy.board_size}x{policy.board_size}，当前棋盘是 {BOARD_SIZE}x{BOARD_SIZE}")
                self.policy = policy
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self.model_path = Path(path)
        self.model_var.set(self._model_label())
        self.status_var.set("模型已加载，点击新对局开始")

    def draw_board(self) -> None:
        self.canvas.delete("all")
        p = self.padding
        end = p + self.cell * (BOARD_SIZE - 1)

        for index in range(BOARD_SIZE):
            offset = p + index * self.cell
            self.canvas.create_line(p, offset, end, offset, fill="#3d2a12", width=1)
            self.canvas.create_line(offset, p, offset, end, fill="#3d2a12", width=1)

        for row, col in self._star_points():
            x, y = self._to_canvas(row, col)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#3d2a12", outline="")

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stone = self.board.get(row, col)
                if stone:
                    self._draw_stone(row, col, stone)

        if self.board.last_move:
            row, col = self.board.last_move
            x, y = self._to_canvas(row, col)
            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, outline="#c4372f", width=2)

    def _star_points(self) -> list[tuple[int, int]]:
        if BOARD_SIZE == 15:
            return [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        center = BOARD_SIZE // 2
        return [(center, center)]

    def _to_canvas(self, row: int, col: int) -> tuple[int, int]:
        return self.padding + col * self.cell, self.padding + row * self.cell

    def _from_canvas(self, x: int, y: int) -> tuple[int, int] | None:
        col = round((x - self.padding) / self.cell)
        row = round((y - self.padding) / self.cell)
        if not self.board.in_bounds(row, col):
            return None
        cx, cy = self._to_canvas(row, col)
        if abs(x - cx) > self.cell * 0.42 or abs(y - cy) > self.cell * 0.42:
            return None
        return row, col

    def _draw_stone(self, row: int, col: int, player: int) -> None:
        x, y = self._to_canvas(row, col)
        radius = self.cell * 0.42
        if player == BLACK:
            fill, outline = "#1e1e1e", "#050505"
        else:
            fill, outline = "#f5f1e7", "#a79c89"
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, width=2)

    def on_click(self, event: tk.Event) -> None:
        if self.game_over or self.current_player != self.human_color:
            return
        move = self._from_canvas(event.x, event.y)
        if move is None:
            return
        row, col = move
        if not self.board.place(row, col, self.current_player):
            return
        self.finish_turn()
        if not self.game_over:
            self.status_var.set("AI 思考中...")
            self.after(250, self.ai_move)

    def ai_move(self) -> None:
        if self.game_over or self.current_player == self.human_color:
            return
        try:
            if isinstance(self.policy, NeuralPolicy):
                move, _ = self.policy.choose_move(self.board, self.current_player)
            else:
                move, _ = self.policy.choose_move(
                    self.board,
                    self.current_player,
                    epsilon=0.0,
                    temperature=0.0,
                    candidate_radius=2,
                )
        except RuntimeError:
            self.game_over = True
            self.status_var.set("平局")
            return
        self.board.place(move[0], move[1], self.current_player)
        self.finish_turn()

    def finish_turn(self) -> None:
        self.draw_board()
        winner = self.board.winner()
        if winner:
            self.game_over = True
            side = "黑棋" if winner == BLACK else "白棋"
            owner = "你" if winner == self.human_color else "AI"
            self.status_var.set(f"{side}获胜，{owner}赢了")
            return
        if self.board.is_full():
            self.game_over = True
            self.status_var.set("平局")
            return
        self.current_player = opponent(self.current_player)
        side = "黑棋" if self.current_player == BLACK else "白棋"
        if self.current_player == self.human_color:
            self.status_var.set(f"轮到你下，当前 {side}")
        else:
            self.status_var.set(f"轮到 AI，当前 {side}")


def main(preferred_model_path: str | Path | None = None) -> None:
    app = GomokuApp(preferred_model_path)
    app.mainloop()
