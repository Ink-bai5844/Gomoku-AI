from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import BLACK, Board


NEURAL_FORMAT = "gomoku_policy_value_v1"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 15,
        channels: int = 96,
        blocks: int = 6,
        architecture: str = "resnet",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.architecture = architecture

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        if architecture == "transformer":
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.action_size, channels))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=transformer_heads,
                dim_feedforward=channels * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        elif architecture != "resnet":
            raise ValueError("architecture must be 'resnet' or 'transformer'")

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.action_size, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.residual_tower(x)
        if self.architecture == "transformer":
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2) + self.pos_embedding
            tokens = self.transformer(tokens)
            x = tokens.transpose(1, 2).reshape(b, c, h, w)

        logits = self.policy_head(x)
        value = self.value_head(x).view(-1)
        return logits, value


def board_to_tensor(board: Board, player: int, device: torch.device | str) -> torch.Tensor:
    arr = np.array(board.grid, dtype=np.int8)
    own = (arr == player).astype(np.float32)
    opp = (arr == -player).astype(np.float32)
    empty = (arr == 0).astype(np.float32)
    stacked = np.stack([own, opp, empty], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).to(device=device, dtype=torch.float32)


def canonical_batch_to_tensor(boards: np.ndarray, device: torch.device | str) -> torch.Tensor:
    own = (boards == 1).astype(np.float32)
    opp = (boards == -1).astype(np.float32)
    empty = (boards == 0).astype(np.float32)
    stacked = np.stack([own, opp, empty], axis=1)
    return torch.from_numpy(stacked).to(device=device, dtype=torch.float32)


@dataclass
class NeuralPolicy:
    net: PolicyValueNet
    device: torch.device

    @classmethod
    def create(
        cls,
        board_size: int = 15,
        channels: int = 96,
        blocks: int = 6,
        architecture: str = "resnet",
        device: str | None = None,
    ) -> "NeuralPolicy":
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = PolicyValueNet(
            board_size=board_size,
            channels=channels,
            blocks=blocks,
            architecture=architecture,
        ).to(torch_device)
        return cls(net=net, device=torch_device)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "NeuralPolicy":
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        if checkpoint.get("format") != NEURAL_FORMAT:
            raise ValueError("not a Gomoku-AI neural checkpoint")
        config = checkpoint["config"]
        policy = cls.create(
            board_size=int(config["board_size"]),
            channels=int(config["channels"]),
            blocks=int(config["blocks"]),
            architecture=str(config["architecture"]),
            device=str(torch_device),
        )
        policy.net.load_state_dict(checkpoint["state_dict"])
        policy.net.eval()
        return policy

    @property
    def board_size(self) -> int:
        return self.net.board_size

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": NEURAL_FORMAT,
                "config": {
                    "board_size": self.net.board_size,
                    "channels": self.net.stem[0].out_channels,
                    "blocks": len(self.net.residual_tower),
                    "architecture": self.net.architecture,
                },
                "state_dict": self.net.state_dict(),
                "meta": meta or {},
            },
            target,
        )

    def predict(self, board: Board, player: int) -> tuple[np.ndarray, float]:
        if board.size != self.board_size:
            raise ValueError(f"model board size is {self.board_size}, got board size {board.size}")
        self.net.eval()
        with torch.no_grad():
            x = board_to_tensor(board, player, self.device)
            logits, value = self.net(x)
            policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return policy, float(value.cpu().numpy()[0])

    def choose_move(
        self,
        board: Board,
        player: int = BLACK,
        *,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[int, int], list[float]]:
        policy, _ = self.predict(board, player)
        legal = np.zeros(board.size * board.size, dtype=np.float32)
        for row, col in board.legal_moves():
            legal[row * board.size + col] = 1.0
        masked = policy * legal
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            raise RuntimeError("no legal moves available")

        legal_probs = masked[legal_actions].astype(np.float64)
        if float(legal_probs.sum()) <= 0.0 or not np.isfinite(legal_probs).all():
            legal_probs = np.ones(len(legal_actions), dtype=np.float64)

        if temperature <= 0:
            action = int(legal_actions[int(np.argmax(legal_probs))])
            return (action // board.size, action % board.size), []

        adjusted = np.zeros_like(legal_probs, dtype=np.float64)
        positive = legal_probs > 0.0
        scaled = np.log(legal_probs[positive]) / max(float(temperature), 1e-6)
        scaled -= float(np.max(scaled))
        adjusted[positive] = np.exp(scaled)
        total = float(adjusted.sum())
        if total <= 0.0 or not np.isfinite(total):
            adjusted = np.ones(len(legal_actions), dtype=np.float64)
            total = float(adjusted.sum())

        probs = adjusted / total
        choice = rng.choice if rng is not None else np.random.choice
        action = int(choice(legal_actions, p=probs))
        return (action // board.size, action % board.size), []


def infer_legacy_checkpoint_board_size(path: str | Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    action_size = int(state_dict["fc3.weight"].shape[0])
    board_size = int(round(math.sqrt(action_size)))
    if board_size * board_size != action_size:
        raise ValueError(f"cannot infer board size from action size {action_size}")
    return board_size


class LegacyAlphaZeroNet(nn.Module):
    """Network definition compatible with alphazero-gomoku/temp/best.pth.tar."""

    def __init__(self, board_size: int, channels: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.channels = channels
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(channels * (board_size - 4) * (board_size - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1, self.board_size, self.board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, self.channels * (self.board_size - 4) * (self.board_size - 4))
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.dropout, training=self.training)
        return self.fc3(x), torch.tanh(self.fc4(x)).view(-1)


@dataclass
class LegacyAlphaZeroPolicy:
    net: LegacyAlphaZeroNet
    device: torch.device

    @classmethod
    def load(cls, path: str | Path, board_size: int, device: str | None = None) -> "LegacyAlphaZeroPolicy":
        inferred = infer_legacy_checkpoint_board_size(path)
        if inferred != board_size:
            raise ValueError(
                f"alphazero opponent checkpoint is {inferred}x{inferred}, "
                f"but current training board is {board_size}x{board_size}"
            )
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        net = LegacyAlphaZeroNet(board_size=board_size).to(torch_device)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        return cls(net=net, device=torch_device)

    def predict(self, board: Board, player: int) -> tuple[np.ndarray, float]:
        arr = np.array(board.grid, dtype=np.float32) * player
        x = torch.from_numpy(arr).view(1, board.size, board.size).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return policy, float(value.cpu().numpy()[0])


class GomokuZeroNet(nn.Module):
    """Network definition compatible with GomokuZeroAI checkpoints."""

    def __init__(self, in_channels: int, channels: int, board_height: int, board_width: int) -> None:
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.action_size = board_height * board_width
        self.backend = nn.Sequential(
            self._block(in_channels, channels, board_height, board_width),
            self._block(channels, channels, board_height, board_width),
            self._block(channels, channels, board_height, board_width),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        value_channels = max(4, channels // 16)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(value_channels * self.action_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, board_height: int, board_width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LayerNorm([out_channels, board_height, board_width]),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backend(x)
        return self.policy_head(x), self.value_head(x).view(-1)


def infer_gomokuzero_checkpoint_board_size(path: str | Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    height = int(config.get("board_height", 0))
    width = int(config.get("board_width", 0))
    if height and width:
        if height != width:
            raise ValueError(f"only square GomokuZeroAI boards are supported, got {height}x{width}")
        return height
    state_dict = checkpoint.get("model_state", checkpoint)
    action_size = int(state_dict["policy_head.net.3.weight"].shape[0])
    board_size = int(round(math.sqrt(action_size)))
    if board_size * board_size != action_size:
        raise ValueError(f"cannot infer GomokuZeroAI board size from action size {action_size}")
    return board_size


@dataclass
class GomokuZeroPolicy:
    net: GomokuZeroNet
    device: torch.device
    board_size: int

    @classmethod
    def load(cls, path: str | Path, board_size: int, device: str | None = None) -> "GomokuZeroPolicy":
        checkpoint_size = infer_gomokuzero_checkpoint_board_size(path)
        if checkpoint_size != board_size:
            raise ValueError(
                f"GomokuZeroAI checkpoint is {checkpoint_size}x{checkpoint_size}, "
                f"but current training board is {board_size}x{board_size}"
            )
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        config = checkpoint.get("config", {})
        model = GomokuZeroNet(
            in_channels=int(config.get("in_channels", 2)),
            channels=int(config.get("channels", 128)),
            board_height=int(config.get("board_height", board_size)),
            board_width=int(config.get("board_width", board_size)),
        ).to(torch_device)
        model.load_state_dict(_remap_gomokuzero_state_dict(checkpoint["model_state"]))
        model.eval()
        return cls(net=model, device=torch_device, board_size=board_size)

    def predict(self, board: Board, player: int) -> tuple[np.ndarray, float]:
        if board.size != self.board_size:
            raise ValueError(f"model board size is {self.board_size}, got board size {board.size}")
        arr = np.array(board.grid, dtype=np.int8)
        own = (arr == player).astype(np.float32)
        opp = (arr == -player).astype(np.float32)
        x = torch.from_numpy(np.stack([own, opp], axis=0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, win_rate = self.net(x)
            policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # GomokuZeroAI value head is [0, 1] win rate; our MCTS expects [-1, 1].
        value = float(win_rate.cpu().numpy()[0]) * 2.0 - 1.0
        return policy, value


def _remap_gomokuzero_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("backend.layers.", "backend.")
        new_key = new_key.replace("policy_head.net.", "policy_head.")
        new_key = new_key.replace("value_head.net.", "value_head.")
        remapped[new_key] = value
    return remapped
