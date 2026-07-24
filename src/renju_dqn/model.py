"""ResNet-based Deep Q-Network mapping a board tensor to per-cell Q-values."""

from __future__ import annotations

import torch
from torch import nn

from .rules import BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class RenjuResNetDQN(nn.Module):
    """Stem -> stacked residual blocks -> Q-value head over `num_move_labels` cells.

    `dueling` splits the head into a scalar state value `V(s)` and a per-action
    advantage `A(s,a)`, combined as `Q = V + (A - mean(A))` to fix the well-known
    identifiability issue of a plain `V + A` sum.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_blocks: int,
        head_channels: int,
        num_move_labels: int,
        board_size: int = BOARD_SIZE,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.head_conv = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        head_dim = head_channels * board_size * board_size
        if dueling:
            self.value_head = nn.Linear(head_dim, 1)
            self.advantage_head = nn.Linear(head_dim, num_move_labels)
        else:
            self.q_head = nn.Linear(head_dim, num_move_labels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(state)
        hidden = self.blocks(hidden)
        hidden = self.head_conv(hidden)
        hidden = hidden.flatten(start_dim=1)

        if self.dueling:
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_head(hidden)
