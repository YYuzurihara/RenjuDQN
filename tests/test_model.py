from __future__ import annotations

import torch

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.model import RenjuResNetDQN, ResidualBlock
from renju_dqn.rules import BOARD_CELLS, BOARD_SIZE


def _build_model(dueling: bool = False) -> RenjuResNetDQN:
    return RenjuResNetDQN(
        in_channels=NUM_CHANNELS,
        channels=8,
        num_blocks=2,
        head_channels=2,
        num_move_labels=BOARD_CELLS,
        dueling=dueling,
    )


def test_output_shape_matches_num_move_labels():
    model = _build_model()
    state = torch.zeros(3, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    q_values = model(state)
    assert q_values.shape == (3, BOARD_CELLS)


def test_dueling_head_output_shape_matches_non_dueling():
    model = _build_model(dueling=True)
    state = torch.zeros(2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    q_values = model(state)
    assert q_values.shape == (2, BOARD_CELLS)


def test_residual_block_preserves_shape():
    block = ResidualBlock(channels=6)
    x = torch.randn(2, 6, BOARD_SIZE, BOARD_SIZE)
    assert block(x).shape == x.shape


def test_dueling_advantage_is_centered_relative_to_value():
    model = _build_model(dueling=True)
    model.eval()
    state = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        hidden = model.head_conv(model.blocks(model.stem(state))).flatten(start_dim=1)
        value = model.value_head(hidden)
        advantage = model.advantage_head(hidden)
        expected = value + (advantage - advantage.mean(dim=1, keepdim=True))
        actual = model(state)
    assert torch.allclose(actual, expected)
