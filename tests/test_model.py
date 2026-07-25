from __future__ import annotations

import pytest
import torch

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.model import NoisyLinear, RenjuResNetDQN, ResidualBlock
from renju_dqn.rules import BOARD_CELLS, BOARD_SIZE


def _build_model(dueling: bool = False, noisy: bool = False) -> RenjuResNetDQN:
    return RenjuResNetDQN(
        in_channels=NUM_CHANNELS,
        channels=8,
        num_blocks=2,
        head_channels=2,
        num_move_labels=BOARD_CELLS,
        dueling=dueling,
        noisy=noisy,
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


def test_noisy_model_output_shape_matches_num_move_labels():
    model = _build_model(noisy=True)
    state = torch.zeros(3, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    q_values = model(state)
    assert q_values.shape == (3, BOARD_CELLS)
    assert isinstance(model.q_head, NoisyLinear)


def test_noisy_model_is_deterministic_in_eval_mode():
    model = _build_model(noisy=True)
    model.eval()
    state = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        first = model(state)
        model.reset_noise()
        second = model(state)
    assert torch.allclose(first, second)


def test_noisy_model_varies_across_resampled_noise_in_train_mode():
    model = _build_model(noisy=True)
    model.train()
    state = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    model.reset_noise()
    first = model(state)
    model.reset_noise()
    second = model(state)
    assert not torch.allclose(first, second)


def test_set_noise_training_enables_noise_independent_of_eval_mode():
    model = _build_model(noisy=True)
    model.eval()
    model.set_noise_training(True)
    state = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        model.reset_noise()
        first = model(state)
        model.reset_noise()
        second = model(state)
    assert not torch.allclose(first, second)


def test_noisy_sigma_norms_reports_q_head_for_non_dueling_noisy_model():
    model = _build_model(noisy=True)
    norms = model.noisy_sigma_norms()
    assert set(norms) == {"q_head"}
    assert norms["q_head"] > 0.0


def test_noisy_sigma_norms_reports_both_heads_for_dueling_noisy_model():
    model = _build_model(dueling=True, noisy=True)
    norms = model.noisy_sigma_norms()
    assert set(norms) == {"value_head", "advantage_head"}
    assert all(value > 0.0 for value in norms.values())


def test_noisy_sigma_norms_is_empty_for_non_noisy_model():
    model = _build_model(noisy=False)
    assert model.noisy_sigma_norms() == {}


def test_noisy_sigma_norms_tracks_sigma_shrinkage():
    model = _build_model(noisy=True)
    before = model.noisy_sigma_norms()["q_head"]
    with torch.no_grad():
        model.q_head.weight_sigma.mul_(0.1)
        model.q_head.bias_sigma.mul_(0.1)
    after = model.noisy_sigma_norms()["q_head"]
    assert after == pytest.approx(before * 0.1)
