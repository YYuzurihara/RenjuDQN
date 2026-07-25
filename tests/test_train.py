from __future__ import annotations

import torch
from omegaconf import OmegaConf
import pytest

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.model import RenjuResNetDQN
from renju_dqn.train import (
    curriculum_window_for_epoch,
    epsilon_for_curriculum,
    make_noisy_net_policy,
    train_model,
)

BOARD_CELLS = 225


def test_curriculum_window_grows_by_step_size_each_epoch():
    assert curriculum_window_for_epoch(1, step_size=5) == 5
    assert curriculum_window_for_epoch(2, step_size=5) == 10
    assert curriculum_window_for_epoch(9, step_size=5) == 45


def test_curriculum_window_disabled_when_step_size_not_positive():
    assert curriculum_window_for_epoch(1, step_size=0) is None


def test_epsilon_for_curriculum_starts_high_and_ends_low():
    assert epsilon_for_curriculum(0, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1) == pytest.approx(1.0)
    assert epsilon_for_curriculum(
        BOARD_CELLS, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1
    ) == pytest.approx(0.1)


def test_epsilon_for_curriculum_interpolates_linearly_with_window_fraction():
    window = BOARD_CELLS // 2
    result = epsilon_for_curriculum(window, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1)
    fraction = window / BOARD_CELLS
    assert result == pytest.approx(1.0 + fraction * (0.1 - 1.0))


def test_epsilon_for_curriculum_clamps_once_window_exceeds_board():
    assert epsilon_for_curriculum(
        BOARD_CELLS * 2, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1
    ) == pytest.approx(0.1)


def test_epsilon_for_curriculum_returns_end_when_curriculum_disabled():
    assert epsilon_for_curriculum(None, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1) == 0.1


def test_epsilon_for_curriculum_exponential_decay_still_hits_endpoints():
    assert epsilon_for_curriculum(
        0, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1, decay_rate=2.0
    ) == pytest.approx(1.0)
    assert epsilon_for_curriculum(
        BOARD_CELLS, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1, decay_rate=5.0
    ) == pytest.approx(0.1)


def test_epsilon_for_curriculum_exponential_decay_is_front_loaded_vs_linear():
    window = BOARD_CELLS // 2
    linear = epsilon_for_curriculum(window, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1)
    exponential = epsilon_for_curriculum(
        window, BOARD_CELLS, epsilon_start=1.0, epsilon_end=0.1, decay_rate=5.0
    )
    assert exponential < linear


def test_train_model_rejects_unsupported_exploration_strategy():
    cfg = OmegaConf.create({"train": {"exploration": "bogus"}})
    with pytest.raises(ValueError, match="Unsupported exploration strategy"):
        train_model(cfg)


def test_make_noisy_net_policy_always_returns_legal_move():
    model = RenjuResNetDQN(
        in_channels=NUM_CHANNELS,
        channels=8,
        num_blocks=2,
        head_channels=2,
        num_move_labels=BOARD_CELLS,
        noisy=True,
    )
    device = torch.device("cpu")
    select_move = make_noisy_net_policy(model, device)

    board = [0] * BOARD_CELLS
    mask = [False] * BOARD_CELLS
    mask[10] = True
    mask[42] = True

    move = select_move(board, player=1, prev_move=None, mask=mask)
    assert mask[move]
