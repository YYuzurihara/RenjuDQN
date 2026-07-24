from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.model import RenjuResNetDQN
from renju_dqn.predict import build_model_from_checkpoint, parse_board_csv, select_move_from_q
from renju_dqn.rules import BOARD_CELLS

MODEL_KWARGS = {
    "in_channels": NUM_CHANNELS,
    "channels": 4,
    "num_blocks": 1,
    "head_channels": 2,
    "num_move_labels": BOARD_CELLS,
    "dueling": False,
}


def test_parse_board_csv_accepts_full_board():
    csv_text = ",".join(["0"] * BOARD_CELLS)
    board = parse_board_csv(csv_text)
    assert board == [0] * BOARD_CELLS


def test_parse_board_csv_rejects_wrong_length():
    with pytest.raises(ValueError, match="Expected"):
        parse_board_csv("0,1,2")


def test_build_model_from_checkpoint_uses_embedded_config():
    model = RenjuResNetDQN(**MODEL_KWARGS)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {"model": dict(MODEL_KWARGS)},
    }

    rebuilt = build_model_from_checkpoint(checkpoint)

    state = torch.randn(1, NUM_CHANNELS, 15, 15)
    rebuilt.eval()
    model.eval()
    assert torch.allclose(rebuilt(state), model(state))


def test_build_model_from_checkpoint_falls_back_to_cfg_when_config_missing():
    model = RenjuResNetDQN(**MODEL_KWARGS)
    checkpoint = {"model_state_dict": model.state_dict()}
    fallback_cfg = OmegaConf.create({"model": dict(MODEL_KWARGS)})

    rebuilt = build_model_from_checkpoint(checkpoint, fallback_cfg=fallback_cfg)

    state = torch.randn(1, NUM_CHANNELS, 15, 15)
    rebuilt.eval()
    model.eval()
    assert torch.allclose(rebuilt(state), model(state))


def test_build_model_from_checkpoint_without_config_or_fallback_raises():
    checkpoint = {"model_state_dict": {}}
    with pytest.raises(ValueError, match="fallback_cfg"):
        build_model_from_checkpoint(checkpoint)


def test_select_move_from_q_picks_best_legal_move():
    q_values = torch.tensor([10.0, 5.0, 1.0])
    legal_mask = torch.tensor([False, True, True])
    assert select_move_from_q(q_values, legal_mask) == 1


def test_select_move_from_q_raises_when_no_legal_moves():
    q_values = torch.tensor([1.0, 2.0])
    legal_mask = torch.tensor([False, False])
    with pytest.raises(ValueError, match="No legal moves"):
        select_move_from_q(q_values, legal_mask)


def test_select_move_from_q_epsilon_one_stays_legal():
    q_values = torch.tensor([10.0, 5.0, 1.0])
    legal_mask = torch.tensor([False, True, True])
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        move = select_move_from_q(q_values, legal_mask, epsilon=1.0, generator=generator)
        assert legal_mask[move]
