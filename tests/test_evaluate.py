from __future__ import annotations

import math

import torch
from torch import nn

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.evaluate import evaluate_td_error, evaluate_win_rate
from renju_dqn.model import RenjuResNetDQN
from renju_dqn.replay_buffer import ReplayBuffer
from renju_dqn.rules import BLACK, BOARD_CELLS, rc_to_idx

GAMMA = 0.99
CENTER = rc_to_idx(7, 7)


def _tiny_model() -> RenjuResNetDQN:
    return RenjuResNetDQN(
        in_channels=NUM_CHANNELS,
        channels=4,
        num_blocks=1,
        head_channels=2,
        num_move_labels=BOARD_CELLS,
        dueling=False,
    )


def _filled_replay_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=10, gamma=GAMMA)
    board = [0] * BOARD_CELLS
    next_board = list(board)
    next_board[CENTER] = BLACK
    for _ in range(8):
        buffer.push(
            board=board,
            player=BLACK,
            move=CENTER,
            winner=BLACK,
            prev_move=None,
            next_board=next_board,
            done=False,
        )
    return buffer


def test_evaluate_td_error_returns_finite_metrics():
    model = _tiny_model()
    buffer = _filled_replay_buffer()
    generator = torch.Generator().manual_seed(0)

    metrics = evaluate_td_error(
        model,
        buffer,
        nn.SmoothL1Loss(),
        device=torch.device("cpu"),
        gamma=GAMMA,
        num_batches=3,
        batch_size=4,
        generator=generator,
    )

    assert set(metrics) == {"td_loss", "mean_q"}
    assert math.isfinite(metrics["td_loss"])
    assert math.isfinite(metrics["mean_q"])


def test_evaluate_win_rate_counts_add_up_to_num_games():
    model = _tiny_model()
    model.eval()
    generator = torch.Generator().manual_seed(0)

    metrics = evaluate_win_rate(
        model,
        device=torch.device("cpu"),
        num_games=2,
        opponent_epsilon=1.0,
        generator=generator,
    )

    assert metrics["games"] == 2.0
    total_rate = metrics["win_rate"] + metrics["draw_rate"] + metrics["loss_rate"]
    assert math.isclose(total_rate, 1.0, rel_tol=1e-9)
