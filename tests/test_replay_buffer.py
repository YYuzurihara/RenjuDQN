from __future__ import annotations

import csv
import io

import pytest

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.dataset import ReplayDataset
from renju_dqn.replay_buffer import ReplayBuffer
from renju_dqn.rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE, rc_to_idx

GAMMA = 0.99
CENTER = rc_to_idx(7, 7)


def _write_rows(path, rows: list[list[int]]) -> None:
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    path.write_text(buffer.getvalue())


def _two_ply_game_rows() -> list[list[int]]:
    board_before_first_move = [0] * BOARD_CELLS
    board_after_first_move = list(board_before_first_move)
    board_after_first_move[CENTER] = BLACK

    row1 = [1, 1, BLACK, *board_before_first_move, CENTER, BLACK, 0]
    row2 = [1, 2, WHITE, *board_after_first_move, 0, BLACK, 0]
    return [row1, row2]


def test_push_and_sample_shapes():
    buffer = ReplayBuffer(capacity=10, gamma=GAMMA)
    board = [0] * BOARD_CELLS
    next_board = list(board)
    next_board[CENTER] = BLACK

    for _ in range(4):
        buffer.push(
            board=board,
            player=BLACK,
            move=CENTER,
            winner=BLACK,
            prev_move=None,
            next_board=next_board,
            done=False,
        )

    assert len(buffer) == 4

    state, action, reward, next_state, done, next_legal_mask = buffer.sample(4)
    assert state.shape == (4, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert next_state.shape == (4, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert action.shape == (4,)
    assert reward.shape == (4,)
    assert done.shape == (4,)
    assert next_legal_mask.shape == (4, BOARD_CELLS)
    assert not done.any()


def test_capacity_evicts_oldest_transition():
    buffer = ReplayBuffer(capacity=2, gamma=GAMMA)
    board = [0] * BOARD_CELLS
    for move in (0, 1, 2):
        buffer.push(
            board=board,
            player=BLACK,
            move=move,
            winner=BLACK,
            prev_move=None,
            next_board=board,
            done=True,
        )

    assert len(buffer) == 2
    # sample() applies a random D4 transform to the action, so check raw storage instead.
    assert {transition.move for transition in buffer._transitions} == {1, 2}


def test_sample_raises_when_buffer_too_small():
    buffer = ReplayBuffer(capacity=10, gamma=GAMMA)
    with pytest.raises(ValueError, match="Not enough transitions"):
        buffer.sample(1)


def test_push_game_marks_only_final_row_done():
    buffer = ReplayBuffer(capacity=10, gamma=GAMMA)
    board0 = [0] * BOARD_CELLS
    board1 = list(board0)
    board1[CENTER] = BLACK
    rows = [(board0, BLACK, CENTER), (board1, WHITE, 0)]

    buffer.push_game(rows, winner=BLACK)

    assert len(buffer) == 2
    transitions = buffer._transitions
    assert transitions[0].done is False
    assert transitions[0].next_board == board1
    assert transitions[1].done is True
    assert transitions[1].winner == BLACK


def test_warmup_from_dataset_matches_dataset_length(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_rows(csv_path, _two_ply_game_rows())
    dataset = ReplayDataset(csv_path, gamma=GAMMA)

    buffer = ReplayBuffer(capacity=100, gamma=GAMMA)
    buffer.warmup_from_dataset(dataset)

    assert len(buffer) == len(dataset)
