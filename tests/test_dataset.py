from __future__ import annotations

import csv
import gzip
import io

import pytest
import torch

from renju_dqn.board_encoder import NUM_CHANNELS
from renju_dqn.dataset import ReplayDataset
from renju_dqn.reward import compute_reward
from renju_dqn.rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE, rc_to_idx

GAMMA = 0.99
CENTER = rc_to_idx(7, 7)


def _write_rows(path, rows: list[list[int]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    path.write_text(buffer.getvalue())


def _two_ply_game_rows() -> list[list[int]]:
    board_before_first_move = [0] * BOARD_CELLS
    board_after_first_move = list(board_before_first_move)
    board_after_first_move[CENTER] = BLACK

    row1 = [1, 1, BLACK, *board_before_first_move, CENTER, BLACK, 0]
    row2 = [1, 2, WHITE, *board_after_first_move, 0, BLACK, 0]
    return [row1, row2]


def test_dataset_builds_transitions_across_plies(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_rows(csv_path, _two_ply_game_rows())

    dataset = ReplayDataset(csv_path, gamma=GAMMA)

    assert len(dataset) == 2

    state0, action0, reward0, next_state0, done0 = dataset[0]
    assert state0.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert next_state0.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert action0.item() == CENTER
    assert done0.item() is False
    # First mover (black) has no stones on the board yet: shaping-only reward.
    board = [0] * BOARD_CELLS
    next_board = list(board)
    next_board[CENTER] = BLACK
    expected_reward0 = compute_reward(
        board, next_board, player=BLACK, winner=BLACK, done=False, gamma=GAMMA
    )
    assert reward0.item() == pytest.approx(expected_reward0)
    # Opponent's stone (none yet) shouldn't appear on black's own-stone plane before the move.
    assert state0[0].sum() == 0.0
    # next_state is seen from white's perspective: black's new stone is the "opponent" plane.
    assert next_state0[1, 7, 7] == 1.0

    state1, action1, reward1, next_state1, done1 = dataset[1]
    assert action1.item() == 0
    assert done1.item() is True
    assert torch.equal(next_state1, torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    # White made the losing move: terminal reward is negative from white's perspective.
    expected_reward1 = compute_reward(
        next_board, next_board, player=WHITE, winner=BLACK, done=True, gamma=GAMMA
    )
    assert reward1.item() == pytest.approx(expected_reward1)
    assert reward1.item() < 0


def test_dataset_loads_gzip_csv(tmp_path):
    csv_path = tmp_path / "data.csv.gz"
    buffer = io.StringIO()
    csv.writer(buffer).writerows(_two_ply_game_rows())
    csv_path.write_bytes(gzip.compress(buffer.getvalue().encode("utf-8")))

    dataset = ReplayDataset(csv_path, gamma=GAMMA)

    assert len(dataset) == 2


def test_dataset_rejects_malformed_row(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("1,1,1,0,0\n")

    with pytest.raises(ValueError, match="columns"):
        ReplayDataset(csv_path, gamma=GAMMA)


def test_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayDataset(tmp_path / "missing.csv", gamma=GAMMA)
