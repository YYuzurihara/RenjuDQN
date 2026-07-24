from __future__ import annotations

import pytest

from renju_dqn.env import play_self_play_game
from renju_dqn.rules import BOARD_CELLS


def first_legal_move(board, player, prev_move, mask):
    return next(index for index, ok in enumerate(mask) if ok)


def test_play_self_play_game_terminates_and_replays_moves():
    rows, winner = play_self_play_game(first_legal_move)

    assert winner in (0, 1, 2)
    assert 1 <= len(rows) <= BOARD_CELLS

    board = [0] * BOARD_CELLS
    for board_before, player, move in rows:
        assert board_before == board
        assert player in (1, 2)
        assert board_before[move] == 0
        board[move] = player


def test_play_self_play_game_rejects_illegal_move():
    def illegal_move(board, player, prev_move, mask):
        return next(index for index, ok in enumerate(mask) if not ok)

    with pytest.raises(ValueError, match="illegal move"):
        play_self_play_game(illegal_move)
