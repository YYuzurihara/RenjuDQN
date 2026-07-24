"""Cross-checks the pybind11 native rules extension against the pure-Python fallback
it's meant to replace (`rules._xxx_python`). Skipped if the extension isn't built.
"""

from __future__ import annotations

import random

import pytest

from renju_dqn import rules
from renju_dqn.rules import BLACK, BOARD_CELLS, WHITE

pytestmark = pytest.mark.skipif(rules._native is None, reason="native rules extension not built")


def _random_board(num_stones: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    board = [0] * BOARD_CELLS
    indices = rng.sample(range(BOARD_CELLS), num_stones)
    for position, index in enumerate(indices):
        board[index] = BLACK if position % 2 == 0 else WHITE
    return board


@pytest.mark.parametrize("seed", range(20))
def test_legal_move_mask_matches_python_fallback(seed):
    board = _random_board(num_stones=30, seed=seed)
    assert rules._native.legal_move_mask(board) == rules._legal_move_mask_python(board)


@pytest.mark.parametrize("seed", range(20))
def test_count_directions_match_python_fallback(seed):
    board = _random_board(num_stones=30, seed=seed)
    empty_indices = [index for index, cell in enumerate(board) if cell == 0]
    move = random.Random(seed).choice(empty_indices)

    for player in (BLACK, WHITE):
        assert rules._native.count_four_directions(
            board, move, player
        ) == rules._count_four_directions_python(board, move, player)
        assert rules._native.count_open_three_directions(
            board, move, player
        ) == rules._count_open_three_directions_python(board, move, player)


def test_winner_after_move_matches_python_fallback():
    board = [0] * BOARD_CELLS
    for index in (0, 1, 2, 3):
        board[index] = BLACK

    assert rules._native.winner_after_move(board, 4, BLACK) == rules._winner_after_move_python(
        board, 4, BLACK
    )
    assert rules._native.winner_after_move(board, 4, BLACK) == BLACK


def test_infer_player_matches_python_fallback_and_raises_on_invalid_board():
    board = [0] * BOARD_CELLS
    board[0] = BLACK
    board[1] = WHITE
    assert rules._native.infer_player(board) == rules._infer_player_python(board)

    invalid_board = [0] * BOARD_CELLS
    invalid_board[0] = WHITE
    with pytest.raises(ValueError):
        rules._native.infer_player(invalid_board)
    with pytest.raises(ValueError):
        rules._infer_player_python(invalid_board)
