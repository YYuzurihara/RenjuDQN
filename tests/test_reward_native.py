"""Cross-checks the pybind11 native reward-shaping functions against the pure-Python
fallback they're meant to replace (`reward._xxx_python`). Skipped if the extension isn't built.
"""

from __future__ import annotations

import random

import pytest

from renju_dqn import reward
from renju_dqn.rules import BLACK, BOARD_CELLS, WHITE

pytestmark = pytest.mark.skipif(reward._native is None, reason="native rules extension not built")


def _random_board(num_stones: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    board = [0] * BOARD_CELLS
    indices = rng.sample(range(BOARD_CELLS), num_stones)
    for position, index in enumerate(indices):
        board[index] = BLACK if position % 2 == 0 else WHITE
    return board


@pytest.mark.parametrize("seed", range(20))
def test_board_potential_matches_python_fallback(seed):
    board = _random_board(num_stones=40, seed=seed)
    for player in (BLACK, WHITE):
        assert reward._native.board_potential(board, player) == pytest.approx(
            reward._board_potential_python(board, player)
        )


@pytest.mark.parametrize("seed", range(20))
def test_normalized_potential_matches_python_fallback(seed):
    board = _random_board(num_stones=40, seed=seed)
    for player in (BLACK, WHITE):
        assert reward._native.normalized_potential(board, player, reward.POTENTIAL_SCALE) == pytest.approx(
            reward._normalized_potential_python(board, player)
        )


@pytest.mark.parametrize("seed", range(20))
def test_compute_reward_matches_python_fallback(seed):
    board = _random_board(num_stones=40, seed=seed)
    next_board = _random_board(num_stones=41, seed=seed + 1000)
    for player in (BLACK, WHITE):
        for winner in (0, BLACK, WHITE):
            for done in (True, False):
                native_value = reward._native.compute_reward(
                    board, next_board, player, winner, done, 0.99,
                    reward.DEFAULT_SHAPING_COEFFICIENT, reward.POTENTIAL_SCALE,
                )
                python_value = reward._compute_reward_python(
                    board, next_board, player, winner, done, 0.99
                )
                assert native_value == pytest.approx(python_value)
