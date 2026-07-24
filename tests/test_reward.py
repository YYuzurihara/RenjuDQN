from __future__ import annotations

from renju_dqn.reward import (
    DEFAULT_SHAPING_COEFFICIENT,
    DRAW,
    board_potential,
    compute_reward,
    normalized_potential,
    stone_threat_score,
    terminal_reward,
)
from renju_dqn.rules import BLACK, BOARD_CELLS, WHITE, rc_to_idx


def empty_board() -> list[int]:
    return [0] * BOARD_CELLS


def test_board_potential_zero_on_empty_board():
    board = empty_board()
    assert board_potential(board, BLACK) == 0.0
    assert board_potential(board, WHITE) == 0.0


def test_board_potential_open_three_favors_owner():
    board = empty_board()
    for col in (6, 7, 8):
        board[rc_to_idx(7, col)] = BLACK

    black_view = board_potential(board, BLACK)
    white_view = board_potential(board, WHITE)

    assert black_view > 0
    assert white_view == -black_view


def test_stone_threat_score_ignores_opponent_stones():
    board = empty_board()
    board[rc_to_idx(7, 7)] = BLACK
    assert stone_threat_score(board, rc_to_idx(7, 7), WHITE) == 0


def test_normalized_potential_is_bounded():
    board = empty_board()
    for col in range(0, 8, 2):
        board[rc_to_idx(7, col)] = BLACK
    value = normalized_potential(board, BLACK)
    assert -1.0 < value < 1.0


def test_compute_reward_non_terminal_uses_only_shaping():
    board = empty_board()
    next_board = empty_board()
    for col in (6, 7, 8):
        next_board[rc_to_idx(7, col)] = BLACK

    reward = compute_reward(board, next_board, player=BLACK, winner=DRAW, done=False, gamma=0.99)

    expected_shaping = DEFAULT_SHAPING_COEFFICIENT * (
        0.99 * normalized_potential(next_board, BLACK) - normalized_potential(board, BLACK)
    )
    assert reward == expected_shaping
    assert reward > 0


def test_compute_reward_terminal_win_ignores_next_state_potential():
    board = empty_board()
    for col in (6, 7, 8, 9):
        board[rc_to_idx(7, col)] = BLACK
    next_board = empty_board()

    reward = compute_reward(board, next_board, player=BLACK, winner=BLACK, done=True, gamma=0.99)

    expected = 1.0 - DEFAULT_SHAPING_COEFFICIENT * normalized_potential(board, BLACK)
    assert reward == expected


def test_terminal_reward_signs():
    assert terminal_reward(BLACK, winner=BLACK) == 1.0
    assert terminal_reward(BLACK, winner=WHITE) == -1.0
    assert terminal_reward(BLACK, winner=DRAW) == 0.0
