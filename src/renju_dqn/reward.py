"""Potential-based reward shaping for Renju self-play transitions.

r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t)
"""

from __future__ import annotations

import math

from .rules import BLACK, WHITE, count_four_directions, count_open_three_directions

DRAW = 0

# A four is one move from winning; an open three merely forces a response,
# so it is weighted well below a four.
FOUR_WEIGHT = 9
OPEN_THREE_WEIGHT = 3

POTENTIAL_SCALE = 5.0
DEFAULT_SHAPING_COEFFICIENT = 0.1


def stone_threat_score(board: list[int], index: int, player: int) -> int:
    fours = count_four_directions(board, index, player)
    open_threes = count_open_three_directions(board, index, player)
    return FOUR_WEIGHT * fours + OPEN_THREE_WEIGHT * open_threes


def board_potential(board: list[int], player: int) -> float:
    """Raw Phi(s), signed from `player`'s perspective (own threats minus opponent's)."""
    opponent = WHITE if player == BLACK else BLACK
    own_score = sum(
        stone_threat_score(board, index, player)
        for index, cell in enumerate(board)
        if cell == player
    )
    opponent_score = sum(
        stone_threat_score(board, index, opponent)
        for index, cell in enumerate(board)
        if cell == opponent
    )
    return float(own_score - opponent_score)


def normalized_potential(board: list[int], player: int, scale: float = POTENTIAL_SCALE) -> float:
    """Phi(s) squashed to (-1, 1) so shaping stays small relative to the +-1 terminal reward."""
    return math.tanh(board_potential(board, player) / scale)


def terminal_reward(player: int, winner: int) -> float:
    if winner == DRAW:
        return 0.0
    return 1.0 if winner == player else -1.0


def compute_reward(
    board: list[int],
    next_board: list[int],
    player: int,
    winner: int,
    done: bool,
    gamma: float,
    coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
    scale: float = POTENTIAL_SCALE,
) -> float:
    """r'_t for the transition where `player` played `board` -> `next_board`.

    `done` marks the final ply of the game: `winner` only contributes on that ply,
    and Phi(s_{t+1}) is fixed at 0 for the terminal state (required for the shaping
    theorem's policy-invariance guarantee to hold).
    """
    sparse = terminal_reward(player, winner) if done else 0.0
    phi_t = normalized_potential(board, player, scale)
    phi_next = 0.0 if done else normalized_potential(next_board, player, scale)
    return sparse + coefficient * (gamma * phi_next - phi_t)
