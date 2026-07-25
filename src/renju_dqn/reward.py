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


def _board_potential_python(board: list[int], player: int) -> float:
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


def _normalized_potential_python(
    board: list[int], player: int, scale: float = POTENTIAL_SCALE
) -> float:
    return math.tanh(_board_potential_python(board, player) / scale)


def terminal_reward(player: int, winner: int) -> float:
    if winner == DRAW:
        return 0.0
    return 1.0 if winner == player else -1.0


def _compute_reward_python(
    board: list[int],
    next_board: list[int],
    player: int,
    winner: int,
    done: bool,
    gamma: float,
    coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
    scale: float = POTENTIAL_SCALE,
) -> float:
    sparse = terminal_reward(player, winner) if done else 0.0
    phi_t = _normalized_potential_python(board, player, scale)
    phi_next = 0.0 if done else _normalized_potential_python(next_board, player, scale)
    return sparse + coefficient * (gamma * phi_next - phi_t)


def _compute_rewards_batch_python(
    boards: list[list[int]],
    next_boards: list[list[int]],
    players: list[int],
    winners: list[int],
    dones: list[bool],
    gamma: float,
    coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
    scale: float = POTENTIAL_SCALE,
) -> list[float]:
    return [
        _compute_reward_python(board, next_board, player, winner, done, gamma, coefficient, scale)
        for board, next_board, player, winner, done in zip(
            boards, next_boards, players, winners, dones
        )
    ]


# --- Native (pybind11) dispatch -----------------------------------------------------
#
# board_potential is O(stones-on-board) -- the hottest path in replay_buffer._encode_batch,
# called once per transition in a training batch -- so it's also implemented in
# `native/rules_native.cpp`, which computes an entire board (or batch of boards, via
# compute_rewards_batch) per Python<->native call instead of once per stone. The functions
# below route to it when the extension is built, and stay on the pure-Python path above
# otherwise.
try:
    from . import _rules_native as _native
except ImportError:
    _native = None


def board_potential(board: list[int], player: int) -> float:
    if _native is not None:
        return _native.board_potential(board, player)
    return _board_potential_python(board, player)


def normalized_potential(board: list[int], player: int, scale: float = POTENTIAL_SCALE) -> float:
    """Phi(s) squashed to (-1, 1) so shaping stays small relative to the +-1 terminal reward."""
    if _native is not None:
        return _native.normalized_potential(board, player, scale)
    return _normalized_potential_python(board, player, scale)


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
    if _native is not None:
        return _native.compute_reward(
            board, next_board, player, winner, done, gamma, coefficient, scale
        )
    return _compute_reward_python(board, next_board, player, winner, done, gamma, coefficient, scale)


def compute_rewards_batch(
    boards: list[list[int]],
    next_boards: list[list[int]],
    players: list[int],
    winners: list[int],
    dones: list[bool],
    gamma: float,
    coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
    scale: float = POTENTIAL_SCALE,
) -> list[float]:
    """Batched form of `compute_reward`: one Python<->native call for the whole batch instead
    of one per transition.
    """
    if _native is not None:
        return _native.compute_rewards_batch(
            list(boards), list(next_boards), list(players), list(winners), list(dones),
            gamma, coefficient, scale,
        )
    return _compute_rewards_batch_python(
        boards, next_boards, players, winners, dones, gamma, coefficient, scale
    )
