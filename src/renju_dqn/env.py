"""Pure-Python self-play game generator for online DQN data collection.

Slower than the `mcts.cpp` engine (no move search, and rule checks are plain
Python), but self-contained until the rules engine is shared with C++ via
pybind11 (Plan.md phase4 follow-up). A whole game is played and buffered
in-memory before being handed back, mirroring how `mcts.cpp` buffers a game's
rows and only attaches the final `winner` once the game ends.
"""

from __future__ import annotations

from collections.abc import Callable

from .rules import (
    BLACK,
    BOARD_CELLS,
    EMPTY,
    WHITE,
    board_with_move,
    legal_move_mask,
    winner_after_move,
)

DRAW = 0

# (board_before_move, player, move), one entry per ply, in play order.
GameRow = tuple[list[int], int, int]
SelectMove = Callable[[list[int], int, int | None, list[bool]], int]


def play_self_play_game(select_move: SelectMove) -> tuple[list[GameRow], int]:
    """Play one game to completion, calling `select_move` for every ply.

    `select_move(board, player, prev_move, legal_mask)` must return a legal move index.
    Returns `(rows, winner)` with `winner` in `{DRAW, BLACK, WHITE}`.
    """
    board = [EMPTY] * BOARD_CELLS
    player = BLACK
    prev_move: int | None = None
    rows: list[GameRow] = []
    winner: int | None = None
    ply = 0

    while winner is None:
        mask = legal_move_mask(board)
        move = select_move(board, player, prev_move, mask)
        if not (0 <= move < BOARD_CELLS) or not mask[move]:
            raise ValueError(f"select_move returned an illegal move: {move}")

        rows.append((list(board), player, move))
        next_board = board_with_move(board, move, player)
        result = winner_after_move(next_board, move, player)
        board = next_board
        prev_move = move
        ply += 1

        if result is not None:
            winner = result
        elif ply >= BOARD_CELLS:
            winner = DRAW
        else:
            player = WHITE if player == BLACK else BLACK

    return rows, winner
