"""Pure-Python self-play game generator for online DQN data collection.

Slower than the `mcts.cpp` engine (no move search, and rule checks are plain
Python), but self-contained until the rules engine is shared with C++ via
pybind11 (Plan.md phase4 follow-up). A whole game is played and buffered
in-memory before being handed back, mirroring how `mcts.cpp` buffers a game's
rows and only attaches the final `winner` once the game ends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
# Batched form of SelectMove: one call decides a move for every game in the batch at once, so
# the caller (a model-backed policy) can encode/forward all of them as a single batch instead of
# one board at a time.
SelectMovesBatch = Callable[
    [list[list[int]], list[int], list[int | None], list[list[bool]]], list[int]
]


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


@dataclass(slots=True)
class GameState:
    """One in-progress game's mutable state, for lockstep batches of concurrent games."""

    board: list[int] = field(default_factory=lambda: [EMPTY] * BOARD_CELLS)
    player: int = BLACK
    prev_move: int | None = None
    rows: list[GameRow] = field(default_factory=list)
    ply: int = 0


def step_games_batch(
    games: list[GameState],
    select_moves: SelectMovesBatch,
) -> list[tuple[int, list[GameRow], int]]:
    """Advance every game in `games` by exactly one ply, deciding all their moves in a single
    `select_moves` call so a model-backed policy can batch its inference across games instead
    of running it once per game (see `train.self_play_worker`).

    Returns `(index, rows, winner)` for each game in `games` that finished this ply, `index`
    being its position in the input list. Games that haven't finished are mutated in place and
    left in `games` for the next call; finished games are left as-is (caller owns replacing them).
    """
    boards = [game.board for game in games]
    players = [game.player for game in games]
    prev_moves = [game.prev_move for game in games]
    masks = [legal_move_mask(game.board) for game in games]
    moves = select_moves(boards, players, prev_moves, masks)

    finished: list[tuple[int, list[GameRow], int]] = []
    for index, game in enumerate(games):
        move = moves[index]
        mask = masks[index]
        if not (0 <= move < BOARD_CELLS) or not mask[move]:
            raise ValueError(f"select_moves returned an illegal move: {move}")

        game.rows.append((list(game.board), game.player, move))
        next_board = board_with_move(game.board, move, game.player)
        result = winner_after_move(next_board, move, game.player)
        game.board = next_board
        game.prev_move = move
        game.ply += 1

        if result is not None:
            finished.append((index, game.rows, result))
        elif game.ply >= BOARD_CELLS:
            finished.append((index, game.rows, DRAW))
        else:
            game.player = WHITE if game.player == BLACK else BLACK

    return finished
