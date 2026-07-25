from __future__ import annotations

import pytest

from renju_dqn.env import GameState, play_self_play_game, step_games_batch
from renju_dqn.rules import BOARD_CELLS


def first_legal_move(board, player, prev_move, mask):
    return next(index for index, ok in enumerate(mask) if ok)


def first_legal_move_batch(boards, players, prev_moves, masks):
    return [first_legal_move(b, p, pm, m) for b, p, pm, m in zip(boards, players, prev_moves, masks)]


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


def test_step_games_batch_matches_sequential_play():
    """Running several identical-policy games through step_games_batch's lockstep loop should
    reproduce exactly what play_self_play_game produces for the same policy, one game at a
    time.
    """
    expected_rows, expected_winner = play_self_play_game(first_legal_move)

    num_games = 3
    games = {index: GameState() for index in range(num_games)}
    results: dict[int, tuple[list, int]] = {}

    while len(results) < num_games:
        active_indices = list(games.keys())
        active_games = [games[index] for index in active_indices]
        finished = step_games_batch(active_games, first_legal_move_batch)
        for local_index, rows, winner in finished:
            game_index = active_indices[local_index]
            results[game_index] = (rows, winner)
            del games[game_index]

    for game_index in range(num_games):
        rows, winner = results[game_index]
        assert rows == expected_rows
        assert winner == expected_winner


def test_step_games_batch_advances_unfinished_games_in_place():
    games = [GameState(), GameState()]
    finished = step_games_batch(games, first_legal_move_batch)

    assert finished == []
    for game in games:
        assert game.ply == 1
        assert len(game.rows) == 1
        assert game.player == 2  # BLACK's opening move flips the side to move to WHITE


def test_step_games_batch_rejects_illegal_move():
    def illegal_moves(boards, players, prev_moves, masks):
        return [next(index for index, ok in enumerate(mask) if not ok) for mask in masks]

    with pytest.raises(ValueError, match="illegal move"):
        step_games_batch([GameState()], illegal_moves)
