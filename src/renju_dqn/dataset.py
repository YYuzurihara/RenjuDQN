"""Replay dataset built from mcts.cpp self-play CSV logs.

Groups CSV rows by `game_id` and reconstructs `(state, action, reward, next_state, done)`
transitions: `next_state` comes from the following `ply` in the same game (zero-filled at
the final ply), and reward is computed with the potential-based shaping in `reward.py`.
"""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from torch.utils.data import Dataset

from .board_encoder import NUM_CHANNELS, encode_board
from .reward import DEFAULT_SHAPING_COEFFICIENT, POTENTIAL_SCALE, compute_reward
from .rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE

CSV_COLUMNS = BOARD_CELLS + 6  # game_id, ply, player, board(225), move_id, winner, foul_loss


def _other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


@dataclass(slots=True)
class _GameRow:
    ply: int
    player: int
    board: list[int]
    move: int
    winner: int


@dataclass(slots=True)
class _Transition:
    board: list[int]
    player: int
    move: int
    winner: int
    prev_move: int | None
    next_board: list[int] | None
    done: bool


Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ReplayDataset(Dataset[Sample]):
    def __init__(
        self,
        csv_path: str | Path,
        gamma: float,
        coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
        scale: float = POTENTIAL_SCALE,
        max_rows: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.gamma = gamma
        self.coefficient = coefficient
        self.scale = scale
        self.transitions: list[_Transition] = []
        self._load(max_rows=max_rows)

    def _load(self, max_rows: int | None) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        if self.csv_path.suffix == ".gz":
            csv_content = gzip.decompress(self.csv_path.read_bytes()).decode("utf-8")
            with io.StringIO(csv_content, newline="") as handle:
                games = self._read_games(handle)
        else:
            with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                games = self._read_games(handle)

        for rows in games.values():
            rows.sort(key=lambda row: row.ply)
            for position, row in enumerate(rows):
                prev_move = rows[position - 1].move if position > 0 else None
                next_board = rows[position + 1].board if position + 1 < len(rows) else None
                done = next_board is None
                self.transitions.append(
                    _Transition(
                        board=row.board,
                        player=row.player,
                        move=row.move,
                        winner=row.winner,
                        prev_move=prev_move,
                        next_board=next_board,
                        done=done,
                    )
                )
                if max_rows is not None and len(self.transitions) >= max_rows:
                    self._validate_non_empty()
                    return

        self._validate_non_empty()

    def _validate_non_empty(self) -> None:
        if not self.transitions:
            raise ValueError(f"No training samples found in {self.csv_path}.")

    def _read_games(self, handle: TextIO) -> dict[int, list[_GameRow]]:
        games: dict[int, list[_GameRow]] = {}
        reader = csv.reader(handle)
        for row_index, raw_row in enumerate(reader, start=1):
            if not raw_row:
                continue
            if len(raw_row) != CSV_COLUMNS:
                raise ValueError(
                    f"Expected {CSV_COLUMNS} columns, got {len(raw_row)} in row {row_index}."
                )
            try:
                values = [int(value) for value in raw_row]
            except ValueError as exc:
                raise ValueError(f"Non-integer value found in row {row_index}.") from exc

            game_id = values[0]
            ply = values[1]
            player = values[2]
            board = values[3 : 3 + BOARD_CELLS]
            move = values[3 + BOARD_CELLS]
            winner = values[4 + BOARD_CELLS]
            games.setdefault(game_id, []).append(
                _GameRow(ply=ply, player=player, board=board, move=move, winner=winner)
            )
        return games

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> Sample:
        transition = self.transitions[index]
        state = encode_board(transition.board, transition.player, transition.prev_move)

        if transition.done:
            next_state = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            next_board_for_reward = transition.board
        else:
            next_player = _other_player(transition.player)
            next_state = encode_board(transition.next_board, next_player, transition.move)
            next_board_for_reward = transition.next_board

        reward = compute_reward(
            board=transition.board,
            next_board=next_board_for_reward,
            player=transition.player,
            winner=transition.winner,
            done=transition.done,
            gamma=self.gamma,
            coefficient=self.coefficient,
            scale=self.scale,
        )

        return (
            state,
            torch.tensor(transition.move, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            next_state,
            torch.tensor(transition.done, dtype=torch.bool),
        )
