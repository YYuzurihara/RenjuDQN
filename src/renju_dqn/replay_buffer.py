"""Fixed-capacity FIFO replay buffer, warmed up from MCTS logs and grown online.

Stores raw `(board, player, move, winner, prev_move, next_board, done)` transitions
(the same shape `dataset.ReplayDataset` reconstructs from CSV rows) rather than
pre-encoded tensors: a board is 225 small ints, so keeping the buffer at
capacity in raw form is cheap, and it lets sampling apply a fresh random D4
augmentation each time instead of baking one in at insert time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from .board_encoder import (
    apply_transform_batch,
    encode_boards,
    random_transform_index,
)
from .dataset import ReplayDataset
from .env import GameRow
from .reward import DEFAULT_SHAPING_COEFFICIENT, POTENTIAL_SCALE, compute_reward
from .rules import BLACK, BOARD_CELLS, WHITE, legal_move_mask

# state, action, reward, next_state, done, next_legal_mask
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


@dataclass(slots=True)
class _Transition:
    board: list[int]
    player: int
    move: int
    winner: int
    prev_move: int | None
    next_board: list[int] | None
    done: bool
    steps_to_end: int


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        gamma: float,
        coefficient: float = DEFAULT_SHAPING_COEFFICIENT,
        scale: float = POTENTIAL_SCALE,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self.gamma = gamma
        self.coefficient = coefficient
        self.scale = scale
        self._transitions: list[_Transition] = []
        self._write_index = 0  # next slot to overwrite once the buffer is full (FIFO)
        # steps_to_end -> slot indices at that distance from their game's terminal ply.
        self._distance_buckets: dict[int, set[int]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._transitions)

    def push(
        self,
        board: list[int],
        player: int,
        move: int,
        winner: int,
        prev_move: int | None,
        next_board: list[int] | None,
        done: bool,
        steps_to_end: int,
    ) -> None:
        transition = _Transition(
            board=list(board),
            player=player,
            move=move,
            winner=winner,
            prev_move=prev_move,
            next_board=list(next_board) if next_board is not None else None,
            done=done,
            steps_to_end=steps_to_end,
        )
        if len(self._transitions) < self.capacity:
            index = len(self._transitions)
            self._transitions.append(transition)
        else:
            index = self._write_index
            old = self._transitions[index]
            self._distance_buckets[old.steps_to_end].discard(index)
            self._transitions[index] = transition
            self._write_index = (self._write_index + 1) % self.capacity

        self._distance_buckets[steps_to_end].add(index)

    def push_game(self, rows: list[GameRow], winner: int) -> None:
        """Push one finished self-play game's `(board, player, move)` rows, in ply order."""
        game_length = len(rows)
        for position, (board, player, move) in enumerate(rows):
            prev_move = rows[position - 1][2] if position > 0 else None
            next_board = rows[position + 1][0] if position + 1 < len(rows) else None
            done = next_board is None
            self.push(
                board=board,
                player=player,
                move=move,
                winner=winner,
                prev_move=prev_move,
                next_board=next_board,
                done=done,
                steps_to_end=game_length - 1 - position,
            )

    def warmup_from_dataset(self, dataset: ReplayDataset) -> None:
        for transition in dataset.transitions:
            self.push(
                board=transition.board,
                player=transition.player,
                move=transition.move,
                winner=transition.winner,
                prev_move=transition.prev_move,
                next_board=transition.next_board,
                done=transition.done,
                steps_to_end=transition.steps_to_end,
            )

    def _encode_batch(self, transitions: list[_Transition]) -> Batch:
        """Vectorized form of encoding one transition at a time: builds every tensor for the
        whole batch with a handful of batched ops instead of `batch_size` individual ones.
        """
        boards = [t.board for t in transitions]
        players = [t.player for t in transitions]
        prev_moves = [t.prev_move for t in transitions]
        actions = [t.move for t in transitions]
        dones = [t.done for t in transitions]

        # Placeholder next_board/player/move for done transitions (masked out below); avoids
        # branching per sample just to keep encode_boards() fed with well-formed inputs.
        next_boards = [t.next_board if not t.done else t.board for t in transitions]
        next_players = [
            (_other_player(t.player) if not t.done else t.player) for t in transitions
        ]
        next_moves = [t.move if not t.done else None for t in transitions]
        next_legal_masks = [
            (legal_move_mask(t.next_board) if not t.done else [False] * BOARD_CELLS)
            for t in transitions
        ]
        rewards = [
            compute_reward(
                board=t.board,
                next_board=t.board if t.done else t.next_board,
                player=t.player,
                winner=t.winner,
                done=t.done,
                gamma=self.gamma,
                coefficient=self.coefficient,
                scale=self.scale,
            )
            for t in transitions
        ]

        state = encode_boards(boards, players, prev_moves)
        next_state = encode_boards(next_boards, next_players, next_moves)
        done_tensor = torch.tensor(dones, dtype=torch.bool)
        next_state[done_tensor] = 0.0

        return (
            state,
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            next_state,
            done_tensor,
            torch.tensor(next_legal_masks, dtype=torch.bool),
        )

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        max_steps_from_end: int | None = None,
    ) -> Batch:
        """Sample a batch of transitions.

        `max_steps_from_end`, if set, restricts sampling to transitions within that many
        plies of their game's terminal state (curriculum learning: 1 covers only the
        terminal ply, 2 covers the terminal ply and the one before it, and so on).
        """
        if max_steps_from_end is not None:
            pool: list[int] = []
            for distance in range(max_steps_from_end):
                pool.extend(self._distance_buckets.get(distance, ()))
            if len(pool) < batch_size:
                raise ValueError(
                    f"Not enough transitions within {max_steps_from_end} step(s) of the end to "
                    f"sample: have {len(pool)}, need {batch_size}."
                )
            if generator is not None:
                choice = torch.randperm(len(pool), generator=generator)[:batch_size]
            else:
                choice = torch.randperm(len(pool))[:batch_size]
            indices = torch.tensor([pool[i] for i in choice.tolist()], dtype=torch.long)
        elif len(self._transitions) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample: have {len(self._transitions)}, need {batch_size}."
            )
        elif generator is not None:
            indices = torch.randperm(len(self._transitions), generator=generator)[:batch_size]
        else:
            indices = torch.randperm(len(self._transitions))[:batch_size]

        transitions = [self._transitions[index] for index in indices.tolist()]
        state, action, reward, next_state, done, next_legal_mask = self._encode_batch(transitions)

        # Each sample gets its own random D4 symmetry (as before), but samples sharing a
        # transform are augmented together in one vectorized op instead of one-by-one.
        transform_indices = [random_transform_index(generator) for _ in range(len(transitions))]
        buckets: dict[int, list[int]] = defaultdict(list)
        for position, transform_index in enumerate(transform_indices):
            buckets[transform_index].append(position)

        for transform_index, positions in buckets.items():
            rows = torch.tensor(positions, dtype=torch.long)
            new_state, new_next_state, new_action, new_legal_mask = apply_transform_batch(
                transform_index, state[rows], next_state[rows], action[rows], next_legal_mask[rows]
            )
            state[rows] = new_state
            next_state[rows] = new_next_state
            action[rows] = new_action
            next_legal_mask[rows] = new_legal_mask

        return state, action, reward, next_state, done, next_legal_mask
