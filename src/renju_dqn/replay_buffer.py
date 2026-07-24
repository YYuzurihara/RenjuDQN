"""Fixed-capacity FIFO replay buffer, warmed up from MCTS logs and grown online.

Stores raw `(board, player, move, winner, prev_move, next_board, done)` transitions
(the same shape `dataset.ReplayDataset` reconstructs from CSV rows) rather than
pre-encoded tensors: a board is 225 small ints, so keeping the buffer at
capacity in raw form is cheap, and it lets sampling apply a fresh random D4
augmentation each time instead of baking one in at insert time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .board_encoder import NUM_CHANNELS, encode_board, encode_legal_move_mask, random_augment
from .dataset import ReplayDataset
from .env import GameRow
from .reward import DEFAULT_SHAPING_COEFFICIENT, POTENTIAL_SCALE, compute_reward
from .rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE

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
    ) -> None:
        transition = _Transition(
            board=list(board),
            player=player,
            move=move,
            winner=winner,
            prev_move=prev_move,
            next_board=list(next_board) if next_board is not None else None,
            done=done,
        )
        if len(self._transitions) < self.capacity:
            self._transitions.append(transition)
        else:
            self._transitions[self._write_index] = transition
            self._write_index = (self._write_index + 1) % self.capacity

    def push_game(self, rows: list[GameRow], winner: int) -> None:
        """Push one finished self-play game's `(board, player, move)` rows, in ply order."""
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
            )

    def _encode(
        self, transition: _Transition
    ) -> tuple[torch.Tensor, int, float, torch.Tensor, bool, torch.Tensor]:
        state = encode_board(transition.board, transition.player, transition.prev_move)

        if transition.done:
            next_state = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            next_board_for_reward = transition.board
            next_legal_mask = torch.zeros(BOARD_CELLS, dtype=torch.bool)
        else:
            next_player = _other_player(transition.player)
            next_state = encode_board(transition.next_board, next_player, transition.move)
            next_board_for_reward = transition.next_board
            next_legal_mask = encode_legal_move_mask(transition.next_board)

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
        return state, transition.move, reward, next_state, transition.done, next_legal_mask

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        if len(self._transitions) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample: have {len(self._transitions)}, need {batch_size}."
            )

        if generator is not None:
            indices = torch.randperm(len(self._transitions), generator=generator)[:batch_size]
        else:
            indices = torch.randperm(len(self._transitions))[:batch_size]

        states, actions, rewards, next_states, dones, next_legal_masks = [], [], [], [], [], []
        for index in indices.tolist():
            state, action, reward, next_state, done, next_legal_mask = self._encode(
                self._transitions[index]
            )
            state, next_state, action, next_legal_mask = random_augment(
                state, next_state, action, next_legal_mask, generator=generator
            )
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            next_legal_masks.append(next_legal_mask)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool),
            torch.stack(next_legal_masks),
        )
