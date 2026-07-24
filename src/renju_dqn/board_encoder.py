"""Board -> (C, 15, 15) image tensor encoding, plus D4 augmentation utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE, legal_move_mask

NUM_CHANNELS = 4  # own stones / opponent stones / last move / side-to-move constant


def _other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def encode_board(board: Sequence[int], player: int, last_move: int | None) -> torch.Tensor:
    """Encode `board` from `player`'s point of view (about to move) as a (C, 15, 15) tensor."""
    if len(board) != BOARD_CELLS:
        raise ValueError(f"Expected {BOARD_CELLS} cells, got {len(board)}.")

    opponent = _other_player(player)
    board_tensor = torch.tensor(board, dtype=torch.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    own_plane = (board_tensor == player).float()
    opponent_plane = (board_tensor == opponent).float()

    last_move_plane = torch.zeros(BOARD_SIZE, BOARD_SIZE)
    if last_move is not None:
        row, col = divmod(last_move, BOARD_SIZE)
        last_move_plane[row, col] = 1.0

    side_to_move_plane = torch.full((BOARD_SIZE, BOARD_SIZE), float(player == BLACK))

    return torch.stack((own_plane, opponent_plane, last_move_plane, side_to_move_plane), dim=0)


def encode_legal_move_mask(board: Sequence[int]) -> torch.Tensor:
    return torch.tensor(legal_move_mask(list(board)), dtype=torch.bool)


# --- D4 (dihedral group of order 8) board symmetries -----------------------------------

BoardTransform = Callable[[torch.Tensor], torch.Tensor]

D4_TRANSFORMS: tuple[BoardTransform, ...] = (
    lambda x: x,  # identity
    lambda x: torch.rot90(x, k=1, dims=(-2, -1)),  # rotate 90
    lambda x: torch.rot90(x, k=2, dims=(-2, -1)),  # rotate 180
    lambda x: torch.rot90(x, k=3, dims=(-2, -1)),  # rotate 270
    lambda x: torch.flip(x, dims=(-1,)),  # mirror left-right
    lambda x: torch.flip(x, dims=(-2,)),  # mirror top-bottom
    lambda x: x.transpose(-2, -1),  # mirror main diagonal
    lambda x: torch.flip(x.transpose(-2, -1), dims=(-2, -1)),  # mirror anti-diagonal
)


def _build_action_permutations() -> tuple[torch.Tensor, ...]:
    """For each transform, `perm[old_action]` gives the transformed action index."""
    index_grid = torch.arange(BOARD_CELLS).reshape(BOARD_SIZE, BOARD_SIZE)
    permutations = []
    for transform in D4_TRANSFORMS:
        # forward[new_flat] = old_flat that ends up at new_flat after the transform.
        forward = transform(index_grid).reshape(-1)
        permutations.append(torch.argsort(forward))
    return tuple(permutations)


_ACTION_PERMUTATIONS = _build_action_permutations()


def random_transform_index(generator: torch.Generator | None = None) -> int:
    return int(torch.randint(0, len(D4_TRANSFORMS), (1,), generator=generator).item())


def apply_transform(
    transform_index: int,
    state: torch.Tensor,
    next_state: torch.Tensor,
    action: int,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Apply the same D4 symmetry to a transition's state/next_state/action/legal mask."""
    transform = D4_TRANSFORMS[transform_index]
    new_state = transform(state)
    new_next_state = transform(next_state)
    new_action = int(_ACTION_PERMUTATIONS[transform_index][action].item())
    new_legal_mask = transform(legal_mask.reshape(BOARD_SIZE, BOARD_SIZE)).reshape(-1)
    return new_state, new_next_state, new_action, new_legal_mask


def random_augment(
    state: torch.Tensor,
    next_state: torch.Tensor,
    action: int,
    legal_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    transform_index = random_transform_index(generator)
    return apply_transform(transform_index, state, next_state, action, legal_mask)
