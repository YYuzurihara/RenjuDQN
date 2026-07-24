from __future__ import annotations

import torch

from renju_dqn.board_encoder import (
    D4_TRANSFORMS,
    NUM_CHANNELS,
    apply_transform,
    encode_board,
    encode_legal_move_mask,
    random_augment,
)
from renju_dqn.rules import BLACK, BOARD_CELLS, BOARD_SIZE, WHITE, legal_move_mask, rc_to_idx


def empty_board() -> list[int]:
    return [0] * BOARD_CELLS


def test_encode_board_marks_own_and_opponent_planes():
    board = empty_board()
    board[rc_to_idx(3, 4)] = BLACK
    board[rc_to_idx(5, 6)] = WHITE

    state = encode_board(board, player=BLACK, last_move=None)

    assert state.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert state[0, 3, 4] == 1.0
    assert state[1, 5, 6] == 1.0
    assert state[0].sum() == 1.0
    assert state[1].sum() == 1.0


def test_encode_board_last_move_and_side_to_move_planes():
    board = empty_board()
    state_no_last_move = encode_board(board, player=BLACK, last_move=None)
    assert state_no_last_move[2].sum() == 0.0
    assert torch.all(state_no_last_move[3] == 1.0)

    state_white_to_move = encode_board(board, player=WHITE, last_move=rc_to_idx(7, 7))
    assert state_white_to_move[2, 7, 7] == 1.0
    assert state_white_to_move[2].sum() == 1.0
    assert torch.all(state_white_to_move[3] == 0.0)


def test_encode_legal_move_mask_matches_rules():
    board = empty_board()
    board[rc_to_idx(7, 7)] = BLACK
    expected = torch.tensor(legal_move_mask(board), dtype=torch.bool)
    assert torch.equal(encode_legal_move_mask(board), expected)


def test_d4_transforms_keep_action_and_mask_consistent_with_state():
    for transform_index in range(len(D4_TRANSFORMS)):
        for action in (rc_to_idx(0, 0), rc_to_idx(3, 11), rc_to_idx(7, 7), rc_to_idx(14, 2)):
            state = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            row, col = divmod(action, BOARD_SIZE)
            state[0, row, col] = 1.0
            next_state = state.clone()
            legal_mask = torch.zeros(BOARD_CELLS, dtype=torch.bool)
            legal_mask[action] = True

            new_state, new_next_state, new_action, new_legal_mask = apply_transform(
                transform_index, state, next_state, action, legal_mask
            )

            stone_rows, stone_cols = torch.where(new_state[0] == 1.0)
            assert (int(stone_rows.item()), int(stone_cols.item())) == divmod(
                new_action, BOARD_SIZE
            )
            assert torch.equal(new_state, new_next_state)
            assert new_legal_mask.sum().item() == 1
            assert new_legal_mask[new_action].item() is True


def test_identity_transform_is_a_noop():
    state = torch.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    legal_mask = torch.zeros(BOARD_CELLS, dtype=torch.bool)
    legal_mask[rc_to_idx(2, 9)] = True

    new_state, new_next_state, new_action, new_legal_mask = apply_transform(
        0, state, state.clone(), rc_to_idx(2, 9), legal_mask
    )

    assert torch.equal(new_state, state)
    assert torch.equal(new_next_state, state)
    assert new_action == rc_to_idx(2, 9)
    assert torch.equal(new_legal_mask, legal_mask)


def test_random_augment_picks_one_of_the_eight_transforms():
    state = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    action = rc_to_idx(7, 7)
    state[0, 7, 7] = 1.0
    legal_mask = torch.zeros(BOARD_CELLS, dtype=torch.bool)
    legal_mask[action] = True

    generator = torch.Generator().manual_seed(0)
    new_state, _, new_action, new_legal_mask = random_augment(
        state, state.clone(), action, legal_mask, generator=generator
    )

    assert new_state[0].sum() == 1.0
    assert new_legal_mask[new_action].item() is True
