"""Renju move legality helpers used during inference."""

from __future__ import annotations

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
CENTER_INDEX = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)


def idx_to_rc(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)


def rc_to_idx(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def board_with_move(board: list[int], index: int, player: int) -> list[int]:
    next_board = board.copy()
    next_board[index] = player
    return next_board


def stone_counts(board: list[int]) -> tuple[int, int]:
    black_count = sum(1 for cell in board if cell == BLACK)
    white_count = sum(1 for cell in board if cell == WHITE)
    return black_count, white_count


def move_number(board: list[int]) -> int:
    black_count, white_count = stone_counts(board)
    return black_count + white_count


def contiguous_count(board: list[int], index: int, player: int, dr: int, dc: int) -> int:
    total = 1
    row, col = idx_to_rc(index)

    step = 1
    while inside(row + dr * step, col + dc * step):
        if board[rc_to_idx(row + dr * step, col + dc * step)] != player:
            break
        total += 1
        step += 1

    step = 1
    while inside(row - dr * step, col - dc * step):
        if board[rc_to_idx(row - dr * step, col - dc * step)] != player:
            break
        total += 1
        step += 1

    return total


def has_five_or_more(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 5 for dr, dc in DIRECTIONS)


def is_overline(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 6 for dr, dc in DIRECTIONS)


def player_has_five(board: list[int], player: int) -> bool:
    return any(cell == player and has_five_or_more(board, index, player) for index, cell in enumerate(board))


def player_has_overline(board: list[int], player: int) -> bool:
    return any(cell == player and is_overline(board, index, player) for index, cell in enumerate(board))


def line_points_through(index: int, dr: int, dc: int) -> list[int]:
    row, col = idx_to_rc(index)
    while inside(row - dr, col - dc):
        row -= dr
        col -= dc

    points: list[int] = []
    while inside(row, col):
        points.append(rc_to_idx(row, col))
        row += dr
        col += dc
    return points


def immediate_wins_in_direction(board: list[int], player: int, line_points: list[int]) -> set[int]:
    wins: set[int] = set()
    for candidate in line_points:
        if board[candidate] != EMPTY:
            continue
        next_board = board_with_move(board, candidate, player)
        if player == BLACK and is_overline(next_board, candidate, BLACK):
            continue
        if has_five_or_more(next_board, candidate, player):
            wins.add(candidate)
    return wins


def _count_four_directions_python(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points = line_points_through(move, dr, dc)
        if immediate_wins_in_direction(board, player, line_points):
            count += 1
    return count


def _count_open_three_directions_python(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points = line_points_through(move, dr, dc)
        found_open_three = False
        for candidate in line_points:
            if board[candidate] != EMPTY:
                continue
            next_board = board_with_move(board, candidate, player)
            if player == BLACK and is_overline(next_board, candidate, BLACK):
                continue
            winning_points = immediate_wins_in_direction(next_board, player, line_points)
            if len(winning_points) >= 2:
                found_open_three = True
                break
        if found_open_three:
            count += 1
    return count


def _infer_player_python(board: list[int]) -> int:
    black_count, white_count = stone_counts(board)
    if black_count == white_count:
        return BLACK
    if black_count == white_count + 1:
        return WHITE
    raise ValueError(
        f"Invalid board: black_count={black_count}, white_count={white_count}. "
        "Expected black == white or black == white + 1."
    )


def _is_forbidden_for_black_python(board: list[int], index: int) -> bool:
    if board[index] != EMPTY:
        return True

    black_count, white_count = stone_counts(board)
    move_number = black_count + white_count

    if move_number == 0:
        return index != CENTER_INDEX

    next_board = board_with_move(board, index, BLACK)
    if is_overline(next_board, index, BLACK):
        return True
    if count_four_directions(next_board, index, BLACK) >= 2:
        return True
    if count_open_three_directions(next_board, index, BLACK) >= 2:
        return True
    return False


def _legal_move_mask_python(board: list[int]) -> list[bool]:
    player = infer_player(board)
    mask: list[bool] = []
    for index, cell in enumerate(board):
        if cell != EMPTY:
            mask.append(False)
            continue
        if player == BLACK:
            mask.append(not is_forbidden_for_black(board, index))
        else:
            mask.append(True)
    return mask


def _winner_after_move_python(board: list[int], index: int, player: int) -> int | None:
    if player == BLACK and is_overline(board, index, BLACK):
        return WHITE
    if has_five_or_more(board, index, player):
        return player
    return None


def _board_winner_python(board: list[int]) -> int | None:
    if player_has_overline(board, BLACK):
        return WHITE
    if player_has_five(board, BLACK):
        return BLACK
    if player_has_five(board, WHITE):
        return WHITE
    return None


# --- Native (pybind11) dispatch -----------------------------------------------------
#
# The hot paths above (forbidden-move / five-in-a-row scans) are also implemented in
# `native/rules_native.cpp`; when that extension is built, the functions below route to
# it instead of the pure-Python implementations, which stay as the portable fallback.
try:
    from . import _rules_native as _native
except ImportError:
    _native = None


def infer_player(board: list[int]) -> int:
    if _native is not None:
        return _native.infer_player(board)
    return _infer_player_python(board)


def is_forbidden_for_black(board: list[int], index: int) -> bool:
    if _native is not None:
        return _native.is_forbidden_for_black(board, index)
    return _is_forbidden_for_black_python(board, index)


def legal_move_mask(board: list[int]) -> list[bool]:
    if _native is not None:
        return _native.legal_move_mask(board)
    return _legal_move_mask_python(board)


def winner_after_move(board: list[int], index: int, player: int) -> int | None:
    if _native is not None:
        return _native.winner_after_move(board, index, player)
    return _winner_after_move_python(board, index, player)


def board_winner(board: list[int]) -> int | None:
    if _native is not None:
        return _native.board_winner(board)
    return _board_winner_python(board)


def count_four_directions(board: list[int], move: int, player: int) -> int:
    if _native is not None:
        return _native.count_four_directions(board, move, player)
    return _count_four_directions_python(board, move, player)


def count_open_three_directions(board: list[int], move: int, player: int) -> int:
    if _native is not None:
        return _native.count_open_three_directions(board, move, player)
    return _count_open_three_directions_python(board, move, player)
