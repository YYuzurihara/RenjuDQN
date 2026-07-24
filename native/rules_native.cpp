// Native (C++) port of renju_dqn.rules' hot paths (forbidden-move / five-in-a-row
// checks), exposed to Python via pybind11. Kept in exact algorithmic lock-step with
// rules.py so the two stay interchangeable; rules.py falls back to its pure-Python
// implementation when this extension isn't built.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int BOARD_SIZE = 15;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int CENTER_INDEX = (BOARD_SIZE / 2) * BOARD_SIZE + (BOARD_SIZE / 2);

constexpr std::array<std::pair<int, int>, 4> DIRECTIONS = {{
    {1, 0},
    {0, 1},
    {1, 1},
    {1, -1},
}};

using Board = std::vector<int>;

inline std::pair<int, int> idx_to_rc(int index) {
    return {index / BOARD_SIZE, index % BOARD_SIZE};
}

inline int rc_to_idx(int row, int col) {
    return row * BOARD_SIZE + col;
}

inline bool inside(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

int contiguous_count(const Board& board, int index, int player, int dr, int dc) {
    int total = 1;
    auto [row, col] = idx_to_rc(index);

    for (int step = 1;; ++step) {
        int r = row + dr * step;
        int c = col + dc * step;
        if (!inside(r, c) || board[rc_to_idx(r, c)] != player) break;
        ++total;
    }
    for (int step = 1;; ++step) {
        int r = row - dr * step;
        int c = col - dc * step;
        if (!inside(r, c) || board[rc_to_idx(r, c)] != player) break;
        ++total;
    }
    return total;
}

bool has_five_or_more(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {
        if (contiguous_count(board, index, player, dr, dc) >= 5) return true;
    }
    return false;
}

bool is_overline(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {
        if (contiguous_count(board, index, player, dr, dc) >= 6) return true;
    }
    return false;
}

bool player_has_five(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[index] == player && has_five_or_more(board, index, player)) return true;
    }
    return false;
}

bool player_has_overline(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[index] == player && is_overline(board, index, player)) return true;
    }
    return false;
}

std::vector<int> line_points_through(int index, int dr, int dc) {
    auto [row, col] = idx_to_rc(index);
    while (inside(row - dr, col - dc)) {
        row -= dr;
        col -= dc;
    }
    std::vector<int> points;
    while (inside(row, col)) {
        points.push_back(rc_to_idx(row, col));
        row += dr;
        col += dc;
    }
    return points;
}

// Candidate indices along `line_points` where `player` playing there completes 5-in-a-row.
// Takes `board` by mutable reference and restores every cell it touches, so callers can chain
// these checks without paying for a full 225-cell copy at each nesting level.
std::vector<int> immediate_wins_in_direction(
    Board& board, int player, const std::vector<int>& line_points
) {
    std::vector<int> wins;
    for (int candidate : line_points) {
        if (board[candidate] != EMPTY) continue;
        board[candidate] = player;
        bool overline = player == BLACK && is_overline(board, candidate, BLACK);
        bool five = !overline && has_five_or_more(board, candidate, player);
        board[candidate] = EMPTY;
        if (five) wins.push_back(candidate);
    }
    return wins;
}

int count_four_directions_impl(Board& board, int move, int player) {
    int count = 0;
    for (const auto& [dr, dc] : DIRECTIONS) {
        auto line_points = line_points_through(move, dr, dc);
        if (!immediate_wins_in_direction(board, player, line_points).empty()) ++count;
    }
    return count;
}

int count_four_directions(const Board& board, int move, int player) {
    Board working = board;
    return count_four_directions_impl(working, move, player);
}

int count_open_three_directions_impl(Board& board, int move, int player) {
    int count = 0;
    for (const auto& [dr, dc] : DIRECTIONS) {
        auto line_points = line_points_through(move, dr, dc);
        bool found_open_three = false;
        for (int candidate : line_points) {
            if (board[candidate] != EMPTY) continue;
            board[candidate] = player;
            bool overline = player == BLACK && is_overline(board, candidate, BLACK);
            std::size_t winning_count =
                overline ? 0 : immediate_wins_in_direction(board, player, line_points).size();
            board[candidate] = EMPTY;
            if (!overline && winning_count >= 2) {
                found_open_three = true;
                break;
            }
        }
        if (found_open_three) ++count;
    }
    return count;
}

int count_open_three_directions(const Board& board, int move, int player) {
    Board working = board;
    return count_open_three_directions_impl(working, move, player);
}

std::pair<int, int> stone_counts(const Board& board) {
    int black_count = 0;
    int white_count = 0;
    for (int cell : board) {
        if (cell == BLACK) {
            ++black_count;
        } else if (cell == WHITE) {
            ++white_count;
        }
    }
    return {black_count, white_count};
}

int infer_player(const Board& board) {
    auto [black_count, white_count] = stone_counts(board);
    if (black_count == white_count) return BLACK;
    if (black_count == white_count + 1) return WHITE;
    throw std::invalid_argument(
        "Invalid board: black_count=" + std::to_string(black_count) +
        ", white_count=" + std::to_string(white_count) +
        ". Expected black == white or black == white + 1."
    );
}

bool is_forbidden_for_black(const Board& board, int index) {
    if (board[index] != EMPTY) return true;

    auto [black_count, white_count] = stone_counts(board);
    int move_number = black_count + white_count;
    if (move_number == 0) return index != CENTER_INDEX;

    Board next_board = board;
    next_board[index] = BLACK;
    if (is_overline(next_board, index, BLACK)) return true;
    if (count_four_directions_impl(next_board, index, BLACK) >= 2) return true;
    if (count_open_three_directions_impl(next_board, index, BLACK) >= 2) return true;
    return false;
}

std::vector<bool> legal_move_mask(const Board& board) {
    int player = infer_player(board);
    std::vector<bool> mask(BOARD_CELLS);
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[index] != EMPTY) {
            mask[index] = false;
            continue;
        }
        mask[index] = (player == BLACK) ? !is_forbidden_for_black(board, index) : true;
    }
    return mask;
}

std::optional<int> winner_after_move(const Board& board, int index, int player) {
    if (player == BLACK && is_overline(board, index, BLACK)) return WHITE;
    if (has_five_or_more(board, index, player)) return player;
    return std::nullopt;
}

std::optional<int> board_winner(const Board& board) {
    if (player_has_overline(board, BLACK)) return WHITE;
    if (player_has_five(board, BLACK)) return BLACK;
    if (player_has_five(board, WHITE)) return WHITE;
    return std::nullopt;
}

}  // namespace

PYBIND11_MODULE(_rules_native, m) {
    m.doc() = "Native implementation of renju_dqn.rules hot paths.";
    // call_guard<gil_scoped_release> lets these run concurrently with other Python threads
    // (e.g. a training thread's GPU work) instead of serializing on the GIL. Argument/return
    // marshalling to/from the Python board list still happens under the GIL either side of it.
    m.def(
        "legal_move_mask", &legal_move_mask, py::arg("board"), py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "winner_after_move",
        &winner_after_move,
        py::arg("board"),
        py::arg("index"),
        py::arg("player"),
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "board_winner", &board_winner, py::arg("board"), py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "is_forbidden_for_black",
        &is_forbidden_for_black,
        py::arg("board"),
        py::arg("index"),
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "infer_player", &infer_player, py::arg("board"), py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "count_four_directions",
        &count_four_directions,
        py::arg("board"),
        py::arg("move"),
        py::arg("player"),
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "count_open_three_directions",
        &count_open_three_directions,
        py::arg("board"),
        py::arg("move"),
        py::arg("player"),
        py::call_guard<py::gil_scoped_release>()
    );
}
