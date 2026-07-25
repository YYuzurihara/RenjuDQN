// Native (C++) port of renju_dqn.rules' hot paths (forbidden-move / five-in-a-row
// checks), exposed to Python via pybind11. Kept in exact algorithmic lock-step with
// rules.py so the two stay interchangeable; rules.py falls back to its pure-Python
// implementation when this extension isn't built.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
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

// `board[index]` must already be EMPTY. Mutates `board` transiently (sets `index` to BLACK to
// probe, then restores it) instead of copying, and takes `move_number` precomputed by the
// caller instead of rescanning all 225 cells -- both are invariant across every candidate index
// checked for the same board, so hoisting them out of the per-candidate work matters a lot once
// this runs once per empty cell (legal_move_mask below checks up to ~BOARD_CELLS candidates).
bool is_forbidden_for_black_impl(Board& board, int index, int move_number) {
    if (move_number == 0) return index != CENTER_INDEX;

    board[index] = BLACK;
    bool overline = is_overline(board, index, BLACK);
    bool forbidden = overline || count_four_directions_impl(board, index, BLACK) >= 2 ||
                      count_open_three_directions_impl(board, index, BLACK) >= 2;
    board[index] = EMPTY;
    return forbidden;
}

bool is_forbidden_for_black(const Board& board, int index) {
    if (board[index] != EMPTY) return true;
    auto [black_count, white_count] = stone_counts(board);
    Board working = board;
    return is_forbidden_for_black_impl(working, index, black_count + white_count);
}

std::vector<bool> legal_move_mask(const Board& board) {
    int player = infer_player(board);
    std::vector<bool> mask(BOARD_CELLS);
    if (player != BLACK) {
        for (int index = 0; index < BOARD_CELLS; ++index) {
            mask[index] = board[index] == EMPTY;
        }
        return mask;
    }

    auto [black_count, white_count] = stone_counts(board);
    int move_number = black_count + white_count;
    Board working = board;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[index] != EMPTY) {
            mask[index] = false;
            continue;
        }
        mask[index] = !is_forbidden_for_black_impl(working, index, move_number);
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

// --- Reward shaping (renju_dqn.reward's hot path) -----------------------------------
//
// board_potential sums a per-stone threat score over every occupied cell, so it's O(stones)
// native work per board rather than the O(1) rule checks above. reward.py used to do this loop
// in Python, calling count_four_directions/count_open_three_directions (each a separate
// GIL-releasing pybind11 call) once per stone; moving the whole loop here turns "O(stones)
// Python<->native round trips per board" into one round trip per board.
constexpr int FOUR_WEIGHT = 9;
constexpr int OPEN_THREE_WEIGHT = 3;
constexpr double POTENTIAL_SCALE = 5.0;
constexpr double DEFAULT_SHAPING_COEFFICIENT = 0.1;
constexpr int DRAW = 0;

int stone_threat_score_impl(Board& board, int index, int player) {
    int fours = count_four_directions_impl(board, index, player);
    int open_threes = count_open_three_directions_impl(board, index, player);
    return FOUR_WEIGHT * fours + OPEN_THREE_WEIGHT * open_threes;
}

// `working` is mutated transiently per stone and always restored by the impl functions above,
// so one buffer can be reused across every stone instead of copying `board` per candidate.
double board_potential_impl(const Board& board, int player) {
    int opponent = (player == BLACK) ? WHITE : BLACK;
    Board working = board;
    long long own_score = 0;
    long long opponent_score = 0;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        int cell = board[index];
        if (cell == player) {
            own_score += stone_threat_score_impl(working, index, player);
        } else if (cell == opponent) {
            opponent_score += stone_threat_score_impl(working, index, opponent);
        }
    }
    return static_cast<double>(own_score - opponent_score);
}

double board_potential(const Board& board, int player) {
    return board_potential_impl(board, player);
}

double normalized_potential_impl(const Board& board, int player, double scale) {
    return std::tanh(board_potential_impl(board, player) / scale);
}

double normalized_potential(const Board& board, int player, double scale) {
    return normalized_potential_impl(board, player, scale);
}

double terminal_reward(int player, int winner) {
    if (winner == DRAW) return 0.0;
    return winner == player ? 1.0 : -1.0;
}

double compute_reward_impl(
    const Board& board,
    const Board& next_board,
    int player,
    int winner,
    bool done,
    double gamma,
    double coefficient,
    double scale
) {
    double sparse = done ? terminal_reward(player, winner) : 0.0;
    double phi_t = normalized_potential_impl(board, player, scale);
    double phi_next = done ? 0.0 : normalized_potential_impl(next_board, player, scale);
    return sparse + coefficient * (gamma * phi_next - phi_t);
}

double compute_reward(
    const Board& board,
    const Board& next_board,
    int player,
    int winner,
    bool done,
    double gamma,
    double coefficient,
    double scale
) {
    return compute_reward_impl(board, next_board, player, winner, done, gamma, coefficient, scale);
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
    m.def(
        "board_potential",
        &board_potential,
        py::arg("board"),
        py::arg("player"),
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "normalized_potential",
        &normalized_potential,
        py::arg("board"),
        py::arg("player"),
        py::arg("scale") = POTENTIAL_SCALE,
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "compute_reward",
        &compute_reward,
        py::arg("board"),
        py::arg("next_board"),
        py::arg("player"),
        py::arg("winner"),
        py::arg("done"),
        py::arg("gamma"),
        py::arg("coefficient") = DEFAULT_SHAPING_COEFFICIENT,
        py::arg("scale") = POTENTIAL_SCALE,
        py::call_guard<py::gil_scoped_release>()
    );
}
