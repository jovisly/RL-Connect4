import random

from utils import NUM_COLS, NUM_ROWS

LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.8
NUM_GAMES = 10_000

def convert_positions_to_board(positions, player):
    """Given {"P1": [(2, 3), ...], "P2": [(1, 1), ...]}, convert to board.

    Board is a list with 42 elements. Each element is 1 if it's occupied by the
    player; -1 if it's occupied by the opponent; and 0 if it's empty.
    """
    opponent = "P1" if player == "P2" else "P2"
    player_moves = positions[player]
    opp_moves = positions[opponent]
    board = []

    for r in range(NUM_ROWS):
        row_num = r + 1

        for c in range(NUM_COLS):
            col_num = c + 1

            if (row_num, col_num) in player_moves:
                board.append(1)
            elif (row_num, col_num) in opp_moves:
                board.append(-1)
            else:
                board.append(0)
    return board


def mini_tests():
    # Small tests.
    positions = {
        "P1": [(1, 4), (2, 4), (3, 4), (4, 4)],
        "P2": [(1, 3), (2, 3), (3, 3)]
    }
    player = "P1"
    board = convert_positions_to_board(positions, player)
    assert board == [
        0, 0, -1, 1, 0, 0, 0,
        0, 0, -1, 1, 0, 0, 0,
        0, 0, -1, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
    ]

    player = "P2"
    board = convert_positions_to_board(positions, player)
    assert board == [
        0, 0, 1, -1, 0, 0, 0,
        0, 0, 1, -1, 0, 0, 0,
        0, 0, 1, -1, 0, 0, 0,
        0, 0, 0, -1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
    ]


if __name__ == "__main__":
    mini_tests()
