from utils import NUM_ROWS, NUM_COLS

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
