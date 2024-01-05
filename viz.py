import curses
from time import sleep

from utils import NUM_ROWS, NUM_COLS, get_row_num

# Horizontal and vertical padding.
PY = 2
PX = 6

def get_symbol(player):
    return "X" if player == "P1" else "O"


def convert_to_grid_pos(row_num, col_num):
    """Given row_num and col_num, convert to y and x position on grid."""
    y = 2 + (NUM_ROWS - row_num) * 2
    x = (col_num - 1) * 4 + 2
    return y, x


def draw_grid(stdscr, positions):
    """Draw the game grid.

         1   2   3
     0 |   |   |   |     6
     1 +---+---+---+...
     2 |   |   |   |     5
     3 +---+---+---+...
y    4 |   |   |   |     4
axis 5 +---+---+---+...
     6 |   |   | O |     3
     7 +---+---+---+...
     8 |   | X | O |     2
     9 +---+---+---+...
    10 | X | X | O |     1
    11 +---+---+---+...
       0123456789012...
       x axis
    """

    # The horizontal lines.
    for i in range(NUM_ROWS + 1):
        y = 1 + i * 2
        for x in range(NUM_COLS * 4 + 1):
            symbol = "+" if x % 4 == 0 else "-"
            stdscr.addstr(y + PY, x + PX, symbol)

    # The vertical lines; eight of them.
    for i in range(NUM_ROWS):
        y = 2 + i * 2
        for i in range(NUM_COLS + 1):
            x = i * 4
            stdscr.addstr(y + PY, x + PX, "|")

    # Add the column numbers on top.
    for i in range(NUM_COLS):
        x = i * 4 + 2
        stdscr.addstr(0 + PY, x + PX, str(i + 1))


    for pos in positions["P1"]:
        y, x = convert_to_grid_pos(*pos)
        stdscr.addstr(y + PY, x + PX, get_symbol("P1"), curses.color_pair(1) | curses.A_BOLD)


    for pos in positions["P2"]:
        y, x = convert_to_grid_pos(*pos)
        stdscr.addstr(y + PY, x + PX, get_symbol("P2"), curses.color_pair(2) | curses.A_BOLD)



def user_prompt_which_col(stdscr, curr_player):
    symbol = get_symbol(curr_player)
    stdscr.addstr(
        (NUM_ROWS + 2) * 2 + PY,
        0 + PX,
        f"Enter column number for {curr_player} ({symbol}): "
    )


def user_prompt_winner(stdscr, curr_player):
    stdscr.addstr(
        (NUM_ROWS + 2) * 2 + PY,
        0 + PX,
        f"Congratulations, {curr_player}! You won!"
    )
    stdscr.addstr(
        (NUM_ROWS + 2) * 2 + PY + 1,
        0 + PX,
        f"Press any key to exit."
    )


def user_prompt_tied(stdscr):
    stdscr.addstr(
        (NUM_ROWS + 2) * 2 + PY,
        0 + PX,
        f"Wow how peaceful, you tied!"
    )
    stdscr.addstr(
        (NUM_ROWS + 2) * 2 + PY + 1,
        0 + PX,
        f"Press any key to exit."
    )


def draw_game_board(stdscr, positions, curr_player, drop_to_col=None):
    symbol = get_symbol(curr_player)
    color = curses.color_pair(1) if curr_player == "P1" else curses.color_pair(2)

    if drop_to_col is None:
        stdscr.clear()
        draw_grid(stdscr, positions)
        user_prompt_which_col(stdscr, curr_player)
        stdscr.refresh()
    else:
        row_num = get_row_num(positions, drop_to_col)
        curr_row_num = NUM_ROWS
        while curr_row_num >= row_num:
            stdscr.clear()
            y, x = convert_to_grid_pos(curr_row_num, drop_to_col)
            draw_grid(stdscr, positions)
            stdscr.addstr(y + PY, x + PX, symbol, color | curses.A_BOLD)
            stdscr.move((NUM_ROWS + 2) * 2 + PY, 0 + PX)
            curr_row_num -= 1
            sleep(0.1)
            stdscr.refresh()



