# Horizontal and vertical padding.
PY = 2
PX = 6

NUM_COLS = 7
NUM_ROWS = 6

def draw_grid(stdscr, drop_to_col=None):
    """Draw the game grid.

         1   2   3
     0 |   |   |   |
     1 +---+---+---+...
     2 |   |   |   |
     3 +---+---+---+...
y    4 |   |   |   |
axis 5 +---+---+---+...
     6 |   |   | O |
     7 +---+---+---+...
     8 |   | X | O |
     9 +---+---+---+...
    10 | X | X | O |
    11 +---+---+---+...
       0123456789012...
       x axis
    """
    stdscr.clear()

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

    # Add message for player.
    if drop_to_col is None:
        stdscr.addstr((NUM_ROWS + 2) * 2 + PY, 0 + PX, "Enter column number: ")
    else:
        # If user is making a drop, then show the column number.
        stdscr.addstr((NUM_ROWS + 2) * 2 + PY, 0 + PX, f"Dropping to column: {drop_to_col}")
    stdscr.refresh()

