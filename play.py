import curses

from viz import draw_grid

POSITIONS = {"P1": [], "P2": []}



def main(stdscr):
    # This allows input to be printed on screen.
    curses.echo()

    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)

    draw_grid(stdscr)

    drop_to_col = stdscr.getstr().decode(encoding="utf-8")
    draw_grid(stdscr, drop_to_col=drop_to_col)
    stdscr.getch()


curses.wrapper(main)
