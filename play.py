import curses

from viz import (
    draw_game_board,
    user_prompt_winner,
    user_prompt_tied,
)
from utils import (
    is_won,
    get_available_col_nums,
    get_row_num,
    switch_player
)


def main(stdscr):
    # This allows input to be printed on screen.
    curses.echo()

    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    curr_player = "P1"
    positions = {"P1": [], "P2": []}

    while True:
        draw_game_board(stdscr, positions=positions, curr_player=curr_player)

        col_num = int(stdscr.getstr().decode(encoding="utf-8"))
        available_col_nums = get_available_col_nums(positions)

        while col_num not in available_col_nums:
            draw_game_board(stdscr, positions=positions, curr_player=curr_player, invalid_col_num=col_num)
            col_num = int(stdscr.getstr().decode(encoding="utf-8"))

        row_num = get_row_num(positions, col_num)
        draw_game_board(
            stdscr, positions=positions, curr_player=curr_player, drop_to_col=col_num
        )
        positions[curr_player].append((row_num, col_num))

        if is_won(positions):
            user_prompt_winner(stdscr, curr_player)
            break

        if len(get_available_col_nums(positions)) == 0:
            user_prompt_tied(stdscr)
            break

        curr_player = switch_player(curr_player)


    stdscr.getch()


curses.wrapper(main)
