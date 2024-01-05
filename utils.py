NUM_COLS = 7
NUM_ROWS = 6



def get_row_num(positions, col_num):
    """Given currently occupied position, and a col_num, return row_num."""
    all_pos = positions["P1"] + positions["P2"]
    pos_this_col = [pos[0] for pos in all_pos if pos[1] == col_num]
    if pos_this_col:
        return max(pos_this_col) + 1
    else:
        return 1


def get_available_col_nums(positions):
    all_pos = positions["P1"] + positions["P2"]
    available_col_nums = []
    for col_num in range(1, NUM_COLS + 1):
        pos_this_col = [pos[0] for pos in all_pos if pos[1] == col_num]
        if len(pos_this_col) < NUM_ROWS:
            available_col_nums.append(col_num)
    return available_col_nums



def pos_is_chain_of_four(pos, positions):
    """Check if position is the start of a chain of four."""
    chain_pos = [(pos[0] + 1 + i, pos[1]) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    chain_pos = [(pos[0] - 1 - i, pos[1]) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    chain_pos = [(pos[0], pos[1] + 1 + i) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    chain_pos = [(pos[0], pos[1] - 1 - i) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    chain_pos = [(pos[0] + i + 1, pos[1] + i + 1) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    chain_pos = [(pos[0] - i - 1, pos[1] - i - 1) for i in range(3)]
    if all([p in positions for p in chain_pos]):
        return True

    return False



def is_won_for_player(player_positions):
    for pos in player_positions:
        if pos_is_chain_of_four(pos, player_positions):
            return True
    return False



def is_won(positions):
    """Check if any player has won the game."""
    if is_won_for_player(positions["P1"]):
        return True
    elif is_won_for_player(positions["P2"]):
        return True
    else:
        return False



def switch_player(curr_player):
    if curr_player == "P1":
        return "P2"
    else:
        return "P1"
