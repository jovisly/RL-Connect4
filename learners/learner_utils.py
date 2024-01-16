from utils import (
  NUM_ROWS,
  NUM_COLS,
  get_available_col_nums,
  get_row_num,
  is_won_for_player
)

REWARD_ILLEGAL_MOVE = -10
REWARD_LOSS = -10
REWARD_WIN = 10


def is_won_for_opponent_next_move(positions, player):
  opponent = "P1" if player == "P2" else "P2"
  opponent_positions = positions[opponent]
  col_nums = get_available_col_nums(positions)

  for c in col_nums:
    new_positions = [p for p in opponent_positions] + [(get_row_num(positions, c), c)]
    if is_won_for_player(new_positions):
      return True

  return False



class Connect4Game:
  def __init__(self):
    self.positions = {"P1": [], "P2": []}
    self.player = "P1"

  def switch_player(self):
    if self.player == "P1":
      self.player = "P2"
    else:
      self.player = "P1"

  def get_available_moves(self):
    return get_available_col_nums(self.positions)

  def get_state(self):
    """Converts positions into a state representation for DQN."""
    return convert_positions_to_board(self.positions, self.player)

  def take_action(self, action):
    """Action is col_num between 1 to 7.

    Returns (next_state, reward, done).
    """
    # Illegal move means game ends with penalty.
    if action not in get_available_col_nums(self.positions):
      return (self.get_state(), REWARD_ILLEGAL_MOVE, True)

    # Game didn't end, so let's update positions.
    row_num = get_row_num(self.positions, action)
    new_pos = (row_num, action)
    self.positions[self.player].append(new_pos)

    # Now check if this player has won.
    if is_won_for_player(self.positions[self.player]):
      return (self.get_state(), REWARD_WIN, True)
    else:
      # Check if the other player can win in the next move. But don't end the
      # game in this case.
      if is_won_for_opponent_next_move(self.positions, self.player):
        return (self.get_state(), REWARD_LOSS, False)
      else:
        return (self.get_state(), 0, False)


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
