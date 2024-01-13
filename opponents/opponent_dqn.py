import torch

from utils import get_available_col_nums

from learners.learner_dqn import DQN
from learners.learner_utils import convert_positions_to_board

model = DQN()
model.load_state_dict(torch.load("./opponents/models/dqn.pt"))


def get_move(positions, player):
    state = convert_positions_to_board(positions, player)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    available_cols = get_available_col_nums(positions)
    with torch.no_grad():
        q_values = model(state_tensor)
        q_values_list = q_values.squeeze().tolist()
        q_dict = {
            i + 1: q
            for i, q in enumerate(q_values_list)
            if i + 1 in available_cols
        }
        # Take the key with the biggest value.
        action = max(q_dict, key=q_dict.get)

    return action



if __name__ == "__main__":
    positions = {"P1": [(1, 4)], "P2": []}
    player = "P2"

    move = get_move(positions, player)
    print(move)
