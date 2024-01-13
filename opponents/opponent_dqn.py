import torch

from learners.learner_dqn import DQN
from learners.learner_utils import convert_positions_to_board

model = DQN()
model.load_state_dict(torch.load("./opponents/models/dqn.pt"))


def get_move(positions, player):
    state = convert_positions_to_board(positions, player)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        q_values = model(state_tensor)
        # Plus 1 because we are 1 to 7.
        action = torch.argmax(q_values).item() + 1

    return action


