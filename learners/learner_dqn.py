"""A DQN learner trained to play Connect 4."""
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils import NUM_ROWS, NUM_COLS
from learners.learner_utils import Connect4Game

# Model Parameters.
FCL_SIZE = 128
# Learning parameters.
NUM_EPISODES = 500_000
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100_000
LR = 1e-2


MODEL_OUTPUT = "../opponents/models/dqn.pt"
MAKE_PLOTS = False

class DQN(nn.Module):
    def __init__(self, fcl_size=FCL_SIZE):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(NUM_ROWS * NUM_COLS, fcl_size)
        self.fc2 = nn.Linear(fcl_size, fcl_size)
        self.fc3 = nn.Linear(fcl_size, NUM_COLS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    dqn = DQN()
    # Loss function and optimizer were chosen based on
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # Loss function.
    loss_fn = nn.SmoothL1Loss()
    # Optimizer.
    optimizer = optim.AdamW(dqn.parameters(), lr=LR, amsgrad=True)

    print("Training DQN... number of parameters in the model:", sum(p.numel() for p in dqn.parameters()))

    # Training loop.
    list_losses = []
    list_epsilon = []

    for episode in range(NUM_EPISODES):
        game = Connect4Game()
        done = False
        num_moves = 0
        total_loss = 0

        # Randomly assign "smart" player for this game.
        trained_player = random.choice(["P1", "P2"])

        while not done:
            # Exploration vs Exploitation - Choose an action
            state = game.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # If not the smart player, use sub-optimal strategy, i.e., mostly random
            # but once a while the actual model.
            epsilon = 1 - (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY))
            if game.player != trained_player:
                if random.random() < epsilon * 0.5:
                    with torch.no_grad():
                        q_values = dqn(state_tensor)
                        # Plus 1 because we are 1 to 7.
                        action = torch.argmax(q_values).item() + 1
                else:
                    action = random.randint(1, 7)
                _, _, done = game.take_action(action)
                game.switch_player()
                continue

            if random.random() < epsilon:
                q_values = dqn(state_tensor)
                # Plus 1 because we are 1 to 7.
                action = torch.argmax(q_values).item() + 1
            else:
                action = random.randint(1, 7)


            # Take the selected action and observe the next state and reward
            next_state, reward, done = game.take_action(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # Compute the TD target
            # TD target = immediate reward + (discount factor * max Q-value of next state)
            q_values_next = dqn(next_state_tensor)
            td_target = reward + GAMMA * torch.max(q_values_next).item()

            # Get the current Q-value
            q_values_current = dqn(state_tensor)
            # Subtract the 1 to go back to index-0. Why did we choose this?
            td_current = q_values_current[action - 1].item()

            # Compute the loss
            loss = loss_fn(torch.tensor([td_current]), torch.tensor([td_target]))
            total_loss += loss.item()
            num_moves += 1

            # Backpropagate and update the model
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

            # Switch player
            game.switch_player()

        if MAKE_PLOTS == True:
            list_losses.append(round(total_loss / num_moves, 6))
            list_epsilon.append(epsilon)
        if episode % 1000 == 0:
            print(f"Episode {episode}: average loss {round(total_loss / num_moves, 6)}")


    if MAKE_PLOTS == True:
        plt.plot([i + 1 for i in range(len(list_losses))], list_losses)
        plt.xlabel('Episode')
        plt.ylabel('Avg. Loss')
        plt.title('DQN Training Loss')
        plt.savefig('dqn.png')

        plt.clf()

        plt.plot([i + 1 for i in range(len(list_epsilon))], list_epsilon)
        plt.xlabel('Episode')
        plt.ylabel('epsilon')
        plt.title('Epsilon')
        plt.savefig('dqn_epsilon.png')

    # Save model.
    torch.save(dqn.state_dict(), MODEL_OUTPUT)
