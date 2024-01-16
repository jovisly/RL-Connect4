"""A policy gradient learner trained to play Connect 4."""
from collections import deque
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import NUM_ROWS, NUM_COLS
from learners.learner_utils import Connect4Game

H_SIZE = 128
# Learning parameters.
NUM_EPISODES = 500_000
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100_000
LR = 1e-2

MODEL_OUTPUT = "../opponents/models/REINFORCE.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# See pyTorch example:
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
class Policy(nn.Module):
    def __init__(self, s_size=NUM_COLS*NUM_ROWS, a_size=NUM_COLS, h_size=H_SIZE):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)



if __name__ == "__main__":
    scores_deque = deque(maxlen=100)
    scores = []

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    print("Training REINFORCE... number of parameters in the model:", sum(p.numel() for p in policy.parameters()))

    for episode in range(NUM_EPISODES):
        saved_log_probs = []
        rewards = []

        game = Connect4Game()
        done = False
        num_games = 0
        total_loss = 0

        # Randomly assign "smart" player for this game.
        trained_player = random.choice(["P1", "P2"])

        # Game loop.
        while not done:
            state = game.get_state()
            state = np.array(state)
            epsilon = 1 - (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY))
            if game.player != trained_player:
                if random.random() < epsilon * 0.5:
                    with torch.no_grad():
                        action, _ = policy.act(state)
                        action += 1
                else:
                    action = random.randint(1, 7)

                _, _, done = game.take_action(action)
                game.switch_player()
                continue


            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            _, reward, done = game.take_action(action + 1)
            rewards.append(reward)
            game.switch_player()

        # End of game loop.
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        n_steps = len(rewards)
        returns = deque(maxlen=n_steps)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(GAMMA * disc_return_t + rewards[t])

        # Avoid dividing by zero.
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)

        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 1000 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

    # Save model.
    torch.save(policy.state_dict(), MODEL_OUTPUT)

