import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, fcl_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(42, fcl_size)
        self.fc2 = nn.Linear(fcl_size, fcl_size)
        self.fc3 = nn.Linear(fcl_size, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
