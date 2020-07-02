import torch
import torch.nn as nn
import properties as prop


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        input_size = prop.POLICY_INPUT_SIZE
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
