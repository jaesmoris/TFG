import os
import torch
from torch import nn

class GrainClassifierSH(nn.Module):
    def __init__(self, l_max, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((l_max+1)**2, 512, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.double),
            nn.ReLU()
        )
        self.classification_layer = nn.Linear(512, n_classes, dtype=torch.double)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.classification_layer(x)
        return x
