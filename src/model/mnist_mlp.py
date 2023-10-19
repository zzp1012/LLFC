import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
