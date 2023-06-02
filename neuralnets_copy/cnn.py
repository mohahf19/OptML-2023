import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes: int, num_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = nn.Sequential(nn.LazyLinear(out_features=512), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
