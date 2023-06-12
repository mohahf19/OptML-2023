import torch
from torch import nn
import torch.nn.functional as F

# LeNet5 of Lecun et al.
class LeNet(nn.Module):

    def __init__(self, normbatch: bool = False, num_channels: int = 3):
        super(LeNet, self).__init__()
        self.normbatch = normbatch

        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if self.normbatch:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)
            self.bn3 = nn.BatchNorm1d(120)
            self.bn4 = nn.BatchNorm1d(84)

    def forward(self, x):
        nonlinearity = F.relu

        x = self.conv1(x)
        if self.normbatch:
            x = self.bn1(x)
        x = nonlinearity (x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.normbatch:
            x = self.bn2(x)
        x = nonlinearity (x)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)

        if self.normbatch:
            x = self.bn3(x)
        x = nonlinearity (x)
        x = self.fc2(x)

        if self.normbatch:
            x = self.bn4(x)
        x = nonlinearity (x)
        x = self.fc3(x)
        return x

# a CNN with ~150k parameters
class CNN150(nn.Module):
    def __init__(self, num_classes: int = 10, num_channels: int = 3):
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
