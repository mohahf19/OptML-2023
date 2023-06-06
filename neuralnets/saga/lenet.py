import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, normbatch):
        super(LeNet, self).__init__()
        self.normbatch = normbatch

        self.conv1 = nn.Conv2d(3, 6, 5)
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