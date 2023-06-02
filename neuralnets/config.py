from pathlib import Path

import torch
import torchinfo
from cnn import CNN
from torchvision import datasets, transforms
import copy


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


train_dataset, test_dataset = get_data()
batch_size = 1
num_epochs = 100
device = "cpu"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loader_temp = torch.utils.data.DataLoader(test_dataset, batch_size=len(train_loader.dataset), shuffle=True)

network = CNN(num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0])
network_temp = CNN(
    num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0]
)
network_temp.load_state_dict(network.state_dict())


print(torchinfo.summary(network))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)
