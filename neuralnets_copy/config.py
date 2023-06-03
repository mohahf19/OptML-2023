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
num_epochs = 100
device = "cpu"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=True)

train_dataset, train_labels = next(iter(train_loader))
train_dataset, train_labels = train_dataset.to(device), train_labels.to(device)
test_dataset, test_labels = next(iter(train_loader))
test_dataset, test_labels = test_dataset.to(device), test_labels.to(device)


network = CNN(num_channels=1, num_classes=torch.unique(train_labels).shape[0])
network_temp = CNN(
    num_channels=1, num_classes=torch.unique(train_labels).shape[0]
)
network_temp.load_state_dict(network.state_dict())


print(torchinfo.summary(network))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)
