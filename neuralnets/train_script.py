# Get MNIST data

import torch
from cnn import CNN
from torchvision import datasets, transforms


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
batchs_size = 64
num_epochs = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchs_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchs_size)

network = CNN(num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0])
print(network)
print(train_dataset)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
