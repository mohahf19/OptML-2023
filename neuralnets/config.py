import copy
from pathlib import Path

import numpy as np
import torch
from cnn import CNN
#from lenet import LeNet
from indexed_dataset import IndexedDataset
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

    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)


# Set reproducaible
seed = 41
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

train_dataset, test_dataset = get_data()
batch_size = 2**10
num_steps = 20000
test_every_n_steps = 100

device = "mps"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

train_loader_temp = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

# You can use the dataset like such:
# data, target, index = next(iter(train_loader_temp))
# print(data.shape, target.shape, index)

network = CNN(num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0])
network_temp = CNN(
    num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0]
)
network_temp.load_state_dict(network.state_dict())


output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)
