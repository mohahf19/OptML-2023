import copy
from pathlib import Path

import numpy as np
import torch
from lenet import LeNet
#from lenet import LeNet
from indexed_dataset import IndexedDataset
from torchvision import datasets, transforms


def get_data():
    transform_train = transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    
    train_dataset = datasets.CIFAR10(
            root="data", train=True,
            download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)


# Set reproducaible
seed = 41
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

train_dataset, test_dataset = get_data()
batch_size = 128
batch_full_grads = 2**20
num_epochs = 250
test_every_x_steps = (len(train_dataset)//batch_size)//10

device = "mps"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_full_grads, shuffle=True
)

train_loader_temp = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_full_grads, shuffle=True
)

# You can use the dataset like such:
# data, target, index = next(iter(train_loader_temp))
# print(data.shape, target.shape, index)

network = LeNet(normbatch=True)
network_temp = LeNet(normbatch=True)
network_temp.load_state_dict(network.state_dict())

output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)
