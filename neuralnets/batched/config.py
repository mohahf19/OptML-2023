import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from indexed_dataset import IndexedDataset
from lenet import LeNet
from torchvision import datasets, transforms


def get_data():
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--num_epochs", type=int, default=250)


args = parser.parse_args()
# Set reproducaible
seed = args.seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

train_dataset, test_dataset = get_data()
batch_size_c = args.batch_size
batch_full_grads = 2**20
num_epochs_c = args.num_epochs
test_every_x_steps_c = (len(train_dataset) // batch_size_c) // 10

device_c = args.device
train_loader_c = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size_c, shuffle=True
)
test_loader_c = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_full_grads, shuffle=True
)

train_loader_temp_c = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_full_grads, shuffle=True
)

# You can use the dataset like such:
# data, target, index = next(iter(train_loader_temp))
# print(data.shape, target.shape, index)

network_c = LeNet(normbatch=True)
network_temp_c = LeNet(normbatch=True)
network_temp_c.load_state_dict(network_c.state_dict())

output_dir_c = Path("output")
output_dir_c.mkdir(exist_ok=True, parents=True)
