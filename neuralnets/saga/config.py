import copy
from pathlib import Path

import numpy as np
import torch
from lenet import LeNet
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
batch_size_full = 2**10
num_steps = 20000
test_every_n_steps = 10
num_parts = 10

device = "mps"

perm = torch.randperm(len(train_dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(train_dataset, perm)

train_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size_full, shuffle=True
)

train_loader_temp = torch.utils.data.DataLoader(shuffled_dataset, batch_size=len(train_dataset)//num_parts, shuffle=False)

assignment = [-1 for _ in range(len(train_dataset))]
batches = []
p = 0
for data, targets, indices in train_loader_temp:
    batches.append(indices)
    print(indices)
    for j in indices:
        assignment[j] = p
    p += 1
assert len([p for p in assignment if p == -1]) == 0

s = len(train_dataset)//num_parts
train_loder_partitions = []
for p in range(num_parts):
    B = torch.utils.data.Subset(shuffled_dataset, [p*s + x for x in range(s)])
    train_loder_partitions.append(torch.utils.data.DataLoader(B, batch_size=s, shuffle=False))

print()
for p in range(num_parts):
    print("------", p)
    for a, b, i in train_loder_partitions[p]:
        print(i)


network = LeNet(normbatch=False)
network.to(device)

network_temp  = []
for p in range(num_parts):
    network_temp.append(LeNet(normbatch=False))
    network_temp[p].load_state_dict(network.state_dict())
    network_temp[p].to(device)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)