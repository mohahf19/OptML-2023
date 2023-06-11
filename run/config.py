import argparse
from pathlib import Path

import numpy as np
import torch
from nns import LeNet

import datasets

# set reproducible randomness
seed = 41
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

# get train and test dataset
train_dataset, test_dataset = datasets.get_cifar10_data()

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--test-every-x-steps", type=int, default=20)
parser.add_argument("--num-runs", type=int, default=5)
parser.add_argument("--seed", type=int, default=41)

args = parser.parse_args()

args = parser.parse_args()
# Set reproducaible
seed = args.seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True

np.random.seed(seed)
device = args.device
print(device)
num_steps = args.num_steps
test_every_x_steps = args.test_every_x_steps
num_runs = args.num_runs

# loss function
criterion = torch.nn.CrossEntropyLoss()

# neural network class
NN = LeNet

# make folder for logging
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)

network = LeNet(normbatch=True)
network_temp = LeNet(normbatch=True)
network_temp.load_state_dict(network.state_dict())
network.to(device)
network_temp.to(device)
