import argparse
from pathlib import Path

import numpy as np
import torch
from nns import *

import datasets

# set reproducible randomness
seed = 41
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

# get train and test dataset
train_dataset, test_dataset = datasets.get_mnist_data()

# get parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--test-every-x-steps", type=int, default=20)
parser.add_argument("--num-runs", type=int, default=5)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--num-parts", type=int, default=10)  # clustered saga only
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)

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
num_parts = args.num_parts
learning_rate = args.lr
batch_size = args.batch_size

# loss function
criterion = torch.nn.CrossEntropyLoss()

# neural network class
NN = LeNet

# make folder for logging
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)
