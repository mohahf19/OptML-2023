import copy
from pathlib import Path

import numpy as np
import torch
import datasets
from nns import LeNet

# set reproducible randomness
seed = 41
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.mps.deterministic = True
np.random.seed(seed)

# get train and test dataset
train_dataset, test_dataset = datasets.get_cifar10_data()

# set device: cpu, mps, cuda
device = "mps"

# loss function
criterion = torch.nn.CrossEntropyLoss()

# neural network class
NN = LeNet

# make folder for logging
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)