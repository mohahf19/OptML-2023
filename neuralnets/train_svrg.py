# Get MNIST data


import pickle

import numpy as np
import torch
from cnn import CNN
from config import (
    device,
    network,
    network_temp,
    num_epochs,
    output_dir,
    test_loader,
    train_loader,
    train_loader_temp,
)
from svrg import SVRG
from tqdm import tqdm

network.to(device)
network_temp.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = SVRG(
    network.parameters(),
    lr=0.001,
    prob=1 / np.sqrt(len(train_loader.dataset)),
    nn=network_temp,
    loss_func=criterion,
    data_loader=train_loader_temp,
    device=device,
)


def train_epoch(device: str = "cpu"):
    network.train()
    running_loss = 0.0
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step(x=data, y=target)
    running_loss += loss.item()
    return running_loss / len(train_loader)


def test(device: str = "cpu"):
    network.train(False)
    running_loss = 0.0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    return running_loss / len(test_loader)


def compute_entire_gradient(network: CNN, criterion, device: str = "cpu"):
    print("Computing entire gradient..")
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        loss.backward()

    for param in network.parameters():
        param.grad /= len(train_loader)


def train(num_epochs: int = 1, device: str = "cpu"):
    tr_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(device)
        test_loss = test(device)
        tr_losses.append(train_loss)
        val_losses.append(test_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")

    return tr_losses, val_losses


print("Started training..")
tr_losses, val_losses = train(num_epochs=10, device="mps")

with open(output_dir / "svrg_losses.pkl", "wb") as f:
    pickle.dump({"train": tr_losses, "val": val_losses}, f)
