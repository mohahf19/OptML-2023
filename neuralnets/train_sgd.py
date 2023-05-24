# Get MNIST data

import torch
from cnn import CNN
from config import batch_size, device, network, num_epochs, test_loader, train_loader
from saga import SAGA
from tqdm import tqdm

network.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)


def train_epoch(device: str = "cpu"):
    network.train()
    running_loss = 0.0
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
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


def train(num_epochs: int = 10, device: str = "cpu"):
    tr_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(device)
        test_loss = test(device)
        tr_losses.append(train_loss)
        val_losses.append(test_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
    print(tr_losses)
    print(val_losses)


print("Started training..")
train(num_epochs=num_epochs, device="mps")
