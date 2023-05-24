# Get MNIST data

import torch
from cnn import CNN
from torchvision import datasets, transforms
from tqdm import tqdm


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
batchs_size = 2**11
num_epochs = 10
device = "mps"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchs_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchs_size)

network = CNN(num_channels=1, num_classes=torch.unique(train_dataset.targets).shape[0])
network.to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()


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
