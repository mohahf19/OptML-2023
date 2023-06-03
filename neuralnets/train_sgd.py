# Get MNIST data

import pickle
from pathlib import Path

import torch
from config import (
    device,
    network,
    num_steps,
    output_dir,
    test_every_n_steps,
    test_loader,
    train_loader,
)
from tqdm import tqdm
from utils import visualize_losses

network.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001)


def train_step(device: str = "cpu"):
    network.train()
    data, target, _ = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    output = network(data)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(device: str = "cpu"):
    network.train(False)
    running_loss = 0.0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    return running_loss / len(test_loader)


def train(weights_folder: Path, num_steps: int = 1, device: str = "cpu"):
    tr_losses = []
    val_losses = []
    torch.save(network.state_dict(), weights_folder / "initial_weights.pt")
    for step in tqdm(range(num_steps)):
        train_loss = train_step(device)
        tr_losses.append((step, train_loss))
        if step % test_every_n_steps == 0:
            torch.save(network.state_dict(), weights_folder / f"weights_{step}.pt")
            val_loss = test(device)
            val_losses.append((step, val_loss))
            print(f"Step: {step}, Train Loss: {train_loss}, Test Loss: {val_loss}")

    return tr_losses, val_losses


output_dir = output_dir / "sgd"
output_dir.mkdir(exist_ok=True, parents=True)

weights_dir = output_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)
print("Started training..")
tr_losses, val_losses = train(
    weights_folder=weights_dir, num_steps=num_steps, device=device
)

with open(output_dir / "results.pkl", "wb") as f:
    pickle.dump({"train": tr_losses, "val": val_losses}, f)

vis_path = output_dir / "loss.png"
visualize_losses(
    output_path=str(vis_path),
    tr_losses=tr_losses,
    val_losses=val_losses,
    title="SGD Losses",
)
