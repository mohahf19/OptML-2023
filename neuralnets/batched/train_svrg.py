import pickle
from collections.abc import Iterator
from pathlib import Path
import numpy as np
import torch

from config import (
    device,
    network,
    network_temp,
    num_epochs,
    output_dir,
    batch_size,
    test_loader,
    train_loader,
    train_loader_temp,
    test_every_x_steps
)
from svrg import SVRG
from tqdm import tqdm
from utils import visualize_losses

network.to(device)
network_temp.to(device)

criterion = torch.nn.CrossEntropyLoss()


steps_per_epoch = len(train_loader.dataset)//batch_size
decrease_lr_epochs = [150,220]
optimizer = SVRG(
    network.parameters(),
    lr=0.1,
    weight_decay = 0.0001,
    snapshot_rand = False,
    prob_snapshot=2 / steps_per_epoch, # twice per epoch in expectation
    steps_per_snapshot = steps_per_epoch,
    nn=network_temp,
    loss_func=criterion,
    data_loader=train_loader_temp,
    device=device,
    lr_decrease=0.1,
    decrease_step=[(epoch-1)*steps_per_epoch for epoch in decrease_lr_epochs],
    momentum=0.9
)

def tensor_to_arr_or_scalar(tensor: torch.Tensor) -> np.ndarray | float:
    if tensor.numel() == 1:
        return tensor.item()
    return tensor.detach().cpu().numpy()


def train_step(tr_loader_iterator: Iterator, device: str = "cpu"):
    network.train()
    data, target, indices = next(tr_loader_iterator)
    data, target = data.to(device), target.to(device)
    output = network(data)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    took_snapshot = optimizer.step(x=data, y=target)
    return loss.item(), took_snapshot, indices

def test(device: str = "cpu"):
    network.train(False)
    running_loss = 0.0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    return running_loss / len(test_loader)


def train(weights_folder: Path, num_epochs: int = 1, device: str = "cpu"):
    tr_losses = []
    val_losses = []
    took_snapshots = [(-1, True)]
    indices = []
    tr_loader_iterator = iter(train_loader)
    # Save initial weights
    torch.save(network.state_dict(), weights_folder / "initial_weights.pt")
    step = 0
    for epoch in tqdm(range(num_epochs)):
        for data, target, index in train_loader:
            network.train()
            data, target = data.to(device), target.to(device)
            output = network(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            took_snapshot = optimizer.step(step=step,x=data, y=target)

            indices.append((step, tensor_to_arr_or_scalar(index)))
            tr_losses.append((step, loss.item()))
            took_snapshots.append((step, took_snapshot))
            if step % test_every_x_steps == 0:
                torch.save(network.state_dict(), weights_folder / f"weights_{step}.pt")
                val_loss = test(device)
                val_losses.append((step, val_loss))
                print(f"Step: {step}, Train Loss: {loss.item()}, Test Loss: {val_loss}")
            step += 1

    return tr_losses, val_losses, took_snapshots, indices


output_dir = output_dir / "svrg"
output_dir.mkdir(exist_ok=True, parents=True)

weights_dir = output_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)

print("Started training..")
tr_losses, val_losses, took_snapshots, indices = train(
    num_epochs=num_epochs, weights_folder=weights_dir, device=device
)

with open(output_dir / "results.pkl", "wb") as f:
    pickle.dump(
        {
            "train": tr_losses,
            "val": val_losses,
            "took_snapshots": took_snapshots,
            "sampled_indices": indices,
        },
        f,
    )

vis_path = output_dir / "loss.png"
visualize_losses(
    output_path=str(vis_path),
    tr_losses=tr_losses,
    val_losses=val_losses,
    snapshots=took_snapshots,
    title="SVRG Losses",
)
