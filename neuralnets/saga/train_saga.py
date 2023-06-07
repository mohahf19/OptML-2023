import pickle
from collections.abc import Iterator
from pathlib import Path
import numpy as np
import torch

from config import (
    device,
    network,
    network_temp,
    num_steps,
    output_dir,
    num_parts,
    test_every_n_steps,
    test_loader,
    train_loader,
    train_loader_temp,
    assignment,
    train_loder_partitions
)
from saga import SAGA
from tqdm import tqdm
from utils import visualize_losses

criterion = torch.nn.CrossEntropyLoss()

optimizer = SAGA(
    network.parameters(),
    lr=0.001,
    prob=1,
    nns=network_temp,
    loss_func=criterion,
    data_loader=train_loader_temp,
    device=device,
    assignment=assignment,
    num_parts=num_parts,
    train_partitions=train_loder_partitions
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
    took_snapshot = optimizer.step(x=data, y=target, i=indices[0])
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


# def compute_entire_gradient(network: CNN, criterion, device: str = "cpu"):
#     print("Computing entire gradient..")
#     for data, target in tqdm(train_loader):
#         # data, target = data.to(device), target.to(device)
#         output = network(data)
#         loss = criterion(output, target)
#         loss.backward()
#
#     for param in network.parameters():
#         param.grad /= len(train_loader)


def train(weights_folder: Path, num_steps: int = 1, device: str = "cpu"):
    tr_losses = []
    val_losses = []
    took_snapshots = [(-1, True)]
    indices = []
    tr_loader_iterator = iter(train_loader)
    # Save initial weights
    torch.save(network.state_dict(), weights_folder / "initial_weights.pt")
    for step in tqdm(range(num_steps)):
        try:
            train_loss, took_snapshot, sampled_indices = train_step(
                tr_loader_iterator, device
            )
        except StopIteration:
            tr_loader_iterator = iter(train_loader)
            train_loss, took_snapshot, sampled_indices = train_step(
                tr_loader_iterator, device
            )
        indices.append((step, tensor_to_arr_or_scalar(sampled_indices)))
        tr_losses.append((step, train_loss))
        took_snapshots.append((step, took_snapshot))
        if step % test_every_n_steps == 0:
            torch.save(network.state_dict(), weights_folder / f"weights_{step}.pt")
            val_loss = test(device)
            val_losses.append((step, val_loss))
            print(f"Step: {step}, Train Loss: {train_loss}, Test Loss: {val_loss}")

    return tr_losses, val_losses, took_snapshots, indices


output_dir = output_dir / "svrg"
output_dir.mkdir(exist_ok=True, parents=True)

weights_dir = output_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)

if True:
    print("Started training..")
    tr_losses, val_losses, took_snapshots, indices = train(
        num_steps=num_steps, weights_folder=weights_dir, device=device
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
