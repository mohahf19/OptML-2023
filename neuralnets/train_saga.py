# Get MNIST data


import pickle
from pathlib import Path

import numpy as np
import torch
from config_saga import (
    device,
    network,
    networks_temp,
    num_steps,
    output_dir,
    test_every_n_steps,
    test_loader,
    train_loader,
    train_loader_temp_full,
    train_loader_temp_single,
    partitions,
    X,
    y
)
X = X.to(device)
y = y.to(device)

partitions = 1
print("**************", X.device, X.shape)
X_block = torch.zeros([partitions, int(np.ceil(X.shape[0]/partitions)), 1] + list(X.shape[1:]))
X_block[0] = X.reshape([int(np.ceil(X.shape[0]/partitions)), 1] + list(X.shape[1:]))
X_block = X_block.to(device)
y_block = torch.zeros([partitions, int(np.ceil(y.shape[0]/partitions))] + list(y.shape[1:]))
y_block[0] = y
y_block = y_block.type(torch.int8)
X_block = X_block.to(device)
y_block = y_block.to(device)
print("1 ---------------------------------", X_block.device, X_block.shape)
print("2 ---------------------------------", y_block.device, y_block.shape, y_block[0,0])

from sagapartition import SAGAPartition
from tqdm import tqdm
from utils import visualize_losses

network.to(device)
for i in range(len(networks_temp)):
    networks_temp[i] = networks_temp[i].to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = SAGAPartition(
    network.parameters(),
    lr=0.001,
    prob=1,
    nns=networks_temp,
    loss_func=criterion,
    data_loader_part=train_loader_temp_full,
    device=device,
    num_part=partitions,
    data = X_block,
    targets = y_block
)

def tensor_to_arr_or_scalar(tensor: torch.Tensor) -> np.ndarray | float:
    if tensor.numel() == 1:
        return tensor.item()
    return tensor.detach().cpu().numpy()


def train_step(it_train, device: str = "cpu"):
    network.train()
    data, target, indices = next(it_train)
    data, target = data.to(device), target.to(device)
    output = network(data)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    took_snapshot, variance_term, grad_term, snap_dist, dist, sgd_step = optimizer.step(x=data, y=target, i=indices[0])
    return loss.item(), took_snapshot, indices, variance_term, grad_term, snap_dist, dist, sgd_step


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
    moving_variance = []
    moving_variance_sgd = []
    snap_distances = []
    distances = []
    avg_grad = []
    count = 0.0
    alpha = 0.25
    # Save initial weights
    torch.save(network.state_dict(), weights_folder / "initial_weights.pt")
    for step in tqdm(range(num_steps)):
        try:
            train_loss, took_snapshot, sampled_indices, variance_term, grad_term, snap_dist, dist, sgd_step = train_step(
                tr_loader_iterator, device
            )
        except StopIteration:
            tr_loader_iterator = iter(train_loader)
            train_loss, took_snapshot, sampled_indices, variance_term, grad_term, snap_dist, dist, sgd_step = train_step(
                tr_loader_iterator, device
            )
        indices.append((step, tensor_to_arr_or_scalar(sampled_indices)))
        tr_losses.append((step, train_loss))
        snap_distances.append(snap_dist)
        distances.append(dist)
        took_snapshots.append((step, took_snapshot))

        if len(avg_grad) == 0:
                    avg_grad = grad_term
        else:
            for j, g in enumerate(grad_term):
                avg_grad[j] += g
        count +=1
        var_term = 0.0
        for var, avg_g in zip(variance_term, avg_grad):
            var_term += torch.norm(var - avg_g/count)**2
        var_term_sgd = 0.0
        for sgd, avg_g in zip(sgd_step, avg_grad):
            var_term_sgd += torch.norm(sgd - avg_g/count)**2

        if len(moving_variance) == 0:
            moving_variance.append(var_term)
        else:
            new_variance = (1 - alpha)*moving_variance[-1] + alpha*var_term
            moving_variance.append(new_variance)

        if len(moving_variance_sgd) == 0:
            moving_variance_sgd.append(var_term_sgd)
        else:
            new_variance = (1 - alpha)*moving_variance_sgd[-1] + alpha*var_term_sgd
            moving_variance_sgd.append(new_variance)
        print(moving_variance[-1], moving_variance_sgd[-1])

        if step % test_every_n_steps == 0:
            torch.save(network.state_dict(), weights_folder / f"weights_{step}.pt")
            val_loss = test(device)
            val_losses.append((step, val_loss))
            print(f"Step: {step}, Train Loss: {train_loss}, Test Loss: {val_loss}")

    return tr_losses, val_losses, took_snapshots, indices, moving_variance, snap_distances, distances


output_dir = output_dir / "svrg"
output_dir.mkdir(exist_ok=True, parents=True)

weights_dir = output_dir / "weights"
weights_dir.mkdir(exist_ok=True, parents=True)

train_b = True
if train_b:
    print("Started training..")
    tr_losses, val_losses, took_snapshots, indices, moving_variance, snap_distances, distances = train(
        num_steps=num_steps, weights_folder=weights_dir, device=device
    )

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "train": tr_losses,
                "val": val_losses,
                "took_snapshots": took_snapshots,
                "sampled_indices": indices,
                "variances": moving_variance,
                "snap_distances": snap_distances,
                "distances": distances
            },
            f,
        )

    vis_path = output_dir / "loss.png"
    visualize_losses(
        output_path=str(vis_path),
        tr_losses=tr_losses,
        val_losses=val_losses,
        snapshots=took_snapshots,
        title="SAGA_partition Losses",
    )
