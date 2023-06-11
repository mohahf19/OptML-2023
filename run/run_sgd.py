import argparse
import pickle
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from config import (
    NN,
    criterion,
    device,
    num_runs,
    num_steps,
    output_dir,
    test_dataset,
    test_every_x_steps,
    train_dataset,
)
from numpy._typing import _16Bit
from sgd import SGD
from tqdm import tqdm
from train_utils import tensor_to_arr_or_scalar, test

# TODO: add command line argument parse for parameters

print("Training with SGD")
batch_size = (
    2  # We use Lenet with batchnorm, so we need more than one sample per batch..
)
batch_size_full_grads = 2**20


## Define the training methods
def train_step(netowrk, train_loader_iterator, device, optimizer, criterion, step):
    network.train()
    data, target, index = next(train_loader_iterator)

    data, target = data.to(device), target.to(device)
    output = network(data)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    _, grad_term, dist = optimizer.step(step=step)
    return grad_term, dist, loss.item(), index


def train(
    network,
    train_loader,
    device,
    optimizer,
    criterion,
    num_steps,
    test_every_x_steps,
    test_loader,
):
    train_loader_iterator = iter(train_loader)
    train_losses = []
    test_losses = []
    indices = []
    moving_variance = []
    avg_grad = []
    distances = []
    count = 0
    alpha = 0.25

    for step in tqdm(range(num_steps)):
        grad_term, dist, train_loss, index = train_step(
            network, train_loader_iterator, device, optimizer, criterion, step
        )

        # append
        indices.append((step, tensor_to_arr_or_scalar(index)))
        train_losses.append((step, train_loss))
        distances.append((step, dist))
        if step % test_every_x_steps == 0:
            test_loss = test(network, test_loader, criterion, device)
            test_losses.append((step, test_loss))

        # Moving averages
        if len(avg_grad) == 0:
            avg_grad = deepcopy(grad_term)
        else:
            for j, g in enumerate(grad_term):
                avg_grad[j] += g

        count += 1
        var_term = 0
        for var, avg_g in zip(grad_term, avg_grad):
            var_term += torch.norm(var - avg_g / count) ** 2

        if len(moving_variance) == 0:
            moving_variance.append(var_term)
        else:
            new_variance = (1 - alpha) * moving_variance[-1] + alpha * var_term
            moving_variance.append(new_variance)

    return train_losses, test_losses, indices, moving_variance, distances


for run_id in range(num_runs):
    print("Run", run_id)
    criterion = deepcopy(criterion)
    network = NN()
    network_temp = NN()
    network.to(device)
    network_temp.load_state_dict(network.state_dict())
    network_temp.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_temp = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_full_grads, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False
    )

    optimizer = SGD(
        network.parameters(),
        lr=0.1,
        weight_decay=0.0001,
        nn=network_temp,
        loss_func=criterion,
        data_loader=train_loader_temp,
        device=device,
        lr_decrease=1,
        decrease_step=[],
        momentum=0,
    )

    train_losses, test_losses, indices, moving_variance, distances = train(
        network,
        train_loader,
        device,
        optimizer,
        criterion,
        num_steps,
        test_every_x_steps,
        test_loader,
    )

    run_output_dir = output_dir / "sgd_runs" / f"{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    with open(run_output_dir / "train_data.pkl", "wb") as f:
        pickle.dump(
            {
                "train": train_losses,
                "val": test_losses,
                "sampled_indices": indices,
                "variances": moving_variance,
                "distances": distances,
            },
            f,
        )
