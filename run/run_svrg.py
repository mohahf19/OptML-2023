import pickle
from copy import deepcopy

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
from svrg import SVRG
from tqdm import tqdm
from train_utils import tensor_to_arr_or_scalar, test

print("Training with SVRG")
batch_size = (
    1  # We use Lenet with batchnorm, so we need more than one sample per batch..
)
batch_size_full_grads = 512


## Define the training methods
def train_step(network, train_loader_iterator, device, optimizer, criterion, step):
    network.train()
    data, target, index = next(train_loader_iterator)

    data, target = data.to(device), target.to(device)
    output = network(data)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    (
        took_snapshot,
        variance_term,
        grad_term,
        snap_dist,
        dist,
        sgd_step,
    ) = optimizer.step(step=step, x=data, y=target)
    return (
        loss.item(),
        took_snapshot,
        index,
        variance_term,
        grad_term,
        snap_dist,
        dist,
        sgd_step,
    )


def train(
    network,
    train_loader,
    device,
    optimizer,
    criterion,
    num_steps,
    test_every_x_steps,
    test_loader,
    weights_folder,
):
    train_loader_iterator = iter(train_loader)
    train_losses = []
    test_losses = []
    indices = []
    moving_variance = []
    moving_variance_sgd = []
    avg_grad = []
    distances = []
    count = 0
    alpha = 0.25
    snap_distances = []
    snapshot_steps = []

    for step in tqdm(range(num_steps)):
        (
            train_loss,
            took_snapshot,
            index,
            variance_term,
            grad_term,
            snap_dist,
            dist,
            sgd_step,
        ) = train_step(
            network, train_loader_iterator, device, optimizer, criterion, step
        )

        # append
        if took_snapshot:
            snapshot_steps.append(step)
        
        if step % test_every_x_steps == 0:
            # torch.save(network.state_dict(), weights_folder / f"weights_{step}.pt")
            test_loss = test(network, test_loader, criterion, device)
            test_losses.append((step, test_loss))
            train_loss_full = test(network, train_loader_full, criterion, device)
            train_losses.append((step, train_loss_full))
            print("############", train_loss_full)
        indices.append((step, tensor_to_arr_or_scalar(index)))
        distances.append((step, dist))
        snap_distances.append((step, snap_dist))

        # Moving averages
        if len(avg_grad) == 0:
            avg_grad = deepcopy(grad_term)
        else:
            for j, g in enumerate(grad_term):
                avg_grad[j] += g

        count += 1
        var_term = 0
        for var, avg_g in zip(variance_term, avg_grad):
            var_term += torch.norm(var - avg_g / count) ** 2
        var_term_sgd = 0
        for sgd, avg_g in zip(sgd_step, avg_grad):
            var_term_sgd += torch.norm(sgd - avg_g / count) ** 2

        if len(moving_variance) == 0:
            moving_variance.append(var_term)
        else:
            new_variance = (1 - alpha) * moving_variance[-1] + alpha * var_term
            moving_variance.append(new_variance)

        if len(moving_variance_sgd) == 0:
            moving_variance_sgd.append(var_term_sgd)
        else:
            new_variance = (1 - alpha) * moving_variance_sgd[-1] + alpha * var_term_sgd
            moving_variance_sgd.append(new_variance)

    return (
        train_losses,
        test_losses,
        indices,
        moving_variance,
        moving_variance_sgd,
        distances,
        snap_distances,
    )


for run_id in range(num_runs):
    print("Run", run_id)
    run_output_dir = output_dir / "svrg_runs" / f"{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    weights_folder = run_output_dir / "weights"
    weights_folder.mkdir(parents=True, exist_ok=True)

    criterion = deepcopy(criterion)
    network = NN()
    network_temp = NN()
    network.to(device)
    network_temp.load_state_dict(network.state_dict())
    network_temp.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_full = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_full_grads, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_full_grads, shuffle=False
    )

    optimizer = SVRG(
        network.parameters(),
        lr=0.1,
        weight_decay=0, #0.0001,
        snapshot_rand=True,
        prob_snapshot=1 / 10,
        steps_per_snapshot=10,  # TODO: put the right thing here
        nn=network_temp,
        loss_func=criterion,
        data_loader=train_loader_full,
        device=device,
        lr_decrease=1,
        decrease_step=[],
        momentum=0,
    )
    (
        train_losses,
        test_losses,
        indices,
        moving_variance,
        moving_variance_sgd,
        distances,
        snap_distances,
    ) = train(
        network,
        train_loader,
        device,
        optimizer,
        criterion,
        num_steps,
        test_every_x_steps,
        test_loader,
        weights_folder,
    )

    with open(run_output_dir / "train_data.pkl", "wb") as f:
        pickle.dump(
            {
                "train": train_losses,
                "val": test_losses,
                "sampled_indices": indices,
                "variances": moving_variance,
                "variances_sgd": moving_variance_sgd,
                "snap_distances": snap_distances,
                "distances": distances,
            },
            f,
        )
