# batch_size_full = 10**20
# num_steps = 20000
# num_parts = 10
#
# # make data loader for testing
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size_full, shuffle=True
# )
#
# # make data loaders for training
# perm = torch.randperm(len(train_dataset)).tolist()
# shuffled_dataset = torch.utils.data.Subset(train_dataset, perm)
#
# train_loader = torch.utils.data.DataLoader(
#     shuffled_dataset, batch_size=1, shuffle=True
# )
#
# train_loader_temp = torch.utils.data.DataLoader(
#     shuffled_dataset, batch_size=len(train_dataset) // num_parts, shuffle=False
# )
#
# assignment = [-1 for _ in range(len(train_dataset))]
# batches = []
# p = 0
# for data, targets, indices in train_loader_temp:
#     batches.append(indices)
#     print(indices)
#     for j in indices:
#         assignment[j] = p
#     p += 1
# assert len([p for p in assignment if p == -1]) == 0
#
# s = len(train_dataset) // num_parts
# train_loder_partitions = []
# for p in range(num_parts):
#     B = torch.utils.data.Subset(shuffled_dataset, [p * s + x for x in range(s)])
#     train_loder_partitions.append(
#         torch.utils.data.DataLoader(B, batch_size=s, shuffle=False)
#     )
#
# network = NN()
# network.to(device)
#
# network_temp = []
# for p in range(num_parts):
#     network_temp.append(NN())
#     network_temp[p].load_state_dict(network.state_dict())
#     network_temp[p].to(device)
import pickle
from copy import deepcopy

import torch
from config import (
    NN,
    criterion,
    device,
    num_parts,
    num_runs,
    num_steps,
    output_dir,
    test_dataset,
    test_every_x_steps,
    train_dataset,
)
from saga import SAGA
from tqdm import tqdm
from train_utils import tensor_to_arr_or_scalar, test

print("Training with SAGA")
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
    ) = optimizer.step(x=data, y=target, i=index[0])
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
        snapshot_steps
    )


for run_id in range(num_runs):
    print("Run", run_id)
    run_output_dir = output_dir / "saga_runs" / f"{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    weights_folder = run_output_dir / "weights"
    weights_folder.mkdir(parents=True, exist_ok=True)

    criterion = deepcopy(criterion)
    network = NN()
    network_temp = []
    network.to(device)

    network_temp = []
    for p in range(num_parts):
        network_temp.append(NN())
        network_temp[p].load_state_dict(network.state_dict())
        network_temp[p].to(device)

    perm = torch.randperm(len(train_dataset)).tolist()
    shuffled_dataset = torch.utils.data.Subset(train_dataset, perm)
    train_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size=1, shuffle=True)

    train_loader_temp = torch.utils.data.DataLoader(shuffled_dataset, batch_size=len(train_dataset)//num_parts, shuffle=False)

    assignment = [-1 for _ in range(len(train_dataset))]
    batches = []
    p = 0
    for data, targets, indices in train_loader_temp:
        batches.append(indices)
        #print(indices)
        for j in indices:
            assignment[j] = p
        p += 1
    assert len([p for p in assignment if p == -1]) == 0

    s = len(train_dataset)//num_parts
    train_loder_partitions = []
    for p in range(num_parts):
        B = torch.utils.data.Subset(shuffled_dataset, [p*s + x for x in range(s)])
        train_loder_partitions.append(torch.utils.data.DataLoader(B, batch_size=s, shuffle=False))

    #print("Setting up splits..")
    #for p in range(num_parts):
    #    print("------", p)
    #    for a, b, i in train_loder_partitions[p]:
    #        print(i)
    
    train_loader_full = torch.utils.data.DataLoader(shuffled_dataset, batch_size=batch_size_full_grads, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_full_grads, shuffle=False)

    optimizer = SAGA(
        network.parameters(),
        lr=0.1,
        prob=1,
        nns=network_temp,
        loss_func=criterion,
        data_loader=train_loader_full,
        device=device,
        assignment=assignment,
        num_parts=num_parts,
        train_partitions=train_loder_partitions,
    )

    (
        train_losses,
        test_losses,
        indices,
        moving_variance,
        moving_variance_sgd,
        distances,
        snap_distances,
        snapshots
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
                "snapshot_steps": snapshots
            },
            f,
        )
