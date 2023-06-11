import numpy as np
import torch


def tensor_to_arr_or_scalar(tensor: torch.Tensor) -> np.ndarray | float:
    if tensor.numel() == 1:
        return tensor.item()
    return tensor.detach().cpu().numpy()


def test(network: torch.nn.Module, test_loader, criterion, device: str = "cpu"):
    network.train(False)
    running_loss = 0.0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    return running_loss / len(test_loader)
