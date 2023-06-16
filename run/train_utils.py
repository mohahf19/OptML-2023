import numpy as np
import torch


def tensor_to_arr_or_scalar(tensor: torch.Tensor) -> np.ndarray | float:
    """Convert a tensor to a numpy array or scalar."""
    if tensor.numel() == 1:
        return tensor.item()
    return tensor.detach().cpu().numpy()


def test(network: torch.nn.Module, test_loader, criterion, device: str = "cpu"):
    """Test the network on the test set.

    Args:
        network: The network to test
        test_loader: The test data loader
        criterion: The loss function
        device: The device to test on

    Returns:
        (float) The average loss on the test set
    """
    network.train(False)
    running_loss = 0.0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    return running_loss / len(test_loader)
