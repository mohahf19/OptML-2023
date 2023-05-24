from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required


class SAGA(Optimizer):
    """WIP"""

    def __init__(self, params, lr=required):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

        # Save the average gradient

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            pass

        return loss
