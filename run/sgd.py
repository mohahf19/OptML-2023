from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    """Implements SGD algorithm."""

    def __init__(
        self,
        params,
        nn,
        loss_func,
        device,
        lr_decrease,
        decrease_step,
        momentum,
        weight_decay,
        lr=required,
    ):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.lr = lr
        self.nn_temp = nn  # backup neural networ, can be useful for logging
        self.loss_func = loss_func
        self.device = device

        self.lr_decrease = lr_decrease
        self.decrease_step = decrease_step
        self.momentum = momentum
        self.weight_decay = weight_decay

        if self.momentum != 0:
            self.momentum_mem = {p: p.data.clone().zero_() for p in self.params}

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, step, closure=None):
        """Does one step of SGD.

        Args:
            step: current step number
            closure: function to evaluate loss. Defaults to None.
        """
        grad_term = []
        dist = 0.0

        params_old = []
        if step in self.decrease_step:
            self.lr = self.lr * self.lr_decrease
        for p in self.params:
            grad_term.append(p.grad.clone())
            params_old.append(p.clone())
            update = p.grad + self.weight_decay * p.data
            if self.momentum != 0:
                update += self.momentum * self.momentum_mem[p]
                self.momentum_mem[p] = update
            p.data = p.data - self.lr * update

        for p, p_old in zip(self.params, params_old):
            dist += torch.norm(p.data - p_old.data)

        return None, grad_term, dist
