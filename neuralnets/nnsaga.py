from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SVRG(Optimizer):
    def __init__(self, params, nn, loss_func, data_loader, device, lr=required):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.lr = lr
        self.nn_temp = [deepcopy(nn) for _ in range(len(data_loader.dataset))]  # stored snapshot neural networ, i.e. store the weight
        self.grad_avg = []  # full gradient at stored snapshot weight
        self.data_loader = (
            data_loader  # access to full dataset to compute average gradient
        )
        self.loss_func = loss_func
        self.device = device

        self.prev_snapshot = False

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, i, closure=None) -> bool:
        if self.prev_snapshot:
           var_red = self.variance_reduction_stoch_grad(x, y)

        for p, var_red_term in zip(self.params, var_red):
            if self.prev_snapshot:
                update = p.grad - var_red_term
            else:
                update = p.grad
            p.data = p.data - self.lr * update

        self.take_snapshot()
        self.prev_snapshot = True
        return True

    def variance_reduction_stoch_grad(self, x, y, i):
        # zeroing the gradients
        for p in self.nn_temp.parameters():
            p.grad = None

        # recompute gradient at sampled data point
        outputs = self.nn_temp(x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        for p, g_avg in zip(self.nn_temp.parameters(), self.grad_avg):
            grad_list.append(p.grad - g_avg)

        return grad_list

    def take_snapshot(self):
        print("Taking snapshot..")
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nn_temp.parameters()):
            p_temp = deepcopy(p_local)
        # zeroing the gradients
        for p in self.nn_temp.parameters():
            p.grad = None

        # compute full gradient at snapshot point
        self.nn_temp = self.nn_temp.to(self.device)
        for data, labels, _ in self.data_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nn_temp(data)
            loss = self.loss_func(output, labels)
            loss.backward()

        # copy full gradient
        self.grad_avg = []
        for p in self.nn_temp.parameters():
            self.grad_avg.append(deepcopy(p.grad) / len(self.data_loader.dataset))
