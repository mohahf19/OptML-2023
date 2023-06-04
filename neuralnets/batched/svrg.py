from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SVRG(Optimizer):
    def __init__(self, params, prob_snapshot, nn, loss_func, steps_per_snapshot, snapshot_rand, data_loader, device, lr_decrease, decrease_step, momentum, weight_decay, lr=required):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.lr = lr
        self.nn_temp = nn  # stored snapshot neural networ, i.e. store the weight
        self.grad_avg = []  # full gradient at stored snapshot weight
        self.prob = prob_snapshot  # probability of updating snapshot
        self.snapshot_rand = snapshot_rand
        self.steps_per_snapshot = steps_per_snapshot
        self.data_loader = (
            data_loader  # access to full dataset to compute average gradient
        )
        self.loss_func = loss_func
        self.device = device

        self.prev_snapshot = False

        self.lr_decrease = lr_decrease
        self.decrease_step = decrease_step
        self.momentum = momentum
        self.weight_decay = weight_decay

        if self.momentum != 0:
            self.momentum_mem = {p:p.data.clone().zero_() for p in self.params}

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, step, closure=None) -> bool:
        if step in self.decrease_step:
            self.lr = self.lr*self.lr_decrease
        if self.prev_snapshot:
            var_red = self.variance_reduction_stoch_grad(x, y)
            for p, var_red_term in zip(self.params, var_red):
                update = (p.grad+self.weight_decay*p.data) - var_red_term
                if self.momentum != 0:
                    update += self.momentum*self.momentum_mem[p]
                    self.momentum_mem[p] = update
                p.data = p.data - self.lr * update
        else:
            for p in self.params:
                update = (p.grad+self.weight_decay*p.data)
                if self.momentum != 0:
                    update += self.momentum*self.momentum_mem[p]
                    self.momentum_mem[p] = update
                p.data = p.data - self.lr * update
        if (self.snapshot_rand and np.random.rand() <= self.prob) or (not self.snapshot_rand and (step+1) % self.steps_per_snapshot == 0):  # coin flip
            self.take_snapshot()
            self.prev_snapshot = True
            return True
        return False

    def variance_reduction_stoch_grad(self, x, y):
        # zeroing the gradients
        for p in self.nn_temp.parameters():
            p.grad = None

        # recompute gradient at sampled data point
        outputs = self.nn_temp(x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        for p, g_avg in zip(self.nn_temp.parameters(), self.grad_avg):
            grad_list.append((p.grad+self.weight_decay*p.data) - g_avg)

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
            self.grad_avg.append((p.grad+self.weight_decay*p.data) / len(self.data_loader.dataset))
