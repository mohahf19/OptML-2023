from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SAGA(Optimizer):
    def __init__(self, params, prob, nns, loss_func, data_loader, device, num_parts, assignment, train_partitions, lr=required):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.lr = lr
        self.nns = nns  # stored snapshot neural networ, i.e. store the weight
        self.grad_avg = [None for _ in range(num_parts)]  # full gradient at stored snapshot weight
        self.prob = prob  # probability of updating snapshot
        self.data_loader = (
            data_loader  # access to full dataset to compute average gradient
        )
        self.loss_func = loss_func
        self.device = device
        self.num_parts = num_parts
        self.assignment = assignment
        self.train_partitions = train_partitions

        self.prev_snapshot = [False for _ in range(num_parts)]

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, i, closure=None) -> bool:
        part = self.assignment[i]
        if self.prev_snapshot[part]:
            var_red = self.variance_reduction_stoch_grad(x, y, part)
            for p, var_red_term in zip(self.params, var_red):
                update = p.grad - var_red_term
                p.data = p.data - self.lr * update
        else:
            for p in self.params:
                update = p.grad
                p.data = p.data - self.lr * update
        if np.random.rand() <= self.prob:  # coin flip
            self.take_snapshot(part)
            self.prev_snapshot[part] = True
            return True
        return False

    def variance_reduction_stoch_grad(self, x, y, part):
        # zeroing the gradients
        for p in self.nns[part].parameters():
            p.grad = None

        # recompute gradient at sampled data point
        outputs = self.nns[part](x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        for p, g_avg in zip(self.nns[part].parameters(), self.grad_avg):
            grad_list.append(p.grad - g_avg)

        return grad_list

    def take_snapshot_costly(self, part):
        print("Taking snapshot..")
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nns[part].parameters()):
            p_temp = deepcopy(p_local)
        # zeroing the gradients
        for p in self.nns[part].parameters():
            p.grad = None
        self.prev_snapshot[part] = True

        # compute full gradient at snapshot point
        self.nns[part] = self.nns[part].to(self.device)
        part_avg = 0
        for data, labels, _ in self.data_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nns[part_avg](data)
            loss = self.loss_func(output, labels)
            loss.backward()

            part_avg += 1

        # copy full gradient
        self.grad_avg = [None for _ in self.params]
        for part_avg in range(self.num_parts):
            if self.prev_snapshot[part_avg]:
                for p_index, p in enumerate(self.nns[part_avg].parameters()):
                    if self.grad_avg[p_index] == None:
                        self.grad_avg[p_index] = p.grad
                    else:
                        self.grad_avg[p_index] += p.grad
        for p_index in range(len(self.grad_avg)):
            self.grad_avg[p_index] /= ((len(self.data_loader.dataset)//self.num_parts)*len([t for t in self.prev_snapshot if t==True]))

    def take_snapshot(self, part):
        print("Taking snapshot..")
        for p in self.nns[part].parameters():
            p.grad = None

        self.nns[part] = self.nns[part].to(self.device)
        for data, labels, _ in self.train_partitions[part]:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nns[part](data)
            loss = self.loss_func(output, labels)
            loss.backward()
        for global_old, part_old in zip(self.grad_avg, self.nns[part].parameters()):
            global_old = global_old-part_old
        
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nns[part].parameters()):
            p_temp = deepcopy(p_local)
        # zeroing the gradients
        for p in self.nns[part].parameters():
            p.grad = None
        self.prev_snapshot[part] = True

        # compute full gradient at snapshot point
        self.nns[part] = self.nns[part].to(self.device)
        for data, labels, _ in self.train_partitions[part]:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nns[part](data)
            loss = self.loss_func(output, labels)
            loss.backward()

        for global_minus_old, part_new in zip(self.grad_avg, self.nns[part].parameters()):
            global_minus_old = global_minus_old+part_old