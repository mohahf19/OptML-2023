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
        self.grad_sum = []  # full gradient at stored snapshot weight
        self.denom_avg = 0
        self.prob = prob  # probability of updating snapshot
        self.loss_func = loss_func
        self.device = device
        self.num_parts = num_parts
        self.assignment = assignment
        self.train_partitions = train_partitions
        self.params_snap = []

        self.prev_snapshot = [False for _ in range(num_parts)]

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, i, closure=None) -> bool:
        part = self.assignment[i]
        flag = False
        params_old = []
        variance_term = []
        sgd_step = []
        grad_term = []
        snap_dist = 0.0
        dist = 0.0
        print(self.prev_snapshot)
        print(part)
        if self.prev_snapshot[part]:
            var_red = self.variance_reduction_stoch_grad(x, y, part)
            for p, var_red_term in zip(self.params, var_red):
                params_old.append(p.clone())
                sgd_step.append(p.grad.clone())
                variance_term.append((p.grad -var_red_term).clone())
                grad_term.append(p.grad.clone())
                update = p.grad - var_red_term
                p.data = p.data - self.lr * update

            for p, p_snap, p_old  in zip(self.params, self.params_snap, params_old):
                snap_dist += (p.data - p_snap.data).norm()
                dist += (p.data - p_old.data).norm()
        else:
            for p in self.params:
                params_old.append(p.clone())
                grad_term.append(p.grad.clone())
                sgd_step.append(p.grad.clone())
                variance_term.append(p.grad.clone())
                update = p.grad
                p.data = p.data - self.lr * update
            
            for p, p_old  in zip(self.params, params_old):
                dist += (p.data - p_old.data).norm()

        if np.random.rand() <= self.prob:  # coin flip
            self.take_snapshot(part)
            if not self.prev_snapshot[part]:
                self.prev_snapshot[part] = True
                print("incrementing ----------------------------------------")
                self.denom_avg += 1
            
            flag = True

        return flag, variance_term, grad_term, snap_dist, dist, sgd_step

    def variance_reduction_stoch_grad(self, x, y, part):
        # zeroing the gradients
        for p in self.nns[part].parameters():
            p.grad = None

        # recompute gradient at sampled data point
        outputs = self.nns[part](x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        for p, g_avg in zip(self.nns[part].parameters(), self.grad_sum):
            grad_list.append(p.grad - g_avg/self.denom_avg)

        return grad_list

    def take_snapshot(self, part):
        print("Taking snapshot..")
        init_avg = True if len(self.grad_sum) == 0 else False
        for p in self.nns[part].parameters():
            p.grad = None
            if init_avg:
                self.grad_sum.append(None)

        self.nns[part] = self.nns[part].to(self.device)
        for data, labels, _ in self.train_partitions[part]:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nns[part](data)
            loss = self.loss_func(output, labels)
            loss.backward()
        for j, part_old in enumerate(self.nns[part].parameters()):
            if self.grad_sum[j] == None:
                self.grad_sum[j] = deepcopy(part_old.grad)
            else:
                self.grad_sum[j] -= part_old.grad/len(self.train_partitions[part])
        
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nns[part].parameters()):
            p_temp = deepcopy(p_local)
        
        # zeroing the gradients
        for p in self.nns[part].parameters():
            p.grad = None

        # compute full gradient at snapshot point
        self.nns[part] = self.nns[part].to(self.device)
        for data, labels, _ in self.train_partitions[part]:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.nns[part](data)
            loss = self.loss_func(output, labels)
            loss.backward()

        for j, part_new in enumerate(self.nns[part].parameters()):
            self.grad_sum[j] = self.grad_sum[j] + part_new.grad/len(self.train_partitions[part])