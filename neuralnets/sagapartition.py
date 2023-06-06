from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SAGAPartition(Optimizer):
    def __init__(self, params, nns, loss_func, prob, data_loader_part, device, num_part, data, targets, lr=required):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.lr = lr
        self.nn_temp = nns  # stored snapshot neural networ, i.e. store the weight
        self.grad_avg = []  # full gradient at stored snapshot weight
        self.prob = prob  # probability of updating snapshot
        self.data_loader = (
            data_loader_part  # access to full dataset to compute average gradient
        )
        self.loss_func = loss_func
        self.device = device
        print("#########################", data.device, targets.device)
        self.num_part = num_part

        """
        self.partitioned_data = torch.zeros([num_part, int(np.ceil(data.shape[0]/num_part)), 1] + list(data.shape[1:]))
        self.partitioned_targets = torch.zeros([num_part, int(np.ceil(targets.shape[0]/num_part))] + list(targets.shape[1:]))
        self.partitioned_data[0] = data.reshape([int(np.ceil(data.shape[0]/num_part)), 1] + list(data.shape[1:]))
        self.partitioned_targets[0] = targets.reshape([int(np.ceil(targets.shape[0]/num_part))] + list(targets.shape[1:]))
        
        for i in range(num_part):
            print("aaaaaaaaaaaaaaaaaaaaaaa")
            self.partitioned_data[i] = self.partitioned_data[i].to(device)
        for i in range(num_part):
            self.partitioned_targets[i] = self.partitioned_targets[i].to(device)
        self.partitioned_data.to("mps")
        print(self.partitioned_data.device)
        """

        self.partitioned_data = data
        self.partitioned_targets = targets

        print(self.partitioned_data.shape, self.partitioned_data.device)
        print(self.partitioned_targets.shape, self.partitioned_targets.device)
        print(self.partitioned_data[0].shape)
        print(self.partitioned_targets[0].shape)

        self.prev_snapshot = False

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, i, closure=None) -> bool:
        print(x.shape, y.shape)
        if self.prev_snapshot:
            var_red = self.variance_reduction_stoch_grad(x, y)
            for p, var_red_term in zip(self.params, var_red):
                update = p.grad - var_red_term
                p.data = p.data - self.lr * update
        else:
            for p in self.params:
                update = p.grad
                p.data = p.data - self.lr * update
        if np.random.rand() <= self.prob:  # coin flip
            self.take_snapshot()
            self.prev_snapshot = True
            return True
        return False

    def variance_reduction_stoch_grad(self, x, y):
        # zeroing the gradients
        for p in self.nn_temp[0].parameters():
            p.grad = None

        # recompute gradient at sampled data point
        outputs = self.nn_temp[0](x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        for p, g_avg in zip(self.nn_temp[0].parameters(), self.grad_avg):
            grad_list.append(p.grad - g_avg)

        return grad_list

    def take_snapshot(self):
        print("Taking snapshot...")
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nn_temp[0].parameters()):
            p_temp = deepcopy(p_local)
        
        # zeroing the gradients
        for p in self.nn_temp[0].parameters():
            p.grad = None

        # compute full gradient at snapshot point
        self.nn_temp[0] = self.nn_temp[0].to(self.device)
        
        output = self.nn_temp[0](self.partitioned_data[0])
        loss = self.loss_func(output, self.partitioned_targets[0])
        loss.backward()

        # copy full gradient
        self.grad_avg = []
        for p in self.nn_temp[0].parameters():
            self.grad_avg.append(deepcopy(p.grad) / len(self.data_loader.dataset))
