from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SAGAPartition(Optimizer):

    def __init__(self, params, nns, loss_func, prob, data_loader_part, device, num_part, data, targets, lr=required):
        defaults = {"lr": lr}
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.dataset_len = len(data_loader_part.dataset)

        self.part_sizes = []
        l = len(data_loader_part.dataset)
        while l > 0:
            s = min(l, len(data_loader_part.dataset)//num_part)
            self.part_sizes.append(s)
            l = l - s
        self.num_part = len(self.part_sizes)
        self.nn_temp = nns  # stored snapshot neural networ, i.e. store the weight

        for part in range(self.num_part):
            shapes = []
            for p in self.nn_temp[part].parameters():
                shapes.append(p.data.shape)
                p.grad = None
            print("###################", part, shapes)
        self.partition_indicator = [-1 for _ in range(len(data_loader_part.dataset))]
        self.partitioned_data = torch.zeros([self.num_part, np.max(np.array(self.part_sizes)), 1]+list(data[0].shape))
        self.partitioned_targets = torch.zeros([self.num_part, np.max(np.array(self.part_sizes))]+list(targets[0].shape))
        data_iter = iter(data_loader_part)
        quit = False
        for part in range(self.num_part):
            indices = []
            for c in range(self.part_sizes[part]):
                indexed_point = next(data_iter, None)
                if indexed_point == None:
                    quit = True
                    break
                else:
                    datapoint, y, index = indexed_point
                    indices.append(index[0])
                    self.partition_indicator[index[0]] = part
            if quit:
                break
            else:
                self.partitioned_data[part] = torch.reshape(data[indices], [len(indices),1]+list(data[indices[0]].shape))
                self.partitioned_targets[part] = targets[indices]
                
        assert len([part for part in self.partition_indicator if part == -1]) == 0

        self.lr = lr
        self.grad_avg = []  # full gradient at stored snapshot weight
        self.prob = prob
        self.loss_func = loss_func
        self.device = device

        self.prev_snapshot = [False for _ in range(num_part)]

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, x, y, i, closure=None) -> bool:
        part = self.partition_indicator[i]
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
        part = 9
        # zeroing the gradients
        for p in self.nn_temp[part].parameters():
            p.grad = None
        # recompute gradient at sampled data point
        outputs = self.nn_temp[part](x)
        loss = self.loss_func(outputs, y)
        loss.backward()

        grad_list = []
        #for p, g_avg in zip(self.nn_temp[9].parameters(), self.grad_avg):
        #    print(9, p.data.shape, g_avg.shape)
        for p, g_avg in zip(self.nn_temp[part].parameters(), self.grad_avg):
            print(part, p.data.shape, p.grad.shape, g_avg.shape)
            grad_list.append(p.grad - g_avg)
        

        return grad_list

    def take_snapshot(self, part):
        print("Taking snapshot..")
        # update snapshot
        for p_local, p_temp in zip(self.params, self.nn_temp[part].parameters()):
            p_temp = deepcopy(p_local)
        # zeroing the gradients
        shapes = []
        for p in self.nn_temp[part].parameters():
            shapes.append(p.data.shape)
            p.grad = None
        print("@@@@@@@@@@", part, shapes)

        # compute full gradient at snapshot point
        
        for p in range(self.num_part):
            self.nn_temp[p] = self.nn_temp[p].to(self.device)
            data, labels = self.partitioned_data[p].to(self.device), self.partitioned_targets[p].to(self.device)

            output = self.nn_temp[p](data)
            loss = self.loss_func(output, labels)
            loss.backward()


        # copy full gradient
        self.grad_avg = []
        for p_name, shape in zip(self.nn_temp[0].state_dict(), shapes):
            g = torch.zeros([self.num_part]+list(shape))
            # self.grad_avg.append(torch.sum(torch.cat([self.nn_temp[p].state_dict()[p_name].grad for p in range(self.num_part)], out = g), 0) / self.dataset_len)
            self.grad_avg.append(torch.sum(g, 0) / self.dataset_len)
        for i in range(len(self.grad_avg)):
            self.grad_avg[i] = self.grad_avg[i].to(self.device)
