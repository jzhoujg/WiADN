# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

class DWA_Loss(nn.Module):

    def __init__(self, T=0.1, k=2, num_train_instances=1000):
        super(DWA_Loss, self).__init__()
        self.T = T
        self.k = k
        self.last_loss_x = 1
        self.last_loss_y = 1
        self.num_train_instances = num_train_instances

    def forward(self,*x):

        loss_1 = x[0]
        loss_2 = x[1]

        # update parameters
        w1 = math.sqrt(loss_1 / self.last_loss_x / self.num_train_instances)
        w2 = math.sqrt(loss_2 / self.last_loss_y / self.num_train_instances)
        l1 = self.k*math.exp(w1/self.T)/(math.exp(w1/self.T)+math.exp(w2/self.T))
        l2 = self.k*math.exp(w2/self.T)/(math.exp(w1/self.T)+math.exp(w2/self.T))
        loss = l1 * loss_1 + l2 * loss_2

        # record last-epoch loss
        self.last_loss_x = loss_1
        self.last_loss_y = loss_2

        return loss





class DTP_Loss(nn.Module):
    def __init__(self, gamma_0=1):
        super(DTP_Loss, self).__init__()
        self.gamma_0 = gamma_0

    def forward(self,*x):

        loss_1 = x[0]
        loss_2 = x[1]
        acc_1 = x[2]
        acc_2 = x[3]
        w_1 = -(1-acc_1)**self.gamma_0*torch.log(torch.tensor(acc_1))
        w_2 = -(1-acc_2)**self.gamma_0*torch.log(torch.tensor(acc_2))

        return w_1*loss_1 + w_2*loss_2


class Fixed_Weigh_Loss(nn.Module):
    def __init__(self, gamma_0=1):
        super(Fixed_Weigh_Loss, self).__init__()
    def forward(self,*x):
        return x[0]+x[1]




class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x[:2]):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])

        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())