import torch
from torch.autograd import grad
import torch.nn as nn

import numpy as np


##
def d(u, t):
    return grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

#
def Actf(x):
    return torch.tanh(x)

     # x * torch.sigmoid(x)   torch.tanh(x)


##
class Net_g1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 128)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 1)

        ##
        # self.alpha = nn.Parameter(torch.tensor([0.6], requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain 默认为1.0

    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = Actf(self.fc3(x))
        x = self.fc4(x)
        return x ##   torch.sigmoid(x) torch.nn.functional.softplus(x)





class Net_g2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 128)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 1)


        # self.alpha = nn.Parameter(torch.tensor([0.6], requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain 默认为1.0

    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = Actf(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.softplus(x)

