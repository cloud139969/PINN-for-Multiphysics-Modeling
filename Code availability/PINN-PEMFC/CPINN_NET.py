import torch
from torch.autograd import grad
import torch.nn as nn

import numpy as np


## Gradient calculation is used for partial derivative computations.
def d(u, t):
    return grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

#
def Actf(x):
    return torch.tanh(x)

     # x * torch.sigmoid(x)   torch.tanh(x)

##
class Net_g1(nn.Module):  ##
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 128)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 32)
        self.fc5 = torch.nn.Linear(32, 1)

        ## Reverse-solving
        # self.alpha = nn.Parameter(torch.tensor([0.6], requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain 默认为1.0

    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = Actf(self.fc3(x))
        x = Actf(self.fc4(x))
        x = self.fc5(x)
        return torch.nn.functional.softplus(x) ##   torch.sigmoid(x)   torch.nn.functional.softplus(x)


##
class Net_g2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 64)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 64)
        self.fc5 = torch.nn.Linear(64, 1)


        # self.alpha = nn.Parameter(torch.tensor([0.6], requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain 默认为1.0

    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = Actf(self.fc3(x))
        x = Actf(self.fc4(x))
        x = self.fc5(x)
        return x ##   torch.sigmoid(x)    torch.nn.functional.softplus(x)


##
class Net_g3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 256)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 1)

        ##
        # self.alpha = nn.Parameter(torch.tensor([0.6], requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain

    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = Actf(self.fc3(x))
        x = self.fc4(x)
        return x ##   torch.sigmoid(x) torch.nn.functional.softplus(x)


class Net_uP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##

    def forward(self, x):
        x = torch.relu_(self.fc1(x))
        x = self.fc2(x)
        return x



class Net_I(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 256)  # 3 means x, y, z
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight, gain= 5/3)  ##gain 1.0
    def forward(self, x):
        x = Actf(self.fc1(x))
        x = Actf(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.softplus(x)

