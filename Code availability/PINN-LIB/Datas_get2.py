import torch
import numpy as np
from Constants import *


def inner_datas(N_t, N_x):

    #
    t_vec = torch.rand(N_t, 1)

    #
    x_vec = torch.rand(N_x, 1)

    #
    t = t_vec.repeat_interleave(N_x, dim=0)
    x = x_vec.repeat(N_t, 1)

    return x, t


def Lbc_datas(N_t):

    t = torch.rand(N_t, 1)
    x = torch.zeros_like(t)
    return x, t


def Rbc_datas(N_t):

    t = torch.rand(N_t, 1)
    x = torch.ones_like(t)
    return x, t


def t0_datas(N_x):

    x = torch.rand(N_x, 1)
    t = torch.zeros_like(x)
    return x, t