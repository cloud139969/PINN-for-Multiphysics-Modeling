import torch
import numpy as np
from constants import *



# ==============================================================================


def inner_datas(N_t, N_xy):
    """

    """

    #  t_vec ( [N_t, 1])
    # -------------------------------------------------------
    N_t1 = int(0.5 * N_t)
    N_t2 = N_t - N_t1

    t1 = torch.rand(N_t1, 1) * 0.25
    t2 = torch.rand(N_t2, 1) * (1.0 - 0.25) + 0.25
    t_vec = torch.cat([t1, t2], dim=0)  # [N_t, 1]


#########
    t_vec[-1] = 1.0
###########
    # -------------------------------------------------------
    #
    N_x1 = int(0.4 * N_xy)
    N_x2 = int(0.3 * N_xy)
    N_x3 = N_xy - N_x1 - N_x2


    # t_ccl, t_mpl, t_gdl, h_ch = ...
    x_cl = (t_ccl) / (t_ccl + t_mpl + t_gdl + h_ch)
    x_gdl = (t_ccl + t_mpl + t_gdl) / (t_ccl + t_mpl + t_gdl + h_ch)

    x1 = torch.rand(N_x1, 1) * x_cl
    x2 = torch.rand(N_x2, 1) * (x_gdl - x_cl) + x_cl
    x3 = torch.rand(N_x3, 1) * (1 - x_gdl) + x_gdl
    x_vec = torch.cat([x1, x2, x3], dim=0)  # [N_xy, 1]


    y_vec = torch.rand(N_xy, 1)  # shape: [N, 1]


    t = t_vec.repeat_interleave(N_xy, dim=0)


    x = x_vec.repeat(N_t, 1)
    y = y_vec.repeat(N_t, 1)

    return x, y, t


def Lbc_datas(N_t, N_y):

    N_t1 = int(0.5 * N_t)
    N_t2 = N_t - N_t1
    t_vec = torch.cat([
        torch.rand(N_t1, 1) * 0.25,
        torch.rand(N_t2, 1) * 0.75 + 0.25
    ], dim=0)

    #
    y_vec = torch.rand(N_y, 1)
    x_vec = torch.zeros(N_y, 1)

    #
    t = t_vec.repeat_interleave(N_y, dim=0)
    x = x_vec.repeat(N_t, 1)
    y = y_vec.repeat(N_t, 1)

    return x, y, t


def Rbc_datas(N_t, N_y):

    # 1. Time
    N_t1 = int(0.5 * N_t)
    N_t2 = N_t - N_t1
    t_vec = torch.cat([
        torch.rand(N_t1, 1) * 0.25,
        torch.rand(N_t2, 1) * 0.75 + 0.25
    ], dim=0)

    # 2. Space
    y_vec = torch.rand(N_y, 1)
    x_vec = torch.ones(N_y, 1)

    # 3. Product
    t = t_vec.repeat_interleave(N_y, dim=0)
    x = x_vec.repeat(N_t, 1)
    y = y_vec.repeat(N_t, 1)

    return x, y, t


def Ubc_datas(N_t, N_x):

    # 1. Time
    N_t1 = int(0.5 * N_t)
    N_t2 = N_t - N_t1
    t_vec = torch.cat([
        torch.rand(N_t1, 1) * 0.25,
        torch.rand(N_t2, 1) * 0.75 + 0.25
    ], dim=0)

    # 2. Space (x 随机, y=1)
    x_vec = torch.rand(N_x, 1)
    y_vec = torch.ones(N_x, 1)

    # 3. Product
    t = t_vec.repeat_interleave(N_x, dim=0)
    x = x_vec.repeat(N_t, 1)
    y = y_vec.repeat(N_t, 1)

    return x, y, t


def Dbc_datas(N_t, N_x):

    # 1. Time
    N_t1 = int(0.5 * N_t)
    N_t2 = N_t - N_t1
    t_vec = torch.cat([
        torch.rand(N_t1, 1) * 0.25,
        torch.rand(N_t2, 1) * 0.75 + 0.25
    ], dim=0)

    # 2. Space
    x_vec = torch.rand(N_x, 1)
    y_vec = torch.zeros(N_x, 1)

    # 3. Product
    t = t_vec.repeat_interleave(N_x, dim=0)
    x = x_vec.repeat(N_t, 1)
    y = y_vec.repeat(N_t, 1)

    return x, y, t


def t0_datas(N_xy):
    """

    """

    x = torch.rand(N_xy, 1)
    y = torch.rand(N_xy, 1)
    t = torch.zeros(N_xy, 1)
    return x, y, t