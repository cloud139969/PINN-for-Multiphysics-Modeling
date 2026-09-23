import torch
import numpy as np
import matplotlib.pyplot as plt
from Region_Masks import get_mask
from CPINN_NET import d, Net_g1, Net_g2, Net_g3, Net_I
from Datas_get2 import inner_datas,t0_datas,Lbc_datas,Rbc_datas,Ubc_datas,Dbc_datas
from constants import *
import time
import random
from NTK_algo import NTKHandler



# --- Seeds ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  #


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# print("Training device:", device)



epochs = 10000  # 6000
n_t = int(round(t_max / 0.1))  ## t_max/0.1
n_inner = 50  ## 200  300
n_bc = 15      ## 60   100
# print("t_max",t_max)
min_loss1 = float('inf')


####
P_cch_in = Pc_in
P_ach_in = Pa_in
C_h2_in = (P_ach_in - RH_a * P_sat) / (R * T_s)
C_o2_in = 0.21 * (P_cch_in - RH_c * P_sat) / (R * T_s)

u_a_in = ST_a * I_ave * A_cell / (2 * F * C_h2_in * A_ach)  ## 气体流速需要修正
u_c_in = ST_c * I_ave * A_cell / (4 * F * C_o2_in * A_cch)
print("u_c_in,u_a_in", u_c_in, u_a_in)
print("C_o2_in,C_h2_in", C_o2_in,C_h2_in)
####################################################
C_o2_initial = 0.21 * Pc_in / R / T_e # 阴极O₂初始浓度
C_h2_initial = Pa_in / R / T_e # 阳极H₂初始浓度
s_initial = 0.0001
T_initial = T_e ##   273.15 + 70   T_e
# I_initial = I_ave

## Normalization
uu = u_c_in
tt = 10.0
PP = 10.0
CC = C_o2_in     ## 用于o2
CC_sacle = 2.1  ## 1.2
AA = C_h2_in    ## 用于h2
AA_sacle = 20.0
SS1 = 0.04
SS2 = 0.01
# TT = 342.0  # 342.0
TT_sacle = 3.0  # 0.3


## Interface
xc_total = t_ccl + t_mpl + t_gdl + h_ch
xc1 = t_ccl / xc_total
xc2 = (t_ccl+ t_mpl+ t_gdl) / xc_total

xa_total = t_acl + t_mpl + t_gdl + h_ch
xa1 = t_acl / xa_total
xa2 = (t_acl+ t_mpl+ t_gdl) / xa_total

## Stack
def create_component():
    PINN_h2 = Net_g2().to(device)
    PINN_o2 = Net_g2().to(device)
    PINN_s_gdl = Net_g1().to(device)   #Net_g1().double() .to(device)
    PINN_s_cl = Net_g1().to(device)
    PINN_T = Net_g3().to(device)
    PINN_I = Net_I().to(device)


    optimizer1 = torch.optim.Adam([
        {'params': PINN_h2.parameters(), 'lr': 0.0001},   ## 'weight_decay':1e-5
        {'params': PINN_o2.parameters(), 'lr': 0.0001},  ## 'weight_decay':1e-5
        {'params': PINN_s_gdl.parameters(), 'lr': 0.0001},  ## 'weight_decay':1e-5
        {'params': PINN_s_cl.parameters(), 'lr': 0.0001},  ## 'weight_decay':1e-5
        {'params': PINN_T.parameters(), 'lr': 0.0001},  ## 'weight_decay':1e-5
        {'params': PINN_I.parameters(), 'lr': 0.0001},  ## 'weight_decay':1e-5
    ])


    ntk_handler_cl = NTKHandler(update_every=200, momentum=0.8, subset_size=256)  # 128
    ntk_handler_gdl = NTKHandler(update_every=200, momentum=0.8, subset_size=256)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=8000, gamma=0.2)
    # return (PINN_mw, PINN_lq_gdl, PINN_lq_cl, PINN_vp_gdl, PINN_vp_cl, PINN_T_gdl, PINN_T_cl,
    #         awl1, awl2, awl3, awl4, optimizer1)
    return PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I, optimizer1, ntk_handler_cl, ntk_handler_gdl


##
components = [create_component() for _ in range(num_cells)]

###################################################
MODEL_CKPT_MAP = {
    "PINN_o2": "Initialization_o2.pt",
    "PINN_h2": "Initialization_h2.pt",
    "PINN_T": "Initialization_T.pt",
    # "PINN_s_cl": "Initialization_s_cl.pt",
    # "PINN_s_gdl": "Initialization_s_gdl.pt",
    "PINN_I": "Initialization_I.pt"
}


for cell_idx in range(num_cells):
    #
    PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I, optimizer1, ntk_handler_cl, ntk_handler_gdl = components[cell_idx]

    #
    model_instances = {
        "PINN_o2": PINN_o2,
        "PINN_h2": PINN_h2,
        "PINN_T": PINN_T,
        # "PINN_s_cl": PINN_s_cl,
        # "PINN_s_gdl": PINN_s_gdl,
        "PINN_I": PINN_I,
    }


    for model_name, ckpt_path in MODEL_CKPT_MAP.items():
        if model_name not in model_instances:
            print(f"Warning: {model_name} is not defined in the current cell, skipping")
            continue

        model = model_instances[model_name]
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict_key = f"{model_name}_state_dict"
            if state_dict_key not in checkpoint:
                raise KeyError(f"Checkpoint missing key: {state_dict_key}")
            model.load_state_dict(checkpoint[state_dict_key])
            print(f"Successfully loaded parameters for {model_name} of cell {cell_idx} (file: {ckpt_path})")
        except FileNotFoundError:
            print(f"Warning: file {ckpt_path} does not exist! Skipping parameter loading for this model")
        except KeyError as e:
            print(f"Warning: required key not found in {ckpt_path}! Error: {e}")
        except Exception as e:
            print(f"Unknown error while loading {model_name}: {e}")



#
loss_list = []
epoch_list = []
loss_data_list = []
loss_pde_list = []
loss_bc_list = []
all_outputs = []    # epoch and component


##################################################
## collocation point from 0 to 1
def get_resampled_data(device):

    x, y, t = inner_datas(n_t, n_inner)
    x_L, y_L, t_L = Lbc_datas(n_t, n_bc)
    x_R, y_R, t_R = Rbc_datas(n_t, n_bc)
    x_U, y_U, t_U = Ubc_datas(n_t, n_bc)
    x_D, y_D, t_D = Dbc_datas(n_t, n_bc)
    x_t0, y_t0, t_t0 = t0_datas(n_bc)  # 初始时刻只需要空间点

    tensors_list = [
        x, y, t,
        x_L, y_L, t_L,
        x_R, y_R, t_R,
        x_U, y_U, t_U,
        x_D, y_D, t_D,
        x_t0, y_t0, t_t0
    ]

    tensors_on_device = [tensor.to(device) for tensor in tensors_list]

    processed_tensors = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in tensors_on_device
    ]

    return processed_tensors


##################################################
start_time = time.time()
##################################################


is_even = (num_cells % 2 == 0)
previous_T = T_s
cell_inlet_T = [0.0 for _ in range(num_cells)]  # 每个电池的 温度分布
for i in range(num_cells):
    if i == 0:
        current_T = previous_T
    elif i <= ((num_cells - 1) // 2):
        current_T = ((2 * k_T_bp / h_bp / h_T_water + 1) * previous_T - 2 * T_cool) / \
                    (2 * k_T_bp / h_bp / h_T_water - 1)
    else:
        if is_even and i == (num_cells // 2):
            current_T = previous_T
        else:
            current_T = ((2 * k_T_bp / h_bp / h_T_water - 1) * previous_T + 2 * T_cool) / \
                        (2 * k_T_bp / h_bp / h_T_water + 1)
    cell_inlet_T[i] = current_T
    previous_T = current_T



#############################################################
##########################   Main   ##########################
current_data = get_resampled_data(device)
for epoch in range(epochs):
    # if epoch % 2 == 0:
    #     torch.cuda.empty_cache()
    current_epoch_loss = 1.0

    # ======
    if epoch > 0 and epoch % 50 == 0:  ## 每100个epoch重新采样
        current_data = get_resampled_data(device)

    #
    (x, y, t,
     x_L, y_L, t_L,
     x_R, y_R, t_R,
     x_U, y_U, t_U,
     x_D, y_D, t_D,
     x_t0, y_t0, t_t0) = current_data
    ##
    t = t * t_max / tt
    t_L = t_L * t_max / tt
    t_R = t_R * t_max / tt
    t_U = t_U * t_max / tt
    t_D = t_D * t_max / tt


    ##
    for i, (PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I, optimizer1, ntk_handler_cl, ntk_handler_gdl ) in enumerate(components):
        ##
        optimizer1.zero_grad()
        # optimizer2.zero_grad()

        #
        x_ccl = x * xc1
        x_acl = x * xa1
        x_cch_centre = (1 + xc2) / 2 * torch.ones_like(y)
        x_ach_centre = (1 + xa2) / 2 * torch.ones_like(y)
        x_ccl_centre = xc1 / 2 * torch.ones_like(y)
        x_acl_centre = xa1 / 2 * torch.ones_like(y)

        ################  Forward propagation  ################
        ## H2
        C_h2 = C_h2_initial + (1.0 - torch.exp(-tt *t))*AA_sacle* PINN_h2(torch.cat([x, y, t], dim=1))
        C_h2_Lbc = C_h2_initial + (1.0 - torch.exp(-tt *t_L))*AA_sacle* PINN_h2(torch.cat([x_L, y_L, t_L], dim=1))
        C_h2_Ubc = C_h2_initial + (1.0 - torch.exp(-tt *t_U))*AA_sacle* PINN_h2(torch.cat([x_U, y_U, t_U], dim=1))
        xa_D_ch = x_D * (1.0 - xa2) + xa2
        C_h2_Dbc_cch = C_h2_initial + (1.0 - torch.exp(-tt *t_D))*AA_sacle* PINN_h2(torch.cat([xa_D_ch, y_D, t_D], dim=1))
        ## O2
        C_o2 = C_o2_initial+ (1.0 - torch.exp(-tt *t))*CC_sacle* PINN_o2(torch.cat([x, y, t], dim=1))
        C_o2_Lbc = C_o2_initial+ (1.0 - torch.exp(-tt *t_L))*CC_sacle* PINN_o2(torch.cat([x_L, y_L, t_L], dim=1))
        C_o2_Ubc = C_o2_initial + (1.0 - torch.exp(-tt *t_U))*CC_sacle * PINN_o2(torch.cat([x_U, y_U, t_U], dim=1))
        xc_D_ch = x_D * (1.0 - xc2) + xc2
        C_o2_Dbc_cch = C_o2_initial+ (1.0 - torch.exp(-tt *t_D))*CC_sacle* PINN_o2(torch.cat([xc_D_ch, y_D, t_D], dim=1))
        ## Water s
        s_lq_cl = s_initial + (1.0 - torch.exp(-tt * t))*SS1* PINN_s_cl(torch.cat([x, y, t], dim=1))
        s_lq_Lbc_cl = s_initial + (1.0 - torch.exp(-tt * t_L))*SS1* PINN_s_cl(torch.cat([x_L, y_L, t_L], dim=1))
        s_lq_Rbc_cl = s_initial + (1.0 - torch.exp(-tt * t_R))*SS1* PINN_s_cl(torch.cat([x_R, y_R, t_R], dim=1))
        s_lq_Ubc_cl = s_initial + (1.0 - torch.exp(-tt * t_U))*SS1* PINN_s_cl(torch.cat([x_U, y_U, t_U], dim=1))

        s_lq_gdl = s_initial + (1.0 - torch.exp(-tt * t))*SS2* PINN_s_gdl(torch.cat([x, y, t], dim=1))
        s_lq_Lbc_gdl = s_initial + (1.0 - torch.exp(-tt * t_L))*SS2* PINN_s_gdl(torch.cat([x_L, y_L, t_L], dim=1))
        s_lq_Rbc_gdl = s_initial + (1.0 - torch.exp(-tt * t_R))*SS2* PINN_s_gdl(torch.cat([x_R, y_R, t_R], dim=1))
        s_lq_Ubc_gdl = s_initial + (1.0 - torch.exp(-tt * t_U))*SS2* PINN_s_gdl(torch.cat([x_U, y_U, t_U], dim=1))
        ## T
        T = T_initial + (1.0 - torch.exp(-tt *t))*TT_sacle * PINN_T(torch.cat([x, y, t], dim=1))
        T_Lbc = T_initial + (1.0 - torch.exp(-tt *t_L))*TT_sacle* PINN_T(torch.cat([x_L, y_L, t_L], dim=1))
        T_Ubc = T_initial + (1.0 - torch.exp(-tt *t_U))*TT_sacle* PINN_T(torch.cat([x_U, y_U, t_U], dim=1))
        T_Rbc_cch = T_initial + (1.0 - torch.exp(-tt *t_R))*TT_sacle*PINN_T(torch.cat([x_R, y_R, t_R], dim=1))

        ############ Change I ############
        t_step = 5.0 / tt  ##
        k = 200.0  # 50.0
        I_ave = I_base + I_delta * torch.sigmoid(k * (t - t_step))  ##
        I = I_ave * PINN_I(torch.cat([y, t], dim=1))

        I_view = I.view(n_t, n_inner, 1)
        I_mean_pred = torch.mean(I_view, dim=1, keepdim=True)  # [N_t, 1, 1]
        I_mean_broad = I_mean_pred.repeat(1, n_inner, 1).view(-1, 1)
        I_local = I * (I_ave / I_mean_broad)
        I_local_detach = I_local.detach()


        ## Definition of the center of the calculation domain
        C_h2_cl = C_h2_initial+ (1.0 - torch.exp(-tt *t))*AA_sacle* PINN_h2(torch.cat([x_acl, y, t], dim=1))
        C_o2_cl = C_o2_initial+ (1.0 - torch.exp(-tt *t))*CC_sacle* PINN_o2(torch.cat([x_ccl, y, t], dim=1))
        T_ccl = T_initial + (1.0 - torch.exp(-tt *t))*TT_sacle * PINN_T(torch.cat([x_ccl, y, t], dim=1))
        #
        C_o2_xch = C_o2_initial+ (1.0 - torch.exp(-tt *t))*CC_sacle* PINN_o2(torch.cat([x_cch_centre, y, t], dim=1)).detach()
        C_h2_xch = C_h2_initial+ (1.0 - torch.exp(-tt *t))*AA_sacle* PINN_h2(torch.cat([x_ach_centre, y, t], dim=1)).detach()
        C_o2_xcl = C_o2_initial+ (1.0 - torch.exp(-tt *t))*CC_sacle* PINN_o2(torch.cat([x_ccl_centre, y, t], dim=1)).detach()
        C_h2_xcl = C_h2_initial+ (1.0 - torch.exp(-tt *t))*AA_sacle* PINN_h2(torch.cat([x_acl_centre, y, t], dim=1)).detach()

        ## Mean calculation
        T_ccl_reshaped = T_ccl.view(n_t, n_inner, 1)
        mean_T_cl_per_time = torch.mean(T_ccl_reshaped, dim=1, keepdim=True)
        mean_T_cl = mean_T_cl_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()

        C_h2_cl_reshaped = C_h2_cl.view(n_t, n_inner, 1)
        mean_C_h2_cl_per_time = torch.mean(C_h2_cl_reshaped, dim=1, keepdim=True)
        mean_C_h2_cl = mean_C_h2_cl_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()

        C_o2_cl_reshaped = C_o2_cl.view(n_t, n_inner, 1)
        mean_C_o2_cl_per_time = torch.mean(C_o2_cl_reshaped, dim=1, keepdim=True)
        mean_C_o2_cl = mean_C_o2_cl_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()

        s_ch = 1e-6
        s_lq_gdl_reshaped = s_lq_gdl.view(n_t, n_inner, 1)
        mean_s_gdl_per_time = torch.mean(s_lq_gdl_reshaped, dim=1, keepdim=True)
        mean_s_gdl = 3.0* mean_s_gdl_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()

        s_lq_cl_reshaped = s_lq_cl.view(n_t, n_inner, 1)
        mean_s_cl_per_time = torch.mean(s_lq_cl_reshaped, dim=1, keepdim=True)
        mean_s_cl = 3.0* mean_s_cl_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()

        T_cell = cell_inlet_T[i]


        # -------------------------- Calculation of membrane water formation --------------------------
        ##
        term_base = I_base * t
        sp_t = torch.nn.functional.softplus(k * (t - t_step))
        sp_0 = torch.nn.functional.softplus(torch.tensor(-k * t_step, device=t.device))
        term_step = (I_delta / k) * (sp_t - sp_0)

        integral_I_ave = term_base + term_step
        integral_physical = integral_I_ave * tt

        delta_lam_generated = integral_physical / (2 * F * (t_ccl + t_mem)) * (EW / (den_mem * w_ion))
        lam_current = torch.clamp(lam_initial + delta_lam_generated, max=lam_sat)


        lam_reshaped = lam_current.view(n_t, n_inner, 1)
        mean_lam_per_time = torch.mean(lam_reshaped, dim=1, keepdim=True)
        lam = mean_lam_per_time.repeat(1, n_inner, 1).view(-1, 1).detach()


        available_capacity = lam_sat - lam_initial
        lam_overflow = delta_lam_generated - available_capacity
        water_switch = torch.sigmoid(200.0 * lam_overflow)


        # Effective diffusion coefficient
        D_h2_cl = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * por_cl ** 1.5 * (1 - mean_s_cl) ** 1.5
        D_h2_gdl = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * por_gdl ** 1.5 * (1 - mean_s_gdl) ** 1.5
        D_h2_ch = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * (1 - s_ch) ** 1.5

        D_o2_cl = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * por_cl ** 1.5 * (1 - mean_s_cl) ** 1.5
        D_o2_gdl = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * por_gdl ** 1.5 * (1 - mean_s_gdl) ** 1.5
        D_o2_ch = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * (1 - s_ch) ** 1.5
        # print("D_o2_cl,D_o2_gdl",D_o2_cl,D_o2_gdl)

        D_nf_eff = (w_ion ** 1.5) * D_nf
        ## Effective conductivity
        kele_cl_eff = kele_cl * (1 - por_cl) ** 1.5
        kele_mpl_eff = kele_mpl * (1 - por_mpl) ** 1.5
        kele_gdl_eff = kele_gdl * (1 - por_gdl) ** 1.5
        kion_mem = (0.5319 * lam - 0.326) * torch.exp(1268 * (1 / 303.15 - 1 / mean_T_cl))
        kion_mem_eff = kion_mem * w_ion ** 1.5
        ## Effective permeability
        K0_ccl_eff = K0_cl * (s_lq_cl ** 3)
        K0_gdl_eff = K0_gdl * (s_lq_gdl ** 3)
        ten_water = -0.0001676 * mean_T_cl + 0.1218


        # ------------------------------------------------------------------------#
        ####################  Electrochemical calculation  ####################
        ##
        ja = (1 - mean_s_cl) * ia_ref * (C_h2_xcl / C_h2_ref) ** 0.5 * (torch.exp(-1400 * (1 / mean_T_cl - 1 / 298.15)))
        jc = (1 - mean_s_cl) * ic_ref * (C_o2_xcl / C_o2_ref) * (torch.exp(-7900 * (1 / mean_T_cl - 1 / 298.15)))

        ASR = (2 * (h_ch / kele_bp + t_gdl / kele_gdl_eff + t_mpl / kele_mpl_eff)
               + 0.5*t_ccl / kele_cl_eff + 0.5*t_acl / kele_cl_eff
               + 0.5*t_ccl / kion_mem_eff + t_mem / kion_mem + 0.5*t_acl / kion_mem_eff)

        ##
        I_lit_a1 = 2 * F * C_h2_xch / (t_acl / (2 * D_h2_cl) + t_mpl / D_h2_cl + t_gdl / D_h2_gdl)
        I_lit_min = I_ave * torch.ones_like(I_lit_a1) + 0.1
        I_lit_a = torch.clamp(I_lit_a1, min=I_lit_min)

        I_lit_c1 = 4 * F * C_o2_xch / (t_ccl / (2 * D_o2_cl) + t_mpl / D_o2_cl + t_gdl / D_o2_gdl)
        I_lit_c = torch.clamp(I_lit_c1, min=I_lit_min)
        # print("I_lit_c1,I_lit_c", I_lit_c1,I_lit_c)
        P_o2 = mean_C_o2_cl / (mean_C_o2_cl + 0.79 / 0.21 * C_o2_in + RH_c * P_sat / R / mean_T_cl)
        P_h2 = mean_C_h2_cl / (mean_C_h2_cl + RH_a * P_sat / R / mean_T_cl)
        # X_vp = (RH_c * P_sat / R / mean_T_cl) / (mean_C_o2_cl + 0.79 / 0.21 * C_o2_in + RH_c * P_sat / R / mean_T_cl)
        # print("X_vp,P_h2",X_vp,P_h2)
        V_re = 1.229 - 0.846 * 0.001 * (mean_T_cl - 298.0) + (R * mean_T_cl / 2 / F) * (
                torch.log(P_h2) + 0.5 * torch.log(P_o2))

        # act loss
        V_act_a = -(R * mean_T_cl / (2 * Alp_a * F)) * torch.log(I_local / (t_acl * ja))
        V_act_c = -(R * mean_T_cl / (4 * Alp_c * F)) * torch.log(I_local / (t_ccl * jc))
        # conc loss
        V_conc_a = (aaa * R * mean_T_cl / (2 * Alp_a * F)) * torch.log(1 - I_local / I_lit_a)
        V_conc_c = (ccc * R * mean_T_cl / (4 * Alp_c * F)) * torch.log(1 - I_local / I_lit_c)
        # ohm loss
        V_ohm = -ASR * I_local

        V_local = V_re + V_act_a + V_act_c + V_conc_a + V_conc_c + V_ohm

        ###################  Region-mask
        tolerance = 1e-5
        t_norm = t * tt/t_max
        t_max_mask = (torch.abs(t_norm - 1.0) < tolerance).squeeze() # 转为一维布尔值

        V_at_tmax = V_local[t_max_mask]

        if len(V_at_tmax) > 0:
            V_mean_t_max = torch.mean(V_at_tmax).item()
        else:
            V_mean_t_max = 0.8
        ################

        V_view = V_local.view(n_t, n_inner, 1)

        loss_IV = torch.mean(torch.var(V_view, dim=1))


        ################  Region-mask for Coefficient   ################
        ## Source
        S_o2_cl = -I_local_detach / (4 * F * t_ccl)
        S_h2_cl = -I_local_detach / (2 * F * t_acl)
        S_lq_cl = I_local_detach / (2 * F * t_ccl)  # I_local_detach / (2 * F * t_ccl)

        V_act_c_view = V_act_c.view(n_t, n_inner, 1)
        V_act_c_mean_pred = torch.mean(V_act_c_view, dim=1, keepdim=True)  # [N_t, 1, 1]
        mean_V_act_c = V_act_c_mean_pred.repeat(1, n_inner, 1).view(-1, 1)
        q_sur = (mean_T_cl - T_e) * h_sur / (h_bp - h_ch)   ## - q_sur
        S_T_cl = I_local_detach * I_local_detach / kele_cl_eff + I_local_detach * (-mean_V_act_c) / t_ccl  ## 简化处理，认为热源存在于催化层


        mask_ccl, mask_cgdl, mask_cch = get_mask(x, xc1, xc2)
        mask_acl, mask_agdl, mask_ach = get_mask(x, xa1, xa2)

        por_c = por_cl * mask_ccl + por_gdl * mask_cgdl + 1.0 * mask_cch
        s_lqc = mean_s_cl * mask_ccl + mean_s_gdl * mask_cgdl + s_ch * mask_cch
        D_o2 = 2* (D_o2_cl * mask_ccl + D_o2_gdl * mask_cgdl + D_o2_ch * mask_cch)   ## 扩散系数
        uc_y = u_c_in * mask_cch          ## 考虑速度变化时， 将u_c_in 替换为 动量方程求解的 u, 另外压缩性简化为梯度0.1186
        k_T =  k_T_cl * mask_ccl + k_T_gdl * mask_cgdl + k_T_bp * mask_cch
        Cp = Cp_cl * mask_ccl + Cp_gdl * mask_cgdl + Cp_ch * mask_cch
        S_o2 = S_o2_cl * mask_ccl
        S_T = S_T_cl * mask_ccl
        S_lq = S_lq_cl * water_switch

        por_a = por_cl * mask_acl + por_gdl * mask_agdl + 1.0 * mask_ach
        D_h2 = (D_h2_cl * mask_acl + D_h2_gdl * mask_agdl + D_h2_ch * mask_ach)  ## 扩散系数
        ua_y = u_a_in * mask_ach
        S_h2 = S_h2_cl * mask_acl


        # ------------------------------------------------------------------------#
        ####################   H2 transport    ####################
        ## Equations normalization
        P1a = ua_y * tt / l_ch
        P2a = D_h2 * tt / (xa_total * xa_total)
        P3a = S_h2 * tt
        # P4 = D_o2 * t_max / (l_ch * l_ch)
        Scale_Factor_h2 = 15000.0 / (2 * F * t_acl) * tt   # P3 = -Scale_Factor * mask_ccl
        ## pde
        pde_C_h2 = (por_a * (1 - s_lqc) / Scale_Factor_h2 * d(C_h2, t) + P1a / Scale_Factor_h2 * d(C_h2,y)
                   - P2a / Scale_Factor_h2 * d(d(C_h2, x), x)- P3a / Scale_Factor_h2)
        loss_pde_C_h2 = torch.mean(pde_C_h2 ** 2)
        ## bc，ic
        temp_bc = torch.zeros_like(C_h2_Dbc_cch)
        loss_Dbc_C_h2_cch = torch.nn.functional.mse_loss(C_h2_Dbc_cch / AA, C_h2_in / AA - temp_bc)  # 修正 target 写法
        Lbc_C_h2 = d(C_h2_Lbc, x_L) / AA
        loss_Lbc_C_h2 = torch.nn.functional.mse_loss(Lbc_C_h2, temp_bc)
        Ubc_C_h2 = d(C_h2_Ubc, y_U) / AA
        loss_Ubc_C_h2 = torch.nn.functional.mse_loss(Ubc_C_h2, temp_bc)   ## + 2.0 * loss_Ubc_C_h2

        weight_inlet_a = 20.0 * (1.0 + u_a_in * tt / l_ch / 10.0)
        loss_C_h2 = 10.0 * loss_pde_C_h2 + weight_inlet_a * loss_Dbc_C_h2_cch + 2.0 * loss_Lbc_C_h2
        # loss_C_h2 = 10.0 * loss_pde_C_h2 + 20.0 * loss_Dbc_C_h2_cch + 2.0 * loss_Lbc_C_h2

        # ------------------------------------------------------------------------#
        ####################   O2 transport   ####################
        ##
        P1c = uc_y * tt / l_ch
        P2c = D_o2 * tt / (xc_total * xc_total)
        P3c = S_o2 * tt
        # P4 = D_o2 * t_max / (l_ch * l_ch)
        Scale_Factor_o2 = 15000.0 / (4 * F * t_ccl) * tt   #  P3 = -Scale_Factor * mask_ccl
        ## pde
        pde_C_o2 = (por_c*(1 - s_lqc)/Scale_Factor_o2 * d(C_o2, t) + P1c/Scale_Factor_o2 * d(C_o2, y)
                    - P2c/Scale_Factor_o2 * d(d(C_o2, x), x) - P3c/Scale_Factor_o2)
        loss_pde_C_o2 = torch.mean(pde_C_o2 ** 2)

        ## bc，ic
        temp_bc = torch.zeros_like(C_o2_Dbc_cch)
        loss_Dbc_C_o2_cch = torch.nn.functional.mse_loss(C_o2_Dbc_cch / CC, C_o2_in/CC - temp_bc)  # 修正 target 写法
        Lbc_C_o2 = d(C_o2_Lbc,x_L) / CC
        loss_Lbc_C_o2 = torch.nn.functional.mse_loss(Lbc_C_o2, temp_bc)
        Ubc_C_o2 = d(C_o2_Ubc,y_U) / CC
        loss_Ubc_C_o2 = torch.nn.functional.mse_loss(Ubc_C_o2, temp_bc)


        # raw_pde_C_o2 = (por_c * (1 - s_lqc) * d(C_o2, t)
        #                 + P1c * d(C_o2, y)
        #                 - P2c * d(d(C_o2, x), x)
        #                 - P3c)
        # pde_C_o2_normalized = raw_pde_C_o2 / (u_c_in * tt / l_ch * CC)
        # loss_pde_C_o2 = torch.mean(pde_C_o2_normalized ** 2)
        # weight_inlet = 20.0 * (1.0 + u_c_in * tt / l_ch / 10.0)
        # loss_C_o2 = 10.0 * loss_pde_C_o2 + weight_inlet * loss_Dbc_C_o2_cch + 2.0 * loss_Lbc_C_o2

        ##  + 2.0 * loss_Ubc_C_o2
        loss_C_o2 = 10.0 * loss_pde_C_o2 + 20.0 * loss_Dbc_C_o2_cch + 2.0 * loss_Lbc_C_o2



        # ------------------------------------------------------------------------#
        ####################   Water transport   ####################
        ## 方
        P1_lq = u_c_in * tt / l_ch * 0.0
        P2_lq_cl = K0_ccl_eff * tt / (vis_lq) / (t_ccl ** 2)  # K0_eff
        P2_lq_gdl = K0_gdl_eff * tt / (vis_lq) / ((t_gdl + t_mpl) ** 2)
        P3_lq = S_lq * M_H2O * tt / (den_water)
        ## 定义缩
        Scale_cl = PP   # P2_ref_cl * Pc_ref_cl / 10  PP
        Scale_gdl = PP/10  # P2_ref_gdl * Pc_ref_gdl / 1000  PP/10  S_lq_cl* M_H2O/10
        ## pde
        P_c_cl = ten_water * math.cos(angle_cl) * (por_cl / K0_cl) ** 0.5 * (1.42 * s_lq_cl - 2.12 * s_lq_cl ** 2 + 1.26 * s_lq_cl ** 3)
        P_c_gdl = ten_water * math.cos(angle_gdl) * (por_gdl / K0_gdl) ** 0.5 * (1.42 * s_lq_gdl - 2.12 * s_lq_gdl ** 2 + 1.26 * s_lq_gdl ** 3)

        P_lq_cl = P0 - P_c_cl
        P_lq_gdl = P0 - P_c_gdl

        term_diff_cl = d(P2_lq_cl * d(P_lq_cl, x), x)
        term_diff_gdl = d(P2_lq_gdl * d(P_lq_gdl, x), x)
        res_pde_s_cl = (por_cl * d(s_lq_cl, t) - term_diff_cl - P3_lq) / Scale_cl  # + P1_lq * d(s_lq_cl, y)
        res_pde_s_gdl = (por_gdl * d(s_lq_gdl, t) - term_diff_gdl) / Scale_gdl  # + P1_lq * d(s_lq_gdl, y)
        ## bc，ic
        res_Lbc_cl = d(s_lq_Lbc_cl, x_L) / SS1  # Neumann 0
        res_Rbc_gdl = (s_lq_Rbc_gdl - 0.01) / SS2 #
        res_Ubc_cl = d(s_lq_Ubc_cl, y_U) / SS1  #
        res_Ubc_gdl = d(s_lq_Ubc_gdl, y_U) / SS2  #


        ten_water_bc = -0.0001676 * T_cell + 0.1218
        P_c_cl_Rbc = ten_water_bc * math.cos(angle_cl) * (por_cl / K0_cl) ** 0.5 * (
                    1.42 * s_lq_Rbc_cl - 2.12 * s_lq_Rbc_cl ** 2 + 1.26 * s_lq_Rbc_cl ** 3)
        P_c_gdl_Lbc = ten_water_bc * math.cos(angle_gdl) * (por_gdl / K0_gdl) ** 0.5 * (
                    1.42 * s_lq_Lbc_gdl - 2.12 * s_lq_Lbc_gdl ** 2 + 1.26 * s_lq_Lbc_gdl ** 3)
        res_P_cont = (P_c_cl_Rbc - P_c_gdl_Lbc) / 100  # / PP


        K0_Rbc_cl = K0_cl * (s_lq_Rbc_cl ** 3 )
        dP_dX_cl = d(P_c_cl_Rbc, x_R)  # 对归一化坐标求导
        Flux_cl = - (K0_Rbc_cl / vis_lq) * (-dP_dX_cl / t_ccl)  # 除以 L_cl

        K0_Lbc_gdl = K0_gdl * (s_lq_Lbc_gdl ** 3 )
        dP_dX_gdl = d(P_c_gdl_Lbc, x_L)  # 对归一化坐标求导
        Flux_gdl = - (K0_Lbc_gdl / vis_lq) * (-dP_dX_gdl / (t_gdl + t_mpl))  # 除以 L_gdl

        res_Flux_cont = (Flux_cl - Flux_gdl) * 1e2   # * 1e2

        # NTK
        # 1. CL Loss      'Lbc': res_Lbc_cl,
        bc_dict_cl = {
             'Ubc': res_Ubc_cl,
            'P_cont': res_P_cont, 'Flux_cont': res_Flux_cont
        }
        ws_cl = ntk_handler_cl.update_and_get_weights(res_pde_s_cl, bc_dict_cl, PINN_s_cl)

        loss_s_cl = 2.0 * torch.mean(res_pde_s_cl ** 2) + \
                    100.0 * torch.mean(res_Lbc_cl ** 2) + \
                    ws_cl['P_cont'] * torch.mean(res_P_cont ** 2) + \
                    ws_cl['Ubc'] * torch.mean(res_Ubc_cl ** 2) + \
                    ws_cl['Flux_cont'] * torch.mean(res_Flux_cont ** 2)


            # 2. GDL Loss
        bc_dict_gdl = {
            'Rbc': res_Rbc_gdl, 'Ubc': res_Ubc_gdl,
            'P_cont': res_P_cont, 'Flux_cont': res_Flux_cont
        }
        ws_gdl = ntk_handler_gdl.update_and_get_weights(res_pde_s_gdl, bc_dict_gdl, PINN_s_gdl)

        loss_s_gdl = 2.0 * torch.mean(res_pde_s_gdl ** 2) + \
                     ws_gdl['Ubc'] * torch.mean(res_Ubc_gdl ** 2) + \
                     ws_gdl['Rbc'] * torch.mean(res_Rbc_gdl ** 2) + \
                     ws_gdl['P_cont'] * torch.mean(res_P_cont ** 2) + \
                     ws_gdl['Flux_cont'] * torch.mean(res_Flux_cont ** 2)


        loss_Flux_cont = torch.mean(res_Flux_cont ** 2)
        loss_P_cont = torch.mean(res_P_cont ** 2)
        loss_Rbc_gdl = torch.mean(res_Rbc_gdl ** 2)
        loss_Lbc_cl = torch.mean(res_Lbc_cl ** 2)

        loss_s_lq = loss_s_cl + loss_s_gdl


        # ------------------------------------------------------------------------#
        ####################   T transfer   ####################
        ##
        P1_T = k_T * tt / (den_ch * Cp) / (xc_total * xc_total)   ##
        P2_T = S_T * tt / (den_ch * Cp)
        S_T_cl11 = 10000 * 10000 / kele_cl_eff + 10000 * (-mean_V_act_c) / t_ccl
        Scale_Factor_T = S_T_cl11 * tt / (den_ch * Cp_gdl)     ##  S_T_cl * t_max / (den_ch * Cp)
        ## pde
        pde_T = (d(T, t)  - P1_T * d(d(T, x), x) - P2_T)/ Scale_Factor_T
        ## bc，ic
        res_Rbc_T = (T_Rbc_cch - T_cell) / TT_sacle
        res_Lbc_T = (d(T_Lbc, x_L) - 0.0) / TT_sacle
        res_Ubc_T = (d(T_Ubc, y_U) - 0.0) / TT_sacle

        loss_pde_T = torch.mean(pde_T ** 2)
        loss_Rbc_T = torch.mean(res_Rbc_T ** 2)
        loss_Lbc_T = torch.mean(res_Lbc_T ** 2)
        # loss_Ubc_T = torch.mean(res_Ubc_T ** 2)
        ## + 1.0*loss_Ubc_T
        loss_T = 10.0 * loss_pde_T + 100.0 * loss_Rbc_T + 1.0 * loss_Lbc_T


        ##########################################
        ############ Backpropagation
        cycle_epoch = 20
        total_cycle = cycle_epoch * 5  # 完整交替周期
        # loss_total
        if epoch % total_cycle < cycle_epoch:
            loss_total = loss_IV
        else:
            loss_total = loss_C_h2 + loss_C_o2 + loss_s_lq + loss_T


        loss_total.backward()
        optimizer1.step()
        # optimizer2.step()

        current_epoch_loss = 0.002* (loss_C_h2 + loss_C_o2 + loss_s_lq + loss_T/100 + loss_IV)


        if epoch % 200 == 0:
            print(f"Simulation progress：{epoch / epochs * 100:.1f}%")
            print("loss_C_h2, loss_C_o2, loss_IV",loss_C_h2, loss_C_o2, loss_IV)
            print("loss_s_lq, loss_T, current_epoch_loss", loss_s_lq, loss_T, current_epoch_loss)


            # loss_list.append(loss_total.item())
            loss_list.append(current_epoch_loss.item())
            epoch_list.append(epoch)

            ## Save model parameters
            if epoch % 200 == 0:
                torch.save({
                    'epoch': epoch,
                    'PINN_s_gdl_state_dict': PINN_s_gdl.state_dict(),
                    'PINN_s_cl_state_dict': PINN_s_cl.state_dict(),
                    'PINN_T_state_dict': PINN_T.state_dict(),
                    'PINN_o2_state_dict': PINN_o2.state_dict(),
                    'PINN_h2_state_dict': PINN_h2.state_dict(),
                    'PINN_I_state_dict': PINN_I.state_dict(),
                    # 'PINN_V_state_dict': PINN_V.state_dict(),

                }, f'component_{i}.pt')
                # if loss_total < min_loss1:
                #     min_loss1 = loss_total.item()


    # if epoch > 100 and current_epoch_loss <= 0.001:
    if epoch > 3000 and current_epoch_loss <= 0.001 and 0.5129 <= V_mean_t_max <= 0.5338:
        print(f"Training terminated early! Epoch: {epoch}, current loss: {current_epoch_loss:.6f} <= 0.001")
        break



# --------------------------
###################################################
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Runtime: {elapsed_time:.4f} s")
###################################################


# ==========================================
# Visualization
# ==========================================
print("epoch_list:", epoch_list)
print("loss_list:", loss_list)
print("len(epoch_list):", len(epoch_list))
print("len(loss_list):", len(loss_list))


# plot
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, loss_list, 'r-', linewidth=2, label='Training Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Loss vs Epoch', fontsize=16)
plt.yscale('log')  #
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# save fig
# plt.savefig('loss_vs_epoch.png', dpi=300)

# show Fig
plt.show()