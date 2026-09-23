import numpy as np
import matplotlib.pyplot as plt
from Constants import *
from CPINN_Net import d, Net_g1, Net_g2
import torch
import random
from Datas_get2 import inner_datas,t0_datas,Lbc_datas,Rbc_datas
import torch.nn.functional as tF
import time



# --- Set seeds ---
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


##################################################
start_time = time.time()   ##
##################################################


# --------------------------


#
epochs = 8000  ## 8000
t_max = 7000.0
n_t = int(round(t_max / 10.0))
n_inner = 100
n_bc = 20

tt = t_max  # 3600.0
I_app = 1.0*capacity   # [A]
I_app_den = I_app / A_cell  # [A/m^2]
j_BV_avg = -I_app / (A_cell * L * a_s)
J_BV_scale = 0.3

C_s0 = c_s0
C_s_scale = 40000.0
C_e0 = c_e0
C_e_scale = 200.0
P_s0 = 4.2 - 0.5 * (C_s0 / c_max) - 0.1 * np.tan(np.pi * ((C_s0 / c_max) - 0.5))
# print("P_s0",P_s0)
P_s_scale = 1.5
P_e0 = 0.0
P_e_scale = 0.005  # 0.007


#
PINN_C_s = Net_g1().to(device)
PINN_C_e = Net_g1().to(device)
PINN_P_s = Net_g1().to(device)
PINN_J_BV = Net_g1().to(device)

#
optimizer1 = torch.optim.Adam([
    {'params': PINN_C_e.parameters(), 'lr': 0.0001},
    {'params': PINN_C_s.parameters(), 'lr': 0.0001},
    {'params': PINN_P_s.parameters(), 'lr': 0.0001},
    {'params': PINN_J_BV.parameters(), 'lr': 0.0001},
])

# #
# MODEL_CKPT_MAP = {
#     "PINN_C_e": "Initialization_C_e.pt",
#     "PINN_C_s": "Initialization_C_s.pt",
#     "PINN_P_s": "Initialization_P_s.pt",
#     "PINN_J_BV": "Initialization_J_BV.pt"
# }
#
# #
# for name, path in MODEL_CKPT_MAP.items():
#     model = locals()[name]  # 自动获取变量，不用手动映射
#     try:
#         checkpoint = torch.load(path, map_location=device)
#         model.load_state_dict(checkpoint[f"{name}_state_dict"])
#         print(f" {name} 加载成功：{path}")
#     except Exception as e:
#         print(f" {name} 加载失败：{str(e)}")



##################################################
## collocation point
def get_resampled_data(device):
    x, t = inner_datas(n_t, n_inner)
    x_L, t_L = Lbc_datas(n_t)
    x_R, t_R = Rbc_datas(n_t)
    x_t0, t_t0 = t0_datas(n_bc)

    tensors_list = [
        x, t,
        x_L, t_L,
        x_R, t_R,
        x_t0, t_t0
    ]

    tensors_on_device = [tensor.to(device) for tensor in tensors_list]

    processed_tensors = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in tensors_on_device
    ]

    return processed_tensors


loss_history = []
current_data = get_resampled_data(device)
for epoch in range(epochs):
    # if epoch % 2 == 0:
    #     torch.cuda.empty_cache()
    current_epoch_loss = 1.0
    optimizer1.zero_grad()

    # === sampling ===
    if epoch > 0 and epoch % 50 == 0:  ##
        current_data = get_resampled_data(device)

    #
    (x, t,
     x_L, t_L,
     x_R, t_R,
     x_t0, t_t0) = current_data
    ##
    t = t * t_max / tt
    t_L = t_L * t_max / tt
    t_R = t_R * t_max / tt


    #
    fluctuation = J_BV_scale * tF.tanh(PINN_J_BV(torch.cat([x, t], dim=1)))  # shape:[n_t * n_inner, 1]
    shape_raw = 1.0 + fluctuation

    shape_view = shape_raw.view(n_t, n_inner, 1)
    shape_mean = torch.mean(shape_view, dim=1, keepdim=True)
    shape_mean_broad = shape_mean.repeat(1, n_inner, 1).view(-1, 1)

    j_BV = -abs(j_BV_avg) * (shape_raw / shape_mean_broad)
    j_BV_detach = j_BV.detach()

    #
    C_s = C_s0 + C_s_scale * t* PINN_C_s(torch.cat([x, t], dim=1))
    C_s_Lbc = C_s0 + C_s_scale * t_L* PINN_C_s(torch.cat([x_L, t_L], dim=1))
    C_s_Rbc = C_s0 + C_s_scale * t_R* PINN_C_s(torch.cat([x_R, t_R], dim=1))

    C_s_detach = C_s.detach()


    C_e = C_e0 + C_e_scale * t * torch.tanh(PINN_C_e(torch.cat([x, t], dim=1)))
    C_e_Lbc = C_e0 + C_e_scale * t_L * torch.tanh(PINN_C_e(torch.cat([x_L, t_L], dim=1)))
    C_e_Rbc = C_e0 + C_e_scale * t_R * torch.tanh(PINN_C_e(torch.cat([x_R, t_R], dim=1)))

    # P_s
    P_s = P_s0 - P_s_scale * t * tF.softplus(PINN_P_s(torch.cat([x, t], dim=1)))
    P_s_Lbc = P_s0 - P_s_scale * t_L * tF.softplus(PINN_P_s(torch.cat([x_L, t_L], dim=1)))
    P_s_Rbc = P_s0 - P_s_scale * t_R * tF.softplus(PINN_P_s(torch.cat([x_R, t_R], dim=1)))

    # P_e
    x_real = x * L
    P_e = (I_app_den / kappa_e) * (0.5 * (x_real ** 2) / L - x_real)



    # =========================================================
    #
    # =========================================================
    # delta_c = (R_s * j_BV) / (5.0 * F * D_s)
    delta_c = -(R_s * torch.abs(j_BV)) / (5.0 * F * D_s)
    C_s_surf = C_s_detach - delta_c
    C_s_surf = torch.clamp(C_s_surf, min=1.0, max=c_max - 1.0)

    theta = C_s_surf / c_max
    theta = torch.clamp(theta, min=0.01, max=0.99)
    U_eq = 4.2 - 0.5 * theta - 0.1 * torch.tan(np.pi * (theta - 0.5))

    C_e_safe = torch.clamp(C_e, min=0.1)
    i_0 = F * k_norm * torch.sqrt(C_e_safe * (c_max - C_s_surf) * C_s_surf)
    i_0_safe = torch.clamp(i_0, min=1e-6)

    eta_net = P_s - P_e - U_eq
    eta_phys = (R * T / (a_n * F)) * torch.asinh(j_BV / (2.0 * i_0_safe))



    loss_J_BV_match = torch.mean(((eta_net - eta_phys) / 0.001) ** 2)
    # loss_eta_sign = torch.mean(tF.relu(eta_net) ** 2) * 100000.0
    # loss_J_BV = 10.0 * loss_J_BV_match + loss_eta_sign
    loss_J_BV = 10.0 * loss_J_BV_match




    # =========================================================
    # PDE calculate
    # =========================================================
    # C_s
    dCs_dt = d(C_s, t) / tt
    PDE_res_C_s = dCs_dt - (-3.0 / (R_s * F) * j_BV_detach)
    loss_pde_C_s = torch.mean((PDE_res_C_s / 1.0) ** 2)

    loss_C_s = 50.0 * loss_pde_C_s

    # C_e
    dCe_dt = d(C_e, t) / tt
    dCe_dx = d(C_e, x) / L
    d2Ce_dx2 = d(dCe_dx, x) / L
    source_term = (1.0 - t_plus) * a_s / F * j_BV_detach
    PDE_res_C_e = eps_e * dCe_dt - (D_e * d2Ce_dx2 + source_term)
    loss_pde_C_e = torch.mean((PDE_res_C_e / (C_e_scale / tt)) ** 2)

    dCe_dx_L = d(C_e_Lbc, x_L) / L
    flux_in = (I_app / A_cell) * (1 - t_plus) / F
    loss_bc_L_C_e = torch.mean(((-D_e * dCe_dx_L - flux_in) / flux_in) ** 2)

    dCe_dx_R = d(C_e_Rbc, x_R) / L
    ref_grad_Ce = flux_in / D_e
    loss_bc_R_C_e = torch.mean((dCe_dx_R / ref_grad_Ce) ** 2)

    loss_C_e = 5.0 * loss_pde_C_e + 20.0 * loss_bc_L_C_e + 20.0 * loss_bc_R_C_e

    # P_s
    dPs_dx = d(P_s, x) / L
    d2Ps_dx2 = d(dPs_dx, x) / L
    PDE_res_Ps = sigma_s * d2Ps_dx2 - a_s * j_BV_detach
    loss_pde_P_s = torch.mean((PDE_res_Ps / (I_app_den / L)) ** 2)

    dPs_dx_L = d(P_s_Lbc, x_L) / L
    ref_grad_Ps = I_app_den / sigma_s
    loss_bc_L_P_s = torch.mean((dPs_dx_L / ref_grad_Ps) ** 2)

    dPs_dx_R = d(P_s_Rbc, x_R) / L
    loss_bc_R_P_s = torch.mean(((-sigma_s * dPs_dx_R - I_app_den) / I_app_den) ** 2)

    loss_P_s = 5.0 * loss_pde_P_s + 10.0 * loss_bc_L_P_s + 50.0 * loss_bc_R_P_s

    # =========================================================
    # 4. 汇总求导
    # =========================================================
    loss_total = loss_C_s + loss_C_e + loss_P_s + loss_J_BV

    loss_total.backward()
    optimizer1.step()

    loss_history.append(loss_total.item())


    if epoch % 200 == 0:
        print(f"Training progress：{epoch / epochs * 100:.1f}%")
        print("loss_C_s, loss_C_e, loss_P_s, loss_J_BV", loss_C_s, loss_C_e, loss_P_s, loss_J_BV)


    if epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            # 'PINN_C_s_state_dict': PINN_C_s.state_dict(),
            'PINN_C_e_state_dict': PINN_C_e.state_dict(),
            'PINN_P_s_state_dict': PINN_P_s.state_dict(),
            'PINN_J_BV_state_dict': PINN_J_BV.state_dict(),
        }, f'LB_model.pt')


###################################################
end_time = time.time()
elapsed_time = end_time - start_time         ##
print(f"Running duration: {elapsed_time:.4f} seconds")
###################################################





# ======================== plot loss  ========================
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Total Loss ($C_e$)', color='#1f77b4')
plt.yscale('log')  #
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ======================== ========================
#
t1 = torch.linspace(0, t_max, 1000).to(device)  # shape: (1000,)
t2 = (t1 / tt).unsqueeze(1)

#
#
x_R = torch.ones_like(t2)
inputs_R = torch.cat([x_R, t2], dim=1)  # shape: (1000, 2)

#
x_L = torch.zeros_like(t2)
inputs_L = torch.cat([x_L, t2], dim=1)  # shape: (1000, 2)

#
with torch.no_grad():
    #
    P_s_R = P_s0 - P_s_scale * t2 * tF.softplus(PINN_P_s(inputs_R))

    #
    V_out = P_s_R - 0.0

#
t_np = t1.cpu().numpy()
v_np = V_out.cpu().numpy().flatten()



target_times = np.arange(0, t_max + 1, 20)  # 0,10,20,...t_max

indices = [np.argmin(np.abs(t_np - t)) for t in target_times]
indices = sorted(list(set(indices)))

sampled_t = t_np[indices]
sampled_v = v_np[indices]
# save-txt
with open("PINN_voltage_PRA.txt", "w", encoding="utf-8") as f:
    f.write("Time(s)\tCell Voltage(V)\n")
    for t, v in zip(sampled_t, sampled_v):
        f.write(f"{t:.1f}\t{v:.6f}\n")



plt.figure(figsize=(8, 5))
plt.plot(t_np, v_np, 'b-', linewidth=2.5, label='PINN Predicted $V_{cell}$')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Cell Voltage (V)', fontsize=12)
plt.title('Discharge Curve: $V_{cell}$ vs Time', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ================================================
n_x = 10
n_t = int(t_max / 10) + 1

x_test = torch.linspace(0, L, n_x).to(device)
t_test = torch.linspace(0, t_max, n_t).to(device)

X, T = torch.meshgrid(x_test, t_test, indexing='ij')

x_flat = X.reshape(-1, 1) / L
t_flat = T.reshape(-1, 1) / tt

#
inputs = torch.cat([x_flat, t_flat], dim=1)

# ==========================================
with torch.no_grad():

    t_real = t_flat * tt

    # C_s_pred = C_s0 + C_s_scale * t_flat * tF.softplus(PINN_C_s(inputs))
    # shape_J_BV = tF.softplus(PINN_J_BV(inputs))
    # j_BV = j_BV_avg * shape_J_BV
    # C_s_pred = C_s0 - (3.0 / (R_s * F)) * j_BV * t_real

    C_s_pred = C_s0 + C_s_scale * t_flat * PINN_C_s(inputs)
    P_s_pred = P_s0 - P_s_scale * t_flat * tF.softplus(PINN_P_s(inputs))
    # C_e_pred = C_e0 + C_e_scale * t_flat * PINN_C_e(inputs)
    C_e_pred = C_e0 + C_e_scale * t_flat  * torch.tanh(PINN_C_e(inputs))

    x_phy = x_flat * L
    P_e_pred = (I_app_den / kappa_e) * (0.5 * (x_phy ** 2) / L - x_phy)


# ===================== [n_t, n_x]  =====================
C_s_pred_2d = C_s_pred.view(n_x, n_t).T  #
P_s_pred_2d = P_s_pred.view(n_x, n_t).T

# ===================== =====================
np.savetxt("C_s_pred.txt", C_s_pred_2d.cpu().numpy(), fmt="%.6f")
np.savetxt("P_s_pred.txt", P_s_pred_2d.cpu().numpy(), fmt="%.6f")


# =====================Reshape - numpy =====================
x_np = x_test.cpu().numpy()
t_np = t_test.cpu().numpy()

C_s_map = C_s_pred.cpu().reshape(n_x, n_t).numpy()
P_s_map = P_s_pred.cpu().reshape(n_x, n_t).numpy()
C_e_map = C_e_pred.cpu().reshape(n_x, n_t).numpy()
P_e_map = P_e_pred.cpu().reshape(n_x, n_t).numpy()


# ==========================================
plt.figure(figsize=(14, 10))

# FIG1：C_s
plt.subplot(2, 2, 1)
plt.contourf(t_np, x_np, C_s_map, 100, cmap='jet')
plt.colorbar(label='$C_s$ (mol/m³)')
plt.xlabel('Time t (s)')
plt.ylabel('Position x (m)')
plt.title('Spatio-temporal distribution of $C_s$')

# FIG2：P_s
plt.subplot(2, 2, 2)
plt.contourf(t_np, x_np, P_s_map, 100, cmap='jet')
plt.colorbar(label='$P_s$ (mol/m³)')
plt.xlabel('Time t (s)')
plt.ylabel('Position x (m)')
plt.title('Spatio-temporal distribution of $P_s$')

# FIG3：C_e
plt.subplot(2, 2, 3)
plt.contourf(t_np, x_np, C_e_map, 100, cmap='jet')
plt.colorbar(label='$C_e$ (mol/m³)')
plt.xlabel('Time t (s)')
plt.ylabel('Position x (m)')
plt.title('Spatio-temporal distribution of $C_e$')

# FIG4：P_e
plt.subplot(2, 2, 4)
plt.contourf(t_np, x_np, P_e_map, 100, cmap='jet')
plt.colorbar(label='$P_e$ (mol/m³)')
plt.xlabel('Time t (s)')
plt.ylabel('Position x (m)')
plt.title('Spatio-temporal distribution of $P_e$')

plt.tight_layout()
plt.show()