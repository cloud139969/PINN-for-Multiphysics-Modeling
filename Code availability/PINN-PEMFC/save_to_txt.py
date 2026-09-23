import torch
import os
from CPINN_NET import d, Net_g1, Net_g2, Net_g3, Net_I
import numpy as np
import shutil
from constants import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_step = 0.1  #
t_steps = torch.arange(0, t_max + time_step, time_step).to(device)

########################### Initialization  #########################
C_o2_initial = 0.21 * Pc_in / R / T_e
C_h2_initial = Pa_in / R / T_e
s_initial = 0.0001
T_initial = T_e ## 273.15 + 70  T_e


P_cch_in = Pc_in
P_ach_in = Pa_in
C_h2_in = (P_ach_in - RH_a * P_sat) / (R * T_s)
C_o2_in = 0.21 * (P_cch_in - RH_c * P_sat) / (R * T_s)
u_a_in = ST_a * I_ave * A_cell / (2 * F * C_h2_in * A_ach)
u_c_in = ST_c * I_ave * A_cell / (4 * F * C_o2_in * A_cch)

uu = u_c_in
tt = 10.0
PP = 10.0
CC = C_o2_in
CC_sacle = 2.1
AA = C_h2_in
AA_sacle = 20.0
SS1 = 0.04 *3.0
SS2 = 0.01 *3.0
# TT = 342.0
TT_sacle = 3.0



xc_total = t_ccl + t_mpl + t_gdl + h_ch
xc1 = t_ccl / xc_total
xc2 = (t_ccl+ t_mpl+ t_gdl) / xc_total

xa_total = t_acl + t_mpl + t_gdl + h_ch
xa1 = t_acl / xa_total
xa2 = (t_acl+ t_mpl+ t_gdl) / xa_total


#
x_num = 80
y_num = 20
# x_coords = torch.linspace(0, 1, x_num).to(device)
# y_coords = torch.linspace(0, 1, y_num).to(device)

num_0_xc1 = 5   # 0→xc1：
num_xc1_xc2 = 25  # xc1→xc2：
num_xc2_1 = 50  # xc2→1：

x_0_xc1 = torch.linspace(0, xc1, num_0_xc1, device=device)
x_xc1_xc2 = torch.linspace(xc1, xc2, num_xc1_xc2 + 1, device=device)[1:]
x_xc2_1 = torch.linspace(xc2, 1, num_xc2_1 + 1, device=device)[1:]

x_coords = torch.cat([x_0_xc1, x_xc1_xc2, x_xc2_1])
y_coords = torch.linspace(0, 1, y_num).to(device)


n_t = len(t_steps)
n_inner = x_num * y_num






is_even = (num_cells % 2 == 0)
previous_T = T_s
cell_inlet_T = [0.0 for _ in range(num_cells)]
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





# ====================== Create a table of contents ======================
def create_directory_structure(base_dir, num_cells):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)


    os.makedirs(base_dir, exist_ok=True)
    for i in range(num_cells):
        component_dir = os.path.join(base_dir, f"component{i}")
        os.makedirs(component_dir, exist_ok=True)


def save_2d_data_to_txt(file_path, data_dict, x_coords, y_coords):

    with open(file_path, 'w', encoding='utf-8') as f:
        # Header:coordinates
        f.write(f"=== coordinates ===\n")
        f.write(f"x-number：{x_num}，坐标：{[round(x.item(), 4) for x in x_coords]}\n")
        f.write(f"y-number：{y_num}，坐标：{[round(y.item(), 4) for y in y_coords]}\n")
        f.write(f"=====================\n\n")

        for t in sorted(data_dict.keys()):
            f.write(f"【time：{t:.1f}s】\n")
            data_2d = data_dict[t]
            for x_idx in range(x_num):
                row_data = [f"{data_2d[x_idx, y_idx]:.6f}" for y_idx in range(y_num)]
                f.write("\t".join(row_data) + "\n")
            f.write("\n")
    print(f"Save completed：{file_path}")

def save_voltage_avg_to_txt(file_path, voltage_dict):

    with open(file_path, 'w', encoding='utf-8') as f:
        # 表头：列名+说明
        f.write(f"=== Average voltage of fuel cells (V) ===\n")
        f.write(f"time(s)\tV_re\tV_act_a\tV_act_c\tV_conc_a\tV_conc_c\tV_ohm\tV_local\n")
        f.write(f"=====================================\n")

        for t in sorted(voltage_dict.keys()):
            v = voltage_dict[t]
            f.write(f"{t:.1f}\t{v['V_re']:.6f}\t{v['V_act_a']:.6f}\t{v['V_act_c']:.6f}\t{v['V_conc_a']:.6f}\t{v['V_conc_c']:.6f}\t{v['V_ohm']:.6f}\t{v['V_local']:.6f}\n")
    print(f"Voltage save completed：{file_path}")

# ============================================
def create_component():
    PINN_h2 = Net_g2().to(device)
    PINN_o2 = Net_g2().to(device)
    PINN_s_gdl = Net_g1().to(device)
    PINN_s_cl = Net_g1().to(device)
    PINN_T = Net_g3().to(device)
    PINN_I = Net_I().to(device)
    return PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I

# ============================================
if __name__ == "__main__":

    components = [create_component() for _ in range(num_cells)]
    for i in range(num_cells):
        trained_model_path = f"component_{i}.pt"
        try:
            checkpoint = torch.load(trained_model_path, map_location=device)
            PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I = components[i]

            PINN_o2.load_state_dict(checkpoint['PINN_o2_state_dict'])
            PINN_h2.load_state_dict(checkpoint['PINN_h2_state_dict'])
            PINN_I.load_state_dict(checkpoint['PINN_I_state_dict'])
            PINN_T.load_state_dict(checkpoint['PINN_T_state_dict'])
            PINN_s_gdl.load_state_dict(checkpoint['PINN_s_gdl_state_dict'])
            PINN_s_cl.load_state_dict(checkpoint['PINN_s_cl_state_dict'])
            print(f" Successfully loaded parameters for component {i}: {trained_model_path}")
        except FileNotFoundError:
            print(
                f" Parameter file for component {i} does not exist: {trained_model_path}, using initialized parameters")
        except KeyError as e:
            print(
                f" Required key missing in parameter file for component {i}: {e}, loading failed, using initialized parameters")



    base_dir = "components"
    create_directory_structure(base_dir, num_cells)


    X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')  # (x_num, y_num)
    x = X.reshape(-1, 1)  # (n_inner, 1)
    y = Y.reshape(-1, 1)  # (n_inner, 1)


    for comp_idx, (PINN_h2, PINN_o2, PINN_s_gdl, PINN_s_cl, PINN_T, PINN_I) in enumerate(components):

        PINN_h2.eval()
        PINN_o2.eval()
        PINN_s_gdl.eval()
        PINN_s_cl.eval()
        PINN_T.eval()
        PINN_I.eval()


        data_C_h2, data_C_o2 = {}, {}
        data_s_lq_cl, data_s_lq_gdl = {}, {}
        data_T, data_I = {}, {}
        voltage_avg_dict = {}
        T_cell = cell_inlet_T[comp_idx]


        with torch.no_grad():
            for t in t_steps:
                t_val = t.item()
                t_flat = torch.ones_like(x).to(device) * t/tt  # (n_inner, 1)

                # --------------------------------------------
                input_xyz = torch.cat([x, y, t_flat], dim=1)  # (n_inner, 3)
                input_yt = torch.cat([y, t_flat], dim=1)          # (n_inner, 2)


                C_h2_flat = C_h2_initial + (1.0 - torch.exp(-tt *t_flat))*AA_sacle* PINN_h2(input_xyz)
                C_o2_flat = C_o2_initial+ (1.0 - torch.exp(-tt *t_flat))*CC_sacle* PINN_o2(input_xyz)
                s_lq_cl_flat = s_initial + (1.0 - torch.exp(-tt *t_flat))*SS1* PINN_s_cl(input_xyz)
                s_lq_gdl_flat = s_initial + (1.0 - torch.exp(-tt *t_flat))*SS2* PINN_s_gdl(input_xyz)
                T_flat = T_initial + (1.0 - torch.exp(-tt *t_flat))*TT_sacle * PINN_T(input_xyz)


                t_step = 5.0 / tt
                k = 200.0
                I_ave = I_base + I_delta * torch.sigmoid(k * (t_flat - t_step))
                I_flat = I_ave * PINN_I(input_yt)


                t1 = t/tt
                term_base = I_base * t1
                sp_t = torch.nn.functional.softplus(k * (t1 - t_step))
                sp_0 = torch.nn.functional.softplus(torch.tensor(-k * t_step, device=t1.device))
                term_step = (I_delta / k) * (sp_t - sp_0)

                integral_I_ave = term_base + term_step
                integral_physical = integral_I_ave * tt

                delta_lam_generated = integral_physical / (2 * F * (t_ccl + t_mem)) * (EW / (den_mem * w_ion))
                lam = torch.clamp(lam_initial + delta_lam_generated, max=lam_sat)


                # --------------------------------------------
                x_ccl = x * xc1
                x_acl = x * xa1
                x_cch_centre = (1 + xc2) / 2 * torch.ones_like(y)
                x_ach_centre = (1 + xa2) / 2 * torch.ones_like(y)
                x_ccl_centre = xc1 / 2 * torch.ones_like(y)
                x_acl_centre = xa1 / 2 * torch.ones_like(y)


                C_h2_cl = C_h2_initial+ (1.0 - torch.exp(-tt *t_flat))*AA_sacle* PINN_h2(torch.cat([x_acl, y, t_flat], dim=1))
                C_o2_cl = C_o2_initial+ (1.0 - torch.exp(-tt *t_flat))*CC_sacle* PINN_o2(torch.cat([x_ccl, y, t_flat], dim=1))
                T_ccl = T_initial + (1.0 - torch.exp(-tt *t_flat))*TT_sacle * PINN_T(torch.cat([x_ccl, y, t_flat], dim=1))

                C_o2_xch = C_o2_initial+ (1.0 - torch.exp(-tt *t_flat))*CC_sacle* PINN_o2(torch.cat([x_cch_centre, y, t_flat], dim=1))
                C_h2_xch = C_h2_initial+ (1.0 - torch.exp(-tt *t_flat))*AA_sacle* PINN_h2(torch.cat([x_ach_centre, y, t_flat], dim=1))
                C_o2_xcl = C_o2_initial+ (1.0 - torch.exp(-tt *t_flat))*CC_sacle* PINN_o2(torch.cat([x_ccl_centre, y, t_flat], dim=1))
                C_h2_xcl = C_h2_initial+ (1.0 - torch.exp(-tt *t_flat))*AA_sacle* PINN_h2(torch.cat([x_acl_centre, y, t_flat], dim=1))

                # --------------------------------------------
                T_ccl_reshaped = T_ccl.view(1, n_inner, 1)
                mean_T_cl = torch.mean(T_ccl_reshaped, dim=1, keepdim=True).repeat(1, n_inner, 1).view(-1, 1)
                C_h2_cl_reshaped = C_h2_cl.view(1, n_inner, 1)
                mean_C_h2_cl = torch.mean(C_h2_cl_reshaped, dim=1, keepdim=True).repeat(1, n_inner, 1).view(-1, 1)
                C_o2_cl_reshaped = C_o2_cl.view(1, n_inner, 1)
                mean_C_o2_cl = torch.mean(C_o2_cl_reshaped, dim=1, keepdim=True).repeat(1, n_inner, 1).view(-1, 1)
                s_ch = 1e-6
                s_lq_gdl_reshaped = s_lq_gdl_flat.view(1, n_inner, 1)
                mean_s_gdl = torch.mean(s_lq_gdl_reshaped, dim=1, keepdim=True).repeat(1, n_inner, 1).view(-1, 1)
                s_lq_cl_reshaped = s_lq_cl_flat.view(1, n_inner, 1)
                mean_s_cl = torch.mean(s_lq_cl_reshaped, dim=1, keepdim=True).repeat(1, n_inner, 1).view(-1, 1)

                # ---------------------- ----------------------
                D_h2_cl = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * por_cl ** 1.5 * (1 - mean_s_cl) ** 1.5
                D_h2_gdl = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * por_gdl ** 1.5 * (1 - mean_s_gdl) ** 1.5
                D_h2_ch = 1.055e-4 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_ach_in) * (1 - s_ch) ** 1.5
                D_o2_cl = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * por_cl ** 1.5 * (1 - mean_s_cl) ** 1.5
                D_o2_gdl = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * por_gdl ** 1.5 * (1 - mean_s_gdl) ** 1.5
                D_o2_ch = 2.652e-5 * (mean_T_cl / 333.15) ** 1.5 * (P0 / P_cch_in) * (1 - s_ch) ** 1.5
                D_nf_eff = (w_ion ** 1.5) * D_nf
                kele_cl_eff = kele_cl * (1 - por_cl) ** 1.5
                kele_mpl_eff = kele_mpl * (1 - por_mpl) ** 1.5
                kele_gdl_eff = kele_gdl * (1 - por_gdl) ** 1.5
                kion_mem = (0.5319 * lam - 0.326) * torch.exp(1268 * (1 / 303.15 - 1 / mean_T_cl))
                kion_mem_eff = kion_mem * w_ion ** 1.5

                # ----------------------  ----------------------

                I_view = I_flat.view(1, n_inner, 1)
                I_mean_pred = torch.mean(I_view, dim=1, keepdim=True)
                I_mean_broad = I_mean_pred.repeat(1, n_inner, 1).view(-1, 1)
                I_local = I_flat * (I_ave / I_mean_broad)

                ja = (1 - mean_s_cl) * ia_ref * (C_h2_xcl / C_h2_ref) ** 0.5 * (
                    torch.exp(-1400 * (1 / mean_T_cl - 1 / 298.15)))
                jc = (1 - mean_s_cl) * ic_ref * (C_o2_xcl / C_o2_ref) * (
                    torch.exp(-7900 * (1 / mean_T_cl - 1 / 298.15)))

                ASR = (2 * (h_ch / kele_bp + t_gdl / kele_gdl_eff + t_mpl / kele_mpl_eff)
                       + 0.5 * t_ccl / kele_cl_eff + 0.5 * t_acl / kele_cl_eff
                       + 0.5 * t_ccl / kion_mem_eff + t_mem / kion_mem + 0.5 * t_acl / kion_mem_eff)
                ##
                I_lit_a1 = 2 * F * C_h2_xch / (t_acl / (2 * D_h2_cl) + t_mpl / D_h2_cl + t_gdl / D_h2_gdl)
                I_lit_min = I_ave * torch.ones_like(I_lit_a1) + 0.1
                I_lit_a = torch.clamp(I_lit_a1, min=I_lit_min)

                I_lit_c1 = 4 * F * C_o2_xch / (t_ccl / (2 * D_o2_cl) + t_mpl / D_o2_cl + t_gdl / D_o2_gdl)
                I_lit_c = torch.clamp(I_lit_c1, min=I_lit_min)
                #
                P_o2 = mean_C_o2_cl / (mean_C_o2_cl + 0.79/0.21 * C_o2_flat + RH_c * P_sat / R / mean_T_cl)
                P_h2 = mean_C_h2_cl / (mean_C_h2_cl + RH_a * P_sat / R / mean_T_cl)
                #
                V_re = 1.229 - 0.846e-3 * (mean_T_cl - 298.0) + (R * mean_T_cl / (2 * F)) * (torch.log(P_h2) + 0.5 * torch.log(P_o2))
                V_act_a = -(R * mean_T_cl / (2 * Alp_a * F)) * torch.log(I_local / (t_acl * ja))
                V_act_c = -(R * mean_T_cl / (4 * Alp_c * F)) * torch.log(I_local / (t_ccl * jc))
                V_conc_a = (aaa * R * mean_T_cl / (2 * Alp_a * F)) * torch.log(1 - I_local / I_lit_a)
                V_conc_c = (ccc * R * mean_T_cl / (4 * Alp_c * F)) * torch.log(1 - I_local / I_lit_c)
                V_ohm = -ASR * I_local
                V_local = V_re + V_act_a + V_act_c + V_conc_a + V_conc_c + V_ohm


                ###
                V_act_c_view = V_act_c.view(1, n_inner, 1)
                V_act_c_mean_pred = torch.mean(V_act_c_view, dim=1, keepdim=True)  # [N_t, 1, 1]
                mean_V_act_c = V_act_c_mean_pred.repeat(1, n_inner, 1).view(-1, 1)
                q_sur = (mean_T_cl - T_e) * h_sur / (h_bp - h_ch)  ## - q_sur
                S_T_cl = I_local * I_local / kele_cl_eff + I_local * (-mean_V_act_c) / t_ccl - q_sur

                data = T_flat.detach().cpu().numpy()
                np.savetxt("S_T_cl.txt", data, fmt="%.6f")  #

                data2 = input_xyz.detach().cpu().numpy()
                np.savetxt("input_xyz.txt", data2, fmt="%.6f")  #
                #
                C_h2_2d = C_h2_flat.reshape(x_num, y_num).cpu().numpy()
                C_o2_2d = C_o2_flat.reshape(x_num, y_num).cpu().numpy()
                s_lq_cl_2d = s_lq_cl_flat.reshape(x_num, y_num).cpu().numpy()
                s_lq_gdl_2d = s_lq_gdl_flat.reshape(x_num, y_num).cpu().numpy()
                T_2d = T_flat.reshape(x_num, y_num).cpu().numpy()
                I_2d = I_local.reshape(x_num, y_num).cpu().numpy()
                #
                data_C_h2[t_val] = C_h2_2d
                data_C_o2[t_val] = C_o2_2d
                data_s_lq_cl[t_val] = s_lq_cl_2d
                data_s_lq_gdl[t_val] = s_lq_gdl_2d
                data_T[t_val] = T_2d
                data_I[t_val] = I_2d

                # --------------------------------------------
                voltage_avg_dict[t_val] = {
                    "V_re": torch.mean(V_re).cpu().item(),
                    "V_act_a": torch.mean(V_act_a).cpu().item(),
                    "V_act_c": torch.mean(V_act_c).cpu().item(),
                    "V_conc_a": torch.mean(V_conc_a).cpu().item(),
                    "V_conc_c": torch.mean(V_conc_c).cpu().item(),
                    "V_ohm": torch.mean(V_ohm).cpu().item(),
                    "V_local": torch.mean(V_local).cpu().item()
                }

        # 6. Save all the data of the current component to the corresponding directory
        comp_dir = os.path.join(base_dir, f"component{comp_idx}")
        #
        save_2d_data_to_txt(os.path.join(comp_dir, "C_h2.txt"), data_C_h2, x_coords, y_coords)
        save_2d_data_to_txt(os.path.join(comp_dir, "C_o2.txt"), data_C_o2, x_coords, y_coords)
        save_2d_data_to_txt(os.path.join(comp_dir, "s_lq_cl.txt"), data_s_lq_cl, x_coords, y_coords)
        save_2d_data_to_txt(os.path.join(comp_dir, "s_lq_gdl.txt"), data_s_lq_gdl, x_coords, y_coords)
        save_2d_data_to_txt(os.path.join(comp_dir, "T.txt"), data_T, x_coords, y_coords)
        save_2d_data_to_txt(os.path.join(comp_dir, "I.txt"), data_I, x_coords, y_coords)
        #
        save_voltage_avg_to_txt(os.path.join(comp_dir, "voltage_average.txt"), voltage_avg_dict)

    print(f"\n===== All components have been processed!  {base_dir} directory =====")