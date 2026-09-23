import math
## Simulation configuration
num_cells = 1  # Number of single cells
t_min = 0.0
t_max = 20.0   # Simulation runtime
## Current mode
    ## 1. Constant current
I_delta = 0.0
I_base = 3000.0
    ## 2. Square-wave current
# I_delta = 3000.0
# I_base = 10000.0



## Operating conditions
I_ave = I_base      # A/m^2 Operating current density of the cell
V_out_initial = 0.60        # V Cell output voltage
T_s = 273.15 + 80   # K Cell operating temperature
T_e = 273.15 + 25   # K Ambient temperature
T_cool = 273.15 + 28   # K Cooling water temperature
ST_c = 2.0    # Cathode stoichiometry ratio 4
ST_a = 2.0         # Anode stoichiometry ratio 3
Pc_out = 1.0*101325    # atm Cathode outlet back pressure
Pa_out = 1.0*101325    # atm Anode outlet back pressure
Pc_in = 2.0*101325    # atm Cathode outlet back pressure
Pa_in = 1.05*101325    # atm Anode outlet back pressure
# m_c = 2.44e-3          # kg/s Cathode inlet flow rate
# Vc_in = 10          # m^3/min Cathode inlet flow rate
# Va_in = 10          # m^3/min Cathode inlet flow rate
RH_c = 0.6         # Cathode inlet relative humidity
RH_a = 0.8        # Anode inlet relative humidity

## Common fixed parameters
F = 96485            # C/mol Faraday constant
R = 8.314            # J/(mol*K) Ideal gas constant
P0 = 101325          # Pa 1 atm
P_sat = 3231.67      # Saturation pressure in Pa (empirical formula)
C_sat = 35.6          # mol/m^3 saturation concentration (empirical formula)
vis_air = 1.81e-5        # Pa*s Dynamic viscosity of inlet saturated air
vis_o2 = 20.55e-6        # Pa*s Dynamic viscosity of hydrogen
vis_h2 = 8.92e-6        # Pa*s Dynamic viscosity of hydrogen
M_H2O = 0.018    # Molar mass of water (kg/mol)
M_air = 0.02897    # Molar mass of air (kg/mol)
M_o2 = 0.032   # Molar mass of Oxygen (kg/mol)
M_h2 = 0.002016    # Molar mass of Hydrogen (kg/mol)
den_air = 1.166          # kg/m^3 Density of saturated air at 300K
den_o2 = 1.429          # kg/m^3 Oxygen density at standard conditions
den_h2 = 0.0899          # kg/m^3 Hydrogen density at 300K

## Electrochemical parameters -- empirical values, corrected values
ia_ref = 5.0e7        # A/m^3 Anode reference exchange current density     2.0e7
ic_ref = 4.0e-1    # A/m^3 Cathode reference exchange current density     4.0e-2
aaa = 20           # Cathode activation loss correction
ccc = 60             # Cathode activation loss correction
C_h2_ref= 56.4       # mol/m^3 Cathode reference exchange current density
C_o2_ref= 3.39       # mol/m^3 Anode reference exchange current density
A_ecsa = 0.7          # 70 m^2/g Electrochemical active surface area, typically ~0.01 g of catalyst

## Material properties -- electrochemical parameters
# lam0_mem = 20       # (6.2) Initial membrane water content
lam_initial = 6.2       # (6.2) Initial membrane water content
lam_sat = 14.0         # Nafion membrane, typical maximum water uptake 14.0
EW = 1.1             # kg/mol Dry ionomer equivalent weight
den_mem = 1980       # kg/m^3 Membrane density (at operating temperature)
Alp_c = 0.5          # Cathode transfer coefficient in BV
Alp_a = 0.5          # Anode transfer coefficient in BV
kele_all = 4000          # S/m Assumed: average electronic conductivity in the membrane electrode
kele_bp = 20000      # S/m Electronic conductivity of the plate
kele_gdl = 800     # S/m Electronic conductivity of the diffusion layer  8000
kele_mpl = 500     # S/m Electronic conductivity of the microporous layer  5000
kele_cl = 500       # S/m Electronic conductivity of the catalyst layer  5000
w_ion = 0.2        ## Ion volume fraction in the catalyst layer
# kion_mem = 8.3     # S/m Electronic conductivity of the plate  kion_mem = (0.5319*lam0_mem-0.326)*math.exp((12000/R)*(1/T_e - 1/T_s))  10

## Material properties -- porosity parameters and gas diffusion coefficients
por_cl = 0.4         # Porosity
por_mpl = 0.4        # Porosity
por_gdl = 0.6        # Porosity
# D_o2 = 2.894e-4      # m^2/s Oxygen diffusion coefficient

## Material properties -- thermodynamic parameters
k_T_bp = 20         # Thermal conductivity W/(m*K)
k_T_gdl = 12
k_T_cl = 0.5
h_T_water = 1000     # Liquid water heat transfer coefficient at 0.5-1.5 m/s, W/(m2*K)
h_sur = 230    #  Air heat transfer coefficient W/(m2*K)
Cp_cl = 3300       # J/(kg*K)
Cp_gdl = 2000       # J/(kg*K)
Cp_ch = 1580       # J/(kg*K)
den_cl = 1000       # kg/m3
den_gdl = 1000      # kg/m3
den_ch = 1000      # kg/m3


## Parameters related to the three-phase change of water
delta_S_a = 86.0   # Anode total entropy change J/(mol K)  @273K
delta_S_c = -251.9 # Cathode total entropy change J/(mol K)   @273K
den_water = 974   # kg/m^3  Liquid water density (at operating temperature)
k_nf_vp = 1.3  # /s  Membrane water -- water vapor
k_nf_lq = 1.0  # /s  Membrane water -- liquid water
k_vp_lq = 1.0  # /s  Water vapor -- liquid water
angle_cl = 1.6581     ## deg, contact angle   3.1416*95/180
angle_mpl = 1.9199   ## deg, contact angle   3.1416*110/180
angle_gdl = 2.0944   ## deg, contact angle   3.1416*120/180
K0_cl = 6.2e-13   ## m^2 Permeability
K0_mpl = 8.3e-13   ## m^2 Permeability
K0_gdl = 6.2e-12  ## m^2 Permeability
vis_lq = 2.414e-5   ## Pa*s  Dynamic viscosity of liquid water
# ten_water = -0.0001676*T_s + 0.1218  ## N/m  ten_water = -0.0001676*T + 0.1218
D_nf = 1.3e-8       ## m^2/s  Membrane water diffusion coefficient



## Component geometric dimensions
l_ch  = 100e-3       # m Channel length, X axis 50e-3
w_ch  = 1e-3       # m Channel width, Y axis
h_ch  = 1e-3       # m Channel height, Z axis
w_land = 1e-3      # m Land width
h_bp = 2.0e-3        # m Plate height
t_gdl  = 200e-6     # m Diffusion layer thickness
t_mpl  = 30e-6     # m Microporous layer thickness
t_ccl  = 12e-6     # m Cathode catalyst layer thickness
t_acl  = 6e-6     # m Anode catalyst layer thickness
t_mem  = 20e-6     # m Membrane thickness
A_cell_total = 25e-4    ## m^2 Total active area
A_cell = l_ch * w_ch   ##l_ch * w_ch ## m^2 Active area per single channel
A_cch = w_ch * h_ch     ##w_ch * h_ch    ## m^2 Cathode channel inlet area  /2 indicates a 1:1 ratio between channel width and land width
A_ach = w_ch * h_ch    ##w_ch * h_ch    ## m^2 Anode channel inlet area