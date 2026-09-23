L = 200e-6  # Cathode thickness [m]
R_s = 6.0e-6  # Active particle radius [m] 3.26e-6
A_cell = 0.05  # Electrode cross-sectional area [m^2]  0.02
eps_e = 0.3  # Electrolyte porosity
eps_s = 0.5  # Volume fraction of active material 0.5
a_s = 3 * eps_s / R_s  # Specific interfacial area [m^-1]

# Transport parameters
D_e = 7.5e-10  # Effective diffusion coefficient in the liquid phase [m^2/s] 7.5e-11
D_s = 1e-12  # Solid-phase diffusion coefficient [m^2/s] 1e-13
sigma_s = 3.8  # Effective electronic conductivity of the solid phase [S/m]
kappa_e = 1.0  # Effective ionic conductivity of the liquid phase [S/m]
t_plus = 0.38  # Lithium-ion transference number

# Kinetic and thermodynamic parameters
F = 96485.0  # Faraday constant [C/mol]
R = 8.314  # Gas constant [J/(mol*K)]
T = 298.15  # Temperature [K]
k_norm = 2e-9  # Butler-Volmer reaction rate constant (simplified) 4.8e-10
c_max = 46000.0  # Maximum solid-phase concentration [mol/m^3]
a_p = 0.5   # Charge transfer coefficient
a_n = 0.5

# Initial conditions and operating conditions
c_e0 = 1000.0  # Initial liquid-phase concentration [mol/m^3]
c_s0 = 5000.0  # Initial solid-phase concentration [mol/m^3] (assumed high SOC state) 12000.0
capacity = A_cell * L * eps_s * c_max * F / 3600
print("Battery capacity in Ah:", capacity)