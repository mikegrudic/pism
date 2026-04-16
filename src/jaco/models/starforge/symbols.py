"""Definition of symbols that are used throughout the model.

NOTE: all quantites are assumed to be in cgs units!
"""

import sympy as sp
from ...symbols import x_, n_

T = sp.Symbol("T")  # Gas temperature
sqrt_T = sp.sqrt(T)
log_T = sp.log(T, 10.0)
n_Htot = n_("Htot")  # Total number density of H nuclei
X_H = sp.Symbol("X")  # mass fraction of hydrogen
T_dust = sp.Symbol("Td")  # Dust temperature
f_dust = sp.Symbol("f_d")  # Factor accounting for sublimation
Z_dust = sp.Symbol("Z_d")  # Solar-normalized dust abundance. Value of 1 corresponds to Solar neighborhood dust.
G_0 = sp.Symbol("G_0")  # UV radiation field normalized to Habing
f_shield = sp.Symbol("f_shield")  # Lyman-Werner self-shielding factor
grad_v = sp.Symbol("∇v")  # velocity gradient Frobenius norm in CGS (s^-1)
NH = sp.Symbol("N_H")  # column density of H nuclei
dx = sp.Symbol("Δx")  # effective cell size in cm
ISRF = sp.Symbol("ISRF")  # scaling factor for ISRF and cosmic ray background
H2_formation_heat_cgs = 7.2e-12
rho = sp.Symbol("rho")
cs = sp.Symbol("c_s")
T_CMB = sp.Symbol("T_CMB")
A_V = 5.34e-22 * NH * Z_dust * f_dust
cosmicray_attenuation_fac = sp.Min(1, 1e21 / NH * sp.exp(-NH / 1e24))
cosmicray_ionization_rate_H = sp.sqrt(ISRF) * 1.6e-12 * 1e-5 * cosmicray_attenuation_fac
psi_grain = G_0 * sqrt_T / (0.5 * (1.0e-12 + x_("e-")) * n_Htot)  # grain charging parameter
