"""Implementation of CO cooling following Whitworth & Jaffa 2018A&A...611A..20W"""

# TODO: write test against their plots
from jaco.processes import NBodyProcess
from .symbols import T, grad_v, x_, n_Htot

# Eq. 37
lam_CO_lo = 2.16e-27
lam_CO_hi = 2.21e-28
beta_0 = 1.23
beta_nH2 = 0.0533
beta_T = 0.164

# note we are letting 'lambda' be the usual thing defined cm^3 erg s^-1, not their erg s^-1
lambda_CO_lo = lam_CO_lo * T**1.5
# LVG limit: grad_v is in s^-1 (CGS), 3.241e-14 = 1 km/s/pc in s^-1
# using n_H here because they are really using n_H_2 as a proxy for mass density, in practice
lambda_CO_hi = lam_CO_hi * (x_("CO") * 3.241e-14 / grad_v) ** -1 * T**4 / (0.5 * n_Htot) ** 2
beta = 1.23 * (0.5 * n_Htot) ** beta_nH2 * T**beta_T
lambda_CO = (lambda_CO_lo ** (-1 / beta) + lambda_CO_hi ** (-1 / beta)) ** -beta

CO_cooling = NBodyProcess(
    ("CO", "H_2"), heat_rate_coefficient=lambda_CO, name="CO Cooling", bibliography=["2018A&A...611A..20W"]
)
