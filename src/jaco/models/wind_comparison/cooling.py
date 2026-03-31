import numpy as np
from jaco.symbols import piecewise_powerlaw, T, n_
from jaco.processes import ThermalProcess
import sympy as sp
from jaco.math import logistic

T_cooling_curve = np.array(
    [0.99999999e1, 1.0e02, 6.0e03, 1.75e04, 4.0e04, 8.7e04, 2.30e05, 3.6e05, 1.5e06, 3.50e06, 2.6e07, 1.0e12]
)

lambda_cooling_curve = np.array(
    [
        1e-30,
        1.00e-27,
        2.00e-26,
        1.50e-22,
        1.20e-22,
        5.25e-22,
        5.20e-22,
        2.25e-22,
        1.25e-22,
        3.50e-23,
        2.10e-23,
        4.12e-21,
    ]
)

exponent_cooling_curve = np.array(
    [
        3.0,
        0.73167566,
        8.33549431,
        -0.26992783,
        1.89942352,
        -0.00984338,
        -1.8698263,
        -0.41187018,
        -1.50238273,
        -0.25473349,
        0.5000359,
        0.5,
    ]
)


lambda_cooling = piecewise_powerlaw(T_cooling_curve, lambda_cooling_curve, T, extrapolate=True, name="lambda_cooling")
cooling = ThermalProcess(-lambda_cooling * n_("H") ** 2, name="Cooling")
heating = ThermalProcess(2e-26 * n_("H") * logistic(-(T - 15000) / 1000), name="Heating")
