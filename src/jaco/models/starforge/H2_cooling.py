"""Implementation of line cooling for various species"""

from jaco.processes.thermal_process import ThermalProcess
import sympy as sp
from .symbols import T, n_Htot, n_, log_T, x_


# TODO: write test comparing with GA08 Fig 2
# TODO: make sure bibliography is complete


def n_HD_prescription():
    return n_Htot * sp.Min(0.00126 * x_("H_2"), 4.0e-5)


def lambda_H2_thin(collider):
    """Returns symbolic expression for the non-LTE collisional excitation cooling rate for H_2 with a colliding species"""
    q = sp.Min(log_T - 3.0, 2.0)
    match collider:
        case "H":
            return 10 ** (-103.0 + 97.59 * log_T - 48.05 * log_T**2 + 10.8 * log_T**3 - 0.9032 * log_T**4)
        # GA08 Eq 26, Galli & Palla 1998 - for H2-H collisions
        case "He":
            return 10 ** (
                -23.6892
                + 2.18924 * q
                - 0.815204 * q * q
                + 0.290363 * q * q * q
                - 0.165962 * q * q * q * q
                + 0.191914 * q * q * q * q * q
            )
        case "H_2":
            return 10 ** (
                -23.9621
                + 2.09434 * q
                - 0.771514 * q * q
                + 0.436934 * q * q * q
                - 0.149132 * q * q * q * q
                - 0.0336383 * q * q * q * q * q
            )
        case "H+":
            return 10 ** (
                -21.7167
                + 1.38658 * q
                - 0.379153 * q * q
                + 0.114537 * q * q * q
                - 0.232142 * q * q * q * q
                + 0.0585389 * q * q * q * q * q
            )
        case "e-":
            lambda_H2_thin_e_lo = 10 ** (
                -34.2862
                - 48.5372 * q
                - 77.1212 * q * q
                - 51.3525 * q * q * q
                - 15.1692 * q * q * q * q
                - 0.981203 * q * q * q * q * q
            )

            lambda_H2_thin_e_hi = 10 ** (
                -22.1903
                + 1.5729 * q
                - 0.213351 * q * q
                + 0.961498 * q * q * q
                - 0.910232 * q * q * q * q
                + 0.137497 * q * q * q * q * q
            )
            return sp.Piecewise((lambda_H2_thin_e_lo, log_T < 2.30103), (lambda_H2_thin_e_hi, log_T >= 2.30103))


def H2_cooling_rate():
    """Returns expression for the total H_2 cooling per unit volume"""
    T3 = sp.Min(T / 1e3, 10.0)
    # super-critical H2-H cooling rate [per H2 molecule] - HM1979 Eq 6.37, 6.38
    Lambda_H2_thick = (
        6.7e-19 * sp.exp(-5.86 / T3)
        + 1.6e-18 * sp.exp(-11.7 / T3)
        + 3.0e-24 * sp.exp(-0.51 / T3)
        + 9.5e-22 * pow(T3, 3.76) * sp.exp(-0.0022 / (T3 * T3 * T3)) / (1.0 + 0.12 * pow(T3, 2.1))
    ) / n_Htot

    Lambda_HD_thin = (
        (1.555e-25 + 1.272e-26 * pow(T, 0.77)) * sp.exp(-128.0 / T)
        + (2.406e-25 + 1.232e-26 * pow(T, 0.92)) * sp.exp(-255.0 / T)
    ) * sp.exp(
        -T3 * T3 / 25.0
    )  # where does this last factor come from?

    thin_cooling_total_perH2 = sum([lambda_H2_thin(c) * n_(c) for c in ("H", "H_2", "He", "e-", "H+")])

    nH_over_ncrit = thin_cooling_total_perH2 / Lambda_H2_thick
    n_over_ncrit_HD = x_("HD") / x_("H_2") * nH_over_ncrit

    total_cooling = n_("H_2") * thin_cooling_total_perH2 / (1 + nH_over_ncrit) + n_("HD") * Lambda_HD_thin / (
        1 + n_over_ncrit_HD
    )
    return total_cooling


H2_cooling = ThermalProcess(
    -H2_cooling_rate(),
    name="H2 + HD Line Cooling",
    bibliography=["2008MNRAS.388.1627G", "1998A&A...335..403G", "1979ApJS...41..555H"],
)
