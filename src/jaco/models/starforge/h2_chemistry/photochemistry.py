"""Implementation of photodissociation of various hydrogen species"""

from ..symbols import sp, G_0, T, grad_v, dx, NH, x_, n_
from jaco.processes import ChemicalReaction


def f_selfshield_H2(prescription="Gnedin & Draine 2014"):
    """Self-shielding prescription as implemented in GIZMO"""
    v_thermal_rms = 1.11e4 * sp.sqrt(T)  # sqrt(3*k_B*T / (2*m_H)) in cm/s
    match prescription:
        case "Gnedin & Draine 2014":
            surface_density_H2_0 = 5.0e14
            w0 = 0.035
            dv_turb = grad_v * dx  # grad_v in s^-1, dx in cm -> dv_turb in cm/s
            x00 = NH * (1 - x_("H+")) / surface_density_H2_0
            x01 = x00 / (
                sp.sqrt(1.0 + 3.0 * dv_turb * dv_turb / (v_thermal_rms * v_thermal_rms)) * sp.sqrt(2.0) * v_thermal_rms
            )
            fH2 = 2 * n_("H_2") / (2 * n_("H_2") + n_("H"))
            x_ss_1 = 1 + fH2 * x01
            x_ss_sqrt = sp.sqrt(1.0 + fH2 * x00)
            x_exp_fac = 0.00085
            return (1.0 - w0) / (x_ss_1 * x_ss_1) + w0 / x_ss_sqrt * sp.exp(-x_exp_fac * x_ss_sqrt)
        case "Wolcott-Green 2011":
            # modified version of Draine & Bertoldi 1965
            x = sp.Symbol("N_H_2") / 5e14
            b5 = v_thermal_rms / 1e5  # b in units of 10^5 cm/s per Draine & Bertoldi convention
            f = 0.965 / (1 + x / b5) ** 1.1 + 0.035 / (1 + x) ** 0.5 * sp.exp(-8.5e-4 * (1 + x) ** 0.5)
            return f

        case _:
            raise NotImplementedError("self-shielding prescription not implemented")


def photodissociation(molecule):
    """
    Photodissociation processes for a species given a certain UV flux

    Neglects the associated photon absorption.
    """
    match molecule:
        case "H_2":
            Rdiss = 3.3e-11 * G_0
            rate = Rdiss * f_selfshield_H2()
            bib = "2014ApJ...795...37G"
            return ChemicalReaction("H_2 -> H + H", rate, name="Photodissociation of H-", bibliography=[bib])
            # could add heat of UV pumping: heat = 2e-11 Rdiss n_H_2
            # and heat of photo dissociation: heat = 6.4e-13 R_diss n_H_2 n/(n+n_crit)
            # while we're in here.
            # "Our value for ncr is a weighted harmonic mean of the value for H_2 -H collisions given by Lepp
            # & Shull (1983) reduced by a factor of 10, as advised by Martin et al. (1996), and the value for H_2-H_2
            # collisions given by Shapiro & Kang (1987)"


def photodetachment(species):
    match species:
        case "H-":
            # glover & jappsen: heating rate of this not significant
            rate = 3.62e-17
            bib = "2023MNRAS.519.3154H"
            return ChemicalReaction("H- -> H + e-", rate, name="Photodetachment of H-", bibliography=[bib])
