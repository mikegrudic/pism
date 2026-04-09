"""Specification of STARFORGE network including radiation, thermo, cosmic rays, and dust"""

from jaco.processes import CollisionalIonization, GasPhaseRecombination, FreeFreeEmission
from .line_cooling import LineCoolingSimple
from ..model import Model
import sympy as sp
from .h2_chemistry import H2_chemistry
from .H2_cooling import H2_cooling
from .CO_cooling import CO_cooling
from .gas_dust_collisions import gas_dust_collisions
from .cosmic_ray_ionization import cosmic_ray_ionization, cosmic_ray_photoionization
from .photoelectric_heating import photoelectric_heating
from .grain_assisted_recombination import grain_assisted_recombination
from .metal_line_cooling import metal_line_cooling
from jaco.processes import inv_compton_cooling
# import h2_chemistry


def make_model():
    processes = [
        H2_chemistry,
        *[LineCoolingSimple(s) for s in ("H", "He+", "C+")],
        *[CollisionalIonization(s) for s in ("H", "He", "He+")],
        *[GasPhaseRecombination(i) for i in ("H+", "He+", "He++")],
        *[FreeFreeEmission(i) for i in ("H+", "He+", "He++")],
        H2_cooling,
        CO_cooling,
        gas_dust_collisions,
        *[cosmic_ray_ionization(s) for s in ("H", "C")],
        cosmic_ray_photoionization("C"),
        photoelectric_heating,
        inv_compton_cooling,
        grain_assisted_recombination("C+"),
        metal_line_cooling(z=0.0),
    ]

    model = sum(processes)
    return model

    # processes += sum([cosmic_ray_ionization(s) for s in ("H", "C")])
    # processes += sum([grain_assisted_recombination(s) for s in ("C+",)])
    # processes += photon_absorption
    # processes += dust_emission
    # processes += photoelectric_heating

    #    process = sum(processes)

    # assumption: H- in equilibrium
    #    collected = sp.collect(process.network"H-".rhs, n_("H-"))
    #    x_Hminus = collected.coeff(n_("H-"), 0) / collected.coeff(n_("H-"), 1)

    # assumption: C- in equilibrium

    # f_CO given by formula dependent on G0

    # assumption: dust energy in steady state

    # UV background with shieldfac


#    return sum(processes)


#     processes += sum(h2_chemistry.reactions)
# #    processes += CO_cooling(prescription="Whitworth+2018")

#     processes += photon_absorption(band) for band in "EUV", "FUV", "NUV", "OPT", "FIR"
#     processes += dust_emission(band) for band in "EUV", "FUV", "NUV", "OPT", "FIR"

# assumptions: x_D = 2.527e-5 x_H, x_D+ = 2.527e-5 x_H+, zero out associated collisional dissociation rates
