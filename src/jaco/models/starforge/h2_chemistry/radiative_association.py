"""Implementation of radiative association"""

from ..symbols import T, log_T
from jaco.processes import ChemicalReaction
import sympy as sp


def radiative_association(species):
    """
    Symbolic implementation of the rate coefficient for radiative association of a species and e-:

    species + e- -> species- + photon

    where gamma has energy equal to the electron affinity of the species.

    Returns
    -------
    Symbolic expression for radiative association rate coefficient in cgs units
    """

    # TODO (low): implement automatic lookup of electron affinity to get the photon energy
    match species:
        case "H" | "D":
            bib = ["1979MNRAS.187P..59W"]
            k_low = -17.845 + 0.762 * log_T + 0.1523 * log_T * log_T - 0.03274 * log_T * log_T * log_T
            k_high = -16.420 + 0.1998 * log_T**2 - 5.447e-3 * log_T**4 + 4.0415e-5 * log_T**6
            k = 10**sp.Piecewise((k_low, T < 6000), (k_high, T >= 6000))
        case _:
            raise NotImplementedError(f"Radiative association rate not implemented for species {species}.")

    return ChemicalReaction(
        f"{species} + e- -> " + species + "-" + f" + photon_assoc,{species}",
        rate_coefficient=k,
        name=f"Radiative association of {species} with e-",
        bibliography=bib,
    )
