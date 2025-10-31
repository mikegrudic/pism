"""Implementation of recombination process"""

from ..process import Process
from .nbody_process import NBodyProcess
from ..misc import recombine
from ..symbols import T, T3, T6
from .ionization import ionization_energy
import sympy as sp


class Recombination(NBodyProcess):
    """
    Class describing a recombination process: ion + e- -> recombined species + hν

    Implements method for setting the chemistry network terms

    Parameters
    ----------
    ion: str
        Ionic species being recombined
    """

    def __init__(self, ion: str):
        self.ion = ion
        self.recombined_species = recombine(ion)
        self.colliding_species = {ion, "e-"}
        super().__init__(self.colliding_species)
        self.ionization_energy = ionization_energy(self.recombined_species)
        self.__rate_coefficient = 0
        self.heat_rate_coefficient = 0

    @property
    def rate_coefficient(self):
        """Returns the rate coefficient of the recombination process"""
        return self.__rate_coefficient

    @rate_coefficient.setter
    def rate_coefficient(self, value):
        """Ensures that the network is always updated when we update the rate coefficient"""
        self.__rate_coefficient = value
        self.update_network()

    def update_network(self):
        """Sets up rate terms in the associated chemistry network for each ion involved"""
        if self.rate is None:
            return
        self.network[self.ion] -= self.rate
        self.network[self.recombined_species] += self.rate
        self.network["e-"] -= self.rate


def GasPhaseRecombination(ion=None) -> Recombination:
    """Return a recombination process representing gas-phase (e.g. radiative) recombination

    Parameters
    ----------
    ion: str, optional
        Ionic species getting recombined. If None, function will return a composite process of all gas-phase recombination
        processes with known rates.

    Returns
    -------
    process: Recombination
        `Process` instance describing the gas-phase recombination process
    """
    if ion is None:
        return sum([GasPhaseRecombination(s) for s in gasphase_recombination_rates], Process())

    process = Recombination(ion)
    process.name = f"Gas-phase recombination of {ion}"

    if ion not in gasphase_recombination_rates:
        raise NotImplementedError(f"{ion} does not have an available gas-phase recombination coefficient.")
    process.rate_coefficient = gasphase_recombination_rates[ion]
    process.heat_rate_coefficient = -gasphase_recombination_cooling[ion]
    return process


def hydrogenic_recombination_rate(Z):
    """Verner & Ferland 1996"""
    return (
        Z
        * 7.982e-11
        / (
            sp.sqrt(T / 3.148 / Z**2)
            * sp.Pow((1.0 + sp.sqrt(T / 3.148 / Z**2)), 0.252)
            * sp.Pow((1.0 + sp.sqrt(T / 7.036e5 / Z**2)), 1.748)
        )
    )


# All fits below are from Verner & Ferland 1996
gasphase_recombination_rates = {
    "H+": hydrogenic_recombination_rate(1),
    "He+": 9.356e-10
    / (
        sp.sqrt(T / 4.266e-2)
        * sp.Pow((1.0 + sp.sqrt(T / 4.266e-2)), 0.2108)
        * sp.Pow((1.0 + sp.sqrt(T / 3.676e7)), 1.7892)
    )
    + 1.9e-3 * T**-1.5 * sp.exp(-4.7e5 / T) * (1 + 0.3 * sp.exp(-9.4e4 / T)),
    "He++": hydrogenic_recombination_rate(2),
}
# an electron—ion pair removes the mean kinetic energy during recombination
# To a good approximation, the mean energy lost by the gas during dielectronic recombination of He+ is the w = 2 excitation energy of He+.
mean_kinetic_energy = 1.036e-16 * T
gasphase_recombination_cooling = {
    "H+": mean_kinetic_energy * gasphase_recombination_rates["H+"],
    "He+": 1.55e-26 * T**-0.3647,
    "He++": mean_kinetic_energy * gasphase_recombination_rates["He++"],
}
gasphase_recombination_cooling["He++"] = 4 * gasphase_recombination_cooling["H+"]  # H-like
