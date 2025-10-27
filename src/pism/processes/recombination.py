"""Implementation of recombination process"""

from .process import Process
from .nbody_process import NBodyProcess
from ..misc import recombine
from ..symbols import T, T3, T6
from .ionization import ionization_energy
import sympy as sp


class Recombination(NBodyProcess):
    """
    Class describing an recombination process.

    Implements method for setting the chemistry network terms
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
    """Return a recombination process representing gas-phase (e.g. radiative) recombination"""
    if ion is None:
        return sum([GasPhaseRecombination(s) for s in gasphase_recombination_rates], Process())

    process = Recombination(ion)
    process.name = f"Gas-phase recombination of {ion}"

    if ion not in gasphase_recombination_rates:
        raise NotImplementedError(f"{ion} does not have an available gas-phase recombination coefficient.")

    process.rate_coefficient = gasphase_recombination_rates[ion]
    process.heat_rate_coefficient = -gasphase_recombination_cooling[ion]
    return process


gasphase_recombination_rates = {
    "H+": 8.4e-11 / sp.sqrt(T) * T3**-0.2 / (1 + T6**0.7),
    "He+": 1.5e-10 * T**-0.6353 + 1.9e-3 * T**-1.5 * sp.exp(-4.7e5 / T) * (1 + 0.3 * sp.exp(-9.4e4 / T)),
}
gasphase_recombination_rates["He++"] = 4 * gasphase_recombination_rates["H+"]  # H-like

gasphase_recombination_cooling = {
    "H+": 8.7e-27 * sp.sqrt(T) * T3**-0.2 / (1 + T6**0.7),
    "He+": 1.5e-10 * T**-0.3647,
}
gasphase_recombination_cooling["He++"] = 4 * gasphase_recombination_cooling["H+"]  # H-like
