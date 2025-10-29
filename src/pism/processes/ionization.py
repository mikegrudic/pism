"""Implementation of ionization process"""

from ..process import Process
from ..misc import ionize
from ..symbols import T, T5, T3, T6, n_e, n_
import sympy as sp
from astropy import units as u


class Ionization(Process):
    """
    Class describing an ionization process. Could be collisional, photo, or cosmic ray-induced.

    Implements method for setting the chemistry network terms
    """

    def __init__(self, species: str):
        self.species = species
        self.ionized_species = ionize(species)
        self.__ionization_energy = None
        super().__init__()
        self.__rate_per_volume = 0

    @property
    def rate(self):
        return self.__rate_per_volume

    @rate.setter
    def rate(self, value):
        self.__rate_per_volume = value
        self.update_network()

    @property
    def ionization_energy(self):
        if self.__ionization_energy is None:
            self.__ionization_energy = ionization_energy(self.species)
        return self.__ionization_energy

    def update_network(self):
        """Sets up rate terms in the associated chemistry network for each species involved"""
        if self.rate is None:
            return
        self.network[self.species] -= self.rate
        self.network[self.ionized_species] += self.rate
        self.network["e-"] += self.rate


def ionization_energy(species, unit=u.erg):
    """Return the energy in erg required to ionize a species"""
    # NOTE: come back and get this from a proper datafile
    energies_eV = {"H": 13.6, "He": 24.59, "He+": 54.42}
    return energies_eV[species] * u.eV.to(unit)


collisional_ionization_cooling_rates = {
    "H": 1.27e-21 * sp.sqrt(T) * sp.exp(-157809.1 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
    "He": 9.38e-22 * sp.sqrt(T) * sp.exp(-285335.4 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
    "He+": 4.95e-22 * sp.sqrt(T) * sp.exp(-631515 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
}

collisional_ionization_rates = {
    "H": 5.85e-11 * sp.sqrt(T) * sp.exp(-157809.1 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
    "He": 2.38e-11 * sp.sqrt(T) * sp.exp(-285335.4 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
    "He+": 5.68e-12 * sp.sqrt(T) * sp.exp(-631515 / T) / (1 + sp.sqrt(T5)),  # 1996ApJS..105...19K
}


def CollisionalIonization(species=None) -> Ionization:
    """Return an ionization process representing collisional ionization of the input species"""

    if species is None:
        return sum([CollisionalIonization(s) for s in collisional_ionization_rates], Process())

    process = Ionization(species)
    process.name = f"Collisional Ionization of {species}"
    nprod = n_(species) * n_e

    if species not in collisional_ionization_rates:
        raise NotImplementedError(f"{species} does not have an available collisional ionization coefficient.")
    process.rate = collisional_ionization_rates[species] * nprod
    process.heat = -process.ionization_energy * process.rate

    return process
