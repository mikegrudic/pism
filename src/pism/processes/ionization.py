"""Implementation of ionization process"""

from .process import Process
from ..misc import ionize
from ..symbols import T, T5, T3, T6
import sympy as sp


class Ionization(Process):
    """
    Class describing an ionization process. Could be collisional, photo, or cosmic ray-induced.

    Implements the chemistry network terms
    """

    def __init__(self, species):
        super().__init__()
        self.species = species
        self.ionized_species = ionize(species)
        self.__rate_per_volume = None

    @property
    def rate(self):
        return self.__rate_per_volume

    @rate.setter
    def rate(self, value):
        self.__rate_per_volume = value
        self.update_network()

    def update_network(self):
        """Sets up rate terms in the associated chemistry network for each species involved"""
        self.network[self.species] -= self.rate
        self.network[self.ionized_species] += self.rate
        self.network["e-"] += self.rate


collisional_ionization_cooling_rates = {
    "H": 1.27e-21 * sp.sqrt(T) * sp.exp(-157809.1 / T) / (1 + sp.sqrt(T5)),
    "He": 9.38e-22 * sp.sqrt(T) * sp.exp(-285335.4 / T) / (1 + sp.sqrt(T5)),
    "He+": 4.95e-22 * sp.sqrt(T) * sp.exp(-631515 / T) / (1 + sp.sqrt(T5)),
}

collisional_ionization_rates = {
    "H": 5.85e-11 * sp.sqrt(T) * sp.exp(-157809.1 / T) / (1 + sp.sqrt(T5)),
    "He": 2.38e-11 * sp.sqrt(T) * sp.exp(-285335.4 / T) / (1 + sp.sqrt(T5)),
    "He+": 5.68e-12 * sp.sqrt(T) * sp.exp(-631515 / T) / (1 + sp.sqrt(T5)),
}


def CollisionalIonization(species: str) -> Ionization:
    """Return an ionization process representing collisional ionization of the input species"""
    process = Ionization(species)

    nprod = sp.symbols(f"n_{species}") * sp.symbols("n_e-")
    if species not in collisional_ionization_rates:
        raise NotImplementedError(f"{species} does not have an available collisional ionization coefficient.")
    process.rate = collisional_ionization_rates * nprod

    if species in collisional_ionization_cooling_rates:
        process.heat = -collisional_ionization_cooling_rates[species] * nprod
    elif process.ionization_energy is not None:
        process.heat = process.ionization_energy * process.rate
    else:
        raise NotImplementedError(f"{species} collisional ionization cooling rate could not be computed.")

    return process
