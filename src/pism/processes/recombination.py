"""Implemetation of generic recombination process"""

from .nbody_process import *


class Recombination(NBodyProcess):
    """
    Electron-ion recombination process

    Implements the chemistry network and methods for computing rate and cooling rate given recombination
    coefficient and ionization energy
    """

    def __init__(self, ion, catalyst=None):
        super().__init__(colliding_species={ion, "e-"})
        self.ion = ion
        self.recombined_species = recombine(ion)
        self.recombination_coefficient = None
        self.__ionization_energy = None

    @property
    def ionization_energy(self):
        """Lookup from table"""
        #        raise NotImplementedError("ionization energy lookup not yet implemented")
        if self.__ionization_energy is None:
            self.__ionization_energy = lookup_ionization_energy(ion)
        return __ionization_energy

    @ionization_energy.setter
    def ionization_energy(self, value):
        self.__ionization_energy = value
