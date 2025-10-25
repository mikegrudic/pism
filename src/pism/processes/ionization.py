"""Implementation of ionization process"""

from .process import Process


class Ionization(Process):
    """
    Class describing an ionization process. Could be collisional, photo, or cosmic ray-induced.

    Implements the chemistry network terms and calculation of
    """

    def __init__(self, species):
        super().__init__()
        self.species = species
        self.ionized_species = ionize(species)
        self.network = {}


#    def initialize_network(self):
