"""Class specifying an N-body collisional process with generic methods"""

from .process import Process
import sympy as sp


class NBodyProcess(Process):
    def __init__(self, colliding_species, rate_coefficient=None, heat_rate_coefficient=None, name: str = ""):
        self.name = name
        self.dust_heat_per_volume = None
        self.colliding_species = colliding_species
        self.rate_coefficient = rate_coefficient
        self.heat_rate_coefficient = heat_rate_coefficient

    @property
    def rate_per_volume(self):
        """Returns the number of events per unit time and volume"""
        return self.rate_coefficient * sp.prod([sp.symbols("n_" + c) for c in self.colliding_species])

    @property
    def heat_per_volume(self):
        """Returns the number of events per unit time and volume"""
        return self.heat_rate_coefficient * sp.prod([sp.symbols("n_" + c) for c in self.colliding_species])

    @property
    def num_colliding_species(self):
        """Number of colliding species"""
        return len(self.colliding_species)
