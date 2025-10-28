"""Class specifying an N-body collisional process with generic methods"""

from ..process import Process
import sympy as sp


class NBodyProcess(Process):
    """Process implementing special methods specific to n-body processes, whose rates
    all follow the pattern

    rate per volume = k * prod_i(n_i) for i in species

    rate and heat are promoted from attributes to properties implemented to compute this pattern.
    """

    def __init__(self, colliding_species, rate_coefficient=0, heat_rate_coefficient=0, name: str = ""):
        self.name = name
        self.initialize_network()
        self.dust_heat = 0
        self.colliding_species = colliding_species
        self.rate_coefficient = rate_coefficient
        self.heat_rate_coefficient = heat_rate_coefficient

    @property
    def rate(self):
        """Returns the number of events per unit time and volume"""
        if self.rate_coefficient is None:
            return None
        return self.rate_coefficient * sp.prod([sp.symbols("n_" + c) for c in self.colliding_species])

    @property
    def heat(self):
        """Returns the number of events per unit time and volume"""
        if self.heat_rate_coefficient is None:
            return None
        return self.heat_rate_coefficient * sp.prod([sp.symbols("n_" + c) for c in self.colliding_species])

    @property
    def num_colliding_species(self):
        """Number of colliding species"""
        return len(self.colliding_species)
