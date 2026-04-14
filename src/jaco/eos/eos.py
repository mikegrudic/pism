"""Routines to generate the symbolic expression for internal energy, mass density, pressure, and c_v, given the species
in the network.
"""

from ..data.atoms import atomic_weights, atoms
from astropy.constants import k_B, m_p, m_e
from ..symbols import T, n_
from astropy import units as u
from ..species_strings import species_charge, base_species, species_mass
import sympy as sp
from .H2_partition_function import H2_energy_over_kbT

k_B_cgs = k_B.to(u.erg / u.K).value


def species_energy(species):
    """Returns the average thermal energy per particle for a species"""
    if species == "H_2":  # currently the only defined special heat capacity, likely the only one that matters
        fac = H2_energy_over_kbT(T)
    else:
        fac = 1.5

    return fac * k_B_cgs * T


def species_heat_capacity(species):
    """Returns the heat capacity per particle for a species"""
    if species == "H_2":  # currently the only defined special heat capacity, likely the only one that matters
        fac = sp.diff(T * H2_energy_over_kbT(T), T)
    else:
        fac = 1.5
    return fac * k_B_cgs


class EOS:
    def __init__(self, species):
        self.species = species

    @property
    def density(self):
        """Total mass density"""
        return sum([species_mass(s) * n_(s) for s in self.species])

    @property
    def energy_density(self):
        """Thermal energy density"""
        return sum([n_(s) * species_energy(s) for s in self.species])

    @property
    def internal_energy(self):
        """Internal energy per unit mass"""
        return self.energy_density / self.density

    @property
    def pressure(self):
        """Pressure via ideal gas law"""
        return sum([n_(s) for s in self.species]) * k_B_cgs * T

    @property
    def heat_capacity(self):
        """Specific heat capacity at constant volume (per unit mass)"""
        return sum([n_(s) * species_heat_capacity(s) for s in self.species]) / self.density
