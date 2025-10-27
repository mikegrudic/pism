"""Implementation of free-free emission as a 2-body process"""

from .nbody_process import NBodyProcess
from ..misc import species_charge
from ..symbols import T
import sympy as sp


def gaunt_factor(T):
    """Fit to data in table 3.3 of Spitzer (1978)"""
    return 1.1 + 0.34 * sp.exp((-5.5 - sp.log(T) / sp.log(10)) ** 2 / 3.0)


def FreeFreeEmission(ion: str) -> NBodyProcess:
    """Free-free emisison process"""
    process = NBodyProcess({ion, "e-"})
    charge = species_charge(ion)
    process.heat_rate_coefficient = -1.42e-27 * gaunt_factor(T) * charge * sp.sqrt(T)
    return process
