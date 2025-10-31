"""Definition of various sympy symbols used throughout the module"""

import sympy as sp

T = sp.Symbol("T")  # temperature
T5 = T / 10**5
T6 = T / 10**6
T3 = T / 10**3
T4 = T / 10**4
c_s = sp.Symbol("c_s")  # sound speed
G = sp.Symbol("G")  # gravitational constant
ρ = sp.Symbol("ρ")  # total mass density
n_e = sp.Symbol("n_e-")  # electron number density
z = sp.Symbol("z")  # cosmological redshift


def n_(species: str):
    return sp.Symbol(f"n_{species}")
