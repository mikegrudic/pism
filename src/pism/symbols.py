"""Definition of various sympy symbols used throughout the module"""

import sympy as sp

T = sp.symbols("T")  # temperature
T5 = T / 10**5
T6 = T / 10**6
T3 = T / 10**3
T4 = T / 10**4
c_s = sp.symbols("c_s")  # sound speed
G = sp.symbols("G")  # gravitational constant
ρ = sp.symbols("ρ")  # total mass density
n_e = sp.symbols("n_e-")  # electron number density
z = sp.symbols("z")  # cosmological redshift


def n_(species: str):
    return sp.symbols(f"n_{species}")
