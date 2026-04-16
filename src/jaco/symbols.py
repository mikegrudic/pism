"""Definition of various sympy symbols used throughout the module"""

import sympy as sp
from typing import Union
from astropy.constants import k_B, m_p
from astropy import units as u
import numpy as np

from jaco.interpolation import PiecewiseLinearInterp, TableInterp2D, TableInterp3D, register_table
from sympy.core.symbol import Str

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
t = sp.Symbol("t")  # time
dt = sp.Symbol("Δt")
n_Htot = sp.Symbol("n_Htot")


boltzmann_cgs = k_B.to(u.erg / u.K).value
protonmass_cgs = m_p.to(u.g).value
# write down internal energy density in terms of number densities - this defines the EOS


def d_dt(species: Union[str, sp.core.symbol.Symbol]):
    if isinstance(species, str):
        return sp.diff(sp.Function(n_(species))(t), t)
    else:
        return sp.diff(sp.Function(species)(t), t)


def n_(species: str):
    """Returns number density symbol for species"""
    match species:
        case "heat":
            return sp.Symbol(f"⍴u")
        case "dust heat":
            return sp.Symbol(f"⍴u_dust")
        case _:
            return sp.Symbol(f"n_{species}")


def x_(species: str):
    """Returns abundance symbol for species"""
    return sp.Symbol(f"x_{species}")


def BDF(species):
    """Backward-difference formula for time discretization: (n_s - n_s_initial) / Δt.

    The initial-value symbol uses the ``_initial`` suffix (not ``_0``) to avoid
    confusion with species names during the n→x conversion in
    :meth:`EquationSystem.do_conservation_reductions`. A ``_0`` suffix would be
    misidentified as part of a species name (e.g. ``n_H_2_0`` → ``x_H_2_0``
    treated as a species containing 2 H atoms).
    """
    return (n_(species) - sp.Symbol(str(n_(species)) + "_initial")) / dt


#: Character replacements for converting sympy symbol names to valid C identifiers.
#: Greek letters are spelled out, ∇ becomes ``grad_``, Δ becomes ``Delta_``
#: (with trailing underscore to separate from the following character).
_SANITIZE_REPLACEMENTS = {
    "+": "plus", "-": "minus", ",": "_",
    "∇": "grad_",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda_", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta_", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi",
    "Π": "Pi", "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
    "⍴": "rho",
}


def sanitize_name(name):
    """Sanitize a string to be a valid C/C++/Fortran identifier."""
    for old, new in _SANITIZE_REPLACEMENTS.items():
        name = name.replace(old, new)
    return name


def sanitize_symbols(expr):
    """
    Given a symbolic expression, replace all symbols with characters invalid in
    C/C++/Fortran identifiers to avoid syntax errors in code generation.
    """
    if hasattr(expr, "__iter__"):  # if iterable call recursively until we get down to actual expressions
        expr = [sanitize_symbols(e) for e in expr]
    else:
        for s in expr.free_symbols:
            name = str(s)
            new_name = sanitize_name(name)
            if new_name != name:
                expr = expr.subs(s, sp.Symbol(new_name))
    return expr


def piecewise_linear(X, Y, x, extrapolate=False, name=None):
    """
    Returns a symbolic expression describing a piecewise-linear interpolant of data X and Y

    Parameters
    ----------
    X: array_like
        array of X values
    Y: array_like
        array of Y values
    x: sympy.Symbol
        Symbol for the independent variable associated with X
    extrapolate: Boolean, optional
        Whether to use a linear extrapolant past the limits, or simply limit to the outermost values
    name: str, optional
        Descriptive name for the interpolant, used in generated code identifiers
    """
    if not np.all(np.diff(X) > 0):
        raise ValueError("X values passed to piecewise_linear must be monotonic.")
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    X_tuple = sp.Tuple(*[sp.Float(v) for v in X])
    Y_tuple = sp.Tuple(*[sp.Float(v) for v in Y])
    extrap_flag = sp.Integer(1 if extrapolate else 0)
    args = [x, X_tuple, Y_tuple, extrap_flag]
    if name is not None:
        args.append(Str(name))
    return PiecewiseLinearInterp(*args)


def piecewise_powerlaw(X, Y, x, extrapolate=False, name=None):
    """
    Returns a symbolic expression describing a piecewise-powerlaw interpolant of data X and Y

    Parameters
    ----------
    X: array_like
        array of X values
    Y: array_like
        array of Y values
    x: sympy.Symbol
        Symbol for the independent variable associated with X
    extrapolate: Boolean, optional
        Whether to use a powerlaw extrapolant past the limits, or simply limit to the outermost values
    name: str, optional
        Descriptive name for the interpolant, used in generated code identifiers
    """
    return sp.exp(piecewise_linear(np.log(X), np.log(Y), sp.log(x), extrapolate, name=name))


def table_interp_2d(name, data, axes, x, y, log_axes=None):
    """Create a symbolic 2D table interpolation expression.

    Parameters
    ----------
    name : str
        Unique name for the table.
    data : np.ndarray
        2D array of shape (n_axis0, n_axis1).
    axes : list of np.ndarray
        [axis0_values, axis1_values], each uniformly spaced (linear or log).
    x, y : sympy expressions
        Interpolation variables corresponding to axis0 and axis1.
    log_axes : list of bool, optional
        Whether each axis is log-spaced.
    """
    register_table(name, data, axes, log_axes)
    return TableInterp2D(x, y, Str(name))


def table_interp_3d(name, data, axes, x, y, z, log_axes=None):
    """Create a symbolic 3D table interpolation expression.

    Parameters
    ----------
    name : str
        Unique name for the table.
    data : np.ndarray
        3D array of shape (n_axis0, n_axis1, n_axis2).
    axes : list of np.ndarray
        [axis0_values, axis1_values, axis2_values].
    x, y, z : sympy expressions
        Interpolation variables corresponding to axis0, axis1, axis2.
    log_axes : list of bool, optional
        Whether each axis is log-spaced.
    """
    register_table(name, data, axes, log_axes)
    return TableInterp3D(x, y, z, Str(name))
