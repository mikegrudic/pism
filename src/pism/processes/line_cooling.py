import sympy as sp
from .nbody_process import NBodyProcess
from ..symbols import T, T5

# put analytic fits for cooling efficiencies
line_cooling_coeffs = {
    "H": {"e-": 7.5e-19 * sp.exp(-118348 / T) / (1 + sp.sqrt(T5))},
    "He+": {"e-": 5.54e-17 * T**-0.397 * sp.exp(-473638 / T) / (1 + sp.sqrt(T5))},
    "C+": {"e-": 1e-27 * 4890 / sp.sqrt(T) * sp.exp(-91.211 / T), "H": 1e-27 * 0.47 * T**0.15 * sp.exp(-91.211 / T)},
}


def LineCoolingSimple(emitter: str, collider: str) -> NBodyProcess:
    """Returns a 2-body process representing cooling via excitations from collisions of given pair of species

    This is the simple approximation where everything is well below critical density and no ambient radiation field.
    eventually would like to have a class that considers collisions from all available colliders, given just the
    energies, deexcitation coefficients, temperature, and statistical weights...
    """

    process = NBodyProcess({emitter, collider})
    if emitter not in line_cooling_coeffs:
        raise NotImplementedError(f"Line cooling not implemented for {emitter}")
    elif collider not in line_cooling_coeffs[emitter]:
        raise NotImplementedError(f"Excitation by collisions with {collider} not implemented for {emitter}")

    process.heat_rate_coefficient = -line_cooling_coeffs[emitter][collider]
    process.name = f"{emitter}-{collider} Line Cooling"

    return process


# def LineCooling(emitter: str)
