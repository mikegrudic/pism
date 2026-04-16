"""STARFORGE ISM thermochemistry model.

Full network including collisional ionization/recombination of H and He,
H2 formation/destruction, CO cooling, dust-gas collisions, cosmic ray
ionization, photoelectric heating, inverse Compton cooling, grain-assisted
recombination, and metal-line cooling from tabulated rates.

The model declares:

- **Fixed species** (``fixed_species``): C+, H-, H_2+, CO, HD — computed
  by the host code and passed as parameters, not solved by the Newton system.
- **Equilibrium overrides**: H_2+ = 0, HD = 2.527e-5 × x_H2 (deuterium locked in HD).
- **PdV work**: included as a ``pdv_work`` parameter in the heating term.
"""

from jaco.processes import CollisionalIonization, GasPhaseRecombination, FreeFreeEmission, ThermalProcess
from .line_cooling import LineCoolingSimple
from ..model import Model
import sympy as sp
from .h2_chemistry import H2_chemistry
from .H2_cooling import H2_cooling
from .CO_cooling import CO_cooling
from .gas_dust_collisions import gas_dust_collisions
from .cosmic_ray_ionization import cosmic_ray_ionization, cosmic_ray_photoionization
from .photoelectric_heating import photoelectric_heating
from .grain_assisted_recombination import grain_assisted_recombination
from .metal_line_cooling import metal_line_cooling
from jaco.processes import inv_compton_cooling


def make_model():
    """Build the STARFORGE thermochemistry model.

    Returns a :class:`Process` whose network contains all rate equations for
    the ISM species, with ``fixed_species`` and ``equilibrium_overrides``
    configured for use with GIZMO's ``gizmo_to_jaco`` interface.

    Returns
    -------
    Process
        Composite process with the full STARFORGE network.
    """
    processes = [
        H2_chemistry,
        *[LineCoolingSimple(s) for s in ("H", "He+", "C+")],
        *[CollisionalIonization(s) for s in ("H", "He", "He+")],
        *[GasPhaseRecombination(i) for i in ("H+", "He+", "He++")],
        *[FreeFreeEmission(i) for i in ("H+", "He+", "He++")],
        H2_cooling,
        CO_cooling,
        gas_dust_collisions,
        *[cosmic_ray_ionization(s) for s in ("H", "C")],
        cosmic_ray_photoionization("C"),
        photoelectric_heating,
        inv_compton_cooling,
        grain_assisted_recombination("C+"),
        metal_line_cooling(z=0.0),
        ThermalProcess(sp.Symbol("pdv_work"), name="PdV work"),
    ]

    model = sum(processes)

    # Equilibrium overrides for species that can't be auto-linearized.
    from jaco.symbols import x_
    model.network.equilibrium_overrides = {
        "H_2+": sp.S.Zero,                     # negligible in ISM conditions
        "HD": 2.527e-5 * x_("H_2"),            # all deuterium locked in HD (D/H = 2.527e-5, Cooke+ 2018)
    }

    # Fixed species: H_2+ negligible, HD locked in deuterium ratio.
    # C+, H-, CO are derived_params (functions of solve variables) — see below.
    model.network.fixed_species = {
        "H_2+": sp.S.Zero,
        "HD": 2.527e-5 * x_("H_2"),
    }

    # Derived parameters: expressions of solve variables that must be re-evaluated
    # each Newton iteration. Substituted into the equations before codegen so their
    # T-derivatives appear correctly in the Jacobian.
    from .symbols import T, grad_v, dx, n_Htot, G_0

    k_B = 1.380649e-16  # Boltzmann constant in CGS
    m_H = 1.6726e-24    # proton mass in CGS
    cs_sq = k_B * T / (2 * m_H)                    # sound speed squared for molecular H
    dv_sq = (grad_v * dx)**2                         # velocity dispersion squared across cell
    Mach_sq = dv_sq / cs_sq                          # Mach number squared
    C_2_expr = 1 + sp.Rational(1, 4) * Mach_sq      # <n^2>/<n>^2 for lognormal with b=0.5

    # --- C+ and CO: simple analytic estimate from GIZMO cooling.cc ---
    x_C_tot = sp.Symbol("x_C,tot")
    f_Cp = 1 / (1 + (n_Htot / (340 * sp.Max(sp.Rational(1, 10), G_0)))**2 / sp.sqrt(sp.Max(T, 10)))
    x_Cplus_expr = x_C_tot * f_Cp
    x_CO_expr = sp.Max(1e-30, x_C_tot * (1 - f_Cp) * x_("H_2"))

    # --- H-: equilibrium from rate equation d(n_Hm)/dt = 0 ---
    # The H- equation is linear in n_Hm: source + sink_coeff * n_Hm = 0
    # => x_Hm = -source / (sink_coeff * n_Htot)
    # Divide each term individually to ensure n_Hm cancels cleanly.
    nHm = sp.Symbol("n_H-")
    eq_rhs = model.network["H-"].rhs
    source = sp.S.Zero
    sink_coeff = sp.S.Zero
    for term in sp.Add.make_args(eq_rhs):
        if nHm not in term.free_symbols:
            source += term
        else:
            sink_coeff += sp.simplify(term / nHm)
    x_Hminus_expr = sp.Max(0, -source / sink_coeff / n_Htot)

    model.network.fixed_species.update({
        "C+": x_Cplus_expr,
        "H-": x_Hminus_expr,
        "CO": x_CO_expr,
    })

    model.network.derived_params = {
        "C_2": C_2_expr,
        "C_3": C_2_expr**3,                          # <n^3>/<n>^3 = C_2^3 for lognormal
    }

    return model

    # processes += sum([cosmic_ray_ionization(s) for s in ("H", "C")])
    # processes += sum([grain_assisted_recombination(s) for s in ("C+",)])
    # processes += photon_absorption
    # processes += dust_emission
    # processes += photoelectric_heating

    #    process = sum(processes)

    # assumption: H- in equilibrium
    #    collected = sp.collect(process.network"H-".rhs, n_("H-"))
    #    x_Hminus = collected.coeff(n_("H-"), 0) / collected.coeff(n_("H-"), 1)

    # assumption: C- in equilibrium

    # f_CO given by formula dependent on G0

    # assumption: dust energy in steady state

    # UV background with shieldfac


#    return sum(processes)


#     processes += sum(h2_chemistry.reactions)
# #    processes += CO_cooling(prescription="Whitworth+2018")

#     processes += photon_absorption(band) for band in "EUV", "FUV", "NUV", "OPT", "FIR"
#     processes += dust_emission(band) for band in "EUV", "FUV", "NUV", "OPT", "FIR"

# assumptions: x_D = 2.527e-5 x_H, x_D+ = 2.527e-5 x_H+, zero out associated collisional dissociation rates
