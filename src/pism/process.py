# please autodoc me

"""Implementation of base Process class with methods for managing and solving systems of equations"""

from collections import defaultdict
import sympy as sp
import jax
import jax.numpy as jnp
from .numerics import newton_rootsolve
from .symbols import n_
from .misc import is_an_ion


class Process:
    """
    Top-level class containing a description of a microscopic process

    Most importantly, this implements the procedure for combining processes to build up a network for chemistry
    + conservation equations.
    """

    def __init__(self, name="", bibliography={}):
        """Construct an empty Process instance

        Parameters
        ----------
        name: str, optional
            Name of the process
        """
        self.name = name
        self.initialize_network()
        self.dust_heat = 0
        self.rate = 0
        self.heat = 0
        self.bibliography = bibliography
        self.subprocesses = [self]

    def __repr__(self):
        """Print the name in print()"""
        return self.name

    def __add__(self, other):
        """Sum 2 processes together: define a new process whose rates are the sum of the input process"""
        if other == 0:  # necessary for native sum() routine to work
            return self

        attrs_to_sum = "heat", "dust_heat", "subprocesses"  # all energy exchange terms

        sum_process = Process()
        sum_process.rate = None  # "rate" ceases to be meaningful for composite processes
        for summed_attr in attrs_to_sum:
            attr1, attr2 = getattr(self, summed_attr), getattr(other, summed_attr)
            if attr1 is None or attr2 is None:
                setattr(sum_process, summed_attr, None)
            else:
                setattr(sum_process, summed_attr, attr1 + attr2)

        # now combine the networks
        sum_process.network = self.combine_networks(self.network, other.network)
        sum_process.name = f"{self.name} + {other.name}"
        return sum_process

    def __radd__(self, other):
        return self.__add__(other)

    def initialize_network(self):
        self.network = defaultdict(int)  # this is a dict for which unknown keys are initialized to 0 by default

    def combine_networks(self, n1, n2):
        combined_network = defaultdict(int)
        combined_keys = set(tuple(n1.keys())).union(set(tuple(n2.keys())))  # gross?
        for k in combined_keys:
            combined_network[k] = n1[k] + n2[k]
        return combined_network

    def print_network_equations(self):
        """Prints the system of equations in the chemistry network"""
        for k, rhs in self.network.items():
            if k == "heat":
                lhs = sp.Symbol("d(⍴u)/dt")
            else:
                lhs = sp.Symbol(f"dn_{k}/dt")
            print(lhs, "=", rhs)

    def network_species(self):
        return list(self.network.keys())

    @property
    def network_ions(self):
        """Returns the list of ions involved in a process"""
        return [s for s in self.network if is_an_ion(s)]

    @property
    def network_reduction_replacements(self):
        """Replacements for reducing the chemistry network with conservation laws"""
        Y = sp.Symbol("Y")  # general: mass fractions of different atoms other than H. n_i,tot = n_i + sum(n_ions of i)
        nHtot = sp.Symbol("n_Htot")  # basically always want this

        substitutions = {
            n_("e-"): n_("H+") + n_("He+") + 2 * n_("He++"),
            # n_("He+"): n_("e-") - n_("H+") - 2 * n_("He++"),
            n_("H+"): nHtot - n_("H"),
            n_("He++"): Y / (4 - 4 * Y) * nHtot - sp.Symbol("n_He") - sp.Symbol("n_He+"),
        }
        return substitutions

    def apply_network_reductions(self, expr):
        """Applies the replacements given by network_reduction_replacements to a symbolic expression"""
        out = expr
        for _ in range(2):  # 2 passes to avoid ordering issues
            for n, r in self.network_reduction_replacements.items():
                out = out.subs(n, r)
        return out

    @property
    def reduced_network(self):
        """
        Returns the chemistry network after substituting known conservation laws:

        n_atom = sum(n_{species containing atom} * number of atoms in species)
        n_e- = sum(ion charge * n_ion) - want to keep n_e- in the explicit updates, so eliminate the highest ions?

        This reduces the network of N rate equations to N - (num_atoms + 1).
        """

        replacements = self.network_reduction_replacements

        reduced_network = {}
        for s, rhs in self.network.items():
            if n_(s) in replacements:
                continue
            else:
                rhs = self.apply_network_reductions(rhs)
            reduced_network[s] = rhs
        return reduced_network

    def get_thermochem_network(self, reduced=True):
        """Returns the network including all chemical processes plus the gas heating-cooling equation"""
        if reduced:
            network = self.reduced_network
        else:
            network = self.network
        return network | {"T": self.apply_network_reductions(self.heat)}  # combine the dicts

    def steadystate(
        self,
        known_quantities,
        guess,
        input_abundances=True,
        output_abundances=True,
        reduce_network=True,
        tol=1e-3,
        careful_steps=10,
    ):
        """
        Solves for equilibrium after substituting a set of known quantities, e.g. temperature, metallicity,
        etc.

        Parameters
        ----------
        known_quantities: dict
            Dict of symbolic quantities and their values that will be plugged into the network solve as known quantities.
            Can be arrays if you want to substitute multiple values. If T is included here, we solve for chemical
            equilibrium. If T is not included, solve for thermochemical equilibrium.
        guess: dict, optional
            Dict of symbolic quantities and their values that will be plugged into the network solve as guesses for the
            unknown quantities. Can be arrays if you want to substitute multiple values. Will default to trying sensible
            guesses for recognized quantities.
        normalize_to_H: bool, optional
            Whether to return abundances normalized by the number density of H nucleons (default: True)
        reduce_network: bool, optional
            Whether to solve the reduced version of the network substituting conservation laws (default: True)
        tol: float, optional
            Desired relative error in chemical abundances (default: 1e-3)
        careful_steps: int, optional
            Number of careful initial steps in the Newton solve before full step size is used - try increasing this if
            your solve has trouble converging.

        Returns
        -------
        equilibrium_abundances: dict
            Dict of species and their equilibrium abundances relative to H or raw number densities (depending on
            value of normalize_to_H)
        """
        if "T" in known_quantities:
            thermo = False  # do a chemistry solve with T fixed
            if reduce_network:
                network_tosolve = self.reduced_network
            else:
                network_tosolve = self.network
        else:
            thermo = True  # solve for equilibrium T as well
            network_tosolve = self.get_thermochem_network(reduced=reduce_network)

        self.do_solver_value_checks(known_quantities, guess)

        #        for k in known_quantities:
        #            if k in

        # need to implement broadcasting between knowns and guesses...
        # can supply just the species names, will convert to the number density symbol if necessary
        unknowns = [sp.Symbol(f"n_{i}") for i in self.reduced_network]
        if thermo:
            unknowns.append("T")
        known_variables = [sp.Symbol(k) if isinstance(k, str) else k for k in known_quantities]

        func = sp.lambdify(
            unknowns + known_variables, list(network_tosolve.values()), modules="jax"
        )  # , dummify=True)

        @jax.jit
        def f_numerical(X, *params):
            """JAX function to rootfind"""
            return jnp.array(func(*X, *params))

        # We also specify a function of the parameters to use for our stopping criterion:
        # converge electron and H abundance to desired tolerance.
        tolerance_vars = [self.apply_network_reductions(n_("e-")), n_("H")]
        if thermo:
            tolerance_vars += [sp.Symbol("T")]
        tolfunc = sp.lambdify(unknowns + known_variables, tolerance_vars, modules="jax")

        @jax.jit
        def tolerance_func(X, *params):
            """Solution will terminate if the relative change in this quantity is < tol"""
            return jnp.array(tolfunc(*X, *params))

        guesses = []
        for i in network_tosolve:
            guesses.append(guess[i])
            if input_abundances and i in self.network:
                guesses[-1] *= known_quantities["n_Htot"]  # convert to density
        guesses = jnp.array(guesses).T
        params = jnp.array(list(known_quantities.values())).T
        sol = newton_rootsolve(
            f_numerical, guesses, params, tolfunc=tolerance_func, rtol=tol, careful_steps=careful_steps
        )

        # get solution into dict form
        sol = {species: sol[:, i] for i, species in enumerate(network_tosolve)}
        # now get the missing species that we eliminated - this needs to be generalized...
        nHtot = known_quantities["n_Htot"]
        sol["H+"] = nHtot - sol["H"]
        sol["e-"] = sol["H+"]
        if "Y" in known_quantities:
            Y = known_quantities["Y"]
            y = Y / (4 - 4 * Y)
            sol["He++"] = y * nHtot - sol["He"] - sol["He+"]
            sol["e-"] += 2 * sol["He++"] + sol["He+"]

        if output_abundances:
            for species, n in sol.items():
                if species != "T":
                    sol[species] = n / nHtot
        return sol

    def do_solver_value_checks(self, known_quantities, guess):
        if not isinstance(known_quantities, dict):
            raise ValueError("known_quantities argument to chemical_equilibrium must be a dictionary.")
        lengths = [len(k) for k in known_quantities.values()]

        if guess is not None:
            if not isinstance(guess, dict):
                raise ValueError("If supplied, guess argument to chemical_equilibrium must be a dictionary.")
            lengths += [len(g) for g in guess.values()]
        if not all([l == lengths[0] for l in lengths]):
            raise ValueError("All known quantities and guesses must be arrays of equal length.")
