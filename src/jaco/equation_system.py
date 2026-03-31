"""Implementation of EquationSystem for representing, manipulating, and constructing systems of conservation laws"""

import sympy as sp
from .species_strings import species_mass, species_charge, species_counts, total_atom_abundance
from .symbols import d_dt, dt, n_, x_, t, BDF, n_Htot, sanitize_symbols
from .eos import EOS
from .data import SolarAbundances
from .data.atoms import atoms
from jax import numpy as jnp

# import jax
from importlib.metadata import version

# jax.config.update("jax_enable_x64", True)
import numpy as np
from .numerics import newton_rootsolve
from astropy import units
from .equation import Equation
from sympy.codegen.ast import Assignment, Comment


class EquationSystem(dict):
    """Dict of symbolic expressions with certain superpowers for manipulating sets of conservation equations."""

    def copy(self):
        new = EquationSystem()
        for k in self:
            new[k] = self[k]
        return new

    def __getitem__(self, __key: str):
        """Dict getitem method where we initialize a differential equation for the conservation of a species if the key
        does not exist"""
        if __key not in self:
            self.__setitem__(__key, Equation(d_dt(n_(__key)), 0))  # technically should only be n_ if this is a species
            # need to make sure that d/dt's don't add up when composing equations
        return super().__getitem__(__key)

    def __add__(self, other):
        """Return a dict whose values are the sum of the values of the operands"""
        keys = self.keys() | other.keys()
        new = EquationSystem()
        for k in keys:
            new[k] = self[k] + other[k]
        return new

    @property
    def symbols(self):
        """Returns the set of all symbols in the equations"""
        all = set()
        for e in self.values():
            all.update(e.free_symbols)
            if e.lhs.atoms(sp.Function):  # yoink the n_ out of the LHS
                all.update([str(e.lhs.atoms(sp.Function)).replace("(t)", "").replace("{", "").replace("}", "")])
        if t in all:  # leave time out
            all.remove(t)
        return all

    @property
    def jacobian(self):
        """Returns a dict of dicts representing the Jacobian of the RHS of the system. Keys are the names of the
        conserved quantities and subkeys are the variable of differentiation.
        """
        return {k: {s: sp.diff(e.rhs, s) for s in self.symbols} for k, e in self.items()}

    def subs(self, expr, replacement):
        """Substitute symbolic expressions throughout the whole network."""
        for k, e in self.items():
            self[k] = e.subs(expr, replacement)

    def reduced(self, knowns, time_dependent=[]):
        subsystem = self.copy()
        subsystem.set_time_dependence(time_dependent)
        subsystem.do_conservation_reductions(time_dependent)
        if "T" in (str(k) for k in knowns) and "T" not in time_dependent:
            del subsystem["heat"]
        return subsystem

    def set_time_dependence(self, time_dependent_vars):
        """Insert backward-difference formulae or set to steady state"""
        # put in backward differences
        for q in self:
            if q in time_dependent_vars:  # insert backward-difference formula
                self[q] = Equation(BDF(q), self[q].rhs)
            else:
                self[q] = Equation(0, self[q].rhs)

        if "T" in time_dependent_vars:  # special behaviour
            self["heat"] = Equation(
                self.eos.density * (self.eos.internal_energy - sp.Symbol("u_0")) / dt, self["heat"].rhs
            )
            if "u" not in self:
                self["u"] = Equation(0, sp.Symbol("u") - self.eos.internal_energy)

    def do_conservation_reductions(self, time_dependent_vars):
        """Eliminate equations from the system using known conservation laws."""
        self.substitutions = []

        # since we have n_Htot let's convert all other n's to x's
        for s in self.symbols:
            if str(s)[:2] == "n_" and "Htot" not in str(s):
                self.substitutions.append((s, n_Htot * sp.Symbol("x_" + str(s)[2:])))
            # TODO: when we do this we must also fix the LHS of the evolution equation.
            # n_i = n_H x_i -> d(n_i)/dt = x_i d(n_H)/dt + n_H d(x_i)/dt (Lagrangian density evolution term)
            # d(x_i)/dt = d/dt (n_i/n_H) = -x_i dlog(n_H)/dt + d(n_i)/dt / n_H

        # charge neutrality
        if n_("e-") in self.symbols and "e-" not in time_dependent_vars:
            x_ion_sum = 0
            for s in self.chemical_species:
                if s == "e-":
                    continue
                x_ion_sum += species_charge(s) * x_(s)
            self.substitutions.append((x_("e-"), x_ion_sum))
            del self["e-"]

        # atom species conservation
        counts = {j: species_counts(j) for j in self.chemical_species}
        for i in self.chemical_species:
            if i not in atoms:  # i is an atom
                continue
            # get total abundance of species
            x_total = 0
            for j in self.chemical_species:
                if j == i:
                    continue
                if i in counts[j]:
                    x_total += counts[j][i] * x_(j)

            if x_total == 0:  # no other species containing i found
                continue

            xtot = total_atom_abundance(i)
            self.substitutions.append((x_(i), xtot - x_total))
            del self[i]

        for expr, sub in self.substitutions:
            self.subs(expr, sub)

    @property
    def rhs(self):
        """Return as dict of rhs-lhs instead of equations"""
        return {k: e.rhs - e.lhs for k, e in self.items()}

    @property
    def rhs_scaled(self):
        """Returns a scaled version of the the RHS pulling out the usual factors affecting collision rates"""
        return [r for r in self.rhs.values()]

    @property
    def eos(self):
        return EOS(self.chemical_species)

    @property
    def chemical_species(self):
        """Returns a tuple of all chemical species detected within the network"""
        # strategy: look for things with n_ or x_, but not photons
        species = set()
        for s in self.symbols:
            if not ("x_" in str(s) or "n_" in str(s)):
                continue
            s = str(s).replace("x_", "").replace("n_", "")
            if species_mass(s) != 0:
                species.add(s)
        return tuple(species)

    def solve(
        self,
        knowns,
        guesses,
        time_dependent=[],
        dt=None,
        verbose=False,
        tol=1e-3,
        careful_steps=20,
        symbolic_keys=False,
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
        guesses: dict
            Dict of symbolic quantities and their values that will be plugged into the network solve as guesses for the
            unknown quantities. Can be arrays if you want to substitute multiple values. Will default to trying sensible
            guesses for recognized quantities (NOT IMPLEMENTED YET)
        tol: float, optional
            Desired relative error in chemical abundances (default: 1e-3)
        careful_steps: int, optional
            Number of careful initial steps in the Newton solve before full step size is used - try increasing this if
            your solve has trouble converging.

        Returns
        -------
        soldict: dict
            Dict of species and their equilibrium abundances relative to H or raw number densities (depending on
            value of normalize_to_H)
        """

        def printv(*a, **k):
            """Print only if locally verbose=True"""
            if verbose:
                print(*a, **k)

        # first: check knowns and guesses are all same size
        num_params = np.array([len(np.array(guesses[g])) for g in guesses] + [len(np.array(knowns[g])) for g in knowns])
        if not np.all(num_params == num_params[0]):
            raise ValueError("Input parameters and initial guesses must all have the same shape.")
        num_params = num_params[0]

        if dt is not None:
            knowns["Δt"] = np.repeat(dt.to(units.s), num_params)

        if "u" in guesses or "T" in time_dependent:
            self["u"] = Equation(0, self.eos.internal_energy - sp.Symbol("u"))
        subsystem = self.reduced(knowns, time_dependent)
        symbols = subsystem.symbols
        num_equations = len(subsystem)

        # are there any symbols for which we can make a reasonable assumption or directly solve the steady-state approximation?
        # TODO: need to reject symbols in knowns that are not found in the network
        prescriptions = {"y": SolarAbundances.x("He"), "Y": SolarAbundances.mass_fraction["He"], "Z": 1.0, "C_2": 1.0}
        assumed_values = {}
        if len(symbols) > num_equations + len(knowns):
            undetermined_symbols = symbols.difference(set(sp.Symbol(g) for g in guesses))
            printv(f"Undetermined symbols: {undetermined_symbols}")
            for s in undetermined_symbols:
                # if we have a prescription for this quantity, plug it in here. This should eventually be specified at the model level.
                if str(s) in prescriptions:
                    # case 1: we have given a value, which we should add to the list of knowns
                    assumed_values[str(s)] = np.repeat(prescriptions[str(s)], num_params)
                    printv(f"{s} not specified; assuming {s}={prescriptions[str(s)]}.")
                    symbols = subsystem.symbols
                    # case 2: we have given an expression in terms of the other available quantities: we need to subs it

        # ok now we should have number of symbols unknowns + knowns
        printv(
            f"Free symbols: {symbols}\nKnown values: {list(knowns)}\nAssumed values: {list(assumed_values)}\nEquations solved: {list(subsystem.rhs)}"
        )
        if len(symbols) != len(knowns | assumed_values) + len(subsystem):
            raise ValueError(
                f"Number of free symbols is {len(symbols)} != number of knowns {len(knowns)} + number of assumptions {len(assumed_values)} + number of equations {len(subsystem)}\n"
            )
        else:
            printv(
                f"It's solvin time. Solving for {set(guesses)} based on input {set(knowns)} and assumptions about {set(assumed_values)}"
            )

        guessvals = {}
        paramvals = {}
        for s in subsystem.symbols:
            for g in guesses:
                if g == str(s) or f"x_{g}" == str(s):
                    guessvals[s] = guesses[g]
            for k in knowns | assumed_values:
                if k == str(s) or f"x_{k}" == str(s):
                    paramvals[s] = (knowns | assumed_values)[k]

        # tuple: first is list of solved variables, second is list of known parameters
        lambda_args = (list(guessvals.keys()), list(paramvals.keys()))
        # bounds = len(lambda_args[0]) * [sp.oo]  # default upper bound is infinity
        # for i, x in enumerate(lambda_args[0]):
        #     if "x_" in str(x) and str(x).split("x_")[1] in self.chemical_species:
        #         bounds[i] = species_max_abundance(str(x).split("x_")[1])

        def lambdify(expr):
            """Turns an expression into a function for numerical evaluation"""
            return sp.lambdify(sanitize_symbols(lambda_args), sanitize_symbols(expr), modules="jax", cse=True)

        func = lambdify(subsystem.rhs_scaled)
        #        maxfunc = lambdify(bounds)

        tolerance_vars = [x_(s) for s in subsystem.chemical_species]  # default: converge on all abundances
        if "T" in guesses:
            tolerance_vars += [sp.Symbol("T")]
        if "u" in guesses:
            tolerance_vars += [sp.Symbol("u"), subsystem["heat"].rhs]
            # , subsystem["heat"]]  # converge on the internal energy and  cooling rate
        tolfunc = lambdify(tolerance_vars)  # sp.lambdify(lambda_args, tolerance_vars, modules="jax", cse=True)

        def f_numerical(X, *params):
            """JAX function to rootfind"""
            return jnp.array(func(X, params))

        def tolerance_func(X, *params):
            """Solution will terminate if the relative change in this quantity is < tol"""
            return jnp.array(tolfunc(X, params))

        # def max_func(X, *params):
        #     """Function that returns upper bounds"""
        #     return jnp.array(maxfunc(X, params))

        sol, num_iter = newton_rootsolve(
            f_numerical,
            jnp.array([g for g in guessvals.values()]).T,
            jnp.array([p for p in paramvals.values()]).T,
            tolfunc=tolerance_func,
            #            maxfunc=max_func,
            rtol=tol,
            careful_steps=careful_steps,
            nonnegative=True,
            return_num_iter=True,
        )
        printv(f"num_iter average={num_iter.mean()} min={num_iter.min()} max={num_iter.max()}")

        soldict = self.package_solution(sol, guessvals, guesses, paramvals, subsystem, symbolic_keys)

        return soldict

    def package_solution(self, sol, guessvals, guesses, paramvals, subsystem, symbolic_keys):
        """Takes the output of newton_rootsolve and packages it into a dict containing the system solution for all variables,
        including substitutions for eliminated variables.
        """
        # now repack the solution
        soldict = {}
        for i, g in enumerate(guessvals):
            soldict[g] = sol[:, i]
        # do a reverse-pass on the substitutions we made to get all quantities
        values_to_subs = soldict | paramvals
        for expr, sub in reversed(subsystem.substitutions):
            if expr in soldict:
                continue
            if "n_" in str(expr):
                continue
            soldict[expr] = sp.lambdify(list(sub.free_symbols), sub)(
                *[values_to_subs[s] for s in list(sub.free_symbols)]
            )  # should probably make a function of this
            values_to_subs |= soldict
        if not symbolic_keys:
            soldict = {str(k): v for k, v in soldict.items()}
            # if we have a bunch of x_'s, should also link up keys in the original input format, e.g. H->x_H
            if np.any(["x_" in k for k in guesses]):  # if we specified abundances with x_ notation, return same
                return soldict
            soldict2 = {}  # otherwise return with input format where keys are simple species strings
            for k in soldict:
                if "x_" in str(k):
                    soldict2[str(k).replace("x_", "")] = soldict[k]
                else:
                    soldict2[k] = soldict[k]
            soldict = soldict2

        return soldict

    def solver_functions(self, solve_vars, time_dependent=[], return_jac=False, return_dict=False):
        """Returns the RHS of the system to solve and its Jacobian, applying simplifications"""

        solve_vars = list(solve_vars)
        if "u" in solve_vars or "T" in time_dependent:
            self["u"] = Equation(0, self.eos.internal_energy - sp.Symbol("u"))
            solve_vars.append("u")

        knowns = self.symbols.difference(solve_vars)
        subsystem = self.reduced(knowns, time_dependent)

        rhs = {}
        for s in subsystem.symbols:
            for g in solve_vars:
                if str(s) == "T" and "T" in solve_vars:
                    rhs[s] = subsystem.rhs["heat"]
                elif str(g) == str(s) or f"x_{g}" == str(s):
                    rhs[s] = subsystem.rhs[g]

        # Sort rhs to match solve_vars order for deterministic output.
        # Map each rhs key (a Symbol) back to its position in solve_vars.
        def _var_sort_key(sym):
            name = str(sym)
            for i, g in enumerate(solve_vars):
                if name == g or name == f"x_{g}" or (name == "T" and g == "T"):
                    return i
            return len(solve_vars)
        rhs = dict(sorted(rhs.items(), key=lambda kv: _var_sort_key(kv[0])))

        if return_jac:
            jac = {}
            for s, expr in rhs.items():
                jac[s] = {s2: sp.diff(expr, s2) for s2 in rhs}

            if return_dict:
                return rhs, jac
            else:
                return (
                    list(rhs.values()),
                    [[jac[s1][s2] for s2 in rhs] for s1 in jac],
                    {s: i for i, s in enumerate(rhs)},
                )

        if return_dict:
            return rhs
        else:
            return list(rhs.values()), {s: i for i, s in enumerate(rhs)}

    def generate_code(self, solve_vars, time_dependent=[], language="Fortran", jac=True, do_cse=True):
        """Generates numerical code that implements the system RHS and/or Jacobian in the specified language."""
        func, jac, indices = self.solver_functions(solve_vars, time_dependent, return_jac=jac)

        func, jac = sanitize_symbols(func), sanitize_symbols(jac)

        from jaco.codegen.printers import (
            JacoCCodePrinter,
            JacoCppCodePrinter,
            JacoCudaCodePrinter,
            JacoFCodePrinter,
            JacoPythonCodePrinter,
            JacoJuliaCodePrinter,
        )

        def _get_printer(language):
            match language.lower():
                case "fortran":
                    return JacoFCodePrinter()
                case "c":
                    return JacoCCodePrinter()
                case "c++":
                    return JacoCppCodePrinter()
                case "cuda":
                    return JacoCudaCodePrinter()
                case "python":
                    return JacoPythonCodePrinter()
                case "julia":
                    return JacoJuliaCodePrinter()

        code_printer = _get_printer(language)

        def printer(x, language="c"):
            return code_printer.doprint(x)

        codeblocks = []

        header = "Computes the RHS function "
        if jac:
            header += "and Jacobian "
        header += f"to solve for {list(indices.keys())}\n\n"

        header += (
            "This code was auto-generated by jaco v"
            + version("jaco")
            + """ and is not intended to be modified or maintained by human beings.

            """
        )

        header += "INDEX CONVENTION: " + " ".join(f"({i}: {s})" for s, i in indices.items()) + "\n"

        header = printer(Comment(header), language)
        codeblocks.append(header)

        if do_cse:
            cse, (func, jac) = sp.cse((sp.Matrix(func), sp.Matrix(jac)))
            block = []
            for expr in cse:
                block.append(printer(Assignment(*expr), language))
            codeblocks.append(" \n".join(block))

        rhs_result = sp.MatrixSymbol("rhs_result", len(func), 1)
        codeblocks.append(printer(Assignment(rhs_result, func), language))

        if jac:
            jac_result = sp.MatrixSymbol("jac_result", len(func), len(func))
            codeblocks.append(printer(Assignment(jac_result, jac), language))

        code = "\n\n".join(codeblocks)

        # Prepend table declarations and helper functions
        table_preamble = ""
        if language.lower() == "c":
            table_preamble = code_printer.get_table_declarations_c()
        elif language.lower() == "c++":
            table_preamble = code_printer.get_table_declarations_cpp()
        elif language.lower() == "cuda":
            table_preamble = code_printer.get_table_declarations_cuda()
        elif language.lower() == "fortran":
            table_preamble = code_printer.get_table_declarations_fortran()
        if table_preamble:
            code = table_preamble + "\n\n" + code

        return code
