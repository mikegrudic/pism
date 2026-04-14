"""Implementation of EquationSystem for representing, manipulating, and constructing systems of conservation laws"""

import sympy as sp
from .species_strings import species_mass, species_charge, species_counts, total_atom_abundance
from .symbols import d_dt, dt, n_, x_, t, BDF, n_Htot, sanitize_symbols, sanitize_name
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
            # Some species (e.g. metals appearing only in heat terms) are
            # detected by chemical_species but have no evolution equation.
            self.pop(i, None)

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
            if "u" not in solve_vars:
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

    def generate_code(self, solve_vars, time_dependent=[], language="c", jac=True, do_cse=True,
                      minimal=True, func_name="microphysics_func_jac", jac_mode="symbolic"):
        """Generates numerical code that implements the system RHS and/or Jacobian.

        Parameters
        ----------
        minimal : bool
            If True (default), return a bare code string (original behavior).
            If False, return a dict with self-documenting code, C/C++ headers, enums, etc.
        func_name : str
            Name of the generated function (only used when minimal=False)
        jac_mode : str
            'symbolic' or 'autodiff' (only used when minimal=False; autodiff requires C++ or CUDA)
        """
        func, jac_expr, indices = self.solver_functions(solve_vars, time_dependent, return_jac=jac)

        if minimal:
            return self._generate_minimal(func, jac_expr, indices, language, jac, do_cse)
        else:
            return self._generate_full(func, jac_expr, indices, language, do_cse, func_name, jac_mode)

    def _generate_minimal(self, func, jac, indices, language, has_jac, do_cse):
        """Original generate_code behavior: return a bare code string."""
        func, jac = sanitize_symbols(func), sanitize_symbols(jac)

        from jaco.codegen.printers import (
            JacoCCodePrinter, JacoCppCodePrinter, JacoCudaCodePrinter,
            JacoFCodePrinter, JacoPythonCodePrinter, JacoJuliaCodePrinter,
        )

        match language.lower():
            case "fortran": code_printer = JacoFCodePrinter()
            case "c": code_printer = JacoCCodePrinter()
            case "c++": code_printer = JacoCppCodePrinter()
            case "cuda": code_printer = JacoCudaCodePrinter()
            case "python": code_printer = JacoPythonCodePrinter()
            case "julia": code_printer = JacoJuliaCodePrinter()

        def printer(x):
            return code_printer.doprint(x)

        codeblocks = []

        header = "Computes the RHS function "
        if has_jac:
            header += "and Jacobian "
        header += f"to solve for {list(indices.keys())}\n\n"
        header += (
            "This code was auto-generated by jaco v" + version("jaco")
            + " and is not intended to be modified or maintained by human beings.\n\n"
        )
        header += "INDEX CONVENTION: " + " ".join(f"({i}: {s})" for s, i in indices.items()) + "\n"
        header = printer(Comment(header))
        codeblocks.append(header)

        if do_cse:
            cse, (func, jac) = sp.cse((sp.Matrix(func), sp.Matrix(jac)))
            block = [printer(Assignment(*expr)) for expr in cse]
            codeblocks.append(" \n".join(block))

        n = len(func)
        if language.lower() in ("python", "julia"):
            codeblocks.append("\n".join(f"rhs_result[{i}] = {printer(func[i])}" for i in range(n)))
            if has_jac:
                codeblocks.append("\n".join(
                    f"jac_result[{i},{j}] = {printer(jac[i, j])}" for i in range(n) for j in range(n)))
        else:
            rhs_result = sp.MatrixSymbol("rhs_result", n, 1)
            codeblocks.append(printer(Assignment(rhs_result, func)))
            if has_jac:
                jac_result = sp.MatrixSymbol("jac_result", n, n)
                codeblocks.append(printer(Assignment(jac_result, jac)))

        code = "\n\n".join(codeblocks)

        table_preamble = ""
        match language.lower():
            case "c": table_preamble = code_printer.get_table_declarations_c()
            case "c++": table_preamble = code_printer.get_table_declarations_cpp()
            case "cuda": table_preamble = code_printer.get_table_declarations_cuda()
            case "fortran": table_preamble = code_printer.get_table_declarations_fortran()
            case "python": table_preamble = code_printer.get_table_declarations_python()
            case "julia": table_preamble = code_printer.get_table_declarations_julia()
        if table_preamble:
            code = table_preamble + "\n\n" + code

        return code

    def _generate_full(self, func, jac, indices, language, do_cse, func_name, jac_mode):
        """Generate self-documenting code with named variables, split rhs/jac, enums, and headers."""
        from jaco.codegen.printers import (
            JacoCCodePrinter, JacoCudaCodePrinter,
            JacoPythonCodePrinter, JacoJuliaCodePrinter,
        )

        lang = language.lower()
        n_vars = len(indices)

        func_mat = sanitize_symbols(sp.Matrix(func))
        jac_mat = sanitize_symbols(sp.Matrix(jac).reshape(n_vars, n_vars))

        var_names = [sanitize_name(str(s)) for s in indices.keys()]
        var_name_set = set(var_names)
        all_syms = func_mat.free_symbols | jac_mat.free_symbols
        param_syms = sorted([s for s in all_syms if str(s) not in var_name_set], key=str)

        match lang:
            case "cuda": printer = JacoCudaCodePrinter()
            case "python": printer = JacoPythonCodePrinter()
            case "julia": printer = JacoJuliaCodePrinter()
            case _: printer = JacoCCodePrinter()

        if jac_mode == "autodiff":
            if lang == "c":
                raise ValueError("autodiff jac_mode requires C++ or CUDA (not C)")
            code = self._gen_autodiff_code(func_mat, var_names, param_syms, printer, do_cse, lang, func_name)
        else:
            code = self._gen_symbolic_code(func_mat, jac_mat, var_names, param_syms, printer, do_cse, lang, func_name)

        result = {"code": code, "func_name": func_name}

        match lang:
            case "cuda": result["interp_header"] = printer.get_table_declarations_cuda()
            case "python" | "julia": pass
            case _: result["interp_header"] = printer.get_table_declarations_c()

        if lang not in ("python", "julia"):
            result["header"] = self._gen_header(func_name, var_names, param_syms, n_vars, lang)

        result["var_names"] = var_names
        result["param_names"] = [str(p) for p in param_syms]
        result["n_vars"] = n_vars
        result["n_params"] = len(param_syms)
        result["language"] = language

        if lang == "c":
            result["eos_code"] = self._gen_eos(var_names, printer)

        return result

    def _gen_symbolic_code(self, func_mat, jac_mat, var_names, param_syms, printer, cse, lang, func_name):
        """Generate code string with explicit symbolic Jacobian."""
        is_scripting = lang in ("python", "julia")
        n_vars = len(var_names)

        if cse:
            cse_exprs, (func_cse, jac_cse) = sp.cse((func_mat, jac_mat))
        else:
            cse_exprs, func_cse, jac_cse = [], func_mat, jac_mat

        if is_scripting:
            ind, comment, semi = "    ", "#", ""
        else:
            ind, comment, semi = "   ", "//", ";"

        jidx = (lambda i, j: f"jac[IDX_{i}, IDX_{j}]") if lang == "julia" \
            else (lambda i, j: f"jac[IDX_{i}][IDX_{j}]")

        lines = []
        lines.append(f"{ind}{comment} Solve variables")
        for var in var_names:
            if is_scripting:
                lines.append(f"{ind}{var} = X[IDX_{var}]")
            elif lang == "c":
                lines.append(f"{ind}const double {var} = vars->{var};")
            else:
                lines.append(f"{ind}const double {var} = X[IDX_{var}];")

        lines.append(f"{ind}{comment} Parameters")
        for p in param_syms:
            if is_scripting:
                lines.append(f"{ind}{p} = params[PARAM_{p}]")
            elif lang == "c":
                lines.append(f"{ind}const double {p} = params->{p};")
            else:
                lines.append(f"{ind}const double {p} = params[PARAM_{p}];")
        lines.append("")

        for sym, expr in cse_exprs:
            if is_scripting:
                lines.append(f"{ind}{printer.doprint(sym)} = {printer.doprint(expr)}")
            else:
                lines.append(f"{ind}const double {printer.doprint(sym)} = {printer.doprint(expr)};")
        lines.append("")

        for i, var in enumerate(var_names):
            if lang == "c":
                lines.append(f"{ind}rhs->{var} = {printer.doprint(func_cse[i])}{semi}")
            else:
                lines.append(f"{ind}rhs[IDX_{var}] = {printer.doprint(func_cse[i])}{semi}")
        lines.append("")

        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                lines.append(f"{ind}{jidx(vi, vj)} = {printer.doprint(jac_cse[i, j])}{semi}  {comment} d(rhs_{vi})/d({vj})")

        body = "\n".join(lines)
        return self._wrap_source(body, func_name, lang, var_names, param_syms, n_vars, printer)

    def _gen_autodiff_code(self, func_mat, var_names, param_syms, printer, cse, lang, func_name):
        """Generate code string with forward-mode autodiff Jacobian."""
        n_vars = len(var_names)
        device = "__device__ " if lang == "cuda" else ""

        if cse:
            cse_exprs, (func_cse,) = sp.cse((func_mat,))
        else:
            cse_exprs, func_cse = [], func_mat

        rhs_lines = []
        rhs_lines.append("   // Solve variables")
        for var in var_names:
            rhs_lines.append(f"   const Scalar {var} = X[IDX_{var}];")
        rhs_lines.append("   // Parameters")
        for p in param_syms:
            rhs_lines.append(f"   const Scalar {p} = params[PARAM_{p}];")
        rhs_lines.append("")
        for sym, expr in cse_exprs:
            rhs_lines.append(f"   const Scalar {printer.doprint(sym)} = {printer.doprint(expr)};")
        rhs_lines.append("")
        for i, var in enumerate(var_names):
            rhs_lines.append(f"   rhs[IDX_{var}] = {printer.doprint(func_cse[i])};")
        rhs_body = "\n".join(rhs_lines)

        math_include = "#include <math.h>" if lang == "cuda" else "#include <cmath>"
        return f"""{math_include}
#include "{func_name}.h"
#include "jaco_interp.h"
#include "jaco_dual.h"

template <typename Scalar>
{device}void {func_name}_rhs(const Scalar* X, const Scalar* params, Scalar* rhs) {{
{rhs_body}
}}

template {device}void {func_name}_rhs<double>(const double*, const double*, double*);

{device}void {func_name}(const double* X, const double* params, double* rhs, double jac[{n_vars}][{n_vars}]) {{
   {func_name}_rhs(X, params, rhs);

   dual X_d[N_VARS], params_d[N_PARAMS], rhs_d[N_VARS];
   for (int i = 0; i < N_PARAMS; i++) params_d[i] = dual(params[i]);
   for (int col = 0; col < N_VARS; col++) {{
      for (int i = 0; i < N_VARS; i++) X_d[i] = dual(X[i], i == col ? 1.0 : 0.0);
      {func_name}_rhs(X_d, params_d, rhs_d);
      for (int i = 0; i < N_VARS; i++) jac[i][col] = rhs_d[i].dot;
   }}
}}
"""

    def _wrap_source(self, body, func_name, lang, var_names, param_syms, n_vars, printer):
        """Wrap a function body with language-appropriate boilerplate."""
        if lang == "python":
            preamble = printer.get_table_declarations_python() if printer else ""
            idx_lines = [f"IDX_{v} = {i}" for i, v in enumerate(var_names)]
            idx_lines += [f"PARAM_{p} = {i}" for i, p in enumerate(param_syms)]
            imports = "import math"
            if preamble:
                imports += "\nimport numpy as np"
            return f"""{imports}
{preamble}

N_VARS = {n_vars}
N_PARAMS = {len(param_syms)}
{chr(10).join(idx_lines)}

def {func_name}(X, params, rhs, jac):
{body}
"""
        elif lang == "julia":
            preamble = printer.get_table_declarations_julia() if printer else ""
            idx_lines = [f"const IDX_{v} = {i + 1}" for i, v in enumerate(var_names)]
            idx_lines += [f"const PARAM_{p} = {i + 1}" for i, p in enumerate(param_syms)]
            return f"""{preamble}

const N_VARS = {n_vars}
const N_PARAMS = {len(param_syms)}
{chr(10).join(idx_lines)}

function {func_name}(X, params, rhs, jac)
{body}
end
"""
        else:
            if lang == "c":
                c_sig = f"void {func_name}(const SolveVars *vars, const Params *params, SolveVars *rhs, double jac[N_VARS][N_VARS])"
                return f'#include <math.h>\n#include "{func_name}.h"\n#include "jaco_interp.h"\n\n{c_sig} {{\n{body}\n}}\n'
            else:
                c_sig = f"void {func_name}(const double* X, const double* params, double* rhs, double jac[N_VARS][N_VARS])"
                if lang == "cuda":
                    return f'#include <math.h>\n#include "{func_name}.h"\n#include "jaco_interp.h"\n\n__device__ {c_sig} {{\n{body}\n}}\n'
                else:
                    return f'#include <cmath>\n#include "{func_name}.h"\n#include "jaco_interp.h"\n\n{c_sig} {{\n{body}\n}}\n'

    def _gen_header(self, func_name, var_names, param_syms, n_vars, lang):
        """Generate C/C++/CUDA header content with enums, structs, and declarations."""
        is_cuda = lang == "cuda"
        is_c = lang == "c"
        device = "__device__ " if is_cuda else ""
        param_names = [str(p) for p in param_syms]

        lines = ["#pragma once", ""]
        var_entries = ", ".join(f"IDX_{v} = {i}" for i, v in enumerate(var_names))
        lines.append(f"enum SolveVarIndex {{ {var_entries}, N_VARS = {n_vars} }};")
        param_entries = ", ".join(f"PARAM_{p} = {i}" for i, p in enumerate(param_names))
        lines.append(f"enum ParamIndex {{ {param_entries}, N_PARAMS = {len(param_names)} }};")
        lines.append("")

        if is_c:
            lines.append("typedef union {")
            lines.append("    struct {")
            for v in var_names:
                lines.append(f"        double {v};")
            lines.append("    };")
            lines.append(f"    double data[{n_vars}];")
            lines.append("} SolveVars;")
            lines.append("")
            lines.append("typedef union {")
            lines.append("    struct {")
            for p in param_names:
                lines.append(f"        double {p};")
            lines.append("    };")
            lines.append(f"    double data[{len(param_names)}];")
            lines.append("} Params;")
        else:
            lines.append(f"struct SolveVars {{")
            for v in var_names:
                lines.append(f"    double {v};")
            lines.append(f"    {device}const double* data() const {{ return &{var_names[0]}; }}")
            lines.append(f"    {device}double* data() {{ return &{var_names[0]}; }}")
            lines.append("};")
            lines.append("")
            lines.append(f"struct Params {{")
            for p in param_names:
                lines.append(f"    double {p};")
            lines.append(f"    {device}const double* data() const {{ return &{param_names[0]}; }}")
            lines.append("};")
        lines.append("")

        if is_c:
            ptr_sig = f"void {func_name}(const SolveVars *vars, const Params *params, SolveVars *rhs, double jac[N_VARS][N_VARS])"
        else:
            ptr_sig = f"void {func_name}(const double* X, const double* params, double* rhs, double jac[N_VARS][N_VARS])"
        lines.append(f"__device__ {ptr_sig};" if is_cuda else f"{ptr_sig};")
        lines.append("")

        if not is_c:
            lines.append("// Convenience overload: accepts SolveVars/Params structs directly")
            lines.append(f"{device}inline void {func_name}(const SolveVars& X, const Params& params, SolveVars& rhs, double jac[N_VARS][N_VARS]) {{")
            lines.append(f"    {func_name}(X.data(), params.data(), rhs.data(), jac);")
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _gen_eos(self, var_names, printer):
        """Generate C code for jaco EOS functions.

        Produces:
          - jaco_eos_pressure(sv, pr)  — P = sum(n_s * k_B * T) in CGS
          - jaco_T_to_u(T, pr, *cv)   — specific internal energy u(T) in CGS, with cv output
          - jaco_u_to_T(u, pr)         — Newton-Raphson inversion of u(T)

        Rewrites species number densities n_s as n_Htot * x_s so the expression
        uses quantities that exist in the SolveVars/Params structs.
        Returns the full file content as a string.
        """
        from .eos.eos import k_B_cgs, EOS
        from .symbols import T as T_sym

        species = self.chemical_species
        eos = EOS(species) if species else None

        var_name_set = set(var_names)

        def _unpack_syms(syms, src_sv="sv", src_pr="pr", skip=frozenset()):
            """Generate local variable declarations unpacking symbols from structs.
            If src_sv is None, all symbols are read from src_pr."""
            decls = []
            for s in sorted(syms, key=str):
                name = sanitize_name(str(s))
                if name in skip:
                    continue
                if src_sv is not None and name in var_name_set:
                    decls.append(f'    const double {name} = {src_sv}->{name};')
                else:
                    decls.append(f'    const double {name} = {src_pr}->{name};')
            return decls

        def _rewrite_for_abundances(expr):
            """Substitute n_(s) -> n_Htot * x_(s) in an expression."""
            subs = {n_(s): n_Htot * x_(s) for s in species} if species else {}
            return expr.subs(subs)

        lines = []
        lines.append('/* Generated by jaco codegen — EOS functions from species contributions. */')
        lines.append('#include "microphysics_func_jac.h"')
        lines.append('#include <math.h>')
        lines.append('#include <stdio.h>')
        lines.append('')

        # --- jaco_eos_pressure ---
        if not species:
            P_expr = n_Htot * x_("H") * k_B_cgs * T_sym
        else:
            P_expr = _rewrite_for_abundances(eos.pressure)
        P_code = printer.doprint(sanitize_symbols(P_expr))
        P_syms = P_expr.free_symbols

        lines.append('double jaco_eos_pressure(const SolveVars *sv, const Params *pr) {')
        lines.extend(_unpack_syms(P_syms, skip={"T"}))
        lines.append('    const double T = sv->T;')
        lines.append(f'    return {P_code};')
        lines.append('}')
        lines.append('')

        # --- jaco_T_to_u: u(T) and cv(T) ---
        if not species:
            # Monatomic ideal H: u = 3/2 k_B T / m_H, cv = 3/2 k_B / m_H
            m_H = species_mass("H")
            u_expr = sp.Rational(3, 2) * k_B_cgs * T_sym / m_H
            cv_expr = sp.Rational(3, 2) * k_B_cgs / m_H
        else:
            u_expr = _rewrite_for_abundances(eos.internal_energy)
            cv_expr = _rewrite_for_abundances(eos.heat_capacity)

        # Apply CSE for potentially complex expressions (e.g. H2 partition function)
        cse_pairs, reduced = sp.cse(sanitize_symbols([u_expr, cv_expr]))
        u_reduced = reduced[0]
        cv_reduced = reduced[1]

        u_syms = u_expr.free_symbols | cv_expr.free_symbols
        lines.append('double jaco_T_to_u(double T, const Params *pr, double *cv_out) {')
        lines.extend(_unpack_syms(u_syms, src_sv=None, skip={"T"}))
        for sym, expr in cse_pairs:
            lines.append(f'    const double {printer.doprint(sym)} = {printer.doprint(expr)};')
        lines.append(f'    if (cv_out) {{ *cv_out = {printer.doprint(cv_reduced)}; }}')
        lines.append(f'    return {printer.doprint(u_reduced)};')
        lines.append('}')
        lines.append('')

        # --- jaco_u_to_T: Newton-Raphson inversion ---
        lines.append('double jaco_u_to_T(double u, const Params *pr) {')
        lines.append('    double T = u * %.16e;' % (species_mass("H") / (1.5 * k_B_cgs)))
        lines.append('    double cv, du, T_lo = 1e-3, T_hi = 1e12;')
        lines.append('    for (int iter = 0; iter < 100; iter++) {')
        lines.append('        double u_of_T = jaco_T_to_u(T, pr, &cv);')
        lines.append('        du = u_of_T - u;')
        lines.append('        if (fabs(du) < 1e-10 * fabs(u)) return T;')
        lines.append('        if (du > 0) T_hi = fmin(T_hi, T); else T_lo = fmax(T_lo, T);')
        lines.append('        double dT = -du / cv;')
        lines.append('        double T_new = T + dT;')
        lines.append('        if (T_new <= T_lo || T_new >= T_hi) T_new = sqrt(T_lo * T_hi);')
        lines.append('        T = T_new;')
        lines.append('    }')
        lines.append('    printf("jaco_u_to_T failed to converge: u=%g T=%g du=%g\\n", u, T, du);')
        lines.append('    return T;')
        lines.append('}')
        lines.append('')
        return '\n'.join(lines)
