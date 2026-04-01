"""Routines for generating a cooling solver function for GIZMO."""

from jaco.models import starforge
import sympy as sp


def generate_funcjac_code(system, solve_vars=None, time_dependent=None,
                          cse=True, language="c", jac_mode="symbolic",
                          func_name="microphysics_func_jac"):
    """Generates self-documenting source files for the RHS + Jacobian function.

    Parameters
    ----------
    system : Process
        The system to generate code for.
    solve_vars : list, optional
        Variables to solve for. Defaults to ["u", "T"].
    time_dependent : list, optional
        Time-dependent variables. Defaults to ["T"].
    cse : bool
        Whether to apply common subexpression elimination.
    language : str
        'c', 'c++', 'cuda', 'python', or 'julia'.
    jac_mode : str
        'symbolic' for explicit Jacobian, 'autodiff' for forward-mode dual numbers.
    func_name : str
        Name of the generated function.
    """
    if solve_vars is None:
        solve_vars = ["u", "T"]
    if time_dependent is None:
        time_dependent = ["T"]

    system.heat += sp.Symbol("pdv_work")

    result = system.generate_code(
        solve_vars, time_dependent, language=language, jac=True, cse=cse,
        minimal=False, func_name=func_name, jac_mode=jac_mode,
    )

    lang = language.lower()

    # Determine file extension
    ext = {"c": ".c", "c++": ".cpp", "cuda": ".cu", "python": ".py", "julia": ".jl"}[lang]

    # Write main source file
    with open(func_name + ext, "w") as f:
        f.write(result["code"])

    # Write interp helper header (C/C++/CUDA only)
    if "interp_header" in result and result["interp_header"]:
        with open("jaco_interp.h", "w") as f:
            f.write("#pragma once\n")
            f.write(result["interp_header"] + "\n")

    # Write type/enum header (C/C++/CUDA only)
    if "header" in result:
        with open(func_name + ".h", "w") as f:
            f.write(result["header"])


if __name__ == "__main__":
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "c"
    jac = sys.argv[2] if len(sys.argv) > 2 else "symbolic"
    solve = ["u", "T", "H+", "He+", "He++"]
    generate_funcjac_code(starforge, solve_vars=solve, language=lang, jac_mode=jac)
