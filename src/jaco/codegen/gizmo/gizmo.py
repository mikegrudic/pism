"""Routines for generating a cooling solver function for GIZMO"""

from jaco.models.wind_comparison import cooling, heating
from sympy.utilities.codegen import C99CodeGen
from sympy.codegen.ast import Assignment
from jaco.codegen.printers import JacoCCodePrinter
import sympy as sp


def generate_funcjac_code(system, cse=True):
    """Generates a .h and .c pair of C source files specifying the function funcjac"""

    system = cooling + heating
    system.heat += sp.Symbol("pdv_work")
    solve_vars = ["u", "T"]
    time_dependent = ["T"]
    func, jac, indices = system.solver_functions(solve_vars, time_dependent, return_jac=True)
    # TODO: keep track of which func element is the heat equation - need this to generate cooling function
    funcjac = sp.Matrix(func + sp.flatten(jac))
    X = sp.MatrixSymbol("X", len(indices), 1)

    index_defs = []
    for var in indices:
        funcjac = funcjac.subs(var, X[indices[var]])
        index_defs.append(f"#define INDEX_{var} {indices[var]}")
    paramsvars = funcjac.free_symbols.copy()
    paramsvars.remove(X)
    P = sp.MatrixSymbol("params", len(paramsvars), 1)
    assignments = []
    for i, p in enumerate(paramsvars):
        funcjac = funcjac.subs(p, P[i])
        assignments.append(Assignment(P[i], p))
        index_defs.append(f"#define INDEX_{p} {i}")
    c_printer = JacoCCodePrinter()
    cg = C99CodeGen(cse=cse, printer=c_printer)
    routine = cg.routine("microphysics_func_jac", funcjac)  # cg.routine("func",sp.Matrix(func + sp.flatten(jac)))
    cg.write([routine], "microphysics_func_jac", to_files=True)

    # Write table declarations and helper functions to a separate header
    table_preamble = c_printer.get_table_declarations_c()
    if table_preamble:
        with open("jaco_interp.h", "w") as F:
            F.write("#pragma once\n")
            F.write(table_preamble + "\n")

    with open("indices.h", "w") as F:
        for i in index_defs:
            F.write(i + "\n")
        F.write(f"#define NUM_VARS {len(indices)}")

    with open("assignments.h", "w") as F:
        F.write(f"double params[{len(paramsvars)}];\n")
        F.write(sp.ccode(Assignment(P, sp.Matrix(list(paramsvars)))) + "\n")
        F.write(f"double X[{len(indices)}];\n")
        F.write(sp.ccode(Assignment(X, sp.Matrix(list(indices)))))


if __name__ == "__main__":
    generate_funcjac_code(cooling + heating, cse=True)
