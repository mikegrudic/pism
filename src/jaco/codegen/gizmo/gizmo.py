"""Routines for generating a cooling solver function for GIZMO"""

from jaco.models.wind_comparison import cooling, heating
from sympy.utilities.codegen import C99CodeGen
from sympy.codegen.ast import Assignment
from jaco.codegen.printers import JacoCCodePrinter, JacoCppCodePrinter, JacoCudaCodePrinter
import sympy as sp


def _build_system():
    """Build the equation system and return func, jac, indices, and substituted expressions."""
    system = cooling + heating
    system.heat += sp.Symbol("pdv_work")
    solve_vars = ["u", "T"]
    time_dependent = ["T"]
    func, jac, indices = system.solver_functions(solve_vars, time_dependent, return_jac=True)
    return func, jac, indices


def _substitute_vars(func, jac, indices):
    """Substitute solve variables into X[i] and free symbols into params[i].

    Returns (func_expr, jac_expr, index_defs, paramsvars_ordered, X, P)
    """
    n_vars = len(indices)
    func_mat = sp.Matrix(func)
    jac_mat = sp.Matrix(jac).reshape(n_vars, n_vars)

    X = sp.MatrixSymbol("X", n_vars, 1)
    index_defs = []
    for var in indices:
        func_mat = func_mat.subs(var, X[indices[var]])
        jac_mat = jac_mat.subs(var, X[indices[var]])
        index_defs.append(f"#define INDEX_{var} {indices[var]}")

    paramsvars = func_mat.free_symbols | jac_mat.free_symbols
    paramsvars.discard(X)
    paramsvars = sorted(paramsvars, key=str)

    P = sp.MatrixSymbol("params", len(paramsvars), 1)
    for i, p in enumerate(paramsvars):
        func_mat = func_mat.subs(p, P[i])
        jac_mat = jac_mat.subs(p, P[i])
        index_defs.append(f"#define INDEX_{p} {i}")

    return func_mat, jac_mat, index_defs, paramsvars, X, P


def _write_indices_and_assignments(index_defs, paramsvars, indices, P, X):
    """Write indices.h and assignments.h."""
    with open("indices.h", "w") as F:
        for i in index_defs:
            F.write(i + "\n")
        F.write(f"#define NUM_VARS {len(indices)}")

    with open("assignments.h", "w") as F:
        F.write(f"double params[{len(paramsvars)}];\n")
        F.write(sp.ccode(Assignment(P, sp.Matrix(list(paramsvars)))) + "\n")
        F.write(f"double X[{len(indices)}];\n")
        F.write(sp.ccode(Assignment(X, sp.Matrix(list(indices)))))


def _generate_c(func, jac, indices, cse):
    """Generate plain C output using C99CodeGen (flat output array)."""
    func_mat, jac_mat, index_defs, paramsvars, X, P = _substitute_vars(func, jac, indices)

    n_vars = len(indices)
    funcjac = sp.Matrix(list(func_mat) + list(jac_mat.reshape(n_vars * n_vars, 1)))

    c_printer = JacoCCodePrinter()
    cg = C99CodeGen(cse=cse, printer=c_printer)
    basename = "microphysics_func_jac"
    routine = cg.routine(basename, funcjac)
    cg.write([routine], basename, to_files=True)

    # Add interp header include
    c_file = basename + ".c"
    with open(c_file, "r") as f:
        src = f.read()
    if "jaco_interp.h" not in src:
        src = src.replace('#include <math.h>', '#include <math.h>\n#include "jaco_interp.h"')
        with open(c_file, "w") as f:
            f.write(src)

    table_preamble = c_printer.get_table_declarations_c()
    if table_preamble:
        with open("jaco_interp.h", "w") as F:
            F.write("#pragma once\n")
            F.write(table_preamble + "\n")

    _write_indices_and_assignments(index_defs, paramsvars, indices, P, X)


def _write_split_header(basename, var_names, param_syms, n_vars, language):
    """Write a header with enums, structs, and function declaration."""
    lang = language.lower()
    is_cuda = lang == "cuda"
    is_c = lang == "c"
    device = "__device__ " if is_cuda else ""
    n_params = len(param_syms)
    param_names = [str(p) for p in param_syms]

    lines = ["#pragma once", ""]

    if not is_cuda and not is_c:
        lines.append("#include <array>")
        lines.append("#include <cstddef>")
        lines.append("")

    # Enums
    var_entries = ", ".join(f"IDX_{v} = {i}" for i, v in enumerate(var_names))
    lines.append(f"enum SolveVarIndex {{ {var_entries}, N_VARS = {n_vars} }};")

    param_entries = ", ".join(f"PARAM_{p} = {i}" for i, p in enumerate(param_names))
    lines.append(f"enum ParamIndex {{ {param_entries}, N_PARAMS = {n_params} }};")
    lines.append("")

    # Structs — C uses typedef struct
    if is_c:
        lines.append("typedef struct {")
        for v in var_names:
            lines.append(f"    double {v};")
        lines.append("} SolveVars;")
    else:
        lines.append("struct SolveVars {")
        for v in var_names:
            lines.append(f"    double {v};")
        lines.append(f"    {device}const double* data() const {{ return &{var_names[0]}; }}")
        lines.append(f"    {device}double* data() {{ return &{var_names[0]}; }}")
        lines.append("};")
    lines.append("")

    if is_c:
        lines.append("typedef struct {")
        for p in param_names:
            lines.append(f"    double {p};")
        lines.append("} Params;")
    else:
        lines.append("struct Params {")
        for p in param_names:
            lines.append(f"    double {p};")
        lines.append(f"    {device}const double* data() const {{ return &{param_names[0]}; }}")
        lines.append("};")
    lines.append("")

    # Function declaration
    ptr_sig = f"void {basename}(const double* X, const double* params, double* rhs, double jac[N_VARS][N_VARS])"
    if is_cuda:
        lines.append(f"__device__ {ptr_sig};")
    elif is_c:
        lines.append(f"{ptr_sig};")
    else:
        lines.append(f"void {basename}(")
        lines.append("    const std::array<double, N_VARS>& X,")
        lines.append("    const std::array<double, N_PARAMS>& params,")
        lines.append("    std::array<double, N_VARS>& rhs,")
        lines.append("    std::array<std::array<double, N_VARS>, N_VARS>& jac")
        lines.append(");")

    lines.append("")

    # Convenience overload (C++ and CUDA only — C doesn't have overloading)
    if not is_c:
        lines.append("// Convenience overload: accepts SolveVars/Params structs directly")
        if is_cuda:
            lines.append(f"__device__ inline void {basename}(const SolveVars& X, const Params& params, SolveVars& rhs, double jac[N_VARS][N_VARS]) {{")
            lines.append(f"    {basename}(X.data(), params.data(), rhs.data(), jac);")
        else:
            lines.append(f"inline void {basename}(const SolveVars& X, const Params& params, SolveVars& rhs, std::array<std::array<double, N_VARS>, N_VARS>& jac) {{")
            lines.append(f"    {basename}(reinterpret_cast<const std::array<double, N_VARS>&>(X),")
            lines.append(f"        reinterpret_cast<const std::array<double, N_PARAMS>&>(params),")
            lines.append(f"        reinterpret_cast<std::array<double, N_VARS>&>(rhs), jac);")
        lines.append("}")
        lines.append("")

    with open(f"{basename}.h", "w") as f:
        f.write("\n".join(lines))


def _generate_split(func, jac, indices, cse, language):
    """Generate C, C++, or CUDA output with separate rhs and jac outputs.

    Keeps original symbol names in the expressions and maps them to
    X[IDX_*]/params[PARAM_*] via const double aliases at the top of the function.
    """
    from jaco.symbols import sanitize_symbols

    n_vars = len(indices)
    lang = language.lower()
    is_cuda = lang == "cuda"
    is_c = lang == "c"

    func_mat = sanitize_symbols(sp.Matrix(func))
    jac_mat = sanitize_symbols(sp.Matrix(jac).reshape(n_vars, n_vars))

    # Identify parameter symbols (everything that's not a solve variable)
    solve_syms = set(indices.keys())
    all_syms = func_mat.free_symbols | jac_mat.free_symbols
    param_syms = sorted(all_syms - solve_syms, key=str)

    if is_cuda:
        printer = JacoCudaCodePrinter()
    elif is_c:
        printer = JacoCCodePrinter()
    else:
        printer = JacoCppCodePrinter()

    # CSE across both func and jac (expressions still use original symbol names)
    if cse:
        cse_exprs, (func_cse, jac_cse) = sp.cse((func_mat, jac_mat))
    else:
        cse_exprs, func_cse, jac_cse = [], func_mat, jac_mat

    # Build function body
    lines = []
    var_names = list(indices.keys())

    # Map solve variables from X array
    lines.append("   // Solve variables")
    for var in var_names:
        lines.append(f"   const double {var} = X[IDX_{var}];")

    # Map parameters from params array
    lines.append("   // Parameters")
    for p in param_syms:
        lines.append(f"   const double {p} = params[PARAM_{p}];")

    lines.append("")

    # CSE temporaries
    for sym, expr in cse_exprs:
        lines.append(f"   const double {printer.doprint(sym)} = {printer.doprint(expr)};")

    lines.append("")

    # RHS assignments
    for i, var in enumerate(var_names):
        lines.append(f"   rhs[IDX_{var}] = {printer.doprint(func_cse[i])};")

    lines.append("")

    # Jacobian assignments
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            lines.append(f"   jac[IDX_{vi}][IDX_{vj}] = {printer.doprint(jac_cse[i, j])};  // d(rhs_{vi})/d({vj})")

    body = "\n".join(lines)

    # Build function signature and file
    basename = "microphysics_func_jac"

    c_ptr_sig = f"void {basename}(const double* X, const double* params, double* rhs, double jac[N_VARS][N_VARS])"

    if is_cuda:
        sig = f"__device__ {c_ptr_sig}"
        header_include = f'#include <math.h>\n#include "{basename}.h"\n#include "jaco_interp.h"'
        ext = ".cu"
    elif is_c:
        sig = c_ptr_sig
        header_include = f'#include <math.h>\n#include "{basename}.h"\n#include "jaco_interp.h"'
        ext = ".c"
    else:
        sig = (
            f'#include "{basename}.h"\n'
            f'#include "jaco_interp.h"\n\n'
            f"void {basename}(\n"
            f"    const std::array<double, N_VARS>& X,\n"
            f"    const std::array<double, N_PARAMS>& params,\n"
            f"    std::array<double, N_VARS>& rhs,\n"
            f"    std::array<std::array<double, N_VARS>, N_VARS>& jac\n"
            f")"
        )
        header_include = None
        ext = ".cpp"

    src = ""
    if header_include:
        src += header_include + "\n\n"
    src += sig + " {\n" + body + "\n}\n"

    with open(basename + ext, "w") as f:
        f.write(src)

    # Write interp header
    if is_cuda:
        table_preamble = printer.get_table_declarations_cuda()
    elif is_c:
        table_preamble = printer.get_table_declarations_c()
    else:
        table_preamble = printer.get_table_declarations_cpp()
    if table_preamble:
        with open("jaco_interp.h", "w") as F:
            F.write("#pragma once\n")
            F.write(table_preamble + "\n")

    # Generate header with enums, structs, and function declaration
    _write_split_header(basename, var_names, param_syms, n_vars, language)

    # Still write indices.h and assignments.h for compatibility
    index_defs = []
    for var, idx in indices.items():
        index_defs.append(f"#define INDEX_{var} {idx}")
    for i, p in enumerate(param_syms):
        index_defs.append(f"#define INDEX_{p} {i}")

    X = sp.MatrixSymbol("X", n_vars, 1)
    P = sp.MatrixSymbol("params", len(param_syms), 1)
    _write_indices_and_assignments(index_defs, param_syms, indices, P, X)


def generate_funcjac_code(system, cse=True, language="c"):
    """Generates C, C++, or CUDA source files specifying the function funcjac.

    Parameters
    ----------
    system : EquationSystem
        The system to generate code for.
    cse : bool, optional
        Whether to apply common subexpression elimination.
    language : str, optional
        'c' for CPU C code, 'c++' for CPU C++ code, 'cuda' for GPU-compatible code.
    """
    func, jac, indices = _build_system()

    if language.lower() in ("c", "c++", "cuda"):
        _generate_split(func, jac, indices, cse, language)
    else:
        raise ValueError(f"Unsupported language: {language}")


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "c"
    generate_funcjac_code(cooling + heating, cse=True, language=lang)
