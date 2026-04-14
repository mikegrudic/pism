"""Routines for writing jaco-generated source files to disk."""

import sympy as sp


def generate_funcjac_code(
    system,
    solve_vars=None,
    time_dependent=None,
    cse=True,
    language="c",
    jac_mode="symbolic",
    func_name="microphysics_func_jac",
    source_ext=None,
    output_dir=".",
):
    """Generates source files for the RHS + Jacobian function and writes them
    to the specified output directory.

    Produces (for C/C++):
      - microphysics_func_jac.{c,cc,cpp}  — RHS + Jacobian function
      - microphysics_func_jac.h            — union types, enums, declaration
      - jaco_interp.h                      — interpolation helpers + static table data
      - jaco_eos.{c,cc,cpp}               — EOS functions from species

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
    source_ext : str, optional
        Override the source file extension (e.g. '.cc'). If None, uses the
        language default.
    output_dir : str
        Directory to write generated files into. Defaults to current directory.
    """
    import os

    if solve_vars is None:
        solve_vars = ["u", "T"]
    if time_dependent is None:
        time_dependent = ["T"]

    system.heat += sp.Symbol("pdv_work")

    result = system.network.generate_code(
        solve_vars,
        time_dependent,
        language=language,
        jac=True,
        do_cse=cse,
        minimal=False,
        func_name=func_name,
        jac_mode=jac_mode,
    )

    lang = language.lower()
    if source_ext is None:
        source_ext = {"c": ".c", "c++": ".cpp", "cuda": ".cu", "python": ".py", "julia": ".jl"}[lang]

    def outpath(name):
        return os.path.join(output_dir, name)

    with open(outpath(func_name + source_ext), "w") as f:
        f.write(result["code"])

    if "interp_header" in result and result["interp_header"]:
        with open(outpath("jaco_interp.h"), "w") as f:
            f.write("#pragma once\n")
            f.write(result["interp_header"] + "\n")

    if "header" in result:
        with open(outpath(func_name + ".h"), "w") as f:
            f.write(result["header"])

    if "eos_code" in result:
        with open(outpath("jaco_eos" + source_ext), "w") as f:
            f.write(result["eos_code"])


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Generate jaco microphysics source files for GIZMO")
    parser.add_argument("model", help="Model name (e.g. wind_comparison)")
    parser.add_argument("--language", default="c", help="Language: c, c++, cuda, python, julia")
    parser.add_argument("--ext", default=None, help="Override source file extension (e.g. .cc)")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated files")
    parser.add_argument("--jac-mode", default="symbolic", help="Jacobian mode: symbolic or autodiff")
    args = parser.parse_args()

    from importlib import import_module
    model_mod = import_module(f"jaco.models.{args.model}")
    # Model modules expose a Process object, either directly or via make_model()
    if hasattr(model_mod, "make_model"):
        model = model_mod.make_model()
    else:
        # look for the first Process-like object in the module
        from jaco.process import Process
        model = None
        for name in dir(model_mod):
            obj = getattr(model_mod, name)
            if isinstance(obj, Process):
                model = obj
                break
        if model is None:
            print(f"Error: could not find a jaco Process in jaco.models.{args.model}", file=sys.stderr)
            sys.exit(1)

    generate_funcjac_code(
        model,
        language=args.language,
        source_ext=args.ext,
        output_dir=args.output_dir,
        jac_mode=args.jac_mode,
    )
