API Documentation
=================

Process
-------

The ``Process`` class is the top-level building block for assembling microphysics
models. Processes represent individual physical mechanisms (ionization,
recombination, cooling, heating) and can be composed with ``+`` to build
composite networks.

.. autoclass:: jaco.Process
   :members:
   :undoc-members:

Specialized process types:

.. automodule:: jaco.processes
   :members: ThermalProcess, NBodyProcess, ChemicalReaction, CollisionalIonization, GasPhaseRecombination, FreeFreeEmission, LineCoolingSimple
   :undoc-members:

EquationSystem
--------------

The ``EquationSystem`` is the core symbolic engine that stores the rate
equations for all species and provides methods for reducing, solving, and
generating code from the network.

.. autoclass:: jaco.EquationSystem
   :members:
   :undoc-members:

Key concepts:

- **fixed_species** (``dict``): Maps species names to fixed values or
  parameter symbols. These species' equations are removed from the solve
  system during reduction, and their abundances are substituted throughout
  the remaining equations. Set at the model level to declare which species
  are computed externally by the host code (e.g. ``gizmo_to_jaco``).

  Example::

      model.network.fixed_species = {
          "C+": sp.Symbol("x_Cplus_fixed"),
          "H-": 1e-15,
          "H_2+": 0,
      }

- **equilibrium_overrides** (``dict``): Maps species names to sympy
  expressions for their equilibrium abundance. Used when the automatic
  steady-state linearization cannot derive the equilibrium (e.g. nonlinear
  or implicit expressions).

  Example::

      model.network.equilibrium_overrides = {
          "H_2+": sp.S.Zero,
          "HD": 2.527e-5 * x_("H_2"),
      }

- **Reduction pipeline** (``reduced()``):

  1. Set time dependence (BDF for evolved species, steady-state for others)
  2. Conservation reductions (n->x conversion, charge neutrality, atom conservation)
  3. Fix species (substitute fixed_species values, remove their equations)
  4. Prune decoupled equations (remove orphaned equations like dust heat)

- **Code generation** (``generate_code()``): Produces C/C++/CUDA/Python/Julia
  source files with the RHS function, Jacobian, EOS functions, and
  interpolation table infrastructure.

Models
------

Models are assembled from processes and declare their configuration:

.. automodule:: jaco.models
   :members:

Starforge
^^^^^^^^^

The STARFORGE model implements ISM thermochemistry with H2, metals, dust,
cosmic rays, and radiation. It is the primary model for star formation
simulations in GIZMO.

.. autofunction:: jaco.models.starforge.starforge.make_model

The model declares:

- **Solve variables**: ``u, T, H+, He+, He++, H_2``
- **Time-dependent**: ``T, H_2`` (backward Euler); ions in steady-state equilibrium
- **Fixed species**: ``C+, H-, H_2+, CO, HD`` (computed by host code)
- **Equilibrium overrides**: ``H_2+ = 0``, ``HD = 2.527e-5 * x_H2``

Wind Comparison
^^^^^^^^^^^^^^^

A simple atomic-H cooling model for wind bubble tests.

.. autofunction:: jaco.models.wind_comparison.cooling.make_model

GIZMO Code Generation
----------------------

The GIZMO-specific codegen layer writes source files for integration into GIZMO's
build system.

.. automodule:: jaco.codegen.gizmo
   :members: generate_funcjac_code

Model-specific defaults for solve variables and time-dependent species:

.. data:: jaco.codegen.gizmo._MODEL_DEFAULTS

   Dictionary mapping model names to their default ``solve_vars`` and
   ``time_dependent`` lists. Used when the codegen is invoked from the
   command line (``python -m jaco.codegen.gizmo <model_name>``).

Equation
--------

.. autoclass:: jaco.Equation
   :members:
   :undoc-members:

Numerical Solvers
-----------------

.. automodule:: jaco.numerics
   :members: newton_rootsolve

EOS
---

.. automodule:: jaco.eos.eos
   :members:
   :undoc-members:

Symbols
-------

.. automodule:: jaco.symbols
   :members: T, n_, x_, n_Htot, sanitize_name, sanitize_symbols, table_interp_2d
   :undoc-members:

Interpolation
-------------

.. automodule:: jaco.interpolation
   :members: PiecewiseLinearInterp, PiecewiseConstantInterp, TableInterp2D, register_table, save_table_hdf5
   :undoc-members:
