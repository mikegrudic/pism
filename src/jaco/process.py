"""Implementation of base Process class with methods for managing and solving systems of equations"""

from .symbols import n_, d_dt
from .equation import Equation
from .equation_system import EquationSystem


class Process:
    """
    Top-level class containing a description of a microscopic process

    Most importantly, this implements the procedure for combining processes to build up a network for chemistry
    + conservation equations.
    """

    # TODO: want to define an equivalence relation between processes that compares only the equations/rates, so that
    # process composition is commutative

    def __init__(self, name="", bibliography=[]):
        """Construct an empty Process instance

        Parameters
        ----------
        name: str, optional
            Name of the process
        """
        self.name = name
        self.initialize_network()
        self.rate = 0
        self.heat = 0
        self.bibliography = bibliography
        self.subprocesses = [self]
        self.equations = []

    def __repr__(self):
        """Print the name in print()"""
        return self.name

    def __add__(self, other):
        """Sum 2 processes together: define a new process whose rates are the sum of the input process"""
        if other == 0:  # necessary for native sum() routine to work
            return self

        attrs_to_sum = "heat", "subprocesses", "network", "bibliography", "equations"  # all rates

        sum_process = Process()
        sum_process.rate = None  # "rate" ceases to be meaningful for composite processes
        for summed_attr in attrs_to_sum:
            attr1, attr2 = getattr(self, summed_attr), getattr(other, summed_attr)
            if attr1 is None or attr2 is None:
                setattr(sum_process, summed_attr, None)
            else:
                setattr(sum_process, summed_attr, attr1 + attr2)

        sum_process.name = f"{self.name} +\n{other.name}"
        return sum_process

    def __radd__(self, other):
        return self.__add__(other)

    def initialize_network(self):
        self.network = EquationSystem()  # this is a dict for which unknown keys are initialized to 0 by default

    @property
    def heat(self):
        """Energy lost from gas per unit volume and time"""
        return self.__heat

    @heat.setter
    def heat(self, value):
        """Ensures that the network is always updated when we update the heat"""
        self.__heat = value
        self.network["heat"] = Equation(d_dt(n_("heat")), value)

    def solve(
        self,
        known_quantities,
        guess,
        time_dependent=[],
        dt=None,
        verbose=False,
        tol=1e-3,
        careful_steps=10,
    ):
        """
        Solves the equations for a set of desired quantities given a set of known quantities

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
        soldict: dict
            Dict of solved quantities
        """

        return self.network.solve(
            known_quantities,
            guess,
            time_dependent=time_dependent,
            tol=tol,
            careful_steps=careful_steps,
            dt=dt,
            verbose=verbose,
        )

    def solver_functions(self, solve_vars, time_dependent=[], return_jac=False, return_dict=False):
        """Returns the RHS of the system to solve and its Jacobian, applying simplifications"""
        return self.network.solver_functions(solve_vars, time_dependent, return_jac, return_dict)

    def generate_code(self, solve_vars=[], time_dependent=[], language="c", cse=True,
                       func_name="microphysics_func_jac", jac_mode="symbolic"):
        """Generates source files for the RHS + Jacobian function and writes them
        to the current directory.

        For C, produces:
          - microphysics_func_jac.c  — RHS + Jacobian function
          - microphysics_func_jac.h  — union types, enums, declaration
          - jaco_interp.h            — interpolation helpers + static table data
          - jaco_eos.c               — EOS pressure function

        Parameters
        ----------
        solve_vars : list
            Variables to solve for. If empty, solves for all species.
        time_dependent : list
            Variables that are time-dependent (get BDF discretization)
        language : str
            Target language: 'c', 'c++', 'cuda', 'python', 'julia'
        cse : bool
            Whether to apply common subexpression elimination
        func_name : str
            Name of the generated function.
        jac_mode : str
            'symbolic' or 'autodiff'
        """
        from .codegen.gizmo import generate_funcjac_code
        generate_funcjac_code(
            self, solve_vars=solve_vars or None,
            time_dependent=time_dependent or None,
            cse=cse, language=language, jac_mode=jac_mode,
            func_name=func_name,
        )
