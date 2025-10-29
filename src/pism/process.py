"""Implementation of basic Process class"""

from collections import defaultdict
import sympy as sp


class Process:
    """
    Top-level class containing a description of a microscopic process

    Most importantly, this implements the procedure for combining processes to build up a network for chemistry
    + conservation equations.
    """

    def __init__(self, name="", bibliography={}):
        self.name = name
        self.initialize_network()
        self.dust_heat = 0
        self.rate = 0
        self.heat = 0
        self.bibliography = bibliography
        self.subprocesses = [self]

    def initialize_network(self):
        self.network = defaultdict(int)  # this is a dict for which unknown keys are initialized to 0 by default

    #    def finialize_network(self):
    #        """Adds finishing simplifications on the network - e.g. substituting conservation equations for certain
    #  species, and deleting extraneous equations in turn """
    #    .network.replace(sp.symbols("n_e-"), sp.symbols("n_Htot")*sp.symbols("x_e")).replace(sp.symbols("n_H+"), sp.symbols("n_Htot")*sp.symbols("x_e")).replace(sp.symbols("n_H"), sp.symbols("n_Htot")*sp.symbols("(1-x_e)")

    def __add__(self, other):
        """Sum 2 processes together: define a new process whose rates are the sum of the input process"""
        if other == 0:  # necessary for native sum() routine to work
            return self

        quantities_to_sum = "heat", "dust_heat", "subprocesses"  # all energy exchange terms

        sum_process = Process()
        sum_process.rate = None  # no longer meaningful to define a single rate
        for summed_quantity in quantities_to_sum:
            attr1, attr2 = getattr(self, summed_quantity), getattr(other, summed_quantity)
            if attr1 is None or attr2 is None:
                setattr(sum_process, summed_quantity, None)
            else:
                setattr(sum_process, summed_quantity, attr1 + attr2)

        # now combine the networks
        sum_process.network = self.combine_networks(self.network, other.network)
        sum_process.name = f"{self.name} + {other.name}"
        return sum_process

    def __radd__(self, other):
        return self.__add__(other)

    def combine_networks(self, n1, n2):
        combined_network = defaultdict(int)
        combined_keys = set(tuple(n1.keys())).union(set(tuple(n2.keys())))  # gross?
        for k in combined_keys:
            combined_network[k] = n1[k] + n2[k]
        return combined_network

    def print_network_equations(self):
        """Prints the system of equations in the chemistry network"""
        for k, rhs in self.network.items():
            print(sp.symbols(f"dn_{k}/dt"), "=", sp.simplify(rhs))

    def network_species(self):
        return list(self.network.keys())

    def solve_steadystate(self, known_quantities={}, x0={}):
        """Solves for a steady state of the network"""
        return

    def __repr__(self):
        return self.name
