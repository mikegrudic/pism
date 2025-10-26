"""Implementation of basic Process class"""

from collections import defaultdict


class Process:
    """
    Top-level class containing a description of a microscopic process

    Most importantly, this implements the procedure for combining processes to build up a network for chemistry
    + conservation equations.
    """

    def __init__(self, name=""):
        self.name = name
        self.initialize_network()
        self.dust_heat = 0
        self.rate = 0
        self.heat = 0

    def initialize_network(self):
        self.network = defaultdict(int)  # this is a dict for which unknown keys are initialized to 0 by default

    #    def finialize_network(self):
    #        """Adds finishing simplifications on the network - e.g. substituting conservation equations for certain species"""

    def __add__(self, other):
        """Sum 2 processes together: define new functions that"""
        sum_process = Process()
        for summed_quantity in "heat", "dust_heat", "rate":
            attr1, attr2 = getattr(self, summed_quantity), getattr(other, summed_quantity)
            if attr1 is None or attr2 is None:
                setattr(sum_process, summed_quantity, None)
            else:
                setattr(sum_process, summed_quantity, attr1 + attr2)

        # now combine the networks
        sum_process.network = self.combine_networks(self.network, other.network)
        sum_process.name = f"{self.name} + {other.name}"
        return sum_process

    def combine_networks(self, n1, n2):
        combined_network = defaultdict(int)
        combined_keys = set(tuple(n1.keys())).union(set(tuple(n2.keys())))  # gross?
        for k in combined_keys:
            combined_network[k] = n1[k] + n2[k]
        return combined_network
