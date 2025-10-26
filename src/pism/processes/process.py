"""Implementation of basic Process class"""


class Process:
    """
    Top-level class containing a description of a microscopic process

    Most importantly, this implements the procedure for combining processes to build up a network for chemistry
    + conservation equations.
    """

    def __init__(self, name=""):
        self.name = name
        self.network = {}
        self.dust_heat = None
        self.rate = None
        self.heat = None

    def __add__(self, other):
        """Sum 2 processes together: define new functions that"""
        sum_process = Process()
        for summed_quantity in "heat", "dust_heat", "rate":
            setattr(sum_process, summed_quantity, getattr(self, summed_quantity) + getattr(other, summed_quantity))
        sum_process.name = f"{self.name} + {other.name}"
        return sum_process
