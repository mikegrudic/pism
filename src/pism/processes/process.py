"""Top-level class describing a microscopic process"""


class Process:
    def __init__(self, name=""):
        # .symbols = sp.symbols("T u")
        self.name = name
        self.RHS = {}
        self.dust_heat_per_volume = None
        self.rate_per_volume = None
        self.heat_per_volume = None

    def __add__(self, other):
        """Sum 2 processes together: define new functions that"""
        sum_process = Process()
        for summed_quantity in "heat_per_volume", "dust_heat_per_volume", "rate_per_volume":
            setattr(sum_process, summed_quantity, getattr(self, summed_quantity) + getattr(other, summed_quantity))
        sum_process.name = f"{self.name} + {other.name}"
        return sum_process

    # def inverse(self):
    #     """Returns the inverse process"""
    #     inverse_process = deepcopy(self)
    #     return inverse_process

    # @property
    # def rate_per_volume(self):
    #     """Returns the volumetric rate in events per cubic volume as a function of the local state"""
    #     return 0

    # @property
    # def heat_per_volume(self):
    #     """Returns the net energy thermalized into the gas per unit time and volume"""
    #     return 0

    # @property
    # def dust_heat(self):
    #     """Returns the heat imparted to dust per volume. Convention: positive results in dust heating, negative in dust cooling"""
    #     return 0

    # @property
    # def RHS(self):
    #     """Returns a dict whose keys are the state quantities and whose entries are the corresponding equation"""
    #     return {}
