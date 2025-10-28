"""Class describing a generic heating/cooling process with no associated radiation or chemistry"""

from ..process import Process
from ..symbols import c_s, G, ρ, T, n_e, z
import sympy as sp


class ThermalProcess(Process):
    """Generic heating/cooling process"""

    def __init__(self, heating_rate, name=""):
        super().__init__(name=name)
        self.heat_per_volume = heating_rate


PdV_heating = ThermalProcess(
    sp.symbols("C_1") * c_s**2 * sp.sqrt(4 * sp.pi * G * ρ), name="Grav. Compression"
)  # 1998ApJ...495..346M

inv_compton_cooling = ThermalProcess(
    5.41e-36 * n_e * T * (1 + z) ** 4, name="Inverse Compton Cooling"
)  # 1986ApJ...301..522I
