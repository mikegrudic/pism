"""Class describing a generic heating/cooling process with no associated radiation or chemistry"""

from .process import Process
from ..symbols import c_s, G, ρ
import sympy as sp


class ThermalProcess(Process):
    def __init__(self, heating_rate, name=""):
        super().__init__(name=__name__)
        self.heat_per_volume = heating_rate


PdV_heating = ThermalProcess(sp.symbols("C1") * c_s**2 * sp.sqrt(4 * sp.pi * G * ρ), name="Grav. Compression")
