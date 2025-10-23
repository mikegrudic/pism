"""Top-level class describing a microscopic process"""

from dataclasses import dataclass


class Process:
    def __init__(self, name: str = ""):
        self.name = ""

    def vol_rate(self, localstate):
        """Returns the volumetric rate in events per cubic volume as a function of the local state"""
        return 0.0

    def heat(self):
        """Returns the thermalized energy per event. Convention: positive results in gas heating, negative in gas cooling"""
        return 0.0

    def dust_heat(self):
        """Returns the heat imparted to dust per event. Convention: positive results in dust heating, negative in dust cooling"""
        return 0.0
