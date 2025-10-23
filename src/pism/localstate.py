"""Specifies class for the local thermal and chemical state"""


class State:
    """Internal energy, abundances, dust mass fraction, radiation field, cosmic ray ionization rate,
    ortho-para ratio,"""

    f_ortho: float = 0.75
    hydrogen_massfrac: float = 0.7381
    metallicity: float = 0.0134

    def __init__(self):
        pass

    @property
    def density(self, species):
        return 0


#   @property
#    def
