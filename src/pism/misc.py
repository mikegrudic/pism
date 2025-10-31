"""Various convenience routines used throughout the package"""

from string import digits


def species_charge(species: str) -> int:
    """Returns the charge number of a species from its name"""
    if species[-1] not in ("-", "+"):
        return 0
    if "++" in species:
        return 2
    elif "--" in species:
        return -2

    base = base_species(species)
    suffix = species.split(base)[-1]
    if suffix == "+":
        return 1
    if suffix == "-":
        return -1
    elif "+" in suffix:
        return int(suffix.rstrip("+"))
    else:
        return -int(suffix.rstrip("-"))


def is_an_ion(species: str) -> bool:
    return species_charge(species) != 0 and species != "e-"


def base_species(species: str) -> str:
    """Removes the charge suffix from a species"""
    base = species.rstrip(digits + "-+")
    return base


def charge_suffix(charge: int) -> str:
    """Returns the suffix for an input charge number"""
    match charge:
        case 0:
            return ""
        case 1:
            return "+"
        case 2:
            return "++"
        case -1:
            return "-"
        case -2:
            return "--"
        case _:
            if charge < -2:
                return str(abs(charge)) + "-"
            else:
                return str(abs(charge)) + "+"


def ionize(species: str) -> str:
    """Returns the symbol of the species produced by removing an electron from the input species"""
    charge = species_charge(species)
    return base_species(species) + charge_suffix(charge + 1)


def recombine(species: str) -> str:
    """Returns the symbol of the species produced by adding an electron to the input species"""
    charge = species_charge(species)
    return base_species(species) + charge_suffix(charge - 1)
