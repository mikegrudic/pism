"""Implementation of Equation class for representing conservation laws"""

import sympy as sp


class Equation(sp.core.relational.Equality):
    """Sympy equation where we overload addition/subtraction to apply those operations to the RHS, for summing rate
    equations"""

    def __new__(cls, lhs, rhs, **kwargs):
        # Force evaluate=False so trivial equalities like Eq(0, 0) don't
        # collapse into a BooleanTrue and lose their lhs/rhs structure.
        kwargs.setdefault("evaluate", False)
        return super().__new__(cls, lhs, rhs, **kwargs)

    def get_summand(self, other):
        """Value-check the operand and return the quantity to be summed in the operation: the expression itself if an expression, or the RHS"""
        if isinstance(other, sp.core.relational.Equality):
            if self.lhs != other.lhs:
                raise ValueError(
                    "Tried to sum incompatible equations. Equation summation only defined for differential equations with the same LHS."
                )
            else:
                return other.rhs
        elif isinstance(other, sp.logic.boolalg.BooleanAtom):
            return 0
        else:
            return other

    def __add__(self, other):
        summand = self.get_summand(other)
        return Equation(self.lhs, self.rhs + summand)

    def __sub__(self, other):
        summand = self.get_summand(other)
        return Equation(self.lhs, self.rhs - summand)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self + other
        return self

    def __isub__(self, other):
        self = self - other
        return self
