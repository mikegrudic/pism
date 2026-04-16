"""Custom sympy code printers that emit efficient helper-function calls for piecewise interpolation."""

import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.fortran import FCodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.julia import JuliaCodePrinter


# --------------------------------------------------------------------------- #
# C helper functions (emitted once per generated file)
# --------------------------------------------------------------------------- #

_C_INTERP_HELPERS = r"""
static inline double jaco_interp1d(double x, const double *xp, const double *yp, int n, int extrap) {
    if (x <= xp[0]) {
        if (extrap) {
            double slope = (yp[1] - yp[0]) / (xp[1] - xp[0]);
            return yp[0] + slope * (x - xp[0]);
        }
        return yp[0];
    }
    if (x >= xp[n - 1]) {
        if (extrap) {
            double slope = (yp[n - 1] - yp[n - 2]) / (xp[n - 1] - xp[n - 2]);
            return yp[n - 1] + slope * (x - xp[n - 1]);
        }
        return yp[n - 1];
    }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return yp[lo] + t * (yp[hi] - yp[lo]);
}

static inline double jaco_interp1d_const(double x, const double *xp, const double *vp, int n, int extrap) {
    if (x <= xp[0]) return vp[0];
    if (x >= xp[n - 1]) return vp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    return vp[lo];
}
"""

# --------------------------------------------------------------------------- #
# Fortran helper functions
# --------------------------------------------------------------------------- #

_FORTRAN_INTERP_HELPERS = r"""
pure function jaco_interp1d(x, xp, yp, n, extrap) result(y)
  implicit none
  double precision, intent(in) :: x
  integer, intent(in) :: n, extrap
  double precision, intent(in) :: xp(n), yp(n)
  double precision :: y, slope, t
  integer :: lo, hi, mid
  if (x <= xp(1)) then
    if (extrap == 1) then
      slope = (yp(2) - yp(1)) / (xp(2) - xp(1))
      y = yp(1) + slope * (x - xp(1))
    else
      y = yp(1)
    end if
    return
  end if
  if (x >= xp(n)) then
    if (extrap == 1) then
      slope = (yp(n) - yp(n-1)) / (xp(n) - xp(n-1))
      y = yp(n) + slope * (x - xp(n))
    else
      y = yp(n)
    end if
    return
  end if
  lo = 1; hi = n
  do while (hi - lo > 1)
    mid = (lo + hi) / 2
    if (xp(mid) <= x) then
      lo = mid
    else
      hi = mid
    end if
  end do
  t = (x - xp(lo)) / (xp(hi) - xp(lo))
  y = yp(lo) + t * (yp(hi) - yp(lo))
end function jaco_interp1d

pure function jaco_interp1d_const(x, xp, vp, n, extrap) result(v)
  implicit none
  double precision, intent(in) :: x
  integer, intent(in) :: n, extrap
  double precision, intent(in) :: xp(n), vp(n-1)
  double precision :: v
  integer :: lo, hi, mid
  if (x <= xp(1)) then
    v = vp(1); return
  end if
  if (x >= xp(n)) then
    v = vp(n-1); return
  end if
  lo = 1; hi = n
  do while (hi - lo > 1)
    mid = (lo + hi) / 2
    if (xp(mid) <= x) then
      lo = mid
    else
      hi = mid
    end if
  end do
  v = vp(lo)
end function jaco_interp1d_const
"""


def _table_var_name(table_tuple, prefix):
    """Generate a unique C variable name for a table array based on its hash."""
    return f"{prefix}_{abs(hash(table_tuple)) % (10**12)}"


def _interp_table_names(expr, xp_prefix="jaco_xp", yp_prefix="jaco_yp"):
    """Derive descriptive table variable names from a PiecewiseLinearInterp/PiecewiseConstantInterp expression.

    If the expression has a name arg, uses it to produce e.g. 'lambda_cooling_xp', 'lambda_cooling_yp'.
    Otherwise falls back to hash-based names.
    """
    X_tuple = expr.args[1]
    Y_or_V_tuple = expr.args[2]
    name = expr.table_name
    if name is not None:
        name_str = str(name)
        return f"{name_str}_xp", f"{name_str}_yp"
    return _table_var_name(X_tuple, xp_prefix), _table_var_name(Y_or_V_tuple, yp_prefix)


class _InterpPrinterMixin:
    """Mixin that tracks referenced tables and generates declarations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp_tables = {}  # maps (tuple_id) -> (var_name, values)
        self._needs_interp_helpers = False

    def _print_Heaviside(self, expr):
        """Print Heaviside step function as a branchless comparison.

        Sympy's default ccode emits a 3-way ternary because H(0) = 1/2 is
        the default value. At floating point, x == 0.0 is a measure-zero
        event that never occurs in physical computation, so we drop the
        H(0) case and emit just (x > 0). The compiler turns this into a
        branchless cmov/csel.
        """
        arg = self._print(expr.args[0])
        return f"(double)({arg} > 0)"

    def _print_Piecewise(self, expr):
        """Print Piecewise as a single-line ternary expression.

        Compilers reliably turn `cond ? a : b` (where a and b are scalar
        floating-point values) into a branchless cmov/csel, so the only
        real cost is the multi-line formatting that sympy emits by default.
        We collapse it to a single line for readability and to make sure
        the compiler sees it as an expression, not a statement.
        """
        # Validate the last condition is True (sentinel)
        if expr.args[-1].cond is not sp.true:
            return super()._print_Piecewise(expr)

        result = self._print(expr.args[-1].expr)
        for e, c in reversed(expr.args[:-1]):
            cond = self._print(c)
            val = self._print(e)
            result = f"({cond} ? {val} : {result})"
        return result

    def _print_Float(self, expr):
        """Print floats using the shortest clean representation.

        Uses the fewest significant digits that preserve the value to
        within a relative error of 1e-15 (well below any physical
        measurement precision). This cleans up artifacts from sympy
        floating-point arithmetic without meaningful loss of accuracy.
        """
        v = float(expr)
        if v == 0:
            return "0.0"
        for digits in range(4, 17):
            candidate = f"{v:.{digits}g}"
            if abs(float(candidate) - v) / abs(v) < 1e-12:
                return candidate
        return repr(v)

    def _register_table(self, tuple_expr, var_name):
        """Register a table with the given variable name and return the name."""
        key = tuple_expr
        if key not in self._interp_tables:
            values = [float(v) for v in tuple_expr]
            self._interp_tables[key] = (var_name, values)
        return self._interp_tables[key][0]

    # --- Multi-dimensional table interp printing ---

    def _table_name_from_expr(self, expr):
        """Extract the table name string from a TableInterp2D/3D expression."""
        return str(expr.args[-1])  # last arg is always the Str name

    def _print_TableInterp2D(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        x_code = self._print(expr.args[0])
        y_code = self._print(expr.args[1])
        return f"jaco_table2d_interp({x_code}, {y_code}, &{name})"

    def _print_TableInterp2D_dx(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        x_code = self._print(expr.args[0])
        y_code = self._print(expr.args[1])
        return f"jaco_table2d_interp_dx({x_code}, {y_code}, &{name})"

    def _print_TableInterp2D_dy(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        x_code = self._print(expr.args[0])
        y_code = self._print(expr.args[1])
        return f"jaco_table2d_interp_dy({x_code}, {y_code}, &{name})"

    def _print_TableInterp3D(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        x_code = self._print(expr.args[0])
        y_code = self._print(expr.args[1])
        z_code = self._print(expr.args[2])
        return f"jaco_table3d_interp({x_code}, {y_code}, {z_code}, &{name})"

    def _print_TableInterp3D_dx(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        return f"jaco_table3d_interp_dx({self._print(expr.args[0])}, {self._print(expr.args[1])}, {self._print(expr.args[2])}, &{name})"

    def _print_TableInterp3D_dy(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        return f"jaco_table3d_interp_dy({self._print(expr.args[0])}, {self._print(expr.args[1])}, {self._print(expr.args[2])}, &{name})"

    def _print_TableInterp3D_dz(self, expr):
        self._needs_interp_helpers = True
        name = self._table_name_from_expr(expr)
        return f"jaco_table3d_interp_dz({self._print(expr.args[0])}, {self._print(expr.args[1])}, {self._print(expr.args[2])}, &{name})"

    def get_table_declarations_c(self):
        """Return C declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_C_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            vals_str = ", ".join(f"{v:.17e}" for v in values)
            lines.append(f"static const double {name}[] = {{{vals_str}}};")
        return "\n".join(lines)

    def get_table_declarations_fortran(self):
        """Return Fortran declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_FORTRAN_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            n = len(values)
            vals_str = ", ".join(f"{v:.17e}d0" for v in values)
            lines.append(
                f"double precision, parameter :: {name}({n}) = (/ {vals_str} /)"
            )
        return "\n".join(lines)


def _print_interp_linear(printer, expr, func_name="jaco_interp1d"):
    """Shared implementation for printing PiecewiseLinearInterp across languages."""
    printer._needs_interp_helpers = True
    desired_x, desired_y = _interp_table_names(expr)
    x_name = printer._register_table(expr.args[1], desired_x)
    y_name = printer._register_table(expr.args[2], desired_y)
    n = len(expr.args[1])
    x_code = printer._print(expr.args[0])
    extrap_code = printer._print(expr.args[3])
    return f"{func_name}({x_code}, {x_name}, {y_name}, {n}, {extrap_code})"


def _print_interp_const(printer, expr, func_name="jaco_interp1d_const"):
    """Shared implementation for printing PiecewiseConstantInterp across languages."""
    printer._needs_interp_helpers = True
    desired_x, desired_v = _interp_table_names(expr, yp_prefix="jaco_vp")
    x_name = printer._register_table(expr.args[1], desired_x)
    v_name = printer._register_table(expr.args[2], desired_v)
    n = len(expr.args[1])
    x_code = printer._print(expr.args[0])
    extrap_code = printer._print(expr.args[3])
    return f"{func_name}({x_code}, {x_name}, {v_name}, {n}, {extrap_code})"


class JacoCCodePrinter(_InterpPrinterMixin, C99CodePrinter):
    """C99 printer that emits helper-function calls for piecewise interpolation
    and clamps exp() arguments to prevent overflow."""

    def _print_exp(self, expr):
        """Clamp exp() arguments to [-500, 500] to prevent overflow.

        Without this, deeply nested rate coefficient expressions can produce
        exp(x) with |x| > 709 (double overflow), which under -ffast-math
        silently becomes NaN and corrupts the Jacobian.
        """
        arg = self._print(expr.args[0])
        return f"exp(fmax(-500.0, fmin(500.0, {arg})))"

    def _print_PiecewiseLinearInterp(self, expr):
        return _print_interp_linear(self, expr)

    def _print_PiecewiseConstantInterp(self, expr):
        return _print_interp_const(self, expr)


class JacoFCodePrinter(_InterpPrinterMixin, FCodePrinter):
    """Fortran printer that emits helper-function calls for piecewise interpolation."""

    def _print_PiecewiseLinearInterp(self, expr):
        return _print_interp_linear(self, expr)

    def _print_PiecewiseConstantInterp(self, expr):
        return _print_interp_const(self, expr)


# --------------------------------------------------------------------------- #
# CUDA/HIP helper functions
# --------------------------------------------------------------------------- #

_CUDA_INTERP_HELPERS = r"""
__device__ __forceinline__ double jaco_interp1d_extrap(double x, const double *xp, const double *yp, int n) {
    if (x <= xp[0]) {
        double slope = (yp[1] - yp[0]) / (xp[1] - xp[0]);
        return yp[0] + slope * (x - xp[0]);
    }
    if (x >= xp[n - 1]) {
        double slope = (yp[n - 1] - yp[n - 2]) / (xp[n - 1] - xp[n - 2]);
        return yp[n - 1] + slope * (x - xp[n - 1]);
    }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x - xp[lo]) / (xp[lo + 1] - xp[lo]);
    return yp[lo] + t * (yp[lo + 1] - yp[lo]);
}

__device__ __forceinline__ double jaco_interp1d_clamp(double x, const double *xp, const double *yp, int n) {
    if (x <= xp[0]) return yp[0];
    if (x >= xp[n - 1]) return yp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x - xp[lo]) / (xp[lo + 1] - xp[lo]);
    return yp[lo] + t * (yp[lo + 1] - yp[lo]);
}

__device__ __forceinline__ double jaco_interp1d_const_extrap(double x, const double *xp, const double *vp, int n) {
    if (x <= xp[0]) return vp[0];
    if (x >= xp[n - 1]) return vp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    return vp[lo];
}
"""

# Alias: const clamp and const extrap have the same boundary behavior (clamp to first/last value)
_CUDA_INTERP_HELPERS += r"""
#define jaco_interp1d_const_clamp jaco_interp1d_const_extrap
"""


class JacoCudaCodePrinter(_InterpPrinterMixin, C99CodePrinter):
    """CUDA/HIP printer that emits __device__ helper functions for piecewise interpolation."""

    def _print_PiecewiseLinearInterp(self, expr):
        self._needs_interp_helpers = True
        desired_x, desired_y = _interp_table_names(expr)
        x_name = self._register_table(expr.args[1], desired_x)
        y_name = self._register_table(expr.args[2], desired_y)
        n = len(expr.args[1])
        x_code = self._print(expr.args[0])
        extrap = int(expr.args[3])
        suffix = "extrap" if extrap else "clamp"
        return f"jaco_interp1d_{suffix}({x_code}, {x_name}, {y_name}, {n})"

    def _print_PiecewiseConstantInterp(self, expr):
        self._needs_interp_helpers = True
        desired_x, desired_v = _interp_table_names(expr, yp_prefix="jaco_vp")
        x_name = self._register_table(expr.args[1], desired_x)
        v_name = self._register_table(expr.args[2], desired_v)
        n = len(expr.args[1])
        x_code = self._print(expr.args[0])
        extrap = int(expr.args[3])
        suffix = "extrap" if extrap else "clamp"
        return f"jaco_interp1d_const_{suffix}({x_code}, {x_name}, {v_name}, {n})"

    def get_table_declarations_cuda(self):
        """Return CUDA declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_CUDA_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            vals_str = ", ".join(f"{v:.17e}" for v in values)
            lines.append(f"__constant__ double {name}[] = {{{vals_str}}};")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# C++ helper functions
# --------------------------------------------------------------------------- #

_CPP_INTERP_HELPERS = r"""
#include <array>
#include <cmath>

template <std::size_t N>
inline double jaco_interp1d(double x, const std::array<double, N>& xp, const std::array<double, N>& yp, int extrap) {
    constexpr int n = static_cast<int>(N);
    if (x <= xp[0]) {
        if (extrap) {
            double slope = (yp[1] - yp[0]) / (xp[1] - xp[0]);
            return yp[0] + slope * (x - xp[0]);
        }
        return yp[0];
    }
    if (x >= xp[n - 1]) {
        if (extrap) {
            double slope = (yp[n - 1] - yp[n - 2]) / (xp[n - 1] - xp[n - 2]);
            return yp[n - 1] + slope * (x - xp[n - 1]);
        }
        return yp[n - 1];
    }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x - xp[lo]) / (xp[lo + 1] - xp[lo]);
    return yp[lo] + t * (yp[lo + 1] - yp[lo]);
}

template <std::size_t N, std::size_t M>
inline double jaco_interp1d_const(double x, const std::array<double, N>& xp, const std::array<double, M>& vp, int extrap) {
    constexpr int n = static_cast<int>(N);
    if (x <= xp[0]) return vp[0];
    if (x >= xp[n - 1]) return vp[M - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    return vp[lo];
}
"""


class JacoCppCodePrinter(_InterpPrinterMixin, C99CodePrinter):
    """C++ printer that emits constexpr std::array tables and templated interp helpers."""

    def _print_PiecewiseLinearInterp(self, expr):
        return _print_interp_linear(self, expr)

    def _print_PiecewiseConstantInterp(self, expr):
        return _print_interp_const(self, expr)

    def get_table_declarations_cpp(self):
        """Return C++ declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_CPP_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            n = len(values)
            vals_str = ", ".join(f"{v:.17e}" for v in values)
            lines.append(f"constexpr std::array<double, {n}> {name} = {{{{{vals_str}}}}};")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Python helper functions
# --------------------------------------------------------------------------- #

_PYTHON_INTERP_HELPERS = '''
import numpy as np

def jaco_interp1d(x, xp, yp, n, extrap):
    if x <= xp[0]:
        if extrap:
            slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
            return yp[0] + slope * (x - xp[0])
        return yp[0]
    if x >= xp[n - 1]:
        if extrap:
            slope = (yp[n - 1] - yp[n - 2]) / (xp[n - 1] - xp[n - 2])
            return yp[n - 1] + slope * (x - xp[n - 1])
        return yp[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return yp[lo] + t * (yp[hi] - yp[lo])

def jaco_interp1d_const(x, xp, vp, n, extrap):
    if x <= xp[0]:
        return vp[0]
    if x >= xp[n - 1]:
        return vp[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    return vp[lo]
'''


class JacoPythonCodePrinter(_InterpPrinterMixin, PythonCodePrinter):
    """Python printer that emits calls to jaco_interp1d helper functions."""

    def _print_PiecewiseLinearInterp(self, expr):
        return _print_interp_linear(self, expr)

    def _print_PiecewiseConstantInterp(self, expr):
        return _print_interp_const(self, expr)

    def get_table_declarations_python(self):
        """Return Python declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_PYTHON_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            vals_str = ", ".join(repr(v) for v in values)
            lines.append(f"{name} = np.array([{vals_str}])")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Julia helper functions
# --------------------------------------------------------------------------- #

_JULIA_INTERP_HELPERS = '''
function jaco_interp1d(x, xp, yp, n, extrap)
    if x <= xp[1]
        if extrap == 1
            slope = (yp[2] - yp[1]) / (xp[2] - xp[1])
            return yp[1] + slope * (x - xp[1])
        end
        return yp[1]
    end
    if x >= xp[n]
        if extrap == 1
            slope = (yp[n] - yp[n-1]) / (xp[n] - xp[n-1])
            return yp[n] + slope * (x - xp[n])
        end
        return yp[n]
    end
    lo, hi = 1, n
    while hi - lo > 1
        mid = div(lo + hi, 2)
        if xp[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return yp[lo] + t * (yp[hi] - yp[lo])
end

function jaco_interp1d_const(x, xp, vp, n, extrap)
    if x <= xp[1]
        return vp[1]
    end
    if x >= xp[n]
        return vp[end]
    end
    lo, hi = 1, n
    while hi - lo > 1
        mid = div(lo + hi, 2)
        if xp[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    return vp[lo]
end
'''


class JacoJuliaCodePrinter(_InterpPrinterMixin, JuliaCodePrinter):
    """Julia printer that emits calls to jaco_interp1d helper functions."""

    def _print_PiecewiseLinearInterp(self, expr):
        return _print_interp_linear(self, expr)

    def _print_PiecewiseConstantInterp(self, expr):
        return _print_interp_const(self, expr)

    def get_table_declarations_julia(self):
        """Return Julia declarations for all referenced tables + helper functions."""
        if not self._needs_interp_helpers:
            return ""
        lines = [_JULIA_INTERP_HELPERS]
        for key, (name, values) in self._interp_tables.items():
            vals_str = ", ".join(repr(v) for v in values)
            lines.append(f"const {name} = [{vals_str}]")
        return "\n".join(lines)
