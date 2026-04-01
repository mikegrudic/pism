"""Custom sympy Function classes for piecewise interpolation that generate clean code."""

import sympy as sp
from sympy.core.symbol import Str
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# --------------------------------------------------------------------------- #
# Table registry for large multi-dimensional tables
# --------------------------------------------------------------------------- #

_TABLE_REGISTRY = {}


def register_table(name, data, axes, log_axes=None):
    """Register a multi-dimensional table for code generation.

    Parameters
    ----------
    name : str
        Unique name for the table, used as identifier in generated code.
    data : np.ndarray
        N-dimensional array of table values (e.g. shape (ny, nx) for 2D).
    axes : list of np.ndarray
        List of 1D arrays for each axis, in order. Must be uniformly spaced
        (in linear or log space).
    log_axes : list of bool, optional
        Whether each axis is log-spaced. Defaults to False for all axes.
    """
    ndim = data.ndim
    if len(axes) != ndim:
        raise ValueError(f"Expected {ndim} axes for {ndim}D data, got {len(axes)}")
    if log_axes is None:
        log_axes = [False] * ndim

    table = {
        "data": np.ascontiguousarray(data, dtype=np.float64),
        "axes": [np.asarray(a, dtype=np.float64) for a in axes],
        "log_axes": list(log_axes),
        "ndim": ndim,
        "shape": data.shape,
    }

    # Validate uniform spacing per axis
    for i, (ax, is_log) in enumerate(zip(axes, log_axes)):
        vals = np.log(ax) if is_log else np.array(ax)
        diffs = np.diff(vals)
        if not np.allclose(diffs, diffs[0], rtol=1e-10):
            raise ValueError(f"Axis {i} is not uniformly spaced (in {'log' if is_log else 'linear'} space)")
        table[f"axis{i}_min"] = float(ax[0])
        table[f"axis{i}_max"] = float(ax[-1])

    _TABLE_REGISTRY[name] = table
    return name


def get_table(name):
    """Retrieve a registered table by name."""
    return _TABLE_REGISTRY[name]


def save_table_hdf5(name, filename=None):
    """Save a registered table to an HDF5 file.

    Parameters
    ----------
    name : str
        Registered table name.
    filename : str, optional
        Output filename. Defaults to '{name}.h5'.
    """
    import h5py
    table = _TABLE_REGISTRY[name]
    if filename is None:
        filename = f"{name}.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=table["data"])
        for i, ax in enumerate(table["axes"]):
            f.create_dataset(f"axis{i}", data=ax)
        f.attrs["ndim"] = table["ndim"]
        for i, is_log in enumerate(table["log_axes"]):
            f.attrs[f"axis{i}_log"] = int(is_log)


class PiecewiseLinearInterp(sp.Function):
    """Piecewise-linear interpolation of tabulated data.

    Arguments: (x, X_tuple, Y_tuple, extrapolate_flag[, name])
        x: sympy expression for the interpolation variable
        X_tuple: sympy.Tuple of Float values (breakpoints, strictly increasing)
        Y_tuple: sympy.Tuple of Float values (function values at breakpoints)
        extrapolate_flag: sympy.Integer(0) to clamp, sympy.Integer(1) to extrapolate
        name: sympy.core.symbol.Str, descriptive identifier for code generation (optional)
    """

    @classmethod
    def eval(cls, x, X_tuple, Y_tuple, extrapolate_flag, name=None):
        # If x is a pure number, evaluate numerically
        if x.is_Number:
            return cls._evaluate(float(x), X_tuple, Y_tuple, int(extrapolate_flag))
        return None

    @staticmethod
    def _evaluate(xval, X_tuple, Y_tuple, extrapolate):
        """Numerically evaluate the piecewise-linear interpolant at xval."""
        X = [float(v) for v in X_tuple]
        Y = [float(v) for v in Y_tuple]
        n = len(X)
        if xval <= X[0]:
            if extrapolate:
                slope = (Y[1] - Y[0]) / (X[1] - X[0])
                return sp.Float(Y[0] + slope * (xval - X[0]))
            return sp.Float(Y[0])
        if xval >= X[-1]:
            if extrapolate:
                slope = (Y[-1] - Y[-2]) / (X[-1] - X[-2])
                return sp.Float(Y[-1] + slope * (xval - X[-1]))
            return sp.Float(Y[-1])
        # Binary search
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if X[mid] <= xval:
                lo = mid
            else:
                hi = mid
        t = (xval - X[lo]) / (X[hi] - X[lo])
        return sp.Float(Y[lo] + t * (Y[hi] - Y[lo]))

    @property
    def table_name(self):
        """Return the descriptive name string, or None if not set."""
        if len(self.args) > 4:
            return str(self.args[4])
        return None

    def _eval_derivative(self, s):
        x = self.args[0]
        if not x.has(s):
            return sp.S.Zero
        X_tuple = self.args[1]
        Y_tuple = self.args[2]
        extrap = self.args[3]
        X = [float(v) for v in X_tuple]
        Y = [float(v) for v in Y_tuple]
        slopes = [(Y[i + 1] - Y[i]) / (X[i + 1] - X[i]) for i in range(len(X) - 1)]
        slopes_tuple = sp.Tuple(*[sp.Float(s_) for s_ in slopes])
        name = self.table_name
        deriv_args = [x, X_tuple, slopes_tuple, extrap]
        if name is not None:
            deriv_args.append(Str(name + "_slopes"))
        return PiecewiseConstantInterp(*deriv_args) * x.diff(s)

    def _eval_evalf(self, prec):
        x = self.args[0]
        if x.is_Number:
            return self._evaluate(
                float(x), self.args[1], self.args[2], int(self.args[3])
            )
        return self

    def doit(self, **hints):
        """Expand to a sympy Piecewise expression (fallback for printers that don't support this type)."""
        x = self.args[0]
        X = [float(v) for v in self.args[1]]
        Y = [float(v) for v in self.args[2]]
        extrapolate = bool(self.args[3])

        slopes = np.diff(Y) / np.diff(X)
        cases = []
        if extrapolate:
            cases.append((Y[0] + slopes[0] * (x - X[0]), x < X[0]))
            cases.append((Y[-1] + slopes[-1] * (x - X[-1]), x >= X[-1]))
        else:
            cases.append((sp.Float(Y[0]), x < X[0]))
            cases.append((sp.Float(Y[-1]), x >= X[-1]))

        for i in range(len(X) - 1):
            cases.append((Y[i] + slopes[i] * (x - X[i]), (x >= X[i]) & (x < X[i + 1])))
        return sp.Piecewise(*cases)


class PiecewiseConstantInterp(sp.Function):
    """Piecewise-constant interpolation (used for derivatives of piecewise-linear interpolants).

    Arguments: (x, X_tuple, values_tuple, extrapolate_flag[, name])
        x: sympy expression for the interpolation variable
        X_tuple: sympy.Tuple of Float values (breakpoints)
        values_tuple: sympy.Tuple of Float values (constant values on each interval, length = len(X)-1)
        extrapolate_flag: sympy.Integer(0) or sympy.Integer(1)
        name: sympy.core.symbol.Str, descriptive identifier for code generation (optional)
    """

    @property
    def table_name(self):
        """Return the descriptive name string, or None if not set."""
        if len(self.args) > 4:
            return str(self.args[4])
        return None

    @classmethod
    def eval(cls, x, X_tuple, values_tuple, extrapolate_flag, name=None):
        if x.is_Number:
            return cls._evaluate(float(x), X_tuple, values_tuple)
        return None

    @staticmethod
    def _evaluate(xval, X_tuple, values_tuple):
        """Numerically evaluate the piecewise-constant interpolant at xval."""
        X = [float(v) for v in X_tuple]
        vals = [float(v) for v in values_tuple]
        n = len(X)
        if xval <= X[0]:
            return sp.Float(vals[0])
        if xval >= X[-1]:
            return sp.Float(vals[-1])
        # Binary search
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if X[mid] <= xval:
                lo = mid
            else:
                hi = mid
        return sp.Float(vals[lo])

    def _eval_derivative(self, s):
        # Derivative of piecewise-constant is zero
        return sp.S.Zero

    def _eval_evalf(self, prec):
        x = self.args[0]
        if x.is_Number:
            return self._evaluate(float(x), self.args[1], self.args[2])
        return self

    def doit(self, **hints):
        """Expand to a sympy Piecewise expression."""
        x = self.args[0]
        X = [float(v) for v in self.args[1]]
        vals = [float(v) for v in self.args[2]]
        extrapolate = bool(self.args[3])

        cases = []
        # Below first breakpoint
        cases.append((sp.Float(vals[0]), x < X[0]))
        # Above last breakpoint
        cases.append((sp.Float(vals[-1]), x >= X[-1]))
        # Interior intervals
        for i in range(len(X) - 1):
            cases.append((sp.Float(vals[i]), (x >= X[i]) & (x < X[i + 1])))
        return sp.Piecewise(*cases)


# --------------------------------------------------------------------------- #
# Multi-dimensional table interpolation
# --------------------------------------------------------------------------- #

class TableInterp2D(sp.Function):
    """Bilinear interpolation on a 2D regular grid, referencing data in the table registry.

    Arguments: (x, y, table_name)
        x, y: sympy expressions for the interpolation variables
        table_name: Str identifying the registered table
    """

    @classmethod
    def eval(cls, x, y, table_name):
        if x.is_Number and y.is_Number:
            return cls._evaluate(float(x), float(y), str(table_name))
        return None

    @staticmethod
    def _evaluate(xval, yval, name):
        table = get_table(name)
        interp = RegularGridInterpolator(
            tuple(table["axes"]), table["data"],
            method="linear", bounds_error=False, fill_value=None,
        )
        return sp.Float(float(interp([[xval, yval]])[0]))

    @property
    def table_name(self):
        return str(self.args[2])

    def _eval_derivative(self, s):
        x, y = self.args[0], self.args[1]
        table_name = self.args[2]
        result = sp.S.Zero
        if x.has(s):
            result += TableInterp2D_dx(x, y, table_name) * x.diff(s)
        if y.has(s):
            result += TableInterp2D_dy(x, y, table_name) * y.diff(s)
        return result

    def _eval_evalf(self, prec):
        x, y = self.args[0], self.args[1]
        if x.is_Number and y.is_Number:
            return self._evaluate(float(x), float(y), str(self.args[2]))
        return self


class TableInterp2D_dx(sp.Function):
    """Partial derivative of TableInterp2D w.r.t. its first argument (x).

    Computed analytically from the bilinear formula — no extra table needed.
    """

    @classmethod
    def eval(cls, x, y, table_name):
        if x.is_Number and y.is_Number:
            return cls._evaluate(float(x), float(y), str(table_name))
        return None

    @staticmethod
    def _evaluate(xval, yval, name):
        """Evaluate x-partial derivative via finite difference of the bilinear interp."""
        table = get_table(name)
        eps = (table["axis0_max"] - table["axis0_min"]) / (table["shape"][0] - 1) * 1e-6
        f_plus = float(TableInterp2D._evaluate(xval + eps, yval, name))
        f_minus = float(TableInterp2D._evaluate(xval - eps, yval, name))
        return sp.Float((f_plus - f_minus) / (2 * eps))

    @property
    def table_name(self):
        return str(self.args[2])

    def _eval_derivative(self, s):
        return sp.S.Zero  # second derivatives not supported


class TableInterp2D_dy(sp.Function):
    """Partial derivative of TableInterp2D w.r.t. its second argument (y)."""

    @classmethod
    def eval(cls, x, y, table_name):
        if x.is_Number and y.is_Number:
            return cls._evaluate(float(x), float(y), str(table_name))
        return None

    @staticmethod
    def _evaluate(xval, yval, name):
        table = get_table(name)
        eps = (table["axis1_max"] - table["axis1_min"]) / (table["shape"][1] - 1) * 1e-6
        f_plus = float(TableInterp2D._evaluate(xval, yval + eps, name))
        f_minus = float(TableInterp2D._evaluate(xval, yval - eps, name))
        return sp.Float((f_plus - f_minus) / (2 * eps))

    @property
    def table_name(self):
        return str(self.args[2])

    def _eval_derivative(self, s):
        return sp.S.Zero


class TableInterp3D(sp.Function):
    """Trilinear interpolation on a 3D regular grid, referencing data in the table registry.

    Arguments: (x, y, z, table_name)
    """

    @classmethod
    def eval(cls, x, y, z, table_name):
        if x.is_Number and y.is_Number and z.is_Number:
            return cls._evaluate(float(x), float(y), float(z), str(table_name))
        return None

    @staticmethod
    def _evaluate(xval, yval, zval, name):
        table = get_table(name)
        interp = RegularGridInterpolator(
            tuple(table["axes"]), table["data"],
            method="linear", bounds_error=False, fill_value=None,
        )
        return sp.Float(float(interp([[xval, yval, zval]])[0]))

    @property
    def table_name(self):
        return str(self.args[3])

    def _eval_derivative(self, s):
        x, y, z = self.args[0], self.args[1], self.args[2]
        table_name = self.args[3]
        result = sp.S.Zero
        if x.has(s):
            result += TableInterp3D_dx(x, y, z, table_name) * x.diff(s)
        if y.has(s):
            result += TableInterp3D_dy(x, y, z, table_name) * y.diff(s)
        if z.has(s):
            result += TableInterp3D_dz(x, y, z, table_name) * z.diff(s)
        return result

    def _eval_evalf(self, prec):
        x, y, z = self.args[0], self.args[1], self.args[2]
        if x.is_Number and y.is_Number and z.is_Number:
            return self._evaluate(float(x), float(y), float(z), str(self.args[3]))
        return self


class TableInterp3D_dx(sp.Function):
    """Partial derivative of TableInterp3D w.r.t. x."""
    @classmethod
    def eval(cls, x, y, z, table_name):
        return None
    @property
    def table_name(self):
        return str(self.args[3])
    def _eval_derivative(self, s):
        return sp.S.Zero


class TableInterp3D_dy(sp.Function):
    """Partial derivative of TableInterp3D w.r.t. y."""
    @classmethod
    def eval(cls, x, y, z, table_name):
        return None
    @property
    def table_name(self):
        return str(self.args[3])
    def _eval_derivative(self, s):
        return sp.S.Zero


class TableInterp3D_dz(sp.Function):
    """Partial derivative of TableInterp3D w.r.t. z."""
    @classmethod
    def eval(cls, x, y, z, table_name):
        return None
    @property
    def table_name(self):
        return str(self.args[3])
    def _eval_derivative(self, s):
        return sp.S.Zero
