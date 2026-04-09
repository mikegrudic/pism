"""Metal line cooling for the STARFORGE model.

Loads tabulated cooling rate coefficients from spcool_tables.hdf5 and builds
2-body NBodyProcess objects of the form:

    rate per volume = k(n_Htot, T) * n_e * n_X

where X is each metal species. The table values are normalized such that
multiplying by n_e * n_X yields the cooling rate density at solar abundance;
the 3D (z, n_H, T) table is sliced to a 2D (n_Htot, T) grid at fixed redshift.

The HDF5 file is shipped alongside this module.
"""

from importlib.resources import files
import numpy as np
import h5py

from jaco.processes import NBodyProcess
from jaco.symbols import T, table_interp_2d
from .symbols import n_Htot


def _hdf5_path():
    """Locate the spcool_tables.hdf5 file shipped with the package."""
    return files("jaco.models.starforge").joinpath("spcool_tables.hdf5")

# (HDF5 dataset name, species symbol used in the chemical network)
METAL_SPECIES = [
    ("Carbon_cooling", "C"),
    ("Nitrogen_cooling", "N"),
    ("Oxygen_cooling", "O"),
    ("Neon_cooling", "Ne"),
    ("Magnesium_cooling", "Mg"),
    ("Silicon_cooling", "Si"),
    ("Sulfur_cooling", "S"),
    ("Calcium_cooling", "Ca"),
    ("Iron_cooling", "Fe"),
]


def _load_redshift_slice(dataset_name, z=0.0, hdf5_path=None):
    """Read a 2D (n_Htot, T) slice of the cooling table at the closest redshift to z.

    Returns
    -------
    table_2d : np.ndarray, shape (n_nH, n_T)
        Rate coefficient values (erg cm^3 / s).
    nH_axis, T_axis : np.ndarray
        Linear-space axis values (the table is uniform in log space).
    """
    if hdf5_path is None:
        hdf5_path = _hdf5_path()
    with h5py.File(str(hdf5_path), "r") as f:
        log_T = f["log_T"][:]
        log_nH = f["log_nH"][:]
        log_zp1 = f["log_one_plus_z"][:]
        iz = int(np.argmin(np.abs(log_zp1 - np.log10(1.0 + z))))
        table_2d = f[dataset_name][iz].astype(np.float64)
    return table_2d, 10.0**log_nH, 10.0**log_T


def metal_line_cooling_process(species, dataset, z=0.0):
    """Build a 2-body cooling NBodyProcess for one metal element at fixed redshift.

    Parameters
    ----------
    species : str
        Symbol used in the chemical network (e.g. "C", "O", "Fe").
    dataset : str
        HDF5 dataset name in spcool_tables.hdf5 (e.g. "Carbon_cooling").
    z : float
        Redshift slice to use.

    Returns
    -------
    NBodyProcess
        2-body process between e- and the metal species. Cooling rate per
        volume is `k(n_Htot, T) * n_e * n_species`.
    """
    table_2d, nH_axis, T_axis = _load_redshift_slice(dataset, z=z)
    table_name = f"{dataset}_z{int(round(z))}"
    k = table_interp_2d(
        table_name, table_2d, [nH_axis, T_axis], n_Htot, T,
        log_axes=[True, True],
    )
    return NBodyProcess(
        colliding_species=["e-", species],
        heat_rate_coefficient=-k,  # negative -> energy lost from gas
        name=f"{species} line cooling",
        bibliography=["2009MNRAS.393...99W"],  # Wiersma+ tables used in GIZMO
    )


def metal_line_cooling(z=0.0):
    """Build the combined metal-line cooling process for all tabulated species."""
    return sum(metal_line_cooling_process(sp, ds, z=z)
               for ds, sp in METAL_SPECIES)
