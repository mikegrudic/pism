from jaco.processes import CollisionalIonization, GasPhaseRecombination
import numpy as np
from matplotlib import pyplot as plt


def test_CIE():
    """Solve for CIE of a H-He mixture and check all abundances against CHIANTI."""
    chianti_He = np.load("tests/chianti_He_abundances.npy")
    chianti_H = np.load("tests/chianti_H_abundances.npy")

    # Use the He grid (same T range for both)
    N = len(chianti_He)
    Tmin, Tmax = chianti_He[:, 0].min(), chianti_He[:, 0].max()

    processes = [CollisionalIonization(s) for s in ("H", "He", "He+")] + [
        GasPhaseRecombination(i) for i in ("H+", "He+", "He++")
    ]
    system = sum(processes)

    Tgrid = np.logspace(np.log10(Tmin), np.log10(Tmax), N)
    ngrid = np.ones_like(Tgrid)

    knowns = {"T": Tgrid, "n_Htot": ngrid}
    guesses = {
        "H+": 0.9 * np.ones_like(Tgrid),
        "He+": 1e-2 * np.ones_like(Tgrid),
        "He++": 1e-2 * np.ones_like(Tgrid),
    }
    sol = system.solve(knowns, guesses, tol=1e-3)

    # Convert jaco abundances to ion fractions (fraction of element)
    He_total = sol["He"] + sol["He+"] + sol["He++"]
    H_total = sol["H"] + sol["H+"]

    jaco_He_frac = np.column_stack([sol["He"] / He_total, sol["He+"] / He_total, sol["He++"] / He_total])
    jaco_H_frac = np.column_stack([sol["H"] / H_total, sol["H+"] / H_total])

    # Interpolate CHIANTI onto our T grid
    chianti_He_interp = np.column_stack(
        [np.interp(np.log10(Tgrid), np.log10(chianti_He[:, 0]), chianti_He[:, i]) for i in (1, 2, 3)]
    )
    chianti_H_interp = np.column_stack(
        [np.interp(np.log10(Tgrid), np.log10(chianti_H[:, 0]), chianti_H[:, i]) for i in (1, 2)]
    )

    # --- Comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H panel
    ax = axes[0]
    ax.set_title("Hydrogen CIE")
    for i, (label, color) in enumerate(zip(["H", "H+"], ["blue", "orange"])):
        ax.loglog(Tgrid, np.clip(jaco_H_frac[:, i], 1e-8, None), color=color, lw=2, label=f"jaco {label}")
        ax.loglog(Tgrid, np.clip(chianti_H_interp[:, i], 1e-8, None), color=color, ls="--", lw=1.5, label=f"CHIANTI {label}")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Ion fraction")
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=9)

    # He panel
    ax = axes[1]
    ax.set_title("Helium CIE")
    for i, (label, color) in enumerate(zip(["He", "He+", "He++"], ["green", "red", "magenta"])):
        ax.loglog(Tgrid, np.clip(jaco_He_frac[:, i], 1e-8, None), color=color, lw=2, label=f"jaco {label}")
        ax.loglog(Tgrid, np.clip(chianti_He_interp[:, i], 1e-8, None), color=color, ls="--", lw=1.5, label=f"CHIANTI {label}")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Ion fraction")
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("tests/CIE_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved tests/CIE_comparison.png")

    # --- Assertions: check all species ---
    # Tolerance is loose (0.2) because jaco and CHIANTI use different rate coefficients
    tol = 0.15

    for i, name in enumerate(["He", "He+", "He++"]):
        maxerr = np.abs(jaco_He_frac[:, i] - chianti_He_interp[:, i]).max()
        print(f"{name}: max |jaco - CHIANTI| = {maxerr:.4f}")
        assert maxerr < tol, f"{name} ion fraction deviates from CHIANTI by {maxerr:.4f} (tol={tol})"

    for i, name in enumerate(["H", "H+"]):
        maxerr = np.abs(jaco_H_frac[:, i] - chianti_H_interp[:, i]).max()
        print(f"{name}: max |jaco - CHIANTI| = {maxerr:.4f}")
        assert maxerr < tol, f"{name} ion fraction deviates from CHIANTI by {maxerr:.4f} (tol={tol})"


def generate_CIE_testdata():
    import ChiantiPy.core as ch

    Tgrid = np.logspace(3, 6, 10**4)

    he = ch.ioneq(2)
    he.load()
    he.calculate(Tgrid)
    np.save("tests/chianti_He_abundances.npy", np.c_[Tgrid, he.Ioneq])

    h = ch.ioneq(1)
    h.load()
    h.calculate(Tgrid)
    np.save("tests/chianti_H_abundances.npy", np.c_[Tgrid, h.Ioneq.T])
