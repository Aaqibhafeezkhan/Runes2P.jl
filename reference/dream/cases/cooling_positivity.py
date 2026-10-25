#!/usr/bin/env python3
"""
DREAM comparison for the positivity of f in a cooling plasma: a fully kinetic run with a
prescribed, time-dependent T(t) and fixed E, to see whether DREAM's own centred and TCDF
interpolation show the same qualitative picture (central goes negative, TCDF does not)
as Runes2P, independently of it. A reduced grid and step count (Np, Nxi, Nt below) by
default, not a round-off comparison: DREAM's NONLINEAR solver may be slow at the full
resolution, and this is exploratory, not a DREAM-compatible port.

Pure deuterium, n = 1e20 m^-3, one radius, full collision mode, BC_F_0. T(t) = T0
(Tf/T0)^(t/tend), E fixed at E_OVER_EC * E_c(n, thermal lnLambda at T0) -- the same
schedule the Runes2P comparison uses, discretized at NT_PRESCRIBE points and given to
DREAM as prescribed data (DREAM interpolates between them; NT_PRESCRIBE is independent of
the solver's own Nt).

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/cooling_positivity.py out/<set>/dream_cooling_positivity.h5

Switches after the file name (for a comparison at a finer resolution): np=, nxi=, nt=, t_final=, t_end=, reltol=, maxiter=,
nt_prescribe=, save= (number of saved steps, 0 saves every step) and variants= (centred,
tcdf, tcdf_upwind: TCDF with AD_INTERP_JACOBIAN_UPWIND). Without switches the run is the
reduced one described above. The file stores the table of T(t), so that Runes2P can use the
same piecewise-linear temperature.
    python <set>/cases/cooling_positivity.py <file> np=200 nt=1000 t_final=10 reltol=1e-12 save=10
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Solver as Solver
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

N_D = 1.0e20
B0 = 5.3
A_MINOR = 0.22
T0 = 15e3
T_FINAL = 100.0
T_END = 1e-3
E_OVER_EC = 20.0
NP, NXI, PMAX = 40, 5, 2.0
NT_PRESCRIBE = 200   # points describing T(t) to DREAM, independent of the solver's own Nt
NT = 100             # DREAM's own solver steps
RELTOL = 1e-6

SAVE = 0             # saved steps, 0: every step
MAXITER = 100

# name: (interpolation, Jacobian interpolation or None for DREAM's default)
VARIANTS = {
    "centred": (FHot.AD_INTERP_CENTRED, None),
    "tcdf": (FHot.AD_INTERP_TCDF, None),
    "tcdf_upwind": (FHot.AD_INTERP_TCDF, FHot.AD_INTERP_JACOBIAN_UPWIND),
}
SELECTED = ["centred", "tcdf"]


def temperature_table():
    t_grid = np.linspace(0, T_END, NT_PRESCRIBE)
    return t_grid, T0 * (T_FINAL / T0) ** (t_grid / T_END)


def critical_field(n, lnlambda):
    """Connor-Hastie critical field [V/m] (matches critical_field of Runes2P)."""
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    me = 9.1093837015e-31
    c = 299792458.0
    return n * e**3 * lnlambda / (4 * np.pi * eps0**2 * me * c**2)


def thermal_lnlambda(T, n):
    return 14.9 - 0.5 * np.log(n / 1e20) + np.log(T / 1e3)


def settings(ad_int, ad_jac=None):
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_THERMAL

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(1)

    t_grid, T_of_t = temperature_table()
    ds.eqsys.T_cold.setPrescribedData(T_of_t.reshape(-1, 1), times=t_grid)

    lnL0 = thermal_lnlambda(T0, N_D)
    E = E_OVER_EC * critical_field(N_D, lnL0)
    ds.eqsys.E_field.setPrescribedData(E)

    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=N_D)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_PRESCRIBED)
    ds.eqsys.n_cold.setPrescribedData(N_D)

    ds.hottailgrid.setNxi(NXI)
    ds.hottailgrid.setNp(NP)
    ds.hottailgrid.setPmax(PMAX)

    p = (np.arange(NP) + 0.5) * PMAX / NP
    xi_f = np.linspace(-1, 1, NXI + 1)
    xi = 0.5 * (xi_f[1:] + xi_f[:-1])
    theta0 = T0 / 5.10998950e5
    f0 = np.exp(-p**2 / (2 * theta0))
    f0 = np.tile(f0, (NXI, 1))  # (nxi, np), isotropic
    ds.eqsys.f_hot.setInitialValue(f0.reshape(1, NXI, NP), r=[0.0], p=p, xi=xi)
    if ad_jac is None:
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=ad_int)
    else:
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=ad_int, ad_jac=ad_jac)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=RELTOL)
    ds.solver.setMaxIterations(maxiter=MAXITER)
    ds.solver.setSaveStatistics(True)
    ds.timestep.setTmax(T_END)
    ds.timestep.setNt(NT)
    if SAVE > 0:
        ds.timestep.setNumberOfSaveSteps(SAVE)
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "cooling_positivity.py"
        base.provenance(out)
        t_grid, T_of_t = temperature_table()
        out["temperature/t"] = t_grid
        out["temperature/T"] = T_of_t
        out["n"] = N_D
        out["B"] = B0
        out["E"] = E_OVER_EC * critical_field(N_D, thermal_lnlambda(T0, N_D))
        for key, value in (("np", NP), ("nxi", NXI), ("p_max", PMAX), ("nt", NT), ("t_end", T_END),
                           ("t_final", T_FINAL), ("reltol", RELTOL), ("maxiter", MAXITER)):
            out.attrs[key] = value
        for name in SELECTED:
            ad_int, ad_jac = VARIANTS[name]
            ds = settings(ad_int, ad_jac)
            sfile, ofile = Path(tmp) / f"s_{name}.h5", Path(tmp) / f"o_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.output.setTiming(stdout=False, file=False)
            ds.save(str(sfile))
            result = subprocess.run([str(base.DREAMI), str(sfile)], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            print(f"--- {name}: dreami exit code {result.returncode} ---")
            if result.returncode != 0:
                print(result.stdout[-4000:])
                out.create_group(name).attrs["failed"] = result.returncode
                continue
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                if "grid" not in out:
                    for q in ("p1_f", "p2_f"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                g["t"] = res["grid/t"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:], compression="gzip")
                g["n_re"] = res["eqsys/n_re"][:]
                g["newton_iterations"] = res["solver/iterations"][:]
                fmin = res["eqsys/f_hot"][:].min()
                neg = int((res["eqsys/f_hot"][:] < 0).sum())
                print(f"{name}: min(f) = {fmin:.3e}, negative cells (all saves) = {neg}")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    for switch in sys.argv[2:]:
        key, value = switch.split("=", 1)
        if key == "variants":
            SELECTED = value.split(",")
        elif key in ("np", "nxi", "nt", "nt_prescribe", "save", "maxiter"):
            globals()[{"np": "NP", "nxi": "NXI", "nt": "NT", "nt_prescribe": "NT_PRESCRIBE", "save": "SAVE",
                       "maxiter": "MAXITER"}[key]] = int(value)
        elif key in ("t_final", "t_end", "reltol"):
            globals()[{"t_final": "T_FINAL", "t_end": "T_END", "reltol": "RELTOL"}[key]] = float(value)
        else:
            raise SystemExit(f"unknown switch {key}")
    main(sys.argv[1])
