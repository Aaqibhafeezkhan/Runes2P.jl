#!/usr/bin/env python3
"""
DREAM kinetic Dreicer rate at the physics of the kinetic Dreicer test of Runes2P.

Deuterium at 1 keV and 1e20 m^-3, one radius, hot-tail grid only with the same faces as
the Runes2P test (biuniform 110 cells to p_max = 2 with 80 below 0.4, 14 uniform
pitch cells). Full collision mode, complete screening, centred interpolation, f = 0 beyond p_max, no synchrotron, no particle source. The fluid
Dreicer and avalanche sources are off, so n_re only collects the flux through p_max and
its time derivative is the kinetic runaway rate. The bulk is not held. Three variants of
the Coulomb logarithm: constant (relativistic, as in the Runes2P test), thermal and energy
dependent.

The file records the DREAM commit, branch, PETSc version and creation time, and stores the
complete DREAM settings of every run under `<variant>/<case>/settings`, so that the test
(test/test_runaway_physics.jl) compares against recorded values without running DREAM.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/dreicer_kinetic.py out/<set>/dream_dreicer_kinetic.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

# The fields and collision times of runaway_rates.py, from the file of the same data set: the
# directory of the output (run runaway_rates.py first), so that both files come from the same
# DREAM build.
REFERENCE_NAME = "dream_runaway_rates.h5"
TEMPERATURE = 1000.0
DENSITY = 1e20
A_MINOR = 0.5
B0 = 5.0
# (E/E_c, settling time in thermal collision times), as in the Runes2P test
CASES = [(10.0, 1600), (25.0, 600), (40.0, 400)]
NT = 60


def p_faces():
    """biuniform_p_faces(110, 80, 0.0, 0.4, 2.0) of Runes2P."""
    return np.concatenate([np.linspace(0.0, 0.4, 81), np.linspace(0.4, 2.0, 31)[1:]])


def settings(field, tmax, lnlambda):
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
    ds.collisions.lnlambda = lnlambda

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(1)

    ds.eqsys.E_field.setPrescribedData(field)
    ds.eqsys.T_cold.setPrescribedData(TEMPERATURE)
    ds.eqsys.n_cold.setPrescribedData(DENSITY)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_PRESCRIBED, n=DENSITY)

    ds.hottailgrid.setCustomGrid(p_f=list(p_faces()), xi_f=list(np.linspace(-1, 1, 15)))
    ds.eqsys.f_hot.setInitialProfiles(n0=DENSITY, T0=TEMPERATURE)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
    ds.eqsys.n_re.setInitialProfile(density=0)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.timestep.setTmax(tmax)
    ds.timestep.setNt(NT)
    return ds


VARIANTS = {"constant": Collisions.LNLAMBDA_CONSTANT, "thermal": Collisions.LNLAMBDA_THERMAL,
            "energy_dependent": Collisions.LNLAMBDA_ENERGY_DEPENDENT}


def main(outfile):
    with h5py.File(Path(outfile).parent / REFERENCE_NAME, "r") as ref:
        temperatures = ref["temperature"][:]
        over_critical = ref["E_over_Ec"][:]
        fields = ref["connor_hastie_corrected/E_field"][:]
        tau = ref["connor_hastie_corrected/tauEETh"][:]
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "dreicer_kinetic.py"
        base.provenance(out)
        out["temperature"] = TEMPERATURE
        out["density"] = DENSITY
        out["B"] = B0
        out["a"] = A_MINOR
        for name, lnlambda in VARIANTS.items():
            for e_over_ec, times in CASES:
                k = int(np.flatnonzero((temperatures == TEMPERATURE) & (over_critical == e_over_ec))[0])
                ds = settings(fields[k], times * tau[k], lnlambda)
                sfile, ofile = Path(tmp) / "s.h5", Path(tmp) / "o.h5"
                ds.output.setFilename(str(ofile))
                ds.save(str(sfile))
                subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
                g = out.create_group(f"{name}/E_over_Ec_{e_over_ec:g}")
                g.attrs["lnlambda"] = int(lnlambda)
                g.attrs["collision_times"] = times
                base.bare_output_filename(sfile)
                with h5py.File(sfile, "r") as sett:
                    for key in sett:
                        sett.copy(sett[key], g, name=f"settings/{key}")
                with h5py.File(ofile, "r") as res:
                    g["E_field"] = fields[k]
                    g["E_over_Ec"] = e_over_ec
                    g["t"] = res["grid/t"][:]
                    g["n_re"] = res["eqsys/n_re"][:, 0]
                    g.create_dataset("f_hot_initial", data=res["eqsys/f_hot"][0], compression="gzip")
                    g.create_dataset("f_hot_final", data=res["eqsys/f_hot"][-1], compression="gzip")
                    if "grid" not in out:
                        for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                            out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                        out["grid/r_f"] = res["grid/r_f"][:]
                    rate = np.diff(g["n_re"][:]) / np.diff(g["t"][:])
                    print(f"{name:17s} E/Ec={e_over_ec:4g}  rate={rate[-1]:.4e}  "
                          f"last/prev5={rate[-1] / rate[-6]:.4f}")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
