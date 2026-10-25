#!/usr/bin/env python3
"""
The rate equations of the ion charge states in DREAM's fluid mode.

Three radii at prescribed, constant cold temperatures, no field and no kinetic grids. Deuterium
singly ionized as the background, and the deposition of neutral deuterium and argon at the start
of the run, each at the density of the background, uniform in radius (scenario 1 of
examples/Disruptions/generate.py). The ions are dynamic, with
DREAM's ADAS ionization and recombination at (n_cold, T_cold), and the cold density is the
free-electron density of the ions. Backward Euler on equal steps over the 20 µs of the ionization
phase of scenario 1 in 200 steps.

test/test_ion_rates_dream.jl runs the same case in Runes2P and compares the charge states over time.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/ion_rates_fluid.py <output file.h5>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

RADII = 3
MINOR_RADIUS = 0.5
B0 = 1e-6
TEMPERATURE = [20.0, 8.0, 3.0]     # eV, at the cell centres, constant in time
BACKGROUND = 2.6e19                # m^-3, deuterium, singly ionized (ASDEXU.py: ne0)
DEPOSIT_DEUTERIUM = 1.0            # times the background, scenario 1
DEPOSIT_ARGON = 1.0                # times the background, scenario 1
TMAX = 20e-6
STEPS = 200


def settings():
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_THERMAL

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(MINOR_RADIUS)
    ds.radialgrid.setWallRadius(MINOR_RADIUS)
    ds.radialgrid.setNr(RADII)
    r = MINOR_RADIUS * (np.arange(RADII) + 0.5) / RADII

    ds.eqsys.E_field.setPrescribedData(0.0)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=BACKGROUND * np.ones(RADII), r=r)
    ds.eqsys.n_i.addIon(name="Dinj", Z=1, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=DEPOSIT_DEUTERIUM * BACKGROUND)
    ds.eqsys.n_i.addIon(name="Ar", Z=18, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=DEPOSIT_ARGON * BACKGROUND)

    ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)
    ds.eqsys.T_cold.setPrescribedData(np.array(TEMPERATURE), radius=r)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=1e-12)
    ds.solver.setMaxIterations(100)
    ds.timestep.setTmax(TMAX)
    ds.timestep.setNt(STEPS)
    return ds, r


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ds, r = settings()
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        sfile, ofile = Path(tmp) / "settings.h5", Path(tmp) / "output.h5"
        ds.output.setFilename(str(ofile))
        ds.save(str(sfile))
        subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
        out.attrs["source"] = "ion_rates_fluid.py"
        base.provenance(out)
        out["r"], out["temperature"], out["background"] = r, TEMPERATURE, BACKGROUND
        out["deposit_deuterium"], out["deposit_argon"] = DEPOSIT_DEUTERIUM, DEPOSIT_ARGON
        out["minor_radius"] = MINOR_RADIUS
        with h5py.File(ofile, "r") as res:
            out["t"] = res["grid/t"][:]
            out["r_grid"] = res["grid/r"][:]
            # (time, charge state of all species in order D, Dinj, Ar, radius)
            out["n_i"] = res["eqsys/n_i"][:]
            out["n_cold"] = res["eqsys/n_cold"][:]
            out["T_cold"] = res["eqsys/T_cold"][:]
            print("n_i shape", res["eqsys/n_i"].shape)
        print("t[s] n_cold[m^-3] per radius")
        for t, n in zip(out["t"][::20], out["n_cold"][::20]):
            print(f"{t:.3e} " + " ".join(f"{x:.8g}" for x in n))
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
