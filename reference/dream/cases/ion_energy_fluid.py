#!/usr/bin/env python3
"""
The ion charge states and the energy of the cold electrons together in DREAM's fluid mode.

The case of cold_energy_fluid.py with dynamic ions: three radii, deuterium singly ionized and a
deposition of neutral argon at the start of the run, uniform in radius. The cold temperature is
self-consistent with ohmic heating from DREAM's conductivity, the radiated power and binding
energy of the ADAS tables in DREAM's default (recombination radiation off, Gould's
bremsstrahlung) and a prescribed heat diffusivity. The ion densities follow DREAM's ADAS rate
equations, and the cold density is the free-electron density of the ions. The radiation, the
binding energy, the conductivity and the rates thus all read the charge states of the run. No
kinetic grids. Backward Euler on equal steps.

test/test_ion_energy_dream.jl runs the same case in Runes2P and compares T_cold, n_cold and the
charge states over time.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/ion_energy_fluid.py <output file.h5>
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
FIELD = 2.0                        # V/m
TEMPERATURE = [50.0, 40.0, 30.0]   # eV, at the cell centres, initial
DEUTERIUM = 1e20                   # m^-3, singly ionized
ARGON = 2e18                       # m^-3, deposited as neutrals
DIFFUSIVITY = 1.0                  # m^2/s
TMAX = 1e-4
STEPS = 100


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

    ds.eqsys.E_field.setPrescribedData(FIELD)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=DEUTERIUM * np.ones(RADII), r=r)
    ds.eqsys.n_i.addIon(name="Ar", Z=18, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=ARGON)

    ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(np.array(TEMPERATURE), radius=r)
    ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)
    ds.eqsys.T_cold.transport.prescribeDiffusion(drr=DIFFUSIVITY)

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
    ds.other.include("fluid/Tcold_ohmic", "fluid/Tcold_radiation", "fluid/Tcold_binding_energy", "fluid/conductivity")
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
        out.attrs["source"] = "ion_energy_fluid.py"
        base.provenance(out)
        out["r"], out["field"], out["deuterium"], out["argon"] = r, FIELD, DEUTERIUM, ARGON
        out["temperature"], out["diffusivity"], out["minor_radius"] = TEMPERATURE, DIFFUSIVITY, MINOR_RADIUS
        with h5py.File(ofile, "r") as res:
            out["t"] = res["grid/t"][:]
            out["r_grid"] = res["grid/r"][:]
            out["T_cold"] = res["eqsys/T_cold"][:]
            out["n_cold"] = res["eqsys/n_cold"][:]
            # (time, charge state of D and Ar in order, radius)
            out["n_i"] = res["eqsys/n_i"][:]
            for q in ("Tcold_ohmic", "Tcold_radiation", "Tcold_binding_energy", "conductivity"):
                out[q] = res[f"other/fluid/{q}"][:]
        print("t[s] T_cold[eV] per radius, n_cold[m^-3] per radius")
        for t, T, n in zip(out["t"][::10], out["T_cold"][::10], out["n_cold"][::10]):
            print(f"{t:.3e} " + " ".join(f"{x:.8g}" for x in T) + " | " + " ".join(f"{x:.6g}" for x in n))
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
