#!/usr/bin/env python3
"""
DREAM's starts of a run and the heating of the cold electrons by a runaway grid.

`case=coronal`: the coronal start of DREAM's examples/ionequil/generate.py: deuterium singly
ionized at 5e19 m^-3 next to neon at 1e19 m^-3 started in coronal equilibrium (init_equil), at
the 800 eV of the example (group `uniform`) and, as a choice of this package, at a temperature
falling linearly from 800 eV to 2 eV over the radii (group `profile`), so that the charge states
of neon change across the radii. One step of the example (1e-6 s): the comparison is of the
charge states at t = 0.

`case=shapes`: the runaway grid of DREAM's examples/init_f_re/generate.py (deuterium fully ionized
at 5e19 m^-3, 100 eV, 100 x 40 cells to p = 50 with the pitch grid biuniform in theta, n_re =
1e16 m^-3) with f_re started from n_re in each of the four shapes INIT_FORWARD, INIT_XI_POSITIVE,
INIT_XI_NEGATIVE and INIT_ISOTROPIC, one group each, in place of the analytic avalanche start of
the example (line 67), a choice of this package. As a choice of this package two radii with
fields of opposite sign, +1.5 and -1.5 V/m, so that INIT_FORWARD picks both sides. One step of the
example: the comparison is of f_re at t = 0.

`case=heating`: the same plasma on one radius, the field of the example (1.5 V/m), f_re started
in INIT_FORWARD and T_cold self-consistent from 100 eV (choices of this package: the example
starts from the avalanche shape and prescribes T_cold), in the 20 steps of the example over
5e-6 s in place of its 1e-3 s (a choice of this package, so that the ohmic heating keeps T_cold
near the 100 eV of the example: it rises to 101.4 eV). DREAM's equation of T_cold then
holds the collisional heating by f_re (src/Settings/Equations/T_cold.cpp:341-354), saved as
other/fluid/Tcold_fre_coll with the states it is evaluated from.

The tests of Runes2P read these files for the coronal start (fill_coronal_equilibrium!), the
runaway shapes (fill_runaway_distribution!) and the heating by the runaway grid.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <build>/cases/initial_state.py <output file.h5> case=<coronal|shapes|heating>

The citations of DREAM's files are at DREAM v26.5.
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
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectronDistribution as FRe
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

# examples/ionequil/generate.py
EQUIL_FIELD = 0.1          # V/m, line 19
EQUIL_DEUTERIUM = 5e19     # m^-3, line 20
EQUIL_TEMPERATURE = 800.0  # eV, line 21
EQUIL_NEON = 1e19          # m^-3, line 32
EQUIL_TMAX = 1e-4 / 100    # one step of the example: tMax / nt, lines 23-24
EQUIL_B0, EQUIL_A, EQUIL_B, EQUIL_NR = 5.0, 0.42, 0.44, 10   # lines 37-40
PROFILE_LOW = 2.0          # eV, at the edge of the profile, a choice of this package

# examples/init_f_re/generate.py
FRE_FIELD = 1.5            # V/m, line 35
FRE_DENSITY = 5e19         # m^-3, line 36
FRE_TEMPERATURE = 100.0    # eV, line 37
FRE_NRE = 1e16             # m^-3, line 57
FRE_NP, FRE_PMAX, FRE_NXI = 100, 50.0, 40   # lines 61-63
FRE_TMAX, FRE_NT = 1e-3, 20                 # lines 43-44
HEATING_TMAX = 5e-6        # s, the run of case=heating, a choice of this package
FRE_B0, FRE_A = 5.0, 0.22                   # lines 71-73
SHAPES = {"forward": FRe.INIT_FORWARD, "xi_positive": FRe.INIT_XI_POSITIVE,
          "xi_negative": FRe.INIT_XI_NEGATIVE, "isotropic": FRe.INIT_ISOTROPIC}


def run(ds, tmp, name):
    sfile, ofile = Path(tmp) / f"settings_{name}.h5", Path(tmp) / f"output_{name}.h5"
    ds.output.setFilename(str(ofile))
    ds.save(str(sfile))
    subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
    return ofile


def coronal(temperature):
    ds = DREAMSettings()
    r = EQUIL_A * (np.arange(EQUIL_NR) + 0.5) / EQUIL_NR
    ds.radialgrid.setB0(EQUIL_B0)
    ds.radialgrid.setMinorRadius(EQUIL_A)
    ds.radialgrid.setWallRadius(EQUIL_B)
    ds.radialgrid.setNr(EQUIL_NR)
    ds.eqsys.E_field.setPrescribedData(EQUIL_FIELD)
    ds.eqsys.T_cold.setPrescribedData(np.asarray(temperature), radius=r)
    ds.eqsys.n_i.addIon(name="D", Z=1, iontype=Ions.IONS_DYNAMIC, Z0=1, n=EQUIL_DEUTERIUM)
    ds.eqsys.n_i.addIon(name="Ne", Z=10, iontype=Ions.IONS_DYNAMIC, n=EQUIL_NEON, init_equil=True)
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)
    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.other.include("fluid")
    ds.timestep.setTmax(EQUIL_TMAX)
    ds.timestep.setNt(1)
    return ds, r


def runaway_plasma(nr, field):
    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED         # line 31
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_ULTRA_RELATIVISTIC         # line 32
    r = FRE_A * (np.arange(nr) + 0.5) / nr
    ds.radialgrid.setB0(FRE_B0)
    ds.radialgrid.setMinorRadius(FRE_A)
    ds.radialgrid.setWallRadius(FRE_A)
    ds.radialgrid.setNr(nr)
    ds.eqsys.E_field.setPrescribedData(np.asarray(field), radius=r)
    ds.eqsys.T_cold.setPrescribedData(FRE_TEMPERATURE)                                # line 50
    ds.eqsys.n_i.addIon(name="D", Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=FRE_DENSITY)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)            # line 56
    ds.eqsys.n_re.setInitialProfile(FRE_NRE)
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setNp(FRE_NP)
    ds.runawaygrid.setPmax(FRE_PMAX)
    ds.runawaygrid.setNxi(FRE_NXI)
    ds.runawaygrid.setBiuniformGrid(thetasep=0.5, nthetasep_frac=0.8)                # line 64
    ds.eqsys.f_re.setAdvectionInterpolationMethod(DistFunc.AD_INTERP_UPWIND)         # line 68
    ds.solver.setType(Solver.LINEAR_IMPLICIT)                                         # line 77
    ds.solver.preconditioner.setEnabled(False)
    ds.other.include("fluid")
    return ds, r


def write_grid(group, res, grid):
    for q in ("t", "r", "r_f"):
        group[f"grid/{q}"] = res[f"grid/{q}"][:]
    for q in ("p1_f", "p2_f", "p1", "p2"):
        group[f"grid/{grid}/{q}"] = res[f"grid/{grid}/{q}"][:]


def main(outfile, case):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "initial_state.py"
        out.attrs["case"] = case
        base.provenance(out)
        if case == "coronal":
            r = EQUIL_A * (np.arange(EQUIL_NR) + 0.5) / EQUIL_NR
            profiles = {"uniform": np.full(EQUIL_NR, EQUIL_TEMPERATURE),
                        "profile": EQUIL_TEMPERATURE + (PROFILE_LOW - EQUIL_TEMPERATURE) * r / r[-1]}
            for name, temperature in profiles.items():
                ds, _ = coronal(temperature)
                with h5py.File(run(ds, tmp, name), "r") as res:
                    group = out.create_group(name)
                    for q in ("t", "r", "r_f"):
                        group[f"grid/{q}"] = res[f"grid/{q}"][:]
                    # (time, charge state of D then Ne, radius)
                    for q in ("n_i", "n_cold", "T_cold"):
                        group[f"eqsys/{q}"] = res[f"eqsys/{q}"][:]
                    group["ion_names"] = res["ionmeta/names"][()]
                    group["ion_Z"] = res["ionmeta/Z"][:]
        elif case == "shapes":
            for name, inittype in SHAPES.items():
                # fields of both signs, a choice of this package, so that INIT_FORWARD takes both sides
                ds, _ = runaway_plasma(2, [FRE_FIELD, -FRE_FIELD])
                ds.eqsys.f_re.setInitType(inittype)
                ds.timestep.setTmax(FRE_TMAX / FRE_NT)
                ds.timestep.setNt(1)
                with h5py.File(run(ds, tmp, name), "r") as res:
                    group = out.create_group(name)
                    write_grid(group, res, "runaway")
                    group["grid/runaway/Vprime"] = res["grid/runaway/Vprime"][:]
                    group["grid/VpVol"] = res["grid/VpVol"][:]
                    for q in ("f_re", "n_re", "E_field"):
                        group[f"eqsys/{q}"] = res[f"eqsys/{q}"][:]
        elif case == "heating":
            ds, _ = runaway_plasma(1, [FRE_FIELD])
            # the forward shape and T_cold self-consistent, choices of this package
            ds.eqsys.f_re.setInitType(FRe.INIT_FORWARD)
            ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
            ds.eqsys.T_cold.setInitialProfile(FRE_TEMPERATURE)
            ds.timestep.setTmax(HEATING_TMAX)
            ds.timestep.setNt(FRE_NT)
            with h5py.File(run(ds, tmp, "heating"), "r") as res:
                group = out.create_group("heating")
                write_grid(group, res, "runaway")
                group["grid/runaway/Vprime"] = res["grid/runaway/Vprime"][:]
                group["grid/VpVol"] = res["grid/VpVol"][:]
                for q in ("f_re", "n_re", "T_cold", "n_cold", "n_i", "E_field", "W_cold"):
                    group[f"eqsys/{q}"] = res[f"eqsys/{q}"][:]
                for q in res["other/fluid"]:
                    if q.startswith("Tcold_"):
                        group[f"other/fluid/{q}"] = res[f"other/fluid/{q}"][:]
        else:
            raise SystemExit(f"case must be coronal, shapes or heating, not {case}")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    options = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if "case" not in options:
        raise SystemExit("usage: initial_state.py <output file.h5> case=<coronal|shapes|heating>")
    main(sys.argv[1], options["case"])
