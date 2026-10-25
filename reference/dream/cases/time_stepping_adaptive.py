#!/usr/bin/env python3
"""
DREAM reference data for the adaptive time stepper (TYPE_ADAPTIVE, step doubling).

The setup of time_stepping.py, superthermal variant: cylindrical, two radii, hot-tail
grid only, prescribed and constant plasma, the hot-hot Boltzmann knock-on operator
treated explicitly. DREAM's adaptive stepper (TimeStepperAdaptive.cpp) checks every
step, or every CHECK_EVERY + 1 steps: two steps of dt/2 and one of dt from the same
state, the error per non-trivial unknown |x_full - x_half|_2 / (abstol + reltol |x_full|_2),
the step accepted when every error is at most 1 and the next dt = 0.95 dt / max error.

The file stores the accepted time sequence (grid/t holds every accepted step of the
adaptive stepper), the moments at those times, the distribution at the first and the
last time, DREAM's log of every check with its error, and the names of the unknowns, so
that a Runes2P stepper with the same rule reproduces the sequence.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/time_stepping_adaptive.py out/<set>/dream_time_stepping_adaptive.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

import DREAM.Settings.TimeStepper as TimeStepper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
import time_stepping as ts  # noqa: E402

DT0 = 1e-6
T_MAX = 4e-4
RELTOL_STEP = 1e-3
NEWTON_RELTOL = 1e-10
# checkevery > 0 aborts the run in DREAM (a Newton failure in an unchecked step), so
# both variants check every step. The second disables the checks of the
# runaway density and current: DREAM's step update still binds on them, since its
# GetErrorScale reads the tolerances by the position in the non-trivial list while the
# acceptance reads them by the unknown ID, which is what the pair of
# runs shows
VARIANTS = {"check_every_step": {"check_every": 0, "runaway_check": True},
            "no_runaway_check": {"check_every": 0, "runaway_check": False}}


def settings(check_every, runaway_check):
    ds = ts.settings(ts.VARIANTS["superthermal"])
    # The Newton tolerance 1e-12 of time_stepping.py reaches its round-off floor on n_hot
    # in a step the stepper does not check ("Newton solver converged, but the converged
    # point does not solve the system of equations", then the iteration limit), and the
    # adaptive stepper cannot roll back an unchecked step: it aborts the run. 1e-10 stays
    # far below the tolerance of the stepper.
    ds.solver.tolerance.set(reltol=NEWTON_RELTOL)
    # setType takes no keywords for the adaptive stepper: every setting by its setter
    ds.timestep.setType(TimeStepper.TYPE_ADAPTIVE)
    ds.timestep.setTmax(T_MAX)
    ds.timestep.setDt(DT0)
    ds.timestep.setCheckInterval(check_every)
    ds.timestep.setRelativeTolerance(RELTOL_STEP)
    if not runaway_check:
        ds.timestep.tolerance.disable(["n_re", "j_re"])
    ds.timestep.setVerbose(True)
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "time_stepping_adaptive.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in ts.IONS]
        out["ions/Z"] = [Z for _, Z, *_ in ts.IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in ts.IONS]
        out["ions/n"] = np.array([n for *_, n in ts.IONS])
        out["E"] = ts.E_FIELD
        out["B"] = ts.B0
        out["p_cut"] = ts.P_CUT
        out["reltol"] = NEWTON_RELTOL
        out["dt0"] = DT0
        out["t_max"] = T_MAX
        out["reltol_step"] = RELTOL_STEP
        for name, variant in VARIANTS.items():
            ds = settings(**variant)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            log = subprocess.run([str(base.DREAMI), str(sfile)], capture_output=True, text=True)
            if log.returncode != 0:
                print(log.stdout[-4000:], log.stderr[-4000:], file=sys.stderr)
                raise RuntimeError(f"DREAM failed on {name} with exit status {log.returncode}")
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                g.attrs["check_every"] = variant["check_every"]
                g.attrs["runaway_check"] = int(variant["runaway_check"])
                if "grid" not in out:
                    for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r"] = res["grid/r"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                for q in ("T_cold", "n_cold", "n_tot", "n_re", "n_hot", "j_hot", "j_re", "j_tot", "j_ohm", "I_p"):
                    if q in res["eqsys"]:
                        g[q] = res[f"eqsys/{q}"][:]
                # the distribution at the first and the last time only, to keep the file small
                f_hot = res["eqsys/f_hot"]
                g["f_hot_initial"] = f_hot[0]
                g["f_hot_final"] = f_hot[-1]
                g["newton_iterations"] = res["solver/iterations"][:]
                g["unknowns"] = sorted(res["eqsys"].keys())
                # the stepper's own report: every check with its error and the next dt
                g["stepper_log"] = "\n".join(line for line in log.stdout.splitlines() if "[TimeStepper]" in line)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
