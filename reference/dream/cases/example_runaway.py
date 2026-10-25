#!/usr/bin/env python3
"""
DREAM's own Dreicer example, examples/runaway/basic.py (Mathias Hoppe, last changed
2024-12-17), run with the settings the example writes, for the Runes2P test
test/test_example_runaway.jl.

The example is run unchanged in a temporary directory; the settings file it saves is
loaded and only these overrides are applied, so that the comparison does not need the
unknowns n_cold and the f_hot particle source:

- n_cold prescribed at the deuterium density. The example leaves it self-consistent,
  n_cold = n_free - n_hot - n_re, with n_hot the density above 7 thermal momenta, which
  changes the collision frequencies as the tail fills.
- f_hot particle source ZERO. The example leaves DREAM's default (explicit), which moves
  particles between n_cold and f_hot.

Everything else is the example's: the hot-tail grid (Np = 300 to pMax = 1, Nxi = 20), E, n,
T, B0, the minor radius, partial screening, upwind interpolation, f = 0 beyond pMax, no
synchrotron, avalanche neglected, Dreicer disabled (DREAM default), full collision mode
and thermal lnLambda (DREAM defaults), the linearly implicit solver and the 20 time steps
to 1 ms. n_re then only collects the flux through pMax, and its time derivative is the
kinetic runaway rate.

The file records the DREAM commit and the example's path, commit, date and author, and
stores the complete settings, so that the test checks the example's values instead of
restating them.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/example_runaway.py out/<set>/dream_example_runaway.h5
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.HotElectronDistribution as FHot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

EXAMPLE = Path("examples") / "runaway" / "basic.py"


def example_provenance(out, example):
    """Record which example file, at which commit, the settings come from."""
    fmt = "%H%n%ad%n%an"
    log = subprocess.run(["git", "-C", str(base.ROOT / "DREAM"), "log", "-1", "--date=short",
                          f"--format={fmt}", "--", str(example)],
                         check=True, capture_output=True, text=True).stdout.split("\n")
    out.attrs["example"] = str(example)
    out.attrs["example_commit"] = log[0]
    out.attrs["example_date"] = log[1]
    out.attrs["example_author"] = log[2]


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "example_runaway.py"
        base.provenance(out)
        example_provenance(out, EXAMPLE)
        tmp = Path(tmp)
        shutil.copy(base.ROOT / "DREAM" / EXAMPLE, tmp / "basic.py")
        subprocess.run([sys.executable, "basic.py"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)

        # free-electron density Σ Z0 n_Z0 of the prescribed ions (charge states, t, r)
        with h5py.File(tmp / "dream_settings.h5", "r") as sett:
            states = sett["eqsys/n_i/prescribed/x"][:, 0, 0]
        free_density = float(sum(z0 * n for z0, n in enumerate(states)))
        ds = DREAMSettings(str(tmp / "dream_settings.h5"))
        ds.eqsys.n_cold.setType(ColdElectrons.TYPE_PRESCRIBED)
        ds.eqsys.n_cold.setPrescribedData(free_density)
        ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
        sfile, ofile = tmp / "s.h5", tmp / "o.h5"
        ds.output.setFilename(str(ofile))
        ds.output.setTiming(stdout=False, file=False)
        ds.save(str(sfile))
        subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)

        base.bare_output_filename(sfile)
        with h5py.File(sfile, "r") as sett:
            for key in sett:
                sett.copy(sett[key], out, name=f"settings/{key}")
        with h5py.File(ofile, "r") as res:
            out["t"] = res["grid/t"][:]
            out["n_re"] = res["eqsys/n_re"][:, 0]
            out["n_cold"] = res["eqsys/n_cold"][:, 0]
            out["E_field"] = res["eqsys/E_field"][0, 0]
            out["T_cold"] = res["eqsys/T_cold"][0, 0]
            out.create_dataset("f_hot_initial", data=res["eqsys/f_hot"][0], compression="gzip")
            out.create_dataset("f_hot_final", data=res["eqsys/f_hot"][-1], compression="gzip")
            for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
            out["grid/r_f"] = res["grid/r_f"][:]
            rate = (out["n_re"][-1] - out["n_re"][-2]) / (out["t"][-1] - out["t"][-2])
            print(f"runaway rate at t = {out['t'][-1]:.3e} s: {rate:.4e} m^-3 s^-1")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
