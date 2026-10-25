#!/usr/bin/env python3
"""
DREAM's Boltzmann knock-on example, examples/avalanche/generateCylindrical.py (Ola Embreus,
branch boltzmann-eqterm, last changed 2026-03-05), for the operator comparison in
test/test_example_avalanche.jl.

The example is run unchanged in a temporary directory and DREAM is run on the settings it
writes, without overrides: hot-tail grid Np = 200 to pmax = 30, biuniform with 40 cells
below psep = 0.7, 30 pitch faces with nxiexp = 1.5 (29 cells), pCutAvalanche = 0.1,
T = 10 eV, E = 10 E_c,tot, D, D0 and Ar0, 100 steps to 0.2 s. The example's own f_hot
(a 100 keV Maxwellian at t = 0, anisotropic and seeded from n_re later) and C_boltz are
stored at the first, middle and last output times, together with n_tot.

The file records the DREAM commit and the example's path, commit, date and author, and
stores the complete settings, so that the test checks the example's values.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/example_avalanche.py out/<set>/dream_example_avalanche.h5
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
from example_runaway import example_provenance  # noqa: E402

EXAMPLE = Path("examples") / "avalanche" / "generateCylindrical.py"
# output times of f_hot and C_boltz; C_boltz[n] is built from f_hot[n]
STORED = (0, 50, 99)


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "example_avalanche.py"
        base.provenance(out)
        example_provenance(out, EXAMPLE)
        tmp = Path(tmp)
        shutil.copy(base.ROOT / "DREAM" / EXAMPLE, tmp / "generate.py")
        subprocess.run([sys.executable, "generate.py"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        sfile = tmp / "settingsCylindrical.h5"
        subprocess.run([str(base.DREAMI), str(sfile)], cwd=tmp, check=True, stdout=subprocess.DEVNULL)

        with h5py.File(sfile, "r") as sett:
            for key in sett:
                sett.copy(sett[key], out, name=f"settings/{key}")
        with h5py.File(tmp / "outputCylindrical.h5", "r") as res:
            stored = list(STORED)
            out["stored_steps"] = stored
            out["t"] = res["grid/t"][:][stored]
            out["n_tot"] = res["eqsys/n_tot"][:][stored]
            out.create_dataset("f_hot", data=res["eqsys/f_hot"][:][stored], compression="gzip")
            out.create_dataset("C_boltz", data=res["other/hottail/C_boltz"][:][stored], compression="gzip")
            for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
            out["grid/r_f"] = res["grid/r_f"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
