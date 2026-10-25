#!/usr/bin/env python3
"""
DREAM reference data for the small-angle collision coefficients.

Runs one short DREAM simulation per collision-setting variant and writes the
coefficients DREAM saves on the momentum flux grids (nu_s, nu_D, nu_par, Coulomb
logarithms, synchrotron) together with the grid and plasma state into one compact
HDF5 file used by the Runes2P test suite.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/collision_frequencies.py out/<set>/dream_collisions.h5
"""
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Solver as Solver

ROOT = Path(__file__).resolve().parents[1]
DREAMI = ROOT / "build" / "iface" / "dreami"


def provenance(out):
    """Record which DREAM and PETSc produced the data."""
    sha = subprocess.run(["git", "-C", str(ROOT / "DREAM"), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(ROOT / "DREAM"), "rev-parse", "--abbrev-ref", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if branch == "HEAD":
        # a checkout detached at DREAM_COMMIT (scripts/fetch-dream.sh): the local branch at this commit
        names = subprocess.run(["git", "-C", str(ROOT / "DREAM"), "for-each-ref", "--points-at", "HEAD",
                                "--format=%(refname:short)", "refs/heads"],
                               check=True, capture_output=True, text=True).stdout.split()
        branch = names[0] if names else branch
        # a clone of scripts/fetch-dream.sh keeps no branch at the commit: the branch pinned
        # with the commit in pixi.toml, when the checkout is at that commit
        if branch == "HEAD" and sha == os.environ.get("DREAM_COMMIT"):
            branch = os.environ.get("DREAM_BRANCH", branch)
    version = {}
    for line in (ROOT / "petsc" / "include" / "petscversion.h").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and parts[1] in (
                "PETSC_VERSION_MAJOR", "PETSC_VERSION_MINOR", "PETSC_VERSION_SUBMINOR"):
            version[parts[1]] = parts[2]
    out.attrs["dream_commit"] = sha
    out.attrs["dream_branch"] = branch
    out.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out.attrs["petsc_version"] = "{}.{}.{}".format(
        version["PETSC_VERSION_MAJOR"], version["PETSC_VERSION_MINOR"], version["PETSC_VERSION_SUBMINOR"])



def bare_output_filename(settings_file):
    """Replace the output/filename of a DREAM settings file (an absolute path of a temporary
    directory) by its bare file name, before the settings are copied into the reference data,
    so that the data holds no paths of the machine that made it; the old string stays in the
    temporary settings file only. No numeric data changes. Rewriting the name after the copy
    would leave the old path in the freed bytes of the HDF5 file."""
    with h5py.File(settings_file, "r+") as sett:
        if "output/filename" not in sett:
            return
        dtype, value = sett["output/filename"].dtype, sett["output/filename"][()]
        bare = [Path(v.decode() if isinstance(v, bytes) else str(v)).name.encode() for v in value]
        del sett["output/filename"]
        sett.create_dataset("output/filename", data=np.array(bare, dtype=object), dtype=dtype)


NR = 3
A_MINOR = 0.5
B0 = 5.0
T_COLD = np.array([5.0, 50.0, 2000.0])
E_FIELD = np.array([0.5, 1.0, 2.0])
NP, NPSEP, PSEP, PMAX = 40, 15, 0.3, 10.0
XI_F = -np.cos(np.linspace(0, np.pi, 14))  # 13 cells, refined towards xi = ±1

# (name, Z, Z0, density profile)
IONS = [
    ("D", 1, 1, [1e20, 5e19, 2e19]),
    ("Ar", 18, 1, [1e18, 2e18, 3e18]),
    ("Arh", 18, 17, [1e17, 1e17, 1e17]),
    ("Ne", 10, 0, [5e17, 5e17, 5e17]),
    ("W", 74, 3, [1e16, 1e16, 1e16]),
]

VARIANTS = {
    "full_partial_thermal": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_FULL,
        collfreq_type=Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_THERMAL,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
    "superthermal_partial_energy": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_SUPERTHERMAL,
        collfreq_type=Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_ENERGY_DEPENDENT,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
    "full_non_energy_zero": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_FULL,
        collfreq_type=Collisions.COLLFREQ_TYPE_NON_SCREENED,
        lnlambda=Collisions.LNLAMBDA_ENERGY_DEPENDENT,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_ZERO,
    ),
    "ultrarel_complete_constant": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_ULTRA_RELATIVISTIC,
        collfreq_type=Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_CONSTANT,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
}

SAVED = [
    "nu_s_f1", "nu_s_f2", "nu_D_f1", "nu_D_f2", "nu_par_f1", "nu_par_f2",
    "lnLambda_ee_f1", "lnLambda_ee_f2", "lnLambda_ei_f1", "lnLambda_ei_f2",
    "synchrotron_f1", "synchrotron_f2",
]


def settings(variant):
    ds = DREAMSettings()
    for key, val in variant.items():
        setattr(ds.collisions, key, val)

    r = (np.arange(NR) + 0.5) * A_MINOR / NR
    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(NR)

    ds.eqsys.E_field.setPrescribedData(E_FIELD, radius=r)
    ds.eqsys.T_cold.setPrescribedData(T_COLD, radius=r)
    for name, Z, Z0, n in IONS:
        ds.eqsys.n_i.addIon(name=name, Z=Z, Z0=Z0, iontype=Ions.IONS_PRESCRIBED,
                            n=np.array(n), r=r)

    ds.hottailgrid.setNp(NP)
    ds.hottailgrid.setPmax(PMAX)
    ds.hottailgrid.setBiuniformGrid(psep=PSEP, npsep=NPSEP)
    ds.hottailgrid.setCustomGrid(xi_f=list(XI_F))
    ds.eqsys.f_hot.setInitialProfiles(n0=1e10, T0=100.0)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.timestep.setTmax(1e-9)
    ds.timestep.setNt(1)
    ds.other.include(*(f"hottail/{q}" for q in SAVED))
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "collision_frequencies.py"
        provenance(out)
        out["ions/name"] = [n for n, *_ in IONS]
        out["ions/Z"] = [Z for _, Z, *_ in IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in IONS]
        out["ions/n"] = np.array([n for *_, n in IONS])
        out["E"] = E_FIELD
        out["B"] = B0
        for name, variant in VARIANTS.items():
            ds = settings(variant)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                for key, val in variant.items():
                    g.attrs[key] = int(val)
                if "grid" not in out:
                    for q in ("p1", "p1_f", "p2", "p2_f"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r"] = res["grid/r"][:]
                g["T_cold"] = res["eqsys/T_cold"][-1, :]
                g["n_cold"] = res["eqsys/n_cold"][-1, :]
                g["n_i"] = res["eqsys/n_i"][-1, :]
                for q in SAVED:
                    # last saved time, first radius dimension kept: (nr, nxi(+1), np(+1))
                    g.create_dataset(q, data=res[f"other/hottail/{q}"][-1], compression="gzip")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
