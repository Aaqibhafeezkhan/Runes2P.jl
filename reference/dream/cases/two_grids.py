#!/usr/bin/env python3
"""
DREAM reference data for the hot-tail and the runaway grid with the flux between them.

Cylindrical, two radii, both kinetic grids enabled, as in DREAM's examples/2kinetic/generate.py
(Ola Embreus, 2021-02-16), which has the hot-tail grid to p = 2 and the runaway grid to
p = 4 with a uniform momentum grid of twice the cells, and no avalanche. Deviations from the
example, each for the test:

- The plasma, the field and the collision settings are those of time_stepping.py (two
  radii, D and Ar, E of 1 and 2 V/m), whose collision frequencies Runes2P reproduces to
  round-off.
- The cell counts are a tenth of the example's (40 and 80 momentum cells), to keep the
  stored data small. The hot-tail momentum grid is split at p = 0.6 with half of its cells
  below, as in time_stepping.py.
- The pitch grids of the two grids differ (12 cells refined towards xi = ±1 on the hot-tail
  grid, 10 cells uniform in theta in two regions on the runaway grid), so that the flux
  through the interface is mapped between different pitch grids
  (PXiExternalKineticKinetic).
- The initial distributions are prescribed on both grids, a smooth function across the
  interface that is 1e-6 of its peak there, so that advection in both directions and
  diffusion through the interface matter from the first step.
- No particle source.

Small-angle Fokker-Planck operator with centred interpolation and synchrotron losses. The
runaway grid keeps DREAM's default boundary condition at its p_max (BC_PHI_CONST).

Variants: "neglect" without avalanche (the example), and "boltzmann" with the Boltzmann
knock-on operator, which pairs hot-tail with hot-tail, hot-tail with runaway and runaway
with runaway primaries and is explicit in the distributions.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/two_grids.py out/<set>/dream_two_grids.h5
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
import knockon_operator as knockon  # noqa: E402

NR = 2
A_MINOR = 0.5
B0 = 5.0
T_COLD = np.array([1000.0, 3000.0])
N_COLD = np.array([1.2e19, 8e18])
E_FIELD = np.array([1.0, 2.0])
IONS = [("D", 1, 1, [1e19, 5e18]), ("Ar", 18, 1, [2e17, 3e17])]
P_MAX_HOT = 2.0       # examples/2kinetic/generate.py
P_MAX_RUNAWAY = 4.0   # examples/2kinetic/generate.py
P_F_HOT = knockon.p_faces_biuniform(40, 20, 0.6, P_MAX_HOT)
P_F_RUNAWAY = np.linspace(P_MAX_HOT, P_MAX_RUNAWAY, 81)
XI_F_HOT = -np.cos(np.linspace(0, np.pi, 13))
P_CUT = 0.12
DT = 1e-5
NT = 20
RELTOL = 1e-12


def xi_faces_biuniform_theta(nxi, ntheta_sep, theta_sep):
    lower = np.linspace(0, theta_sep, ntheta_sep + 1)
    upper = np.linspace(theta_sep, np.pi, nxi - ntheta_sep + 1)
    faces = np.cos(np.concatenate([lower, upper[1:]]))[::-1].copy()
    faces[0], faces[-1] = -1.0, 1.0
    return faces


XI_F_RUNAWAY = xi_faces_biuniform_theta(10, 6, 0.9)

COLLISIONS = dict(
    collfreq_mode=Collisions.COLLFREQ_MODE_FULL,
    collfreq_type=Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED,
    lnlambda=Collisions.LNLAMBDA_THERMAL,
    screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
)

VARIANTS = {
    "neglect": dict(avalanche=Runaways.AVALANCHE_MODE_NEGLECT),
    # getattr: DREAM v26.5 has no Boltzmann knock-on mode, and radial_transport.py and
    # self_consistent_field.py import this module there
    "boltzmann": dict(avalanche=getattr(Runaways, "AVALANCHE_MODE_BOLTZMANN_KNOCK_ON", None)),
}


def initial_f(p, xi, r):
    """Anisotropic and positive, 1e-6 of its peak at the interface p = 2."""
    P, XI = np.meshgrid(p, xi)  # (nxi, np)
    out = np.empty((len(r), len(xi), len(p)))
    for ir, rr in enumerate(r):
        width = 0.15 + 0.05 * rr
        out[ir] = 1e16 * np.exp(-P / width) * (1 + 0.8 * XI + 0.3 * XI**2) * (1 + P)
    return out


def centers(faces):
    return 0.5 * (faces[1:] + faces[:-1])


def settings(variant):
    ds = DREAMSettings()
    for key, val in COLLISIONS.items():
        setattr(ds.collisions, key, val)

    r = (np.arange(NR) + 0.5) * A_MINOR / NR
    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(NR)

    ds.eqsys.E_field.setPrescribedData(E_FIELD, radius=r)
    ds.eqsys.T_cold.setPrescribedData(T_COLD, radius=r)
    ds.eqsys.n_cold.setPrescribedData(N_COLD, radius=r)
    for name, Z, Z0, n in IONS:
        ds.eqsys.n_i.addIon(name=name, Z=Z, Z0=Z0, iontype=Ions.IONS_PRESCRIBED,
                            n=np.array(n), r=r)

    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setCustomGrid(p_f=list(P_F_HOT), xi_f=list(XI_F_HOT))
    ds.runawaygrid.setEnabled(True)
    ds.runawaygrid.setCustomGrid(p_f=list(P_F_RUNAWAY), xi_f=list(XI_F_RUNAWAY))

    p, xi = centers(P_F_HOT), centers(XI_F_HOT)
    ds.eqsys.f_hot.setInitialValue(initial_f(p, xi, r), r=r, p=p, xi=xi)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

    p, xi = centers(P_F_RUNAWAY), centers(XI_F_RUNAWAY)
    ds.eqsys.f_re.setInitialValue(initial_f(p, xi, r), r=r, p=p, xi=xi)
    ds.eqsys.f_re.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
    ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

    if variant["avalanche"] == Runaways.AVALANCHE_MODE_NEGLECT:
        ds.eqsys.n_re.setAvalanche(avalanche=variant["avalanche"])
    else:
        ds.eqsys.n_re.setAvalanche(avalanche=variant["avalanche"], pCutAvalanche=P_CUT)

    # LINEAR_IMPLICIT segfaults with the Boltzmann operator (see knockon_operator.py)
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=RELTOL)
    ds.solver.setSaveStatistics(True)
    ds.timestep.setTmax(NT * DT)
    ds.timestep.setNt(NT)
    ds.other.include("fluid/gammaFhot")
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "two_grids.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in IONS]
        out["ions/Z"] = [Z for _, Z, *_ in IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in IONS]
        out["ions/n"] = np.array([n for *_, n in IONS])
        out["E"] = E_FIELD
        out["B"] = B0
        out["p_cut"] = P_CUT
        out["reltol"] = RELTOL
        for key, val in COLLISIONS.items():
            out.attrs[key] = int(val)
        for name, variant in VARIANTS.items():
            ds = settings(variant)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                g.attrs["avalanche"] = int(variant["avalanche"])
                if "grid" not in out:
                    for grid in ("hottail", "runaway"):
                        for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                            out[f"grid/{grid}/{q}"] = res[f"grid/{grid}/{q}"][:]
                    out["grid/r"] = res["grid/r"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                for q in ("T_cold", "n_cold", "n_tot", "n_re"):
                    g[q] = res[f"eqsys/{q}"][:]
                g["gammaFhot"] = res["other/fluid/gammaFhot"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:], compression="gzip")
                g.create_dataset("f_re", data=res["eqsys/f_re"][:], compression="gzip")
                g["newton_iterations"] = res["solver/iterations"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
