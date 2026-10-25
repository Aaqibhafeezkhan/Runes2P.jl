#!/usr/bin/env python3
"""
DREAM reference data for the hot-hot Boltzmann knock-on operator (MollerBoltzmannOperator).

Cylindrical, two radii with different total electron densities, hot-tail grid only,
AVALANCHE_MODE_BOLTZMANN_KNOCK_ON. The initial f_hot is anisotropic in pitch so that the
pitch kernel is exercised. DREAM builds the knock-on source explicitly from f_hot of the
previous time level, so other/hottail/C_boltz[n] (saved after step n + 1) belongs to
eqsys/f_hot[n].

Variants:
- pmax5:   biuniform p grid on [0, 5]; no pair is clamped by DREAM's xi* range.
- pmax2p2: uniform p grid on [0, 2.2]; sqrt(3) < p_max < 3, so DREAM clamps a share of
           the contributing pairs to the table edge.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/knockon_operator.py out/<set>/dream_knockon_operator.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

NR = 2
A_MINOR = 0.5
XI_F = -np.cos(np.linspace(0, np.pi, 13))  # 12 cells, refined towards xi = ±1
DT = 1e-9
NT = 2
IONS = [("D", 1, 1, [1e19, 5e18]), ("Ar", 18, 0, [5e17, 2e18])]


def p_faces_biuniform(np_, npsep, psep, pmax):
    lo = np.linspace(0, psep, npsep + 1)
    hi = np.linspace(psep, pmax, np_ - npsep + 1)
    return np.concatenate([lo, hi[1:]])


VARIANTS = {
    "pmax5": dict(p_f=p_faces_biuniform(40, 15, 1.0, 5.0), p_cut=0.1),
    "pmax2p2": dict(p_f=np.linspace(0, 2.2, 31), p_cut=0.15),
}


def initial_f(p, xi, r):
    """Anisotropic, smooth, positive: shape varies with radius."""
    P, XI = np.meshgrid(p, xi)  # (nxi, np)
    out = np.empty((len(r), len(xi), len(p)))
    for ir, rr in enumerate(r):
        width = 0.4 + 0.3 * rr
        out[ir] = 1e16 * np.exp(-P / width) * (1 + 0.8 * XI + 0.3 * XI**2) * (1 + P)
    return out


def settings(variant):
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_SUPERTHERMAL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT

    r = (np.arange(NR) + 0.5) * A_MINOR / NR
    ds.radialgrid.setB0(5.0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(NR)

    ds.eqsys.E_field.setPrescribedData(0.1)
    ds.eqsys.T_cold.setPrescribedData(100.0)
    for name, Z, Z0, n in IONS:
        ds.eqsys.n_i.addIon(name=name, Z=Z, Z0=Z0, iontype=Ions.IONS_PRESCRIBED,
                            n=np.array(n), r=r)

    p_f = variant["p_f"]
    ds.hottailgrid.setCustomGrid(p_f=list(p_f), xi_f=list(XI_F))
    p = 0.5 * (p_f[1:] + p_f[:-1])
    xi = 0.5 * (XI_F[1:] + XI_F[:-1])
    ds.eqsys.f_hot.setInitialValue(initial_f(p, xi, r), r=r, p=p, xi=xi)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_BOLTZMANN_KNOCK_ON,
                               pCutAvalanche=variant["p_cut"])
    ds.eqsys.n_re.setInitialProfile(density=0)
    ds.runawaygrid.setEnabled(False)

    # LINEAR_IMPLICIT segfaults at the first step with the Boltzmann operator (93d27e1)
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.timestep.setTmax(NT * DT)
    ds.timestep.setNt(NT)
    ds.other.include("hottail/C_boltz")
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "knockon_operator.py"
        base.provenance(out)
        for name, variant in VARIANTS.items():
            ds = settings(variant)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                g["p_cut"] = variant["p_cut"]
                for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                    g[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                g["grid/r_f"] = res["grid/r_f"][:]
                g["grid/VpVol"] = res["grid/VpVol"][:]
                g["n_tot"] = res["eqsys/n_tot"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:], compression="gzip")
                g.create_dataset("C_boltz", data=res["other/hottail/C_boltz"][:],
                                 compression="gzip")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
