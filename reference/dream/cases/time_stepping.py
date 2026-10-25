#!/usr/bin/env python3
"""
DREAM reference data for multi-step backward-Euler time stepping.

Cylindrical, two radii, hot-tail grid only: small-angle Fokker-Planck operator (centred
interpolation, synchrotron, f = 0 beyond p_max, no particle source) and the hot-hot
Boltzmann knock-on operator, which DREAM treats explicitly in f_hot. DREAM's NONLINEAR
solver with a tight tolerance advances the system over NT steps. Everything the
coefficients read is prescribed and constant (T_cold, n_cold, ions, E), so each step is
the backward-Euler step of a fixed linear operator plus the knock-on source of the
previous time level.

Without a runaway grid, DREAM also adds a Rosenbluth-Putvinski source from n_re to f_hot.
The initial f_hot falls off far below p_max, so n_re (fed by the flux through p_max)
stays negligible; the file stores n_re so that the test can check this.

Variants: superthermal and full collision modes.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/time_stepping.py out/<set>/dream_time_stepping.h5
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
P_F = knockon.p_faces_biuniform(60, 30, 0.6, 10.0)
XI_F = -np.cos(np.linspace(0, np.pi, 13))  # 12 cells, refined towards xi = ±1
P_CUT = 0.1
DT = 1e-5
NT = 20
RELTOL = 1e-12

VARIANTS = {
    "superthermal": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_SUPERTHERMAL,
        collfreq_type=Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_ENERGY_DEPENDENT,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
    "full": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_FULL,
        collfreq_type=Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_THERMAL,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
}


def initial_f(p, xi, r):
    """Anisotropic and positive, falling to ~1e-27 of its peak at p_max."""
    P, XI = np.meshgrid(p, xi)  # (nxi, np)
    out = np.empty((len(r), len(xi), len(p)))
    for ir, rr in enumerate(r):
        width = 0.15 + 0.05 * rr
        out[ir] = 1e16 * np.exp(-P / width) * (1 + 0.8 * XI + 0.3 * XI**2) * (1 + P)
    return out


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
    ds.eqsys.n_cold.setPrescribedData(N_COLD, radius=r)
    for name, Z, Z0, n in IONS:
        ds.eqsys.n_i.addIon(name=name, Z=Z, Z0=Z0, iontype=Ions.IONS_PRESCRIBED,
                            n=np.array(n), r=r)

    ds.hottailgrid.setCustomGrid(p_f=list(P_F), xi_f=list(XI_F))
    p = 0.5 * (P_F[1:] + P_F[:-1])
    xi = 0.5 * (XI_F[1:] + XI_F[:-1])
    ds.eqsys.f_hot.setInitialValue(initial_f(p, xi, r), r=r, p=p, xi=xi)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_BOLTZMANN_KNOCK_ON,
                               pCutAvalanche=P_CUT)
    ds.eqsys.n_re.setInitialProfile(density=0)
    ds.runawaygrid.setEnabled(False)

    # LINEAR_IMPLICIT segfaults with the Boltzmann operator (see knockon_operator.py)
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=RELTOL)
    ds.solver.setSaveStatistics(True)
    ds.timestep.setTmax(NT * DT)
    ds.timestep.setNt(NT)
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "time_stepping.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in IONS]
        out["ions/Z"] = [Z for _, Z, *_ in IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in IONS]
        out["ions/n"] = np.array([n for *_, n in IONS])
        out["E"] = E_FIELD
        out["B"] = B0
        out["p_cut"] = P_CUT
        out["reltol"] = RELTOL
        for name, variant in VARIANTS.items():
            ds = settings(variant)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                for key, val in variant.items():
                    g.attrs[key] = int(val)
                if "grid" not in out:
                    for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r"] = res["grid/r"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                for q in ("T_cold", "n_cold", "n_tot", "n_re"):
                    g[q] = res[f"eqsys/{q}"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:], compression="gzip")
                g["newton_iterations"] = res["solver/iterations"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
