#!/usr/bin/env python3
"""
The kinetic ionization of DREAM in its fully kinetic mode.

Two radii at prescribed temperatures of 20 and 10 eV, no field, a hot-tail grid that covers the
bulk, and an initial f_hot of a Maxwellian with a tail of 1e-3 of the density at 300 eV. Deuterium
singly ionized and neutral argon deposited, both dynamic, with DREAM's kinetic ionization above the
hot threshold of 3 thermal momenta (KINETIC_APPROX_JAC) next to its fluid ADAS ionization
(`ions.cpp:280`). One backward-Euler step of 1e-9 s, so that n_i(1) − n_i(0) = dt R(x(1)) with the
rates R at the state of the end of the step, which Runes2P evaluates on the same state.

test/test_kinetic_ionization_dream.jl compares the change of the charge states.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/kinetic_ionization.py <output file.h5>
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

RADII = 2
MINOR_RADIUS = 0.5
B0 = 5.0
TEMPERATURE = np.array([20.0, 10.0])   # eV
DEUTERIUM = 4e19                       # m^-3, singly ionized
ARGON = 1e18                           # m^-3, deposited as neutrals
TAIL = 1e-3                            # density fraction of the tail
TAIL_TEMPERATURE = 300.0               # eV
THRESHOLD = 3.0                        # thermal momenta
NP, NXI = 120, 4
PMAX = 0.15
DT = 1e-9


def maxwellian(p, T):
    """Relativistic Maxwellian of density about 1; Runes2P reads f_hot from the output."""
    theta = T / 5.10998950e5
    fine = np.linspace(0.0, 1.0, 200001)
    norm = 4 * np.pi * np.sum(fine**2 * np.exp(-fine**2 / (np.sqrt(1 + fine**2) + 1) / theta)) * (fine[1] - fine[0])
    return np.exp(-p**2 / (np.sqrt(1 + p**2) + 1) / theta) / norm


def settings():
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_THERMAL

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(MINOR_RADIUS)
    ds.radialgrid.setWallRadius(MINOR_RADIUS)
    ds.radialgrid.setNr(RADII)
    r = MINOR_RADIUS * (np.arange(RADII) + 0.5) / RADII

    ds.eqsys.E_field.setPrescribedData(0.0)
    ds.eqsys.T_cold.setPrescribedData(TEMPERATURE, radius=r)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=DEUTERIUM * np.ones(RADII), r=r)
    ds.eqsys.n_i.addIon(name="Ar", Z=18, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=ARGON)
    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_KINETIC_APPROX_JAC)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(NXI)
    ds.hottailgrid.setNp(NP)
    ds.hottailgrid.setPmax(PMAX)
    p = (np.arange(NP) + 0.5) * PMAX / NP
    xi_f = np.linspace(-1, 1, NXI + 1)
    xi = 0.5 * (xi_f[1:] + xi_f[:-1])
    f0 = np.zeros((RADII, NXI, NP))
    for ir in range(RADII):
        bulk = maxwellian(p, TEMPERATURE[ir])
        tail = maxwellian(p, TAIL_TEMPERATURE)
        f0[ir] = np.tile(DEUTERIUM * ((1 - TAIL) * bulk + TAIL * tail), (NXI, 1))
    ds.eqsys.f_hot.setInitialValue(f0, r=r, p=p, xi=xi)
    ds.eqsys.f_hot.setHotRegionThreshold(pThreshold=THRESHOLD, pMode=FHot.HOT_REGION_P_MODE_THERMAL)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=1e-12)
    ds.solver.setMaxIterations(100)
    # Absolute tolerances for the current unknowns, which are zero at E = 0 and hold round-off
    # only (|x| about 1e-11): DREAM accepts |dx| <= abstol + reltol |x| (ConvergenceChecker.cpp),
    # and with abstol 0 the Newton solver of v26.5 and 93d27e17 never converges on them on the
    # machine that generated the reference data (|dx| of j_ohm stays 2.9e-11 per iteration). The values are 1e5
    # times that noise and far below any physical current or flux; none of these unknowns is
    # stored or compared, and E is prescribed, so they do not act on n_i, n_cold, n_hot or f_hot.
    for unknown in ("j_ohm", "j_tot"):
        ds.solver.tolerance.set(unknown=unknown, abstol=1e-6)   # A/m^2
    ds.solver.tolerance.set(unknown="I_p", abstol=1e-6)         # A
    ds.solver.tolerance.set(unknown="psi_p", abstol=1e-12)      # Wb
    ds.timestep.setTmax(DT)
    ds.timestep.setNt(1)
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
        out.attrs["source"] = "kinetic_ionization.py"
        base.provenance(out)
        out["temperature"], out["deuterium"], out["argon"] = TEMPERATURE, DEUTERIUM, ARGON
        out["threshold"], out["minor_radius"], out["dt"] = THRESHOLD, MINOR_RADIUS, DT
        with h5py.File(ofile, "r") as res:
            out["t"] = res["grid/t"][:]
            out["r_grid"] = res["grid/r"][:]
            for q in ("p1_f", "p2_f"):
                out[q] = res[f"grid/hottail/{q}"][:]
            # (time, charge state of D and Ar in order, radius)
            out["n_i"] = res["eqsys/n_i"][:]
            out["n_cold"] = res["eqsys/n_cold"][:]
            out["n_hot"] = res["eqsys/n_hot"][:]
            out["T_cold"] = res["eqsys/T_cold"][:]
            # (time, radius, xi, p)
            out["f_hot"] = res["eqsys/f_hot"][:]
            change = res["eqsys/n_i"][1] - res["eqsys/n_i"][0]
            print("change of the argon states over the step, radius 0:", change[2:8, 0])
            print("n_cold", res["eqsys/n_cold"][:], "n_hot", res["eqsys/n_hot"][:])
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
