#!/usr/bin/env python3
"""
DREAM reference data for the Rosenbluth-Putvinski avalanche of a hot-tail grid without a
runaway grid: the kinetic source on f_hot (`hottail/S_ava`),
the effective critical field, p* and the fluid growth rate of the external avalanche
(`fluid/Eceff`, `fluid/pStar`, `fluid/GammaAva`) at one time.

The settings are those of the disruption reference case (examples/Disruptions/generate.py
--kinetic --cylindrical, scenario 1; disruption_kinetic.py): the
hot-tail grid to p_max = 0.8 with 60 of its 140 momentum cells below 0.05 and 15 pitch
cells, collisions FULL, partially screened, energy-dependent lnΛ, the bremsstrahlung
stopping power, the kinetic RP avalanche with pCutAvalanche = 0.01 and E_ceff in mode FULL
(generate.py:143-146, 313, 329), B = 2.5 T (ASDEXU.py:25), in a cylinder. Deviations, each
for the test: two radii; a prescribed plasma of D and Ar in fixed charge states, a
temperature of 5 and 3 eV and a field of +30 and -0.05 V/m, so that the pitch of the knock-ons
takes both signs and the second radius lies below its effective critical field; a runaway
density of 1e16 and 3e15 m^-3; one step of 1e-12 s, so that the state of the output is the
initial one to round-off.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/rp_avalanche.py out/<set>/dream_rp_avalanche.h5
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
B0 = 2.5                                   # ASDEXU.py:25
TEMPERATURE = np.array([5.0, 3.0])         # eV
FIELD = np.array([30.0, -0.05])            # V/m
RUNAWAYS = np.array([1e16, 3e15])          # m^-3
DEUTERIUM = np.array([[1e18, 2e18], [2.5e19, 2e19]])     # (charge state, radius)
ARGON = np.zeros((19, RADII))
ARGON[1] = [3e18, 1e18]
ARGON[4] = [1e18, 2e18]
NP1, NP2, PSEP, PMAX, NXI = 60, 80, 0.05, 0.8, 15        # generate.py:35-43
P_CUT = 0.01                                              # generate.py:329
DT = 1e-12


def settings():
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONLESS
    ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(MINOR_RADIUS)
    ds.radialgrid.setWallRadius(MINOR_RADIUS)
    ds.radialgrid.setNr(RADII)
    r = MINOR_RADIUS * (np.arange(RADII) + 0.5) / RADII

    ds.eqsys.E_field.setPrescribedData(FIELD, radius=r)
    ds.eqsys.T_cold.setPrescribedData(TEMPERATURE, radius=r)
    # prescribed charge states take a time axis: (charge state, time, radius)
    ds.eqsys.n_i.addIon(name="D", Z=1, iontype=Ions.IONS_PRESCRIBED, n=DEUTERIUM[:, None, :], r=r, t=np.array([0.0]))
    ds.eqsys.n_i.addIon(name="Ar", Z=18, iontype=Ions.IONS_PRESCRIBED, n=ARGON[:, None, :], r=r, t=np.array([0.0]))

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_KINETIC, pCutAvalanche=P_CUT)
    ds.eqsys.n_re.setEceff(Runaways.COLLQTY_ECEFF_MODE_FULL)
    ds.eqsys.n_re.setInitialProfile(density=RUNAWAYS, radius=r)

    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(NXI)
    ds.hottailgrid.setNp(NP1 + NP2)
    ds.hottailgrid.setPmax(PMAX)
    ds.hottailgrid.setBiuniformGrid(psep=PSEP, npsep=NP1)
    nfree = DEUTERIUM[1] + sum(Z0 * ARGON[Z0] for Z0 in range(19))
    ds.eqsys.f_hot.setInitialProfiles(rn0=r, n0=nfree, rT0=r, T0=TEMPERATURE)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_EXPLICIT, shape=FHot.PARTICLE_SOURCE_SHAPE_DELTA)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
    ds.eqsys.f_hot.enableIonJacobian(False)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=1e-10)
    ds.timestep.setTmax(DT)
    ds.timestep.setNt(1)
    ds.other.include("fluid/Eceff", "fluid/GammaAva", "fluid/pStar", "fluid/pCrit", "hottail/S_ava")
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
        out.attrs["source"] = "rp_avalanche.py"
        base.provenance(out)
        out["temperature"], out["field"], out["deuterium"], out["argon"] = TEMPERATURE, FIELD, DEUTERIUM, ARGON
        out["minor_radius"], out["B0"], out["p_cut"] = MINOR_RADIUS, B0, P_CUT
        with h5py.File(ofile, "r") as res:
            out["r_grid"] = res["grid/r"][:]
            for q in ("p1_f", "p2_f", "p1", "p2"):
                out[q] = res[f"grid/hottail/{q}"][:]
            for q in ("n_re", "n_tot", "n_cold", "E_field", "T_cold"):
                out[q] = res[f"eqsys/{q}"][-1]
            # (radius, xi, p) at the end of the step
            out["S_ava"] = res["other/hottail/S_ava"][-1]
            for q in ("Eceff", "GammaAva", "pStar", "pCrit"):
                out[q] = res[f"other/fluid/{q}"][-1]
            print("Eceff", res["other/fluid/Eceff"][-1], "GammaAva", res["other/fluid/GammaAva"][-1],
                  "pStar", res["other/fluid/pStar"][-1])
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
