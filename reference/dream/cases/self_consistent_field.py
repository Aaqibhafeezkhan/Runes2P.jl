#!/usr/bin/env python3
"""
DREAM reference data for the self-consistent electric field with both kinetic grids.

Cylindrical, three radii, fully kinetic: the thermal bulk is on the hot-tail grid, the
runaway grid lies above it, and the electric field follows from Faraday's and Ampère's law
with the currents of the two distributions. The set-up of the field is that of
examples/transport-rr/generate_consistent.py (Mathias Hoppe): TYPE_SELFCONSISTENT with a
prescribed loop voltage of zero at the wall, which is a perfectly conducting wall, the
minor radius 0.22 m, 1 ms in 30 steps, B0 = 5 T. Deviations from the example, each for
the test:

- The run starts from distributions that carry current, where the example restarts from a
  run with a prescribed field: a Maxwellian with a drift on the hot-tail grid (1 MA/m² at
  the axis) and a seed of runaway electrons on the runaway grid. DREAM cannot prescribe the
  initial current of a kinetic run, and a state without current and flux has no field:
  the initial field of a self-consistent run is a first guess alone.
- The temperature is prescribed at 80 to 120 eV, where the example has 10 keV and a
  self-consistent temperature. At 100 eV the resistive time of the plasma is about 15 ms,
  so that the field changes by several percent in the 1 ms of the run. At 10 keV it would
  take seconds.
- No radial transport (radial_transport.py has its own comparison).
- Both kinetic grids, as in two_grids.py (examples/2kinetic/generate.py): the hot-tail grid
  to p = 2 with 40 of its 70 momentum cells below p = 0.12, which resolves the bulk at
  100 eV with 6 cells per thermal momentum, and the runaway grid to p = 12. The seed lies
  around p = 5, so that it meets the interface and stays below the upper boundary of the
  runaway grid: DREAM's runaway density is not lowered by the flux through that boundary.
- The corrected conductivity is off in the first variant: the ohmic current is the
  current of all momenta of the hot-tail grid minus the hot current. The example leaves
  DREAM's default, which is on (the second variant).
- The cold density is prescribed at the free-electron density, so that the collision
  coefficients are constant in time.
- The ions are D and Ar+ with a profile in radius.

Small-angle Fokker-Planck operator with centred interpolation and synchrotron losses, no
avalanche. The hot region starts at 7 thermal momenta (DREAM's default).

The particle source holds the density of the hot-tail grid at n_cold + n_hot, with its
magnitude as an unknown (PARTICLE_SOURCE_IMPLICIT, as in the example). The wall is at 1.1
minor radii (as in the example), with a loop voltage of 2π times 1 V/m, of the size of the
field in the plasma, where the example has none.

Variants:
- "no_source": no particle source, the wall at the edge of the plasma and no loop voltage,
  the corrected conductivity off: the only comparison of the self-consistent system without
  the source and with the wall at the edge.
- "implicit_source": the corrected conductivity off.
- "corrected_conductivity": the same run with DREAM's corrected conductivity
  (`j_ohm.setCorrectedConductivity(True)`, DREAM's default), which adds
  E (sigma - F sigma) to the ohmic current, with the conductivity sigma of Sauter and Redl
  in its collisionless form (DREAM's default, the conductivity of Braams and Karney in a
  cylinder) and DREAM's fit F of the conductivity of its kinetic operator.
  DREAM's conductivity is stored with both variants.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/self_consistent_field.py out/<set>/dream_self_consistent_field.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.ElectricField as Efield
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
import knockon_operator as knockon  # noqa: E402
import two_grids  # noqa: E402

NR = 3
A_MINOR = 0.22   # examples/transport-rr/generate_consistent.py
B0 = 5.0         # examples/transport-rr/generate_consistent.py
T_MAX = 1e-3     # examples/transport-rr/generate_consistent.py
NT = 30          # examples/transport-rr/generate_consistent.py
T_COLD = np.array([120.0, 100.0, 80.0])
IONS = [("D", 1, 1, [5e19, 4.5e19, 4e19]), ("Ar", 18, 1, [5e17, 1e18, 1.5e18])]
N_COLD = np.array([5.05e19, 4.6e19, 4.15e19])   # the free electrons of the ions
E_INITIAL = np.array([0.7, 0.6, 0.5])           # first guess of the field
J_INITIAL = np.array([1.0e6, 0.8e6, 0.5e6])     # current density of the drifting Maxwellian
N_SEED = np.array([1e15, 2e15, 1e15])           # density scale of the runaway seed
P_MAX_HOT = 2.0        # examples/2kinetic/generate.py
P_MAX_RUNAWAY = 12.0
P_F_HOT = knockon.p_faces_biuniform(70, 40, 0.12, P_MAX_HOT)
P_F_RUNAWAY = np.linspace(P_MAX_HOT, P_MAX_RUNAWAY, 41)
XI_F_HOT = two_grids.XI_F_HOT
XI_F_RUNAWAY = two_grids.XI_F_RUNAWAY
RELTOL = 1e-10
SAVED_STEPS = [0, 1, 2, 10, 20, 30]

VARIANTS = {
    "no_source": dict(particle_source=FHot.PARTICLE_SOURCE_ZERO, wall_radius=A_MINOR, loop_voltage=0.0,
                      corrected_conductivity=False),
    "implicit_source": dict(particle_source=FHot.PARTICLE_SOURCE_IMPLICIT, wall_radius=1.1 * A_MINOR,
                            loop_voltage=2 * np.pi * 1.0, corrected_conductivity=False),
    "corrected_conductivity": dict(particle_source=FHot.PARTICLE_SOURCE_IMPLICIT, wall_radius=1.1 * A_MINOR,
                                   loop_voltage=2 * np.pi * 1.0, corrected_conductivity=True),
}

FLUID = ("E_field", "psi_p", "j_tot", "j_ohm", "j_hot", "j_re", "n_hot", "n_cold", "n_re", "T_cold")
SCALARS = ("I_p", "psi_edge", "psi_wall")


MC2 = 510998.95   # eV
EC = 1.60217662e-19 * 299792458.0   # e c


def centers(faces):
    return 0.5 * (faces[1:] + faces[:-1])


def initial_f_hot(p, xi, ir):
    """Maxwellian of the cold density and temperature with a drift that carries J_INITIAL."""
    theta = T_COLD[ir] / MC2
    drift = J_INITIAL[ir] / (EC * N_COLD[ir])
    P, XI = np.meshgrid(p, xi)  # (nxi, np)
    return N_COLD[ir] / (2 * np.pi * theta)**1.5 * np.exp(-(P**2 - 2 * P * XI * drift + drift**2) / (2 * theta))


def initial_f_re(p, xi, ir):
    """Seed around p = 5 that moves along the field, 0.1 of its peak at the interface."""
    P, XI = np.meshgrid(p, xi)
    return N_SEED[ir] / 30 * np.exp(-(P - 5)**2 / 4) * np.exp(3 * (XI - 1))


def settings(variant):
    ds = DREAMSettings()
    for key, val in two_grids.COLLISIONS.items():
        setattr(ds.collisions, key, val)

    r = (np.arange(NR) + 0.5) * A_MINOR / NR
    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(variant["wall_radius"])
    ds.radialgrid.setNr(NR)

    ds.eqsys.T_cold.setPrescribedData(T_COLD, radius=r)
    ds.eqsys.n_cold.setPrescribedData(N_COLD, radius=r)
    for name, Z, Z0, n in IONS:
        ds.eqsys.n_i.addIon(name=name, Z=Z, Z0=Z0, iontype=Ions.IONS_PRESCRIBED,
                            n=np.array(n), r=r)

    ds.eqsys.E_field.setType(Efield.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(E_INITIAL, radius=r)
    ds.eqsys.E_field.setBoundaryCondition(bctype=Efield.BC_TYPE_PRESCRIBED, inverse_wall_time=0,
                                          V_loop_wall_R0=variant["loop_voltage"])
    ds.eqsys.j_ohm.setCorrectedConductivity(variant["corrected_conductivity"])

    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setCustomGrid(p_f=list(P_F_HOT), xi_f=list(XI_F_HOT))
    ds.runawaygrid.setEnabled(True)
    ds.runawaygrid.setCustomGrid(p_f=list(P_F_RUNAWAY), xi_f=list(XI_F_RUNAWAY))

    p, xi = centers(P_F_HOT), centers(XI_F_HOT)
    ds.eqsys.f_hot.setInitialValue(np.array([initial_f_hot(p, xi, ir) for ir in range(NR)]), r=r, p=p, xi=xi)
    p, xi = centers(P_F_RUNAWAY), centers(XI_F_RUNAWAY)
    ds.eqsys.f_re.setInitialValue(np.array([initial_f_re(p, xi, ir) for ir in range(NR)]), r=r, p=p, xi=xi)
    ds.eqsys.f_hot.setParticleSource(variant["particle_source"])
    for f in (ds.eqsys.f_hot, ds.eqsys.f_re):
        f.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
        f.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=RELTOL)
    ds.solver.setSaveStatistics(True)
    ds.timestep.setTmax(T_MAX)
    ds.timestep.setNt(NT)
    ds.other.include("fluid/conductivity")
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "self_consistent_field.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in IONS]
        out["ions/Z"] = [Z for _, Z, *_ in IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in IONS]
        out["ions/n"] = np.array([n for *_, n in IONS])
        out["B"] = B0
        out["minor_radius"] = A_MINOR
        out["reltol"] = RELTOL
        for key, val in two_grids.COLLISIONS.items():
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
                g.attrs["particle_source"] = int(variant["particle_source"])
                g["wall_radius"] = variant["wall_radius"]
                g["loop_voltage_at_wall"] = variant["loop_voltage"]
                g.attrs["corrected_conductivity"] = int(variant["corrected_conductivity"])
                g["conductivity"] = res["other/fluid/conductivity"][:]
                if "grid" not in out:
                    for grid in ("hottail", "runaway"):
                        for q in ("p1_f", "p2_f"):
                            out[f"grid/{grid}/{q}"] = res[f"grid/{grid}/{q}"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                g["saved_steps"] = SAVED_STEPS
                # the fluid unknowns at all time levels, the distributions at the saved steps
                for q in FLUID + SCALARS:
                    g[q] = res[f"eqsys/{q}"][:]
                if variant["particle_source"] != FHot.PARTICLE_SOURCE_ZERO:
                    g["S_particle"] = res["eqsys/S_particle"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:][SAVED_STEPS], compression="gzip")
                g.create_dataset("f_re", data=res["eqsys/f_re"][:][SAVED_STEPS], compression="gzip")
                g["newton_iterations"] = res["solver/iterations"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
