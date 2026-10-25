#!/usr/bin/env python3
"""
DREAM reference data for the radial transport of the distributions.

Cylindrical, three radii, the hot-tail and the runaway grid of two_grids.py with the flux
between them, the small-angle Fokker-Planck operator with centred interpolation and
synchrotron losses, no avalanche, no particle source. Both distributions diffuse in radius
with the Rechester-Rosenbluth coefficient of a magnetic perturbation that is uniform in
radius and constant in time.

Values:
- delta B / B = 1e-3, from DREAM's examples/SPI/SPI_ITER.py:57 (dBOverB).
- Major radius 6.2 m, the size of ITER (M. Shimada et al., Progress in the ITER Physics
  Basis, Chapter 1, Nucl. Fusion 47, S1 (2007), doi:10.1088/0029-5515/47/6/S01, table 2). DREAM sets R0 = 1 in cylindrical geometry
  and has q = 1 fixed (RechesterRosenbluthTransport.cpp), so the level passed to DREAM is
  delta B / B times sqrt(R0).
- The plasma, the grids and the initial distributions are those of two_grids.py, at three
  radii.

Variants by the condition at the edge of the plasma: "f_0" (Transport.BC_F_0, f = 0 outside
the plasma) and "conservative" (Transport.BC_CONSERVATIVE, no flux through the edge, the
default of the Python interface).

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/radial_transport.py out/<set>/dream_radial_transport.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
import two_grids  # noqa: E402

NR = 3
A_MINOR = 0.5
B0 = 5.0
T_COLD = np.array([1000.0, 2000.0, 3000.0])
N_COLD = np.array([1.2e19, 1.0e19, 8e18])
E_FIELD = np.array([1.0, 1.5, 2.0])
IONS = [("D", 1, 1, [1e19, 7e18, 5e18]), ("Ar", 18, 1, [2e17, 2.5e17, 3e17])]
DELTA_B_OVER_B = 1e-3
MAJOR_RADIUS = 6.2
DT = 1e-5
NT = 20
RELTOL = 1e-12

VARIANTS = {
    "f_0": Transport.BC_F_0,
    "conservative": Transport.BC_CONSERVATIVE,
}


def settings(edge):
    ds = DREAMSettings()
    for key, val in two_grids.COLLISIONS.items():
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
    ds.hottailgrid.setCustomGrid(p_f=list(two_grids.P_F_HOT), xi_f=list(two_grids.XI_F_HOT))
    ds.runawaygrid.setEnabled(True)
    ds.runawaygrid.setCustomGrid(p_f=list(two_grids.P_F_RUNAWAY), xi_f=list(two_grids.XI_F_RUNAWAY))

    for f, p_f, xi_f in ((ds.eqsys.f_hot, two_grids.P_F_HOT, two_grids.XI_F_HOT),
                         (ds.eqsys.f_re, two_grids.P_F_RUNAWAY, two_grids.XI_F_RUNAWAY)):
        p, xi = two_grids.centers(p_f), two_grids.centers(xi_f)
        f.setInitialValue(two_grids.initial_f(p, xi, r), r=r, p=p, xi=xi)
        f.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
        f.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
        f.transport.setMagneticPerturbation(dBB=DELTA_B_OVER_B * np.sqrt(MAJOR_RADIUS))
        f.transport.setBoundaryCondition(edge)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

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
        out.attrs["source"] = "radial_transport.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in IONS]
        out["ions/Z"] = [Z for _, Z, *_ in IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in IONS]
        out["ions/n"] = np.array([n for *_, n in IONS])
        out["E"] = E_FIELD
        out["B"] = B0
        out["T_cold"] = T_COLD
        out["n_cold"] = N_COLD
        out["delta_b_over_b"] = DELTA_B_OVER_B
        out["major_radius"] = MAJOR_RADIUS
        for key, val in two_grids.COLLISIONS.items():
            out.attrs[key] = int(val)
        for name, edge in VARIANTS.items():
            ds = settings(edge)
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                g.attrs["transport_boundary_condition"] = int(edge)
                if "grid" not in out:
                    for grid in ("hottail", "runaway"):
                        for q in ("p1_f", "p2_f"):
                            out[f"grid/{grid}/{q}"] = res[f"grid/{grid}/{q}"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                # the first and the last state, and two in between
                saved = [0, 1, NT // 2, NT]
                g["saved_steps"] = saved
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:][saved], compression="gzip")
                g.create_dataset("f_re", data=res["eqsys/f_re"][:][saved], compression="gzip")
                g["newton_iterations"] = res["solver/iterations"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
