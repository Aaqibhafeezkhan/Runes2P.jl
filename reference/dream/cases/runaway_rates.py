#!/usr/bin/env python3
"""
DREAM reference values for the fluid runaway rate formulas.

Fluid mode, no kinetic grid: one radial point per parameter set, so a single short run
gives the critical and Dreicer fields, the thermal collision time, the Dreicer rate
(Connor-Hastie in one variant, the neural network of Hesslow et al. in the other), the
effective critical momentum and the avalanche growth rate for every set.

The parameter sets cover moderate and strong fields at several temperatures, including
the regime of DREAM's CODE benchmark of the runaway rate (low T, high E/E_D), and stay
inside the neural network's range (1 eV to 20 keV, DreicerNeuralNetwork::IsApplicable).

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/runaway_rates.py out/<set>/dream_runaway_rates.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

A_MINOR = 0.5
B0 = 5.0
DT = 1e-9

# (temperature [eV], deuterium density [m^-3], E/E_c); the last two are the CODE regime
CASES = [
    (1000.0, 1e20, 3.0),
    (1000.0, 1e20, 10.0),
    (1000.0, 1e20, 40.0),
    (5000.0, 5e19, 10.0),
    (100.0, 1e20, 40.0),
    (20.0, 1e20, 120.0),
    (100.0, 1e20, 20.0),  # the kinetic avalanche case of test_runaway_physics.jl
    (1000.0, 1e20, 25.0),  # the kinetic Dreicer cases, where the two references agree
    (1000.0, 1e20, 60.0),
]
DREICER_MODES = {
    "connor_hastie": Runaways.DREICER_RATE_CONNOR_HASTIE_NOCORR,
    "connor_hastie_corrected": Runaways.DREICER_RATE_CONNOR_HASTIE,
    "neural_network": Runaways.DREICER_RATE_NEURAL_NETWORK,
}
SAVED = ["Ecfree", "Ectot", "EDreic", "GammaAva", "gammaDreicer", "pCrit", "tauEETh", "tauEERel"]


def critical_field(density, coulomb_log):
    """Connor-Hastie E_c [V/m], to place E relative to it before the run."""
    e, eps0, me, c = 1.602176634e-19, 8.8541878128e-12, 9.1093837015e-31, 299792458.0
    return density * coulomb_log * e**3 / (4 * np.pi * eps0**2 * me * c**2)


def relativistic_coulomb_log(density, temperature):
    """DREAM's lnΛ_c (LNLAMBDA_CONSTANT), as used for E_c."""
    return 14.6 + 0.5 * np.log(temperature / (density / 1e20))


def settings(temperature, density, e_over_ec, dreicer_mode):
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_CONSTANT

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(1)

    field = e_over_ec * critical_field(density, relativistic_coulomb_log(density, temperature))
    ds.eqsys.E_field.setPrescribedData(field)
    ds.eqsys.T_cold.setPrescribedData(temperature)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_PRESCRIBED, n=density)

    ds.eqsys.n_re.setDreicer(dreicer_mode)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID)
    ds.eqsys.n_re.setInitialProfile(density=1.0)
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.timestep.setTmax(DT)
    ds.timestep.setNt(1)
    ds.other.include(*(f"fluid/{q}" for q in SAVED))
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "runaway_rates.py"
        base.provenance(out)
        out["temperature"] = [T for T, _, _ in CASES]
        out["density"] = [n for _, n, _ in CASES]
        out["E_over_Ec"] = [f for _, _, f in CASES]
        for name, mode in DREICER_MODES.items():
            group = out.create_group(name)
            group.attrs["dreicer_mode"] = int(mode)
            values = {q: [] for q in SAVED}
            fields, cold, total = [], [], []
            for temperature, density, e_over_ec in CASES:
                ds = settings(temperature, density, e_over_ec, mode)
                sfile = Path(tmp) / f"settings_{name}.h5"
                ofile = Path(tmp) / f"output_{name}.h5"
                ds.output.setFilename(str(ofile))
                ds.save(str(sfile))
                subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
                with h5py.File(ofile, "r") as res:
                    fields.append(res["eqsys/E_field"][-1, 0])
                    # the fields and rates are evaluated at the cold and total electron
                    # densities of the step, which the runaways generated in it change
                    cold.append(res["eqsys/n_cold"][-1, 0])
                    total.append(res["eqsys/n_tot"][-1, 0])
                    for q in SAVED:
                        values[q].append(res[f"other/fluid/{q}"][-1, 0])
            group["E_field"] = fields
            group["n_cold"] = cold
            group["n_tot"] = total
            for q in SAVED:
                group[q] = values[q]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
