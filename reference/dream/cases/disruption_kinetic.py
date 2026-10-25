#!/usr/bin/env python3
"""
DREAM's disruption of the DREAM paper as the reference case of the coupled system.

The chain of examples/Disruptions/generate.py (Mathias Hoppe, Ola Embréus; DREAM paper,
Hoppe et al., Comput. Phys. Commun. 268, 108098 (2021), section 6) with the parameters of
examples/Disruptions/ASDEXU.py, run as `generate.py --kinetic --cylindrical -s 1` by
default, with these settings:

- scenario 1 ("full-conversion"; a choice of this package; see the docs page *Choices of
  this package*), TCDF throughout, as the scenario sets it;
- DREAM's kinetic Rosenbluth-Putvinski avalanche with pCutAvalanche = 0.01 and the external
  avalanche term above p_max (generate.py:313, 329), the bremsstrahlung stopping power on;
- elongation 1 in place of the linear 1.0 to 1.15 of ASDEXU.py:49 (a departure and a
  choice of this package: the cylinder of Runes2P has no shaping; see the docs page
  *Choices of this package*).

The chain: a one-step run for the conductivity of the initial plasma, the field of the
prescribed current profile of ASDEXU.getCurrentDensity, two runs of 1 s in 3 steps with
that field rescaled to I_p (the initial current), then phase 1 (ionization of the injected
D and Ar) and phase 2 (thermal and current quench). Each run writes its DREAM output into
`--outdir`, and `--run-from` starts the chain at a later run from those files, so a phase
can be repeated without the runs before it.

Every parameter that a run may change is an argument: the radius count, the grid
sizes, the mode, the advection scheme, the scenario, the step counts. The defaults are
those of generate.py and ASDEXU.py, cited next to each.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>), the full case and its extract for the comparison:
    pixi run python <set>/cases/disruption_kinetic.py --outdir <directory>
    pixi run python <set>/cases/disruption_kinetic.py --outdir <directory> --run-from 3 \\
        --extract <directory>/extract.h5
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.constants

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.OhmicCurrent as OhmicCurrent
import DREAM.Settings.Equations.RunawayElectrons as RunawayElectrons
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

# ASDEX Upgrade, examples/Disruptions/ASDEXU.py:23-31
A_MINOR = 0.5      # minor radius (m)
B_WALL = 0.55      # wall radius (m)
B0 = 2.5           # on-axis magnetic field (T)
IP = 800e3         # plasma current (A)
NE0 = 2.6e19       # electron density (m^-3)
TE0 = 5.8e3        # central temperature (eV)
TE_WIDTH = 0.4653  # ASDEXU.getInitialTemperature, c1


def temperature_profile(r):
    """ASDEXU.getInitialTemperature: T0 exp(-((r/a)/0.4653)^2)."""
    return TE0 * np.exp(-((r / A_MINOR) / TE_WIDTH)**2)


def current_profile(r):
    """ASDEXU.getCurrentDensity: (1 - (r/a)^4)^1.5, to be scaled to I_p."""
    return (1 - (r / A_MINOR)**4)**1.5


# generate.py:498-509, the scenarios of the DREAM paper are 1 and 4
SCENARIOS = [
    {"t_ioniz": 5e-6, "t_TQ": 4e-3, "nt_ioniz": 400, "nt_TQ": 2000, "fracDinj": 10, "fracAr": 0.6, "Drr": 4000},
    {"t_ioniz": 2e-5, "t_TQ": 1e-3, "nt_ioniz": 200, "nt_TQ": 1500, "fracDinj": 1, "fracAr": 1, "Drr": 4000},
    {"t_ioniz": 5e-6, "t_TQ": 9e-3, "nt_ioniz": 400, "nt_TQ": 1600, "fracDinj": 10, "fracAr": 0.15, "Drr": 4000},
    {"t_ioniz": 5e-6, "t_TQ": 4e-3, "nt_ioniz": 1600, "nt_TQ": 1500, "fracDinj": 10, "fracAr": 0.3, "Drr": 4000},
    {"t_ioniz": 1e-5, "t_TQ": 5e-3, "nt_ioniz": 400, "nt_TQ": 1600, "fracDinj": 2, "fracAr": 0.2, "Drr": 1000},
]

SCHEMES = {"tcdf": FHot.AD_INTERP_TCDF, "upwind": FHot.AD_INTERP_UPWIND, "central": FHot.AD_INTERP_CENTRED,
           "smart": FHot.AD_INTERP_SMART, "muscl": FHot.AD_INTERP_MUSCL, "ospre": FHot.AD_INTERP_OSPRE,
           "quick": FHot.AD_INTERP_QUICK}
MODES = ("kinetic", "fluid")
RUNS = ("conductivity", "init", "init_rescaled", "ionization", "quench")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", required=True, type=Path, help="directory of the DREAM settings and outputs")
    parser.add_argument("--scenario", type=int, default=1, help="scenario of generate.py (default 1)")
    parser.add_argument("--mode", choices=MODES, default="kinetic", help="generate.py --kinetic or --fluid")
    parser.add_argument("--nr", type=int, default=15, help="radial cells (generate.py:32)")
    parser.add_argument("--np1", type=int, default=60, help="momentum cells below psep (generate.py:35)")
    parser.add_argument("--np2", type=int, default=80, help="momentum cells from psep to pmax (generate.py:36)")
    parser.add_argument("--nxi", type=int, default=15, help="pitch cells (generate.py:40)")
    parser.add_argument("--xi-grid", choices=("uniform", "biuniform_theta"), default="uniform",
                        help="uniform (generate.py:204) or a grid biuniform in theta, 10 of the cells "
                             "below theta = 0.5")
    parser.add_argument("--pmax", type=float, default=0.8, help="p_max of the hot-tail grid (generate.py:42)")
    parser.add_argument("--psep", type=float, default=0.05, help="end of the dense momentum cells (generate.py:43)")
    parser.add_argument("--elongation", choices=("1", "asdexu"), default="1",
                        help="1 (a departure, the cylinder has no shaping) or the linear 1.0 to 1.15 of ASDEXU.py:49")
    parser.add_argument("--scheme", choices=sorted(SCHEMES), default="tcdf",
                        help="advection interpolation of f_hot (generate.py:239, 335; TCDF in scenario 1)")
    parser.add_argument("--nt-ioniz", type=int, default=None, help="steps of phase 1 (default: the scenario's)")
    parser.add_argument("--nt-quench", type=int, default=None, help="steps of phase 2 (default: the scenario's)")
    parser.add_argument("--t-ioniz", type=float, default=None, help="length of phase 1 [s] (default: the scenario's)")
    parser.add_argument("--t-quench", type=float, default=None,
                        help="end time of phase 2 [s] (default: the scenario's t_TQ)")
    parser.add_argument("--save-steps", type=int, default=0, help="saved steps per phase (0: every step)")
    parser.add_argument("--linear-solver", choices=("lu", "mkl"), default="lu",
                        help="lu, PETSc's sequential LU of the example (generate.py:60), or mkl, PARDISO with LU as "
                             "the backup (generate.py:59): the same Newton iteration, a faster factorization")
    parser.add_argument("--verbose", action="store_true", help="print DREAM's Newton iterations of the phases")
    parser.add_argument("--run-from", type=int, default=0, choices=range(len(RUNS)),
                        help="first run of the chain: " + ", ".join(f"{i} {n}" for i, n in enumerate(RUNS)))
    parser.add_argument("--run-to", type=int, default=len(RUNS) - 1, choices=range(len(RUNS)),
                        help="last run of the chain")
    parser.add_argument("--extract", type=Path, default=None,
                        help="write the compact comparison file of the phases to this path")
    parser.add_argument("--test-data", type=Path, default=None,
                        help="write DREAM's outputs of the two phases (the unknowns, the grids) to this path, the "
                             "reference of the Runes2P test of the coupled system (dream_disruption_small.h5)")
    return parser.parse_args(argv)


def phase_times(args):
    scenario = SCENARIOS[args.scenario]
    t_ioniz = scenario["t_ioniz"] if args.t_ioniz is None else args.t_ioniz
    t_quench = scenario["t_TQ"] if args.t_quench is None else args.t_quench
    nt_ioniz = scenario["nt_ioniz"] if args.nt_ioniz is None else args.nt_ioniz
    nt_quench = scenario["nt_TQ"] if args.nt_quench is None else args.nt_quench
    return t_ioniz, nt_ioniz, t_quench, nt_quench


def path(args, run, kind="output"):
    return args.outdir / f"{kind}_{RUNS.index(run)}_{run}.h5"


def set_grid(ds, args):
    """ASDEXU.setMagneticField with toroidal = False: the analytic toroidal grid with R0 = inf,
    psi_p = 0 and uniform radial faces, with the elongation 1."""
    r = np.linspace(0, A_MINOR, 101)                # ASDEXU.py:44-46
    # departure: ASDEXU.py:49 has linspace(1.0, 1.15), kept as --elongation asdexu
    kappa = np.ones(r.shape) if args.elongation == "1" else np.linspace(1.0, 1.15, r.size)
    ds.radialgrid.setType(RadialGrid.TYPE_ANALYTIC_TOROIDAL)
    ds.radialgrid.setWallRadius(B_WALL)
    ds.radialgrid.setMajorRadius(np.inf)
    r_f = np.linspace(0, A_MINOR, args.nr + 1)      # ASDEXU.py:65-67, rGridNonuniformity = 1
    ds.radialgrid.setCustomGridPoints(r_f)
    ds.radialgrid.setShaping(psi=np.zeros(r.shape), rpsi=r, GOverR0=B0, kappa=kappa, rkappa=r)


def baseline(args):
    """generate.py getBaseline, part 1: the conductivity of the initial plasma."""
    ds = DREAMSettings()
    ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER   # generate.py:143
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED            # generate.py:144
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT                        # generate.py:145
    ds.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONLESS                       # generate.py:146
    set_grid(ds, args)
    ds.eqsys.E_field.setPrescribedData(1e-4)                                             # generate.py:152
    rT = np.linspace(0, A_MINOR, 100)                                                    # ASDEXU.py:83-84
    T0 = temperature_profile(rT)
    ds.eqsys.T_cold.setPrescribedData(T0, radius=rT)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, T=T0, n=NE0 * np.ones(rT.size), r=rT)
    ds.eqsys.j_ohm.setConductivityMode(OhmicCurrent.CONDUCTIVITY_MODE_SAUTER_COLLISIONAL)   # generate.py:57, 166
    ds.runawaygrid.setEnabled(False)
    ds.hottailgrid.setEnabled(False)
    ds.timestep.setTmax(1e-11)
    ds.timestep.setNt(1)
    ds.other.include("fluid", "scalar")
    return ds, rT, T0


def earlier_output(args, name):
    """DREAM's output of an earlier run of the chain, which a later --run-from starts from."""
    output = path(args, name)
    if not output.exists():
        raise SystemExit(f"{output} is missing: run {RUNS.index(name)} ({name}) first, "
                         f"with --run-from {RUNS.index(name)} or lower")
    return DREAMOutput(str(output))


def run(ds, args, name):
    settings, output = path(args, name, "settings"), path(args, name)
    ds.output.setFilename(str(output))
    ds.save(str(settings))
    print(f"run {RUNS.index(name)} {name}", flush=True)
    # the arguments of the invocation that made this run: a later --run-from keeps them
    (args.outdir / f"arguments_{RUNS.index(name)}_{name}.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    with open(args.outdir / f"log_{RUNS.index(name)}_{name}.txt", "w") as log:
        subprocess.run([str(base.DREAMI), str(settings)], check=True, stdout=log, stderr=subprocess.STDOUT)
    return output


def chain(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    t_ioniz, nt_ioniz, t_quench, nt_quench = phase_times(args)
    scenario = SCENARIOS[args.scenario]
    kinetic = args.mode == "kinetic"
    todo = range(args.run_from, args.run_to + 1)

    ds, rT, T0 = baseline(args)
    if 0 in todo:
        run(ds, args, "conductivity")
    if args.run_to < 1:
        return
    do = earlier_output(args, "conductivity")
    # nfree of the D background, generate.py:162
    nfree, rn0 = ds.eqsys.n_i.getFreeElectronDensity()

    # generate.py getBaseline, part 2: the initial current
    if kinetic:
        ds.hottailgrid.setEnabled(True)
        ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL                      # generate.py:198
        ds.hottailgrid.setNxi(args.nxi)                                                  # generate.py:204
        ds.hottailgrid.setPmax(args.pmax)                                                # generate.py:214
        ds.hottailgrid.setNp(args.np1 + args.np2)                                        # generate.py:216
        if args.xi_grid == "uniform":
            ds.hottailgrid.setBiuniformGrid(psep=args.psep, npsep=args.np1)              # generate.py:217
        else:
            # the faces of Runes2P's biuniform_theta_xi_faces(nxi, 10, 0.5), as custom faces: DREAM's
            # own setBiuniformGrid(thetasep=...) does not survive the copy of the settings at
            # 93d27e1 (XiGrid.fromdict takes int() of an array)
            ds.hottailgrid.setBiuniformGrid(psep=args.psep, npsep=args.np1)
            theta = np.concatenate([np.linspace(0, 0.5, 11), np.linspace(0.5, np.pi, args.nxi - 10 + 1)[1:]])
            ds.hottailgrid.xigrid.setCustomGridPoints(np.cos(theta)[::-1])
        ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T0)              # generate.py:231
        ds.eqsys.f_hot.setHotRegionThreshold(5)                                          # generate.py:234
        ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)                              # generate.py:237
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=SCHEMES[args.scheme],
                                                        ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)   # generate.py:239
        ds.eqsys.f_hot.enableIonJacobian(False)                                          # generate.py:243

    # generate.py:247-254, prescribedJ (the default of main, generate.py:537)
    r = do.grid.r[:]
    j = current_profile(r)
    j0 = IP * 2.0 * np.pi / do.grid.integrate(j)
    E0 = j0 * j / do.other.fluid.conductivity[-1, :] * np.ones((1, r.size))
    ds.eqsys.E_field.setPrescribedData(E0, radius=r, times=[0])
    do.close()

    ds.timestep.setTmax(1)                                                               # generate.py:270-271
    ds.timestep.setNt(3)
    ds.solver.setType(Solver.NONLINEAR)
    if args.linear_solver == "mkl":                                                      # generate.py:59-60
        ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        ds.solver.setBackupSolver(Solver.LINEAR_SOLVER_LU)                               # generate.py:472-473
    else:
        ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    ds.solver.setMaxIterations(100)
    ds.solver.tolerance.set(reltol=1e-7)
    if kinetic:
        ds.solver.tolerance.set("n_hot", abstol=1e5)                                     # generate.py:282-283
        ds.solver.tolerance.set("j_hot", abstol=1)
    if 1 in todo:
        run(ds, args, "init")
    if args.run_to < 2:
        return
    if 2 in todo:
        do = earlier_output(args, "init")
        E = do.eqsys.E_field[-1, :] * IP / do.eqsys.I_p[-1]                              # generate.py:296
        ds.eqsys.E_field.setPrescribedData(E, radius=do.grid.r, times=[0])
        do.close()
        run(ds, args, "init_rescaled")
    else:
        do = earlier_output(args, "init")
        ds.eqsys.E_field.setPrescribedData(do.eqsys.E_field[-1, :] * IP / do.eqsys.I_p[-1], radius=do.grid.r, times=[0])
        do.close()

    # generate.py getBaseline, part 3
    ds1 = DREAMSettings(ds)
    ignorelist = ["n_i", "N_i", "W_i"]
    ds1.eqsys.n_re.setEceff(RunawayElectrons.COLLQTY_ECEFF_MODE_FULL)                     # generate.py:313
    if kinetic:
        ds1.eqsys.n_re.setAvalanche(RunawayElectrons.AVALANCHE_MODE_KINETIC, pCutAvalanche=0.01)   # generate.py:329
        ds1.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_DISABLED)                 # generate.py:333
        ds1.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=SCHEMES[args.scheme],
                                                         ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)  # generate.py:335
        ds1.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_KINETIC_APPROX_JAC)             # generate.py:340
        ds1.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_EXPLICIT,
                                          shape=FHot.PARTICLE_SOURCE_SHAPE_DELTA)          # generate.py:361
    else:
        ds1.eqsys.n_re.setAvalanche(RunawayElectrons.AVALANCHE_MODE_FLUID_HESSLOW)       # generate.py:316
        ds1.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_NEURAL_NETWORK)          # generate.py:317
        ds1.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T0)             # generate.py:319
        ds1.eqsys.n_re.setHottail(RunawayElectrons.HOTTAIL_MODE_ANALYTIC_ALT_PC)         # generate.py:320
    ds1.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)                         # generate.py:365-369
    ds1.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, inverse_wall_time=0, V_loop_wall_R0=0)
    ds1.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)                            # generate.py:370
    reltol = 2e-6                                                                        # generate.py:58
    ds1.solver.tolerance.set(reltol=reltol)                                              # generate.py:375-387
    ds1.solver.tolerance.set(unknown="n_re", reltol=reltol, abstol=1e5)
    ds1.solver.tolerance.set(unknown="j_re", reltol=reltol, abstol=1e-10 * 1e5)
    if kinetic:
        ds1.solver.tolerance.set(unknown="f_hot", reltol=reltol, abstol=1e5)
        ds1.solver.tolerance.set(unknown="n_hot", reltol=reltol, abstol=1e5)
        ds1.solver.tolerance.set(unknown="j_hot", reltol=reltol, abstol=1e-5)
    ds1.output.setTiming(True, True)                                                     # generate.py:391
    ds1.solver.setVerbose(args.verbose)
    ds1.fromOutput(str(path(args, "init_rescaled")), ignore=ignorelist)                  # generate.py:394

    # generate.py simulate, phase 1: ionization
    ds1.eqsys.n_i.addIon("Dinj", Z=1, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=scenario["fracDinj"] * NE0, T=1)
    ds1.eqsys.n_i.addIon("Ar", Z=18, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=scenario["fracAr"] * NE0, T=1)
    if scenario["Drr"] > 0:
        ds1.eqsys.T_cold.transport.prescribeDiffusion(drr=scenario["Drr"])               # generate.py:443-444
    ds1.timestep.setTmax(t_ioniz)
    ds1.timestep.setNt(nt_ioniz)
    if args.save_steps > 0:
        ds1.timestep.setNumberOfSaveSteps(min(args.save_steps, nt_ioniz))
    if 3 in todo:
        run(ds1, args, "ionization")
    if args.run_to < 4:
        return
    earlier_output(args, "ionization").close()

    # phase 2: thermal and current quench, generate.py:460-487
    ds2 = DREAMSettings(ds1)
    ds2.fromOutput(str(path(args, "ionization")))
    ds2.clearIgnore()
    ds2.timestep.setTmax(t_quench - t_ioniz)
    ds2.timestep.setNt(nt_quench)
    if args.save_steps > 0:
        ds2.timestep.setNumberOfSaveSteps(min(args.save_steps, nt_quench))
    # generate.py:478-480 switches scenario 0 to upwind in kinetic mode; with --scheme the
    # scheme is the argument's in both phases, and scenario 0 keeps the switch of the example
    if args.scenario == 0 and kinetic and args.scheme == "tcdf":
        ds2.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_UPWIND,
                                                         ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)
    if 4 in todo:
        run(ds2, args, "quench")


# the fluid and scalar quantities of the comparison, (time, radius) and (time,)
FLUID = ("T_cold", "E_field", "n_cold", "n_hot", "n_re", "n_tot", "j_ohm", "j_hot", "j_re", "j_tot", "W_cold",
         "psi_p")
SCALARS = ("I_p",)
OTHER_FLUID = ("Tcold_radiation", "Tcold_ohmic", "Tcold_fhot_coll", "Tcold_nre_coll", "Tcold_ion_coll",
               "Tcold_transport", "GammaAva", "conductivity", "Eceff", "nre_avalanche", "Ecfree")


def extract(args):
    """The comparison file: per phase the times, the fluid and scalar unknowns, the ion
    densities and W_i, the parts of the energy equation that DREAM saves, and f_hot at the
    first and last saved time of each phase."""
    args.extract.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.extract, "w") as out:
        out.attrs["source"] = "disruption_kinetic.py"
        out.attrs["arguments"] = json.dumps(stored_arguments(args))
        for name in RUNS:
            made = args.outdir / f"arguments_{RUNS.index(name)}_{name}.json"
            if made.exists():
                out.attrs[f"arguments_{name}"] = made.read_text()
        base.provenance(out)
        for name in ("init_rescaled", "ionization", "quench"):
            with h5py.File(path(args, name), "r") as res:
                group = out.create_group(name)
                group["t"] = res["grid/t"][:]
                group["r"] = res["grid/r"][:]
                group["r_f"] = res["grid/r_f"][:]
                group["dr"] = res["grid/dr"][:]
                group["VpVol"] = res["grid/VpVol"][:]
                for q in FLUID + SCALARS:
                    if f"eqsys/{q}" in res:
                        group[q] = res[f"eqsys/{q}"][:]
                for q in ("n_i", "W_i", "N_i"):
                    if f"eqsys/{q}" in res:
                        group[q] = res[f"eqsys/{q}"][:]
                for q in OTHER_FLUID:
                    if f"other/fluid/{q}" in res:
                        group[f"other_{q}"] = res[f"other/fluid/{q}"][:]
                if "eqsys/f_hot" in res:
                    f = res["eqsys/f_hot"]
                    group["f_hot_first"] = f[0]
                    group["f_hot_last"] = f[-1]
                    for q in ("p1_f", "p2_f", "p1", "p2"):
                        group[q] = res[f"grid/hottail/{q}"][:]
                if "ionmeta/names" in res:
                    group["ion_names"] = res["ionmeta/names"][()]
                    group["ion_Z"] = res["ionmeta/Z"][:]
                if "timings" in res:
                    group.attrs["timings"] = str({k: float(np.sum(res["timings"][k][()])) for k in res["timings"]
                                                  if isinstance(res["timings"][k], h5py.Dataset)})
    print(f"wrote {args.extract}")


TEST_UNKNOWNS = ("f_hot", "T_cold", "n_cold", "n_hot", "n_re", "n_tot", "E_field", "psi_p", "psi_edge", "psi_wall",
                 "I_p", "j_ohm", "j_hot", "j_re", "j_tot", "W_cold", "n_i", "W_i")


def stored_arguments(args):
    """The arguments as stored in the output, with the bare name of each path argument, so that
    the data holds no paths of the machine that made it."""
    return {k: (v.name if isinstance(v, Path) else str(v)) for k, v in vars(args).items()}


def test_data(args):
    """DREAM's outputs of the two phases as the Runes2P test reads them: per phase the unknowns of
    TEST_UNKNOWNS at every saved time, the times and the grids, with the provenance."""
    args.test_data.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.test_data, "w") as out:
        out.attrs["source"] = "disruption_kinetic.py"
        out.attrs["arguments"] = json.dumps(stored_arguments(args))
        base.provenance(out)
        for name in ("ionization", "quench"):
            with h5py.File(path(args, name), "r") as res:
                group = out.create_group(name)
                for q in TEST_UNKNOWNS:
                    group[f"eqsys/{q}"] = res[f"eqsys/{q}"][:]
                for q in ("t", "r", "r_f"):
                    group[f"grid/{q}"] = res[f"grid/{q}"][:]
                for q in ("p1_f", "p2_f"):
                    group[f"grid/hottail/{q}"] = res[f"grid/hottail/{q}"][:]
    print(f"wrote {args.test_data}")


def main(argv=None):
    args = arguments(argv)
    if args.run_from <= args.run_to:
        chain(args)
    if args.extract is not None:
        extract(args)
    if args.test_data is not None:
        test_data(args)


if __name__ == "__main__":
    main()
