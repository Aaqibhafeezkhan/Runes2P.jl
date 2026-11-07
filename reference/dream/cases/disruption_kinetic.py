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
  *Choices of this package*);
- 500 steps of phase 1 at elongation 1, passed as `--nt-ioniz 500`: the default stays the
  200 of scenario 1, with which DREAM's first step fails (see the docs page *Choices of
  this package*).

`--mode superthermal` runs `generate.py --superthermal` (MODE_SUPERTHERMAL): the hot-tail grid
of NP_KIN2 uniform cells without the fine region below psep (generate.py:218-219), the initial
Maxwellian scaled by 0.9999 so that a trace cold population remains, without DREAM's corrected
conductivity (generate.py:221-231), the superthermal collision frequencies in the phases
(generate.py:342-344), T_cold restarted at 1 eV with T_cold, W_cold, n_hot and n_cold not taken
from the initialization (generate.py:346-356), and no particle source: the example sets it in
the kinetic modes only (generate.py:358-361), and DREAM builds it only with the full collision
frequencies (src/Settings/Equations/f_hot.cpp:163-197 at v26.5).

`--runaway-grid` turns on DREAM's runaway grid from run 1 on, which the example leaves off
(generate.py:169): momentum faces uniform from the hot-tail p_max, then growing geometrically to
`--re-pmax`, and one pitch grid biuniform in theta; the settings of f_re are stated where the
grid is set. Without it the settings are those above. The defaults of the runaway grid are a
choice of this package for scenario 1 at the reduced size of the tests: 12 cells of 0.1 from the
hot-tail p_max 0.8 to 2, 20 geometric cells to 20, and 8 pitch cells, 5 below theta = 0.5, unlike
the 4 uniform cells of the hot-tail grid there, so that the test crosses DREAM's mapping between
the two pitch grids. A grid to p_max = 60 held 3e-5 of the density of f_re above p = 20 at 300 us
of the scenario: the avalanche source keeps a small population up to the last cell, so a few cells
more do not change the result (see the docs page *Choices of this package*).

The chain: a one-step run for the conductivity of the initial plasma, the field of the
prescribed current profile of ASDEXU.getCurrentDensity, two runs of 1 s in 3 steps with
that field rescaled to I_p (the initial current), then phase 1 (ionization of the injected
D and Ar) and phase 2 (thermal and current quench). Each run writes its DREAM output into
`--outdir`, and `--run-from` starts the chain at a later run from those files, so a phase
can be repeated without the runs before it.

`--window-start k --window-steps n` splits phase 2 at its saved step k: run 4 then stops at step k,
and run 5 (window) takes n steps from DREAM's state there, with the step length of the unsplit phase
2, so that its states are those of steps k to k + n of the unsplit run up to the tolerance of the
solver. `--test-data` then writes the window only, the reference of a test that starts Runes2P from
DREAM's state partway through the quench. Without it the chain is the one above. `--frac-dinj`
replaces the injected deuterium of the scenario; without it the density is the scenario's.

Every parameter that a run may change is an argument: the radius count, the grid
sizes, the mode, the advection scheme, the scenario, the step counts. The defaults are
those of generate.py and ASDEXU.py, cited next to each, except those of the runaway grid
above.

Usage (from reference/dream, with the environment that run_all.sh sets for <set>), the full case and its extract for the comparison:
    pixi run python <set>/cases/disruption_kinetic.py --outdir <directory> --nt-ioniz 500
    pixi run python <set>/cases/disruption_kinetic.py --outdir <directory> --nt-ioniz 500 \\
        --run-from 3 --extract <directory>/extract.h5
and the window of phase 2 with the runaway grid at the reduced size of the tests, from the 164th of 429 steps:
    pixi run python <set>/cases/disruption_kinetic.py --outdir <directory> --nr 2 --np1 10 --np2 20 --nxi 4 \\
        --nt-ioniz 500 --t-ioniz 2e-5 --nt-quench 429 --t-quench 3e-4 --linear-solver mkl --runaway-grid \\
        --window-start 164 --window-steps 10 --test-data dream_disruption_runaway_small.h5
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
import DREAM.Settings.Equations.DistributionFunction as DistFunc
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
MODES = ("kinetic", "superthermal", "fluid")
RUNS = ("conductivity", "init", "init_rescaled", "ionization", "quench")
# the run of --window-start, after the chain; not in RUNS, whose defaults the files made without it store
WINDOW = "window"
ALL_RUNS = RUNS + (WINDOW,)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", required=True, type=Path, help="directory of the DREAM settings and outputs")
    parser.add_argument("--scenario", type=int, default=1, help="scenario of generate.py (default 1)")
    parser.add_argument("--mode", choices=MODES, default="kinetic",
                        help="generate.py --kinetic, --superthermal or --fluid; superthermal takes np2 uniform "
                             "cells and ignores np1 and psep")
    parser.add_argument("--nr", type=int, default=15, help="radial cells (generate.py:32)")
    parser.add_argument("--np1", type=int, default=60,
                        help="momentum cells below psep (generate.py:35; not in superthermal)")
    parser.add_argument("--np2", type=int, default=80, help="momentum cells from psep to pmax (generate.py:36)")
    parser.add_argument("--nxi", type=int, default=15, help="pitch cells (generate.py:40)")
    parser.add_argument("--xi-grid", choices=("uniform", "biuniform_theta"), default="uniform",
                        help="uniform (generate.py:204) or a grid biuniform in theta, 10 of the cells "
                             "below theta = 0.5")
    parser.add_argument("--pmax", type=float, default=0.8, help="p_max of the hot-tail grid (generate.py:42)")
    parser.add_argument("--psep", type=float, default=0.05,
                        help="end of the dense momentum cells (generate.py:43; not in superthermal)")
    parser.add_argument("--elongation", choices=("1", "asdexu"), default="1",
                        help="1 (a departure, the cylinder has no shaping) or the linear 1.0 to 1.15 of ASDEXU.py:49")
    parser.add_argument("--scheme", choices=sorted(SCHEMES), default="tcdf",
                        help="advection interpolation of f_hot (generate.py:239, 335; TCDF in scenario 1)")
    parser.add_argument("--nt-ioniz", type=int, default=None,
                        help="steps of phase 1 (default: the scenario's); pass 500 with --scenario 1 "
                             "--elongation 1 (docs: Choices of this package)")
    parser.add_argument("--nt-quench", type=int, default=None, help="steps of phase 2 (default: the scenario's)")
    parser.add_argument("--t-ioniz", type=float, default=None, help="length of phase 1 [s] (default: the scenario's)")
    parser.add_argument("--t-quench", type=float, default=None,
                        help="end time of phase 2 [s] (default: the scenario's t_TQ)")
    parser.add_argument("--save-steps", type=int, default=0, help="saved steps per phase (0: every step)")
    parser.add_argument("--linear-solver", choices=("lu", "mkl"), default="lu",
                        help="lu, PETSc's sequential LU of the example (generate.py:60), or mkl, PARDISO with LU as "
                             "the backup (generate.py:59): the same Newton iteration, a faster factorization")
    parser.add_argument("--verbose", action="store_true", help="print DREAM's Newton iterations of the phases")
    parser.add_argument("--run-from", type=int, default=0, choices=range(len(ALL_RUNS)),
                        help="first run of the chain: " + ", ".join(f"{i} {n}" for i, n in enumerate(ALL_RUNS)) +
                             " (5 only with --window-start)")
    parser.add_argument("--run-to", type=int, default=None, choices=range(len(ALL_RUNS)),
                        help="last run of the chain (default: 4, or 5 with --window-start)")
    parser.add_argument("--extract", type=Path, default=None,
                        help="write the compact comparison file of the phases to this path")
    parser.add_argument("--test-data", type=Path, default=None,
                        help="write DREAM's outputs of the runs of the chain in --outdir (the unknowns, the "
                             "conductivity, the grids) to this path, the reference of the Runes2P tests of the "
                             "coupled system (dream_disruption_small.h5, dream_disruption_chain_small.h5, "
                             "dream_disruption_superthermal_small.h5; with --runaway-grid and --window-start "
                             "the window, dream_disruption_runaway_small.h5)")
    parser.add_argument("--test-data-runs", default=None,
                        help="the runs that --test-data writes, separated by commas (default: runs 0 to 4, or "
                             "window with --window-start; dream_disruption_small.h5 holds ionization,quench)")
    parser.add_argument("--test-data-channels", action="store_true",
                        help="write with --test-data also the channels of the equation of T_cold that DREAM saves "
                             "(other/fluid/Tcold_*), such as the heating by f_hot (Tcold_fhot_coll)")
    # the runaway grid, a departure from the example (generate.py:169 turns it off): on from run 1, so
    # that f_re is in every output (without it the phases would start f_re as n_re in the lowest cell,
    # src/Settings/Equations/f_re.cpp:147-206 at v26.5)
    parser.add_argument("--runaway-grid", action="store_true",
                        help="turn on DREAM's runaway grid from p_max of the hot-tail grid, from run 1 on "
                             "(in the kinetic and superthermal modes; the --re-* arguments set it)")
    parser.add_argument("--re-dp-fine", type=float, default=0.1,
                        help="width of the uniform momentum cells of the runaway grid from --pmax to --re-p-fine-to; "
                             "the cells take the width nearest to it that fills the interval with a whole number of "
                             "cells, at least one")
    parser.add_argument("--re-p-fine-to", type=float, default=2.0,
                        help="end of the uniform cells of the runaway grid")
    parser.add_argument("--re-pmax", type=float, default=20.0,
                        help="p_max of the runaway grid; the cells from --re-p-fine-to grow geometrically, the "
                             "first as wide as --re-dp-fine")
    parser.add_argument("--re-np-coarse", type=int, default=20,
                        help="momentum cells of the runaway grid from --re-p-fine-to to --re-pmax")
    parser.add_argument("--re-nxi", type=int, default=8,
                        help="pitch cells of the runaway grid, biuniform in theta, fine near xi = 1 where the "
                             "runaways and the knock-ons of the avalanche source lie")
    parser.add_argument("--re-nxi-sep", type=int, default=5,
                        help="pitch cells of the runaway grid below theta = 0.5")
    parser.add_argument("--re-scheme", choices=sorted(SCHEMES), default=None,
                        help="advection interpolation of f_re in both phases (default: that of f_hot, --scheme, "
                             "with its switch to upwind in phase 2; DREAM's own default is the centred one, "
                             "py/DREAM/Settings/Equations/RunawayElectronDistribution.py:22-28)")
    # the window of phase 2, an addition to the example
    parser.add_argument("--window-start", type=int, default=None,
                        help="split phase 2 at this saved step of the unsplit phase (a step index with "
                             "--save-steps 0): run 4 stops there and run 5 (window) continues from DREAM's state")
    parser.add_argument("--window-steps", type=int, default=10,
                        help="steps of the window run, each as long as a step of the unsplit phase 2")
    # the injected deuterium, an addition to the example
    parser.add_argument("--frac-dinj", type=float, default=None,
                        help="density of the injected deuterium in units of the initial electron density "
                             "(default: the scenario's fracDinj, generate.py:498-509)")
    args = parser.parse_args(argv)
    # the defaults that depend on --window-start, resolved here so that the runs without it store
    # the values they stored before the option existed
    if args.run_to is None:
        args.run_to = len(ALL_RUNS) - 1 if args.window_start is not None else len(RUNS) - 1
    if args.test_data_runs is None:
        args.test_data_runs = WINDOW if args.window_start is not None else ",".join(RUNS)
    check_arguments(args)
    return args


def check_arguments(args):
    """Refuse the combinations that the chain cannot run, before any run or file is written."""
    if args.runaway_grid and args.mode == "fluid":
        raise SystemExit("--runaway-grid needs the kinetic or superthermal mode")
    if args.window_start is None and args.run_to == len(RUNS):
        raise SystemExit("run 5 (window) needs --window-start")
    if args.window_start is not None:
        nt_quench = phase_times(args)[3]
        if not 1 <= args.window_start < nt_quench:
            raise SystemExit(f"--window-start must be a step of phase 2, from 1 to {nt_quench - 1}")
        if not 1 <= args.window_steps <= nt_quench - args.window_start:
            raise SystemExit(f"--window-steps must be from 1 to {nt_quench - args.window_start}, the steps of "
                             "phase 2 after --window-start")
        if args.save_steps > 0:
            raise SystemExit("--window-start counts the saved steps of phase 2 with every step saved: "
                             "--save-steps 0")
    if not args.runaway_grid:
        return
    if not 1 <= args.re_nxi_sep < args.re_nxi:
        raise SystemExit("--re-nxi-sep must be at least 1 and below --re-nxi")
    if args.re_dp_fine <= 0:
        raise SystemExit("--re-dp-fine must be positive")
    if not args.re_p_fine_to > args.pmax:
        raise SystemExit("--re-p-fine-to must be above --pmax")
    runaway_p_faces(args)                                                                # refuses no growing cells


RE_ARGUMENTS = ("runaway_grid", "re_dp_fine", "re_p_fine_to", "re_pmax", "re_np_coarse", "re_nxi", "re_nxi_sep",
                "re_scheme")


WINDOW_ARGUMENTS = ("window_start", "window_steps")


def omitted_arguments(args):
    """The arguments of the runaway grid, left out of the stored arguments without --runaway-grid, those
    of the window without --window-start and --frac-dinj without a value, so that the files made
    without them keep the arguments they were made with."""
    omitted = () if args.runaway_grid else RE_ARGUMENTS
    if args.window_start is None:
        omitted += WINDOW_ARGUMENTS
    if args.frac_dinj is None:
        omitted += ("frac_dinj",)
    return omitted


def biuniform_theta_xi_faces(nxi, nsep, thetasep=0.5):
    """The pitch faces of Runes2P's biuniform_theta_xi_faces(nxi, nsep, thetasep): nsep cells
    uniform in theta to thetasep, the others uniform in theta to pi, as increasing xi."""
    theta = np.concatenate([np.linspace(0, thetasep, nsep + 1), np.linspace(thetasep, np.pi, nxi - nsep + 1)[1:]])
    return np.cos(theta)[::-1]


def runaway_p_faces(args):
    """The momentum faces of the runaway grid: uniform cells of about --re-dp-fine from p_max of the
    hot-tail grid to --re-p-fine-to, then --re-np-coarse cells to --re-pmax that grow by a constant
    ratio, the first as wide as the uniform ones. The first face is the hot-tail p_max, which DREAM
    imposes (src/Settings/KineticGrid.cpp:158-166, 237-240 at v26.5)."""
    nfine = max(1, int(round((args.re_p_fine_to - args.pmax) / args.re_dp_fine)))
    fine = np.linspace(args.pmax, args.re_p_fine_to, nfine + 1)
    dp, length, n = fine[1] - fine[0], args.re_pmax - args.re_p_fine_to, args.re_np_coarse
    if dp * n >= length:
        raise SystemExit("--re-np-coarse cells of --re-dp-fine reach beyond --re-pmax: no growing cells")
    # the ratio q of dp (q^n - 1)/(q - 1) = length, by bisection on (1, 2)
    lo, hi = 1.0 + 1e-12, 2.0
    while dp * (hi**n - 1) / (hi - 1) < length:
        hi *= 2
    for _ in range(200):
        q = 0.5 * (lo + hi)
        lo, hi = (q, hi) if dp * (q**n - 1) / (q - 1) < length else (lo, q)
    widths = dp * q**np.arange(n)
    coarse = args.re_p_fine_to + np.cumsum(widths)
    coarse[-1] = args.re_pmax
    return np.concatenate([fine, coarse])


def phase_times(args):
    scenario = SCENARIOS[args.scenario]
    t_ioniz = scenario["t_ioniz"] if args.t_ioniz is None else args.t_ioniz
    t_quench = scenario["t_TQ"] if args.t_quench is None else args.t_quench
    nt_ioniz = scenario["nt_ioniz"] if args.nt_ioniz is None else args.nt_ioniz
    nt_quench = scenario["nt_TQ"] if args.nt_quench is None else args.nt_quench
    return t_ioniz, nt_ioniz, t_quench, nt_quench


def path(args, run, kind="output"):
    return args.outdir / f"{kind}_{ALL_RUNS.index(run)}_{run}.h5"


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
        raise SystemExit(f"{output} is missing: run {ALL_RUNS.index(name)} ({name}) first, "
                         f"with --run-from {ALL_RUNS.index(name)} or lower")
    return DREAMOutput(str(output))


def run(ds, args, name):
    settings, output = path(args, name, "settings"), path(args, name)
    ds.output.setFilename(str(output))
    ds.save(str(settings))
    index = ALL_RUNS.index(name)
    print(f"run {index} {name}", flush=True)
    # the arguments of the invocation that made this run: a later --run-from keeps them
    (args.outdir / f"arguments_{index}_{name}.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items() if k not in omitted_arguments(args)}, indent=1))
    with open(args.outdir / f"log_{index}_{name}.txt", "w") as log:
        subprocess.run([str(base.DREAMI), str(settings)], check=True, stdout=log, stderr=subprocess.STDOUT)
    return output


def chain(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    t_ioniz, nt_ioniz, t_quench, nt_quench = phase_times(args)
    scenario = SCENARIOS[args.scenario]
    kinetic = args.mode == "kinetic"
    superthermal = args.mode == "superthermal"
    hot = kinetic or superthermal                                                        # mode != MODE_FLUID
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
    if hot:
        ds.hottailgrid.setEnabled(True)
        ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL                      # generate.py:198
        ds.hottailgrid.setNxi(args.nxi)                                                  # generate.py:204
        ds.hottailgrid.setPmax(args.pmax)                                                # generate.py:214
        if kinetic:
            ds.hottailgrid.setNp(args.np1 + args.np2)                                    # generate.py:216
            ds.hottailgrid.setBiuniformGrid(psep=args.psep, npsep=args.np1)              # generate.py:217
        else:
            ds.hottailgrid.setNp(args.np2)                                               # generate.py:218-219
        if args.xi_grid != "uniform":
            # the faces of Runes2P's biuniform_theta_xi_faces(nxi, 10, 0.5), as custom faces: DREAM's
            # own setBiuniformGrid(thetasep=...) does not survive the copy of the settings at
            # 93d27e1 (XiGrid.fromdict takes int() of an array)
            ds.hottailgrid.xigrid.setCustomGridPoints(biuniform_theta_xi_faces(args.nxi, 10))
        # a trace cold population in the superthermal mode, without the corrected conductivity
        scale = 0.9999 if superthermal else 1.0                                          # generate.py:221-225
        if superthermal:
            ds.eqsys.j_ohm.setCorrectedConductivity(False)                               # generate.py:229
        ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=scale * nfree, rT0=rT, T0=T0)      # generate.py:231
        ds.eqsys.f_hot.setHotRegionThreshold(5)                                          # generate.py:234
        ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)                              # generate.py:237
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=SCHEMES[args.scheme],
                                                        ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)   # generate.py:239
        ds.eqsys.f_hot.enableIonJacobian(False)                                          # generate.py:243
    if hot and args.runaway_grid:
        # the runaway grid of --runaway-grid from run 1 on, with custom faces: p from the hot-tail p_max,
        # one pitch grid biuniform in theta. With it DREAM's f_hot takes the kinetic-kinetic interface at
        # p_max in place of its boundary condition (src/Settings/Equations/f_hot.cpp:94-95 at v26.5), and
        # the kinetic avalanche of the phases puts the Rosenbluth-Putvinski source on f_re above p_max
        # in place of the external avalanche term; the heating of T_cold, the kinetic ionization and
        # j_re come from f_re (src/Settings/Equations/T_cold.cpp:341-365, ions.cpp:340-353 and
        # j_re.cpp at v26.5). Nothing below sets them otherwise.
        ds.runawaygrid.setEnabled(True)
        ds.runawaygrid.setCustomGrid(p_f=runaway_p_faces(args),
                                     xi_f=biuniform_theta_xi_faces(args.re_nxi, args.re_nxi_sep))
        # the settings of f_re stated, as those of f_hot above: the scheme of f_hot unless --re-scheme
        # (a choice of this package, so that one scheme holds across the interface at p_max; DREAM's
        # default is the centred one, py/DREAM/Settings/Equations/RunawayElectronDistribution.py:22-28;
        # see the docs page *Choices of this package*), the boundary condition at
        # the runaway p_max that keeps the flux of the last face (BC_PHI_CONST, DREAM's default),
        # synchrotron losses neglected as the example leaves them for f_hot (DREAM's default,
        # py/DREAM/Settings/Equations/DistributionFunction.py:67), and the ion Jacobian off as for
        # f_hot (generate.py:243)
        re_scheme = SCHEMES[args.re_scheme or args.scheme]
        ds.eqsys.f_re.setAdvectionInterpolationMethod(ad_int=re_scheme, ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)
        ds.eqsys.f_re.setBoundaryCondition(bc=DistFunc.BC_PHI_CONST)
        ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
        ds.eqsys.f_re.enableIonJacobian(False)

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
    if hot:
        ds.solver.tolerance.set("n_hot", abstol=1e5)                                     # generate.py:282-283
        ds.solver.tolerance.set("j_hot", abstol=1)
    if hot and args.runaway_grid:
        # a choice of this package: the abstol of f_hot of the phases (generate.py:375-387), as DREAM sets
        # none for f_re, only the relative tolerance of all unknowns (py/DREAM/Settings/ToleranceSettings.py:14-17);
        # see the docs page *Choices of this package*
        ds.solver.tolerance.set("f_re", abstol=1e5)
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
    if hot:
        ds1.eqsys.n_re.setAvalanche(RunawayElectrons.AVALANCHE_MODE_KINETIC, pCutAvalanche=0.01)   # generate.py:329
        ds1.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_DISABLED)                 # generate.py:333
        ds1.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=SCHEMES[args.scheme],
                                                         ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)  # generate.py:335
        ds1.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_KINETIC_APPROX_JAC)             # generate.py:340
    if superthermal:
        ds1.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_SUPERTHERMAL              # generate.py:342-344
        ds1.eqsys.T_cold.setInitialProfile(temperature=1)                                # generate.py:346-356
        ignorelist += ["T_cold", "W_cold", "n_hot", "n_cold"]
    if kinetic:
        ds1.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_EXPLICIT,
                                          shape=FHot.PARTICLE_SOURCE_SHAPE_DELTA)          # generate.py:361
    if not hot:
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
    if hot:
        ds1.solver.tolerance.set(unknown="f_hot", reltol=reltol, abstol=1e5)
        ds1.solver.tolerance.set(unknown="n_hot", reltol=reltol, abstol=1e5)
        ds1.solver.tolerance.set(unknown="j_hot", reltol=reltol, abstol=1e-5)
    if hot and args.runaway_grid:
        # a choice of this package, as in the runs before: generate.py:375-387 sets none for f_re, nor
        # does DREAM (py/DREAM/Settings/ToleranceSettings.py:14-17); the abstol of f_hot
        ds1.solver.tolerance.set(unknown="f_re", reltol=reltol, abstol=1e5)
    ds1.output.setTiming(True, True)                                                     # generate.py:391
    ds1.solver.setVerbose(args.verbose)
    ds1.fromOutput(str(path(args, "init_rescaled")), ignore=ignorelist)                  # generate.py:394

    # generate.py simulate, phase 1: ionization
    frac_dinj = scenario["fracDinj"] if args.frac_dinj is None else args.frac_dinj
    ds1.eqsys.n_i.addIon("Dinj", Z=1, iontype=Ions.IONS_DYNAMIC_NEUTRAL, n=frac_dinj * NE0, T=1)
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
    if args.window_start is None:
        ds2.timestep.setTmax(t_quench - t_ioniz)
        ds2.timestep.setNt(nt_quench)
    else:
        # run 4 to the window start, with the step length (t_quench - t_ioniz)/nt_quench of the unsplit phase
        ds2.timestep.setTmax((t_quench - t_ioniz) * args.window_start / nt_quench)
        ds2.timestep.setNt(args.window_start)
    if args.save_steps > 0:
        ds2.timestep.setNumberOfSaveSteps(min(args.save_steps, nt_quench))
    # generate.py:478-480 switches scenario 0 to upwind in kinetic mode, and generate.py:476-477
    # scenarios 2 and 3 in the superthermal mode; with --scheme the scheme is the argument's in both
    # phases, and these scenarios keep the switch of the example
    if args.scheme == "tcdf" and ((args.scenario == 0 and kinetic) or (args.scenario in (2, 3) and superthermal)):
        ds2.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_UPWIND,
                                                         ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)
        # f_re follows f_hot unless --re-scheme (a choice of this package, as in run 1)
        if args.runaway_grid and args.re_scheme is None:
            ds2.eqsys.f_re.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_UPWIND,
                                                            ad_jac=FHot.AD_INTERP_JACOBIAN_FULL)
    if 4 in todo:
        run(ds2, args, "quench")
    if args.run_to < 5:
        return
    with h5py.File(path(args, "quench"), "r") as res:                                    # not an unsplit run 4
        if res["grid/t"].size != args.window_start + 1:
            raise SystemExit(f"{path(args, 'quench')} holds {res['grid/t'].size - 1} steps, not the "
                             f"--window-start {args.window_start}: run 4 again with --run-from 4")

    # the window: the steps window_start to window_start + window_steps of phase 2 from DREAM's state at
    # the end of run 4, as generate.py:460-470 starts phase 2 from phase 1, with the settings of phase 2
    ds3 = DREAMSettings(ds2)
    ds3.fromOutput(str(path(args, "quench")))
    ds3.timestep.setTmax((t_quench - t_ioniz) * args.window_steps / nt_quench)
    ds3.timestep.setNt(args.window_steps)
    run(ds3, args, WINDOW)


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


TEST_UNKNOWNS = ("f_hot", "f_re", "T_cold", "n_cold", "n_hot", "n_re", "n_tot", "E_field", "psi_p", "psi_edge", "psi_wall",
                 "I_p", "j_ohm", "j_hot", "j_re", "j_tot", "W_cold", "n_i", "W_i")


def stored_arguments(args):
    """The arguments as stored in the output, with the bare name of each path argument, so that
    the data holds no paths of the machine that made it. --test-data-channels is stored only when
    it is given, and the arguments of the runaway grid only with --runaway-grid, so that the files made
    without them keep the arguments they were published with."""
    stored = {k: (v.name if isinstance(v, Path) else str(v)) for k, v in vars(args).items()
              if k not in omitted_arguments(args)}
    if not args.test_data_channels:
        del stored["test_data_channels"]
    return stored


def test_data(args):
    """DREAM's outputs of the runs of the chain that the output directory holds, of those named by
    --test-data-runs, as the Runes2P tests read them: per run the unknowns of TEST_UNKNOWNS that it has at every saved time, the
    conductivity of the fluid quantities where DREAM saved it, with --test-data-channels the channels
    of the equation of T_cold (other/fluid/Tcold_*), the times and the grids, with the
    provenance. The runs before the phases (conductivity, init, init_rescaled) are those of the
    initialization of the case."""
    args.test_data.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.test_data, "w") as out:
        out.attrs["source"] = "disruption_kinetic.py"
        out.attrs["arguments"] = json.dumps(stored_arguments(args))
        base.provenance(out)
        for name in ALL_RUNS:
            if name not in args.test_data_runs.split(",") or not path(args, name).exists():
                continue
            with h5py.File(path(args, name), "r") as res:
                group = out.create_group(name)
                if name == WINDOW:
                    # the times of the window count from its start, as those of each phase do: its start
                    # as a step of phase 2 and as a time from the start of phase 2 and of phase 1
                    t_ioniz, _, t_quench, nt_quench = phase_times(args)
                    t_start = (t_quench - t_ioniz) * args.window_start / nt_quench
                    group.attrs["window_start_step"] = args.window_start
                    group.attrs["window_start_time_quench"] = t_start
                    group.attrs["window_start_time"] = t_ioniz + t_start
                for q in TEST_UNKNOWNS:
                    if f"eqsys/{q}" in res:
                        group[f"eqsys/{q}"] = res[f"eqsys/{q}"][:]
                if "other/fluid/conductivity" in res:
                    group["other/fluid/conductivity"] = res["other/fluid/conductivity"][:]
                if args.test_data_channels and "other/fluid" in res:
                    for q in res["other/fluid"]:
                        if q.startswith("Tcold_"):
                            group[f"other/fluid/{q}"] = res[f"other/fluid/{q}"][:]
                for q in ("t", "r", "r_f"):
                    group[f"grid/{q}"] = res[f"grid/{q}"][:]
                for g in ("hottail", "runaway"):
                    for q in ("p1_f", "p2_f"):
                        if f"grid/{g}/{q}" in res:
                            group[f"grid/{g}/{q}"] = res[f"grid/{g}/{q}"][:]
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
