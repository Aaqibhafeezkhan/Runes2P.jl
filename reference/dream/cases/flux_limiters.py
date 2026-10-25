#!/usr/bin/env python3
"""
DREAM reference data for the advective interpolation schemes and flux limiters, for the
Runes2P test test/test_flux_limiters_dream.jl.

The settings are those of DREAM's flux-limiter example, examples/fluxlimiter/generate_data.py
(Ola Embreus, 2020-11-23), which scans the interpolation schemes: E = 6.745459970079014 V/m,
n = 5e19 m^-3 fully ionized deuterium, T = 100 eV, full collision mode, Np = 500 to
pMax = 0.3, Nxi = 40, f = 0 beyond pMax, one radius (B0 = 5 T, a = 0.22 m), 10 time steps
to 1 ms, avalanche neglected, no runaway grid, and the example's linearly implicit solver.
The example's convergence scan is not run; each scheme is run once at the example's base
resolution. Overrides, as in example_runaway.py, so that the comparison needs neither
unknown: n_cold prescribed at the deuterium density, and no f_hot particle source.

The scheme is set for p and xi together (ad_int). For a scheme linear in f (centred as a
control, QUICK, second-order upwind; upwind is covered by example_runaway.py) the
linearly implicit solver is backward Euler; for a flux limiter it builds the limiter
weights once per step from the previous time level. The file stores the initial state, the
final state of the linear schemes and time levels 1 and 2 of the limiters (the first step
and one lagged step), which is what the test uses, and the settings once. A last run,
QUICK in superthermal mode, where the flux through p = 0 does not vanish, tests the face
value on the p = 0 face.

(With the NONLINEAR solver and reltol = 1e-12, DREAM does not converge on this case: the
centred scheme stagnates at a relative update of 5.6e-7, and QUICK and TCDF fail after 100
iterations.)

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/flux_limiters.py out/<set>/dream_flux_limiters.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
from example_runaway import example_provenance  # noqa: E402

EXAMPLE = Path("examples") / "fluxlimiter" / "generate_data.py"
E = 6.745459970079014
N = 5e19
T = 100
PMAX = 0.3
NP = 500
NXI = 40
TMAX = 1e-3
NT = 10
LIMITERS = ("smart", "muscl", "ospre", "tcdf")

SCHEMES = {
    "centred": FHot.AD_INTERP_CENTRED,
    "quick": FHot.AD_INTERP_QUICK,
    "upwind_second_order": FHot.AD_INTERP_UPWIND_2ND_ORDER,
    "smart": FHot.AD_INTERP_SMART,
    "muscl": FHot.AD_INTERP_MUSCL,
    "ospre": FHot.AD_INTERP_OSPRE,
    "tcdf": FHot.AD_INTERP_TCDF,
}


def settings(scheme, collfreq_mode=Collisions.COLLFREQ_MODE_FULL):
    ds = DREAMSettings()
    ds.eqsys.E_field.setPrescribedData(E)
    ds.eqsys.T_cold.setPrescribedData(T)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=N)
    ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_NEGLECT)
    ds.hottailgrid.setNxi(NXI)
    ds.hottailgrid.setNp(NP)
    ds.hottailgrid.setPmax(PMAX)
    ds.collisions.collfreq_mode = collfreq_mode
    ds.eqsys.f_hot.setInitialProfiles(n0=N, T0=T)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=scheme)
    ds.runawaygrid.setEnabled(False)
    ds.radialgrid.setB0(5)
    ds.radialgrid.setMinorRadius(0.22)
    ds.radialgrid.setWallRadius(0.22)
    ds.radialgrid.setNr(1)
    # overrides (see the module docstring)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_PRESCRIBED)
    ds.eqsys.n_cold.setPrescribedData(N)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.timestep.setTmax(TMAX)
    ds.timestep.setNt(NT)
    return ds


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "flux_limiters.py"
        base.provenance(out)
        example_provenance(out, EXAMPLE)
        out["E"], out["n"], out["T"] = E, N, T
        for name, scheme in SCHEMES.items():
            ds = settings(scheme)
            sfile, ofile = Path(tmp) / f"s_{name}.h5", Path(tmp) / f"o_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.output.setTiming(stdout=False, file=False)
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            g = out.create_group(name)
            g.attrs["ad_int"] = int(scheme)
            # the settings once (they differ only in the interpolation method, kept per scheme)
            base.bare_output_filename(sfile)
            with h5py.File(sfile, "r") as sett:
                if "settings" not in out:
                    for key in sett:
                        sett.copy(sett[key], out, name=f"settings/{key}")
                for q in ("p1", "p2"):
                    g[f"adv_interp/{q}"] = sett[f"eqsys/f_hot/adv_interp/{q}"][()]
            with h5py.File(ofile, "r") as res:
                f = res["eqsys/f_hot"][:, 0]  # (t, xi, p)
                if "grid" not in out:
                    for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                    out["t"] = res["grid/t"][:]
                    out.create_dataset("f_initial", data=f[0], compression="gzip", compression_opts=9, shuffle=True)
                levels = (1, 2) if name in LIMITERS else (NT,)
                for n in levels:
                    g.create_dataset(f"f_{n}", data=f[n], compression="gzip", compression_opts=9, shuffle=True)
            print(f"{name}: done")
        # QUICK in superthermal mode, where the flux through p = 0 does not vanish: it tests
        # the scheme's face value on the p = 0 face (the limiters give f1 there anyway)
        ds = settings(FHot.AD_INTERP_QUICK, Collisions.COLLFREQ_MODE_SUPERTHERMAL)
        sfile, ofile = Path(tmp) / "s_qs.h5", Path(tmp) / "o_qs.h5"
        ds.output.setFilename(str(ofile))
        ds.output.setTiming(stdout=False, file=False)
        ds.save(str(sfile))
        subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
        g = out.create_group("quick_superthermal")
        with h5py.File(sfile, "r") as sett:
            g["collfreq_mode"] = sett["collisions/collfreq_mode"][()]
            g["adv_interp/p1"] = sett["eqsys/f_hot/adv_interp/p1"][()]
        with h5py.File(ofile, "r") as res:
            g.create_dataset(f"f_{NT}", data=res["eqsys/f_hot"][NT, 0], compression="gzip", compression_opts=9,
                             shuffle=True)
        print("quick_superthermal: done")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
