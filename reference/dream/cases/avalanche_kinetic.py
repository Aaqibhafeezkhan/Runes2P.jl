#!/usr/bin/env python3
"""
DREAM's kinetic avalanche growth rate in a uniform plasma with a constant field.

A fully ionized plasma with one ion species at 100 eV and an electron density of 1e20 m^-3,
a constant field, full collision mode, complete screening, the constant Coulomb logarithm,
cylindrical geometry, one radius, hot-tail grid only, f = 0 beyond p_max, no particle source,
no synchrotron losses, no Dreicer source.

Two sources (source=):
  boltzmann  AVALANCHE_MODE_BOLTZMANN_KNOCK_ON, the operator of Runes2P. The primaries are the
             electrons of f_hot. The grid is that of the Runes2P test (60 cells below p = 0.6,
             cells of the width 99.4/60 above, 14 uniform pitch cells), the seed a Gaussian
             at p = 2, n_re starts at zero. The runaways are the electrons of f_hot above
             the test-particle critical momentum plus n_re.
  rp         AVALANCHE_MODE_KINETIC, the Rosenbluth-Putvinski source. The primaries are the
             electrons of n_re, those beyond p_max, with infinite energy. The grid ends at
             p_max (default 10, cells of the width 0.1 above p = 0.6), n_re starts at
             1e10 m^-3 and f_hot at 1 m^-3 (100 keV, as examples/avalanche of DREAM). The
             runaways are n_re.

The growth rate is printed per window. The file holds the time
series and the settings. With store=test it holds the grid and the first and the last
distribution as well, which test/test_avalanche_dream.jl uses:

    pixi run python <set>/cases/avalanche_kinetic.py out/<set>/dream_avalanche_kinetic.h5 \
        source=boltzmann pmax=200 tmax=0.08 nt=192 windows=4 pcut=0.13 store=test

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/avalanche_kinetic.py <output file.h5> source=boltzmann [e_over_ec=20]
        [charge=1] [pmax=200] [tmax=0.16] [nt=384] [windows=8] [pcut=0.13] [ad=centred]
        [split=0.6] [width=] [nxi=14] [store=record]

split is the end of the fine part of the momentum grid (cells of 0.01), width the cell width
above it, nxi the number of uniform pitch cells.

plasma=argon field=<V/m> takes the plasma of Hesslow et al. 2019, Fig. 1, in place of the
fully ionized one: deuterium ions and singly ionized argon with 1e20 m^-3 each at 10 eV,
partial screening and the energy-dependent Coulomb logarithm. The critical momentum of the
count is that of the total electron density.
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
from runaway_rates import A_MINOR, B0, critical_field, relativistic_coulomb_log  # noqa: E402

TEMPERATURE = 100.0
DENSITY = 1e20
WIDTH_BELOW = 0.01
INTERPOLATION = {"centred": FHot.AD_INTERP_CENTRED, "upwind": FHot.AD_INTERP_UPWIND, "tcdf": FHot.AD_INTERP_TCDF}


def argument(name, default):
    for a in sys.argv[2:]:
        if a.startswith(name + "="):
            return a[len(name) + 1:]
    return default


def p_faces(pmax, split, width):
    below = int(round(split / WIDTH_BELOW))
    above = int(round((pmax - split) / width))
    return np.concatenate([np.linspace(0.0, split, below + 1), np.linspace(split, pmax, above + 1)[1:]])


def settings(source, e_over_ec, charge, pmax, tmax, nt, pcut, interpolation, split, width, nxi, argon_field):
    argon = argon_field is not None
    ds = DREAMSettings()
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = (Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED if argon
                                   else Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED)
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT if argon else Collisions.LNLAMBDA_CONSTANT

    ds.radialgrid.setB0(B0)
    ds.radialgrid.setMinorRadius(A_MINOR)
    ds.radialgrid.setWallRadius(A_MINOR)
    ds.radialgrid.setNr(1)

    if argon:
        temperature, free, total = 10.0, 2e20, 1.9e21
        critical = critical_field(total, relativistic_coulomb_log(free, temperature))
        field = argon_field
        ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_PRESCRIBED, n=1e20)
        ds.eqsys.n_i.addIon(name="Ar", Z=18, Z0=1, iontype=Ions.IONS_PRESCRIBED, n=1e20)
    else:
        temperature, free = TEMPERATURE, DENSITY
        critical = critical_field(DENSITY, relativistic_coulomb_log(DENSITY, TEMPERATURE))
        field = e_over_ec * critical
        ds.eqsys.n_i.addIon(name="ion", Z=charge, Z0=charge, iontype=Ions.IONS_PRESCRIBED, n=DENSITY / charge)
    ds.eqsys.E_field.setPrescribedData(field)
    ds.eqsys.T_cold.setPrescribedData(temperature)
    ds.eqsys.n_cold.setPrescribedData(free)

    p_f = p_faces(pmax, split, width)
    xi_f = np.linspace(-1, 1, nxi + 1)
    ds.hottailgrid.setCustomGrid(p_f=list(p_f), xi_f=list(xi_f))
    p = 0.5 * (p_f[1:] + p_f[:-1])
    xi = 0.5 * (xi_f[1:] + xi_f[:-1])
    if source == "boltzmann":
        P, XI = np.meshgrid(p, xi)  # (nxi, np)
        seed = 1e12 * np.exp(-(P - 2.0)**2 / 0.08) * np.exp(4 * (XI - 1))
        ds.eqsys.f_hot.setInitialValue(seed.reshape(1, len(xi), len(p)), r=[0.0], p=p, xi=xi)
        ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_BOLTZMANN_KNOCK_ON, pCutAvalanche=pcut)
        ds.eqsys.n_re.setInitialProfile(density=0)
    else:
        ds.eqsys.f_hot.setInitialProfiles(n0=1, T0=1e5)
        ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_KINETIC, pCutAvalanche=pcut)
        ds.eqsys.n_re.setInitialProfile(density=1e10)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=interpolation)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    ds.runawaygrid.setEnabled(False)

    # LINEAR_IMPLICIT segfaults at the first step with the Boltzmann operator (93d27e1)
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
    ds.solver.tolerance.set(reltol=1e-8)
    ds.timestep.setTmax(tmax)
    ds.timestep.setNt(nt)
    ds.other.include("fluid/GammaAva", "fluid/pCrit", "fluid/Ecfree", "fluid/Eceff")
    return ds, field, critical


def main(outfile):
    source = argument("source", "boltzmann")
    if source not in ("boltzmann", "rp"):
        raise SystemExit("source is boltzmann or rp")
    e_over_ec = float(argument("e_over_ec", "20"))
    charge = int(argument("charge", "1"))
    pmax = float(argument("pmax", "200" if source == "boltzmann" else "10"))
    tmax = float(argument("tmax", "0.16"))
    nt = int(argument("nt", "384"))
    windows = int(argument("windows", "8"))
    pcut = float(argument("pcut", "0.13"))
    interpolation = argument("ad", "centred")
    split = float(argument("split", "0.6"))
    width = float(argument("width", str((100.0 - 0.6) / 60) if source == "boltzmann" else "0.1"))
    nxi = int(argument("nxi", "14"))
    argon_field = float(argument("field", "0")) if argument("plasma", "ionized") == "argon" else None
    if nt % windows:
        raise SystemExit("nt is a multiple of windows")

    ds, field, critical = settings(source, e_over_ec, charge, pmax, tmax, nt, pcut, INTERPOLATION[interpolation],
                                   split, width, nxi, argon_field)
    e_over_ec = field / critical
    p_critical = 1 / np.sqrt(e_over_ec - 1)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "avalanche_kinetic.py"
        out.attrs["arguments"] = " ".join(sys.argv[2:])
        base.provenance(out)
        sfile, ofile = Path(tmp) / "s.h5", Path(tmp) / "o.h5"
        ds.output.setFilename(str(ofile))
        ds.save(str(sfile))
        subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
        base.bare_output_filename(sfile)
        with h5py.File(sfile, "r") as sett:
            for key in sett:
                sett.copy(sett[key], out, name=f"settings/{key}")
        with h5py.File(ofile, "r") as res:
            t = res["grid/t"][:]
            n_re = res["eqsys/n_re"][:, 0]
            f = res["eqsys/f_hot"][:, 0]                 # (time, xi, p)
            p_f, xi_f = res["grid/hottail/p1_f"][:], res["grid/hottail/p2_f"][:]
            p = res["grid/hottail/p1"][:]
            weight = res["grid/hottail/Vprime"][0] * np.diff(xi_f)[:, None] * np.diff(p_f)[None, :] / \
                res["grid/VpVol"][0]
            if argument("store", "record") == "test":
                for q in ("p1_f", "p2_f"):
                    out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                out["grid/r_f"] = res["grid/r_f"][:]
                out["B"] = B0
                out["temperature"], out["density"] = TEMPERATURE, DENSITY
                out.create_dataset("f_hot_initial", data=res["eqsys/f_hot"][0], compression="gzip")
                out.create_dataset("f_hot_final", data=res["eqsys/f_hot"][-1], compression="gzip")
            fluid = {q: res[f"other/fluid/{q}"][-1, 0] for q in ("GammaAva", "pCrit", "Ecfree", "Eceff")}
        above = p > p_critical
        on_grid = np.einsum("tjp,jp->t", f[:, :, above], weight[:, above])
        mean_momentum = np.einsum("tjp,jp->t", f[:, :, above], weight[:, above] * p[above][None, :]) / \
            np.where(on_grid > 0, on_grid, 1)
        count = n_re + on_grid if source == "boltzmann" else n_re
        out["t"], out["n_re"], out["runaways_on_grid"], out["count"] = t, n_re, on_grid, count
        out["mean_momentum_on_grid"] = mean_momentum
        out["E_field"], out["E_c"], out["p_critical"] = field, critical, p_critical
        for q, v in fluid.items():
            out[f"fluid/{q}"] = v
        print(f"source {source}, E/E_c {e_over_ec:g}, Z {charge}, p_max {pmax:g}, {len(p)} x {nxi} cells, "
              f"split {split:g}, width above {width:.4g}, "
              f"cutoff {pcut:g}, {interpolation}, t_max {tmax:g} s, {nt} steps")
        print(f"E {field:.6g} V/m, E_c {critical:.6g}, DREAM Ecfree {fluid['Ecfree']:.6g}, Eceff {fluid['Eceff']:.6g}, "
              f"p_c {p_critical:.4f}, fluid formula of the run's mode: GammaAva {fluid['GammaAva']:.4f} 1/s, "
              f"pCrit {fluid['pCrit']:.4f}")
        per_window = nt // windows
        rates = []
        for w in range(windows):
            a, b = w * per_window, (w + 1) * per_window
            rate = np.log(count[b] / count[a]) / (t[b] - t[a]) if count[a] > 0 and count[b] > 0 else np.nan
            rates.append(rate)
            print(f"window {w + 1}, {1e3 * t[a]:.1f} to {1e3 * t[b]:.1f} ms: rate {rate:.3f}, count {count[b]:.4e}, "
                  f"n_re {n_re[b]:.4e}, mean p on the grid {mean_momentum[b]:.2f}, "
                  f"share in n_re {n_re[b] / count[b]:.2e}")
        out["window_rates"] = rates
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
