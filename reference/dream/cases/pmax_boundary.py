#!/usr/bin/env python3
"""
DREAM reference data for the p_max boundary conditions of the hot-tail equation, for
the Runes2P test test/test_pmax_boundary_dream.jl.

DREAM's own example, examples/basic/basic.py (Embreus, 2020-11-23), run unchanged with
its own settings (full collision mode; ds.hottailgrid.collfreq_mode is ignored by DREAM),
the linearly implicit solver, and its default
BC_PHI_CONST boundary. Its own initial 1 keV Maxwellian on p_max = 5 underflows to exactly
zero on the upper part of the grid, and its own 4 steps of 0.25 microseconds cannot carry
anything out to p = 5 either, so it cannot tell BC_F_0, BC_PHI_CONST and BC_DPHI_CONST
apart. Two overrides are added on top of the example's own settings:

- the initial f_hot, set to the example's own commented-out linear ramp
  (pmax - p) / pmax, isotropic: non-negligible at p = p_max, so the boundary conditions
  give different solutions there. The p and xi node values come from an unmodified run of
  the example (its own grid), read back from that run's output.
- n_cold prescribed at the free-electron density of the example's ions (D fully ionized,
  Ar neutral, so only the deuterium free density) and the particle source set to zero
  (the overrides of example_runaway.py, so that the comparison needs no n_cold or
  particle-source unknown).

Everything else is the example's: the hot-tail grid, E, T, B0, the minor radius, full
collision mode, the linearly implicit solver. The example's own 4 steps to 1 microsecond
are run 8 times over (NT below) so the boundary conditions separate further.

Variants: BC_F_0, BC_PHI_CONST (DREAM's default) and BC_DPHI_CONST, all with the example's
centred interpolation, plus BC_PHI_CONST and BC_DPHI_CONST again with QUICK (linear in f,
so it still runs under the linearly implicit solver; the boundary is tested with the
face coefficients of the scheme, not only centred).

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/pmax_boundary.py out/<set>/dream_pmax_boundary.h5
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.HotElectronDistribution as FHot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402
from example_runaway import example_provenance  # noqa: E402

EXAMPLE = Path("examples") / "basic" / "basic.py"
NT = 8

# name -> (boundary condition, ad_int or None for the example's own centred)
VARIANTS = {
    "f_zero": (FHot.BC_F_0, None),
    "phi_const": (FHot.BC_PHI_CONST, None),
    "dphi_const": (FHot.BC_DPHI_CONST, None),
    "phi_const_quick": (FHot.BC_PHI_CONST, FHot.AD_INTERP_QUICK),
    "dphi_const_quick": (FHot.BC_DPHI_CONST, FHot.AD_INTERP_QUICK),
}


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "pmax_boundary.py"
        base.provenance(out)
        example_provenance(out, EXAMPLE)
        tmp = Path(tmp)
        shutil.copy(base.ROOT / "DREAM" / EXAMPLE, tmp / "basic.py")
        subprocess.run([sys.executable, "basic.py"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)

        with h5py.File(tmp / "dream_settings.h5", "r") as sett:
            # charge-state densities are stacked per ion (Z + 1 states each), not a flat
            # index across ions, so the free density is summed per ion; both the nuclear
            # charges and the per-state densities are also stored below, so the Runes2P test
            # builds its ions from this file instead of restating the example's values
            ion_z = sett["eqsys/n_i/Z"][:]
            raw_name = sett["eqsys/n_i/names"][0]
            raw_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            ion_names = raw_name.rstrip(";").split(";")
            states = sett["eqsys/n_i/prescribed/x"][:, 0, 0]
            dt = float(sett["timestep/tmax"][0]) / float(sett["timestep/nt"][0])
        free_density = 0.0
        idx = 0
        for z in ion_z:
            for z0 in range(int(z) + 1):
                free_density += z0 * states[idx]
                idx += 1

        # an unmodified probe run of the example, to read back its own p, xi grid
        probe_ds = DREAMSettings(str(tmp / "dream_settings.h5"))
        probe_ofile = tmp / "probe_output.h5"
        probe_ds.output.setFilename(str(probe_ofile))
        probe_ds.output.setTiming(stdout=False, file=False)
        probe_sfile = tmp / "probe_settings.h5"
        probe_ds.save(str(probe_sfile))
        subprocess.run([str(base.DREAMI), str(probe_sfile)], check=True, stdout=subprocess.DEVNULL)
        with h5py.File(probe_ofile, "r") as res:
            p = res["grid/hottail/p1"][:]
            xi = res["grid/hottail/p2"][:]
        with h5py.File(tmp / "dream_settings.h5", "r") as sett:
            pmax = float(sett["hottailgrid/pmax"][0])
        ramp = (pmax - p) / pmax
        f0 = np.tile(ramp, (len(xi), 1))  # (nxi, np), isotropic

        out["ions/name"] = ion_names
        out["ions/Z"] = ion_z
        out["ions/states"] = states  # charge-state densities, stacked per ion (Z + 1 rows each)

        for name, (bc, ad_int) in VARIANTS.items():
            ds = DREAMSettings(str(tmp / "dream_settings.h5"))
            ds.eqsys.n_cold.setType(ColdElectrons.TYPE_PRESCRIBED)
            ds.eqsys.n_cold.setPrescribedData(free_density)
            ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
            ds.eqsys.f_hot.setInitialValue(f0.reshape(1, len(xi), len(p)), r=[0.0], p=p, xi=xi)
            ds.eqsys.f_hot.setBoundaryCondition(bc)
            if ad_int is not None:
                ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=ad_int)
            ds.timestep.setTmax(NT * dt)
            ds.timestep.setNt(NT)

            sfile, ofile = tmp / f"s_{name}.h5", tmp / f"o_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.output.setTiming(stdout=False, file=False)
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)

            g = out.create_group(name)
            g.attrs["bc"] = int(bc)
            base.bare_output_filename(sfile)
            with h5py.File(sfile, "r") as sett2:
                for key in sett2:
                    sett2.copy(sett2[key], g, name=f"settings/{key}")
            with h5py.File(ofile, "r") as res:
                if "grid" not in out:
                    # only the face arrays are stored: the test builds MomentumGrid and
                    # RadialGrid from them, not from the cell centres or Vprime
                    for q in ("p1_f", "p2_f"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                g["t"] = res["grid/t"][:]
                g["n_re"] = res["eqsys/n_re"][:]
                fh = res["eqsys/f_hot"]
                g.create_dataset("f_hot_initial", data=fh[0], compression="gzip")
                g.create_dataset("f_hot_final", data=fh[-1], compression="gzip")
                n_re = res["eqsys/n_re"][:]
                t = res["grid/t"][:]
                rate = float(n_re[-1, 0] - n_re[-2, 0]) / float(t[-1] - t[-2])
                print(f"{name}: n_re rate at t = {t[-1]:.3e} s: {rate:.4e} m^-3 s^-1")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
