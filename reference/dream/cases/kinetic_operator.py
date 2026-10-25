#!/usr/bin/env python3
"""
DREAM reference data for the full small-angle kinetic operator.

One linearly implicit (backward Euler) DREAM step with centred interpolation,
f = 0 beyond p_max and no particle source. With coefficients taken from the
previous time step, the saved distributions satisfy
    (f1 - f0)/dt = L f1
to the accuracy of the direct linear solve, which tests the complete
discretization (fluxes, interpolation, boundaries, flux through p = 0).

Usage (from reference/dream, with the environment that run_all.sh sets for <set>):
    pixi run python <set>/cases/kinetic_operator.py out/<set>/dream_kinetic_operator.h5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.HotElectronDistribution as FHot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision_frequencies as base  # noqa: E402

VARIANTS = {
    "full_partial_thermal": base.VARIANTS["full_partial_thermal"],
    "superthermal_complete_energy": dict(
        collfreq_mode=Collisions.COLLFREQ_MODE_SUPERTHERMAL,
        collfreq_type=Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED,
        lnlambda=Collisions.LNLAMBDA_ENERGY_DEPENDENT,
        screened_diffusion=Collisions.SCREENED_DIFFUSION_MODE_MAXWELLIAN,
    ),
}
DT = 1e-6


def main(outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, h5py.File(outfile, "w") as out:
        out.attrs["source"] = "kinetic_operator.py"
        base.provenance(out)
        out["ions/name"] = [n for n, *_ in base.IONS]
        out["ions/Z"] = [Z for _, Z, *_ in base.IONS]
        out["ions/Z0"] = [Z0 for _, _, Z0, _ in base.IONS]
        out["ions/n"] = [n for *_, n in base.IONS]
        out["E"] = base.E_FIELD
        out["B"] = base.B0
        out["dt"] = DT
        for name, variant in VARIANTS.items():
            ds = base.settings(variant)
            ds.eqsys.f_hot.setInitialProfiles(n0=1e17, T0=1000.0)
            ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_CENTRED)
            ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
            ds.eqsys.f_hot.setParticleSource(FHot.PARTICLE_SOURCE_ZERO)
            ds.timestep.setTmax(DT)
            ds.timestep.setNt(1)
            ds.other.include()
            sfile = Path(tmp) / f"settings_{name}.h5"
            ofile = Path(tmp) / f"output_{name}.h5"
            ds.output.setFilename(str(ofile))
            ds.save(str(sfile))
            subprocess.run([str(base.DREAMI), str(sfile)], check=True, stdout=subprocess.DEVNULL)
            with h5py.File(ofile, "r") as res:
                g = out.create_group(name)
                for key, val in variant.items():
                    g.attrs[key] = int(val)
                if "grid" not in out:
                    for q in ("p1", "p1_f", "p2", "p2_f", "Vprime"):
                        out[f"grid/{q}"] = res[f"grid/hottail/{q}"][:]
                    out["grid/r"] = res["grid/r"][:]
                    out["grid/r_f"] = res["grid/r_f"][:]
                    out["grid/VpVol"] = res["grid/VpVol"][:]
                g["t"] = res["grid/t"][:]
                g["T_cold"] = res["eqsys/T_cold"][:]
                g["n_cold"] = res["eqsys/n_cold"][:]
                g.create_dataset("f_hot", data=res["eqsys/f_hot"][:], compression="gzip")
                # DREAM's cell-centre current moment of f_hot (no momentum threshold in the
                # superthermal variant; the full variant carries pThreshold), for the
                # DREAM-compatible pitch rule of current_density of Runes2P
                g["j_hot"] = res["eqsys/j_hot"][:]
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1])
