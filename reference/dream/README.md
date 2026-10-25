# Reference data of DREAM and GSL

This folder holds the generator scripts of the public reference data sets of Runes2P.
The tests compare Runes2P with these data instead of running DREAM.

The cases are DREAM's public examples, run with their own settings and a few stated overrides, and synthetic cases written for one comparison each.
Each script in `cases/` states its case, its overrides and the test that reads its output.
`../gsl/bicubic_reference.c` writes the values of GSL's 2D interpolation as DREAM uses it, on a synthetic table.

## The sets

| Set | Source | Files |
|---|---|---|
| `dream-v26.5` | DREAM release v26.5, commit `ecdd5e146537c77602c9d7cc76b36100200e4b9a` | 16 |
| `dream-boltzmann-eqterm-93d27e17` | DREAM branch `boltzmann-eqterm`, commit `93d27e1763827e19dafb1fe790e99c2688ad6d62` | 6 |
| `gsl-2.8` | GSL 2.8 of conda-forge | 1 |

DREAM is at <https://github.com/chalmersplasmatheory/DREAM>.
The branch `boltzmann-eqterm` is in no release, and v26.5 is not its ancestor.
The cases of the Boltzmann knock-on operator need the branch.

The DREAM file and line citations in the code refer to DREAM commit `93d27e1` (branch `boltzmann-eqterm`) unless they name another version.

Each set is published as a tarball at `https://deep.reak.bme.hu/~data/runes2p/reference/<set>/<set>.tar.gz`:

- https://deep.reak.bme.hu/~data/runes2p/reference/dream-v26.5/dream-v26.5.tar.gz
- https://deep.reak.bme.hu/~data/runes2p/reference/dream-boltzmann-eqterm-93d27e17/dream-boltzmann-eqterm-93d27e17.tar.gz
- https://deep.reak.bme.hu/~data/runes2p/reference/gsl-2.8/gsl-2.8.tar.gz

The files sit at the top level of each tarball.
Every DREAM file records `dream_commit`, `dream_branch` and `petsc_version` as attributes.

## Build

The toolchain comes from the pixi environment of `pixi.toml` and `pixi.lock`: conda-forge GCC, MKL, HDF5 without MPI, GSL 2.8 and Python 3.13.
PETSc v3.25.5 is built from source without MPI and with MKL PARDISO, as DREAM recommends.
One PETSc serves both DREAM builds.

```bash
cd reference/dream
pixi install --frozen
bash build_all.sh          # NJOBS sets the number of compile jobs, default 8
```

`build_all.sh` clones PETSc and DREAM, checks out the commit of each set and builds them.
The logs go to `logs/`.

## Regenerate

```bash
bash run_all.sh                       # all three sets
bash run_all.sh gsl-2.8               # named sets only
ONLY="kinetic_operator" bash run_all.sh dream-v26.5
```

A first argument that is not the name of a set is the data directory, which defaults to this folder.
`run_all.sh` checks that the DREAM checkout of each set is at its commit before it runs a case.

## Layout

- `cases/`: the 22 case scripts of the two DREAM sets.
- `scripts/`: the fetch and build steps that `pixi.toml` runs as tasks.
- `petsc/`: the shared PETSc, made by `build_all.sh`.
- `<set>/DREAM/`, `<set>/build/`: the DREAM checkout and build of a set.
  `<set>/petsc` links to the shared PETSc.
- `<set>/cases/`: the copies of the case scripts that `run_all.sh` runs.
  They find `dreami`, the DREAM checkout and PETSc relative to their own folder.
- `out/<set>/`: the output files of a set, the content of its tarball.
- `work/<set>/` and `logs/`: working files and the log of every case.

None of these build and output folders is tracked.

## Reproducibility

The HDF5 files hold a `created` time stamp.
A rerun reproduces the numbers, not the bytes of the files.

A tarball is packed from `out/<set>/`, with the files at mode 644:

```bash
export LC_ALL=en_US.UTF-8      # the shell expands * in this order
cd out/<set>
tar --sort=name --mtime='2026-01-01 00:00Z' --owner=0 --group=0 --numeric-owner \
    --mode='u=rw,go=r' -cf - * | gzip -n -9 > ../<set>.tar.gz
```

The file order inside the published tarballs is that of a UTF-8 locale (`dream_time_stepping_adaptive.h5` before `dream_time_stepping.h5`).
A repack needs `LC_ALL=en_US.UTF-8` to give the same order.
