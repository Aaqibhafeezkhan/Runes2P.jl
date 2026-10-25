# Reference data

The tests compare Runes2P with the output of DREAM and of GSL stored in three reference sets, so they do not run DREAM themselves.
The DREAM cases are DREAM's public examples, run with their own settings and a few stated overrides, and synthetic cases written for one comparison each.
The generator scripts, the build of DREAM and the regeneration are in `reference/` of the repository.

## The sets

| Set | Source | Files |
|---|---|---|
| `dream-v26.5` | DREAM release v26.5, commit `ecdd5e146537c77602c9d7cc76b36100200e4b9a` | 16 |
| `dream-boltzmann-eqterm-93d27e17` | DREAM branch `boltzmann-eqterm`, commit `93d27e1763827e19dafb1fe790e99c2688ad6d62` | 6 |
| `gsl-2.8` | GSL 2.8 of conda-forge | 1 |

DREAM is at <https://github.com/chalmersplasmatheory/DREAM>.
The branch `boltzmann-eqterm` holds the Boltzmann knock-on operator, which is in no release, so the cases of that operator need the branch.
The release v26.5 is not an ancestor of the branch.
The set `gsl-2.8` holds the values of GSL's two-dimensional interpolation as DREAM uses it for the ADAS coefficients, on a synthetic table (`reference/gsl/bicubic_reference.c`).

The DREAM file and line citations in the code and on these pages refer to DREAM commit `93d27e1` (branch `boltzmann-eqterm`) unless they name another version.

Each set is published as a tarball with its files at the top level:

- <https://deep.reak.bme.hu/~data/runes2p/reference/dream-v26.5/dream-v26.5.tar.gz>
- <https://deep.reak.bme.hu/~data/runes2p/reference/dream-boltzmann-eqterm-93d27e17/dream-boltzmann-eqterm-93d27e17.tar.gz>
- <https://deep.reak.bme.hu/~data/runes2p/reference/gsl-2.8/gsl-2.8.tar.gz>

## The files and their tests

| File | Set | Case script | Test |
|---|---|---|---|
| `dream_collisions.h5` | `dream-v26.5` | `collision_frequencies.py` | `test_collisions_dream.jl` |
| `dream_cold_energy.h5` | `dream-v26.5` | `cold_energy_fluid.py` | `test_cold_energy_dream.jl` |
| `dream_cooling_positivity.h5` | `dream-v26.5` | `cooling_positivity.py` | `test_cooling_dream.jl` |
| `dream_disruption_small.h5` | `dream-v26.5` | `disruption_kinetic.py` | `test_disruption_dream.jl` |
| `dream_dreicer_kinetic.h5` | `dream-v26.5` | `dreicer_kinetic.py` | `test_runaway_physics.jl` |
| `dream_example_runaway.h5` | `dream-v26.5` | `example_runaway.py` | `test_example_runaway.jl` |
| `dream_flux_limiters.h5` | `dream-v26.5` | `flux_limiters.py` | `test_flux_limiters_dream.jl` |
| `dream_ion_energy.h5` | `dream-v26.5` | `ion_energy_fluid.py` | `test_ion_energy_dream.jl` |
| `dream_ion_rates.h5` | `dream-v26.5` | `ion_rates_fluid.py` | `test_ion_rates_dream.jl` |
| `dream_kinetic_ionization.h5` | `dream-v26.5` | `kinetic_ionization.py` | `test_kinetic_ionization_dream.jl` |
| `dream_kinetic_operator.h5` | `dream-v26.5` | `kinetic_operator.py` | `test_kinetic_operator_dream.jl` |
| `dream_pmax_boundary.h5` | `dream-v26.5` | `pmax_boundary.py` | `test_pmax_boundary_dream.jl` |
| `dream_radial_transport.h5` | `dream-v26.5` | `radial_transport.py` | `test_radial_transport_dream.jl` |
| `dream_rp_avalanche.h5` | `dream-v26.5` | `rp_avalanche.py` | `test_avalanche_rp_dream.jl` |
| `dream_runaway_rates.h5` | `dream-v26.5` | `runaway_rates.py` | `test_runaway_rates.jl`, `test_runaway_physics.jl` |
| `dream_self_consistent_field.h5` | `dream-v26.5` | `self_consistent_field.py` | `test_self_consistent_dream.jl` |
| `dream_avalanche_kinetic.h5` | `dream-boltzmann-eqterm-93d27e17` | `avalanche_kinetic.py` | `test_avalanche_dream.jl` |
| `dream_example_avalanche.h5` | `dream-boltzmann-eqterm-93d27e17` | `example_avalanche.py` | `test_example_avalanche.jl` |
| `dream_knockon_operator.h5` | `dream-boltzmann-eqterm-93d27e17` | `knockon_operator.py` | `test_knockon_operator.jl` |
| `dream_time_stepping.h5` | `dream-boltzmann-eqterm-93d27e17` | `time_stepping.py` | `test_time_stepping_dream.jl` |
| `dream_time_stepping_adaptive.h5` | `dream-boltzmann-eqterm-93d27e17` | `time_stepping_adaptive.py` | `test_time_stepping_dream.jl` |
| `dream_two_grids.h5` | `dream-boltzmann-eqterm-93d27e17` | `two_grids.py` | `test_two_grids_dream.jl` |
| `gsl_bicubic_reference.txt` | `gsl-2.8` | `reference/gsl/bicubic_reference.c` | `test_adas.jl` |

The case scripts are in `reference/dream/cases/`, and each states its case, its overrides and the test that reads its output.
The tests are in `test/`.

## How the tests use them

`test/Artifacts.toml` declares each set as a lazy artifact, with the URL of its tarball and its hashes.
`reference_file(name)` of `test/helpers.jl` gives the path of a data file in the artifact of its set, which is downloaded at its first use.
The table `REFERENCE_FILES` of `test/helpers.jl` names the set of every file.
For the output of a DREAM example it also names the last DREAM commit that changed the example.

Every DREAM file records `dream_commit`, `dream_branch` and `petsc_version` as attributes.
`test/test_kinetic_operator_dream.jl` checks the commit and the branch of every DREAM file against its set.
With the DREAM checkout of a set at `reference/dream/<set>/DREAM` it also checks that the checkout is at that commit, and that each example is unchanged since its recorded commit.
Without the checkout these checks are skipped.

Test nodes without network need the sets installed beforehand, on a node that has network:

```sh
julia --project=test -e 'using Pkg; Pkg.Artifacts.ensure_all_artifacts_installed("test/Artifacts.toml"; include_lazy=true)'
```

## Regeneration

`reference/dream/README.md` describes the build and the regeneration.
The toolchain is the pixi environment of `reference/dream/pixi.toml`, with PETSc built from source for both DREAM builds.
`build_all.sh` checks out the commit of each set and builds DREAM, and `run_all.sh` runs the cases of the named sets into `out/<set>/`.
`run_all.sh` checks that the DREAM checkout of each set is at its commit before it runs a case.
The HDF5 files hold a `created` time stamp, so a rerun with the same build gives the same data but not the same bytes.
