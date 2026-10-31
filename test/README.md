# Tests of Runes2P

The test suite: its layout, its tiers and groups, how to run it, and the rules for a new test file.

## Layout

| Path | Content |
|---|---|
| `runtests.jl` | the runner: every test file in a module of its own, selection by group, tier or file, an optional summary file |
| `groups.toml` | the groups of test files, each with its tier |
| `helpers.jl` | the module `TestHelpers`: definitions that several test files use |
| `example_cases.jl` | setups built from DREAM's examples, shared by the tests |
| `example_disruption.jl` | the system of DREAM's disruption case started from DREAM's state, and the budgets of its closures, used by `test_disruption_dream.jl` |
| `test_*.jl` | the test files |
| `*_helpers.jl` | a file that one test file includes, not a test file of its own |
| `Artifacts.toml` | the reference data of the comparison tests, one lazy artifact per set: DREAM's output written by the cases of `reference/dream/cases/`, and GSL's values of `reference/gsl/` |
| `Project.toml`, `Manifest.toml` | the test environment |

## Tiers and groups

Every test file is in exactly one group of `groups.toml`, and every group has a tier.
`runtests.jl` checks both before it runs a test.

- `quick`: the unit tests, which are the component tests and the system tests.
  - The component tests are the group `component` of the tier `quick`: one building block in isolation.
    They cover the grids, the layout, the knock-on kernels, the moments, the operator on a grid, the terms of the fluid equations on a radial grid and the step controllers on small systems.
    They are the smoke tests, run as one process.
  - A system test checks an assembled kinetic system or the machinery of a run: the equation system, the plasma source, the populations, checkpoints, the preconditioner, the records.
- `full`: the physics validation against independent references (analytic limits, the literature, converged results) and the comparisons with DREAM.
  A full run is both tiers.

A tier is defined by what a file tests, not by its run time.
The run times follow from it: assembling a kinetic system is what compiles.

The groups other than `component` balance the run time by the measured run times of their files.
Most take 4 to 9 minutes, and the comment above a group gives its measured run time.
A file belongs to `component` by what it tests, and `packed = false` keeps that group out of the balance.

## Running

From the root of the repository:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                                   # every group
julia --project=. -e 'using Pkg; Pkg.test(test_args=["--tier", "quick"])'      # the unit tests
julia --project=. -e 'using Pkg; Pkg.test(test_args=["quick-3", "full-1"])'    # named groups
julia --project=test test/runtests.jl component                                # the component tests
julia -O0 --project=test test/runtests.jl component                            # the same, compiled faster
julia --project=test test/runtests.jl --files test_grids.jl,test_layout.jl     # named files
```

- Without arguments every group runs, in the order of `groups.toml`.
- `--index <n>` runs the n-th group of `groups.toml`.
- `--results <path>` writes a TOML summary of the run: the commit, local changes, the counts and the time per file.
- The groups are independent, so they can run side by side in separate processes.

The comparison tests read the reference sets of `Artifacts.toml`, which are downloaded at their first use (about 5.5 MB).
Where the test nodes have no network, install them beforehand on a node that has:

```sh
julia --project=test -e 'using Pkg; Pkg.Artifacts.ensure_all_artifacts_installed("test/Artifacts.toml"; include_lazy=true)'
```

The checks of the DREAM commit, of the history of DREAM's examples and of DREAM's table of ADAS files need the DREAM checkout of each set at `reference/dream/<set>/DREAM` (`reference/dream/README.md`).
Without it they are skipped.

## Optimisation level

`julia -O0` compiles the tests several times faster and runs them slower.
The component tests and a run of a few files are faster at `-O0`.

Two kinds of assertion hold for optimised code alone:

- Below `-O2` an assertion on allocations is skipped and reported as broken (`skip = !OPTIMIZED` in the test files).
- Below `-O2` the stored values of `test_regression_values.jl` are compared to 1e-12 in place of bit for bit, since the sums run in another order.

At `-O2` the pure-Julia entries of `test_regression_values.jl` are compared bit for bit, or to the tolerance of `RUNES2P_REGRESSION_RTOL` where it is set.
Such an entry that differs is logged with its relative deviation, also when it passes.
The `:blas` entries are compared to 1e-13 at every level.

## Environment variables

| Variable | Default | Use |
|---|---|---|
| `RUNES2P_ADAS_CACHE` | `~/.cache/Runes2P/adas` | the directory of the OpenADAS ADF11 files |
| `RUNES2P_CHIANTI_DIR` | `~/chianti` | the CHIANTI database, for the regeneration test of the table of CHIANTI's cross sections |
| `RUNES2P_JACOBIAN_CHUNK_SIZE` | none | the upper limit of the chunk size of the dual numbers of the sparse Jacobians |
| `RUNES2P_REGRESSION_RTOL` | none, bit for bit | the relative tolerance of the pure-Julia regression values at `-O2`; the Full suite of CI sets `1e-13` |

- `scripts/fetch_adas.py` fills the ADAS cache.
  No ADAS data is in the repository.
  Without the cache the tests on OpenADAS data are skipped and reported as broken.
- Without the CHIANTI files the regeneration of `data/chianti_ionization.toml` by `scripts/generate_chianti_ionization.jl` is skipped.
  The table itself is in the repository and is tested without them.
- `RUNES2P_JACOBIAN_CHUNK_SIZE=8` is recommended for the test suite: `env RUNES2P_JACOBIAN_CHUNK_SIZE=8 julia ...`.

## Why the tests compile long

Most of the run time of a test file is compilation, not the size of its problems.
The grids of the tests have 120 to 1 540 unknowns.

- The automatic Jacobians use the fixed tags `JacobianTag` and `OperatorJacobianTag`.
  The code of a term is then compiled once for all systems, not once per system.
- ForwardDiff chooses the chunk size of the dual numbers as the smaller of 12 and the number of colors of a system.
  Systems with 9, 10 and 11 colors then have three types of dual numbers, and the code of the terms is compiled for each.
- With `RUNES2P_JACOBIAN_CHUNK_SIZE=8` every system with 8 colors or more has the chunk size 8, and that code is compiled once.
  The values of the Jacobian do not depend on the chunk size.
  A run without the variable keeps ForwardDiff's choice, which needs the fewest evaluations.

What is still compiled per case:

- `integrate`, `rhs!`, the Jacobian operator and keyword calls, once per type of `EquationSystem`;
- a Jacobian that a test file takes itself with `ForwardDiff.jacobian` of a closure, once per closure.

## Rules for a test file

- A test file may use Runes2P, Test, `Diagonal`, `example_cases.jl` and `TestHelpers`, and nothing that another test file defines.
  The runner includes every file into a module of its own, so a missing definition fails at once.
- A file that needs `example_disruption.jl` includes it itself.
- A file loads the packages it uses itself (`using HDF5`, `using Random: Xoshiro`).
  A package that only the tests use goes into `test/Project.toml`.
- A helper that a second file needs moves into `helpers.jl`, with a comment naming the file it came from, and its name into the export list.
- A new test file goes into a group of its tier in `groups.toml`, or into the group `component` when it tests one building block in isolation.
  A file that runs longer than about ten minutes alone gets a group of its own.
- A test that reads a docstring (`@doc name`) adds the file of that docstring to `DOCSTRINGS_READ_BY_TESTS` of `scripts/code_changes.jl`, so that a change of the docstring runs the tests.
- A test of a record (a settings record, a type name in an output file) does not depend on the module the test runs in.
- Each new piece of physics gets a check against a reference independent of DREAM: an analytic limit, the literature, a manufactured solution or a converged result.
  An agreement with DREAM shows that both codes use the same method, not that the method is right.
- A comparison with DREAM reads DREAM's output, written by a case script of `reference/dream/cases/`, with `reference_file` of `helpers.jl`, from the artifact of its set.
  The set's DREAM branch and commit are in the table of `helpers.jl`.
