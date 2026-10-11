# Tests of Runes2P

The test suite, its layout and the rules for a new test file.
On a cluster the suite runs as an array job, one task per group.

## Layout

| Path | Content |
|---|---|
| `runtests.jl` | the runner: every test file in a module of its own, selection by group, tier or file, a summary file for the array job |
| `groups.toml` | the groups of test files, each with its tier, one array task per group |
| `helpers.jl` | the module `TestHelpers`: definitions that several test files use |
| `example_cases.jl` | setups built from DREAM's examples, shared by the tests |
| `example_disruption.jl` | the system of the disruption case from DREAM's state, used by `test_disruption_dream.jl` |
| `test_*.jl` | the test files, 66 on 2026-10-01 |
| `*_helpers.jl` | a file that one test file includes, not a test file of its own |
| `data/` | DREAM's output for the comparison tests, written by `reference/dream/cases/` |
| `drafts/` | test files for a step that is not built yet, outside the suite until then |
| `Project.toml`, `Manifest.toml` | the test environment |

## Running

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                                   # every group
julia --project=. -e 'using Pkg; Pkg.test(test_args=["--tier", "quick"])'      # the unit tests
julia --project=. -e 'using Pkg; Pkg.test(test_args=["quick-3", "full-1"])'    # named groups
julia --project=test test/runtests.jl component                                # the component tests
julia -O0 --project=test test/runtests.jl component                            # the same, compiled faster
julia --project=test test/runtests.jl --files test_grids.jl,test_layout.jl     # named files
```

- Without arguments every group runs, in the order of `groups.toml`.
- `--results <path>` writes a summary of the run: the commit, local changes, the counts and the time per file.
- On a cluster the full run is the array job, about 20 minutes against 70 to 160 minutes as one process.
  On a workstation the groups run side by side in separate processes.
  As one process the suite runs only after a change of `runtests.jl` or `helpers.jl`, where files of one process could affect each other.

## Tiers

- The component tests, the group `component` of `groups.toml`: one building block in isolation.
  They cover the grids, the layout, the knock-on kernels, the moments, the operator on a grid, the terms of the fluid equations on a radial grid and the step controllers on small systems.
  They are the smoke tier: one process, run before a commit (decision of the author).
  A file belongs to the group by what it tests, and `packed = false` keeps the group out of the packing.
- `quick`: the unit tests.
  These are the component tests and the system tests.
  A system test checks an assembled kinetic system or the machinery of a run: the equation system, the plasma source, the populations, checkpoints, the preconditioner, the records.
  They run while one works on a module.
- `full`: the physics validation against independent references (analytic limits, the literature, converged results) and the comparisons with DREAM.
  The full run is both tiers, and it passes before a change is merged.
- A change in progress runs only the test files it adds or changes, and `test_settings.jl`, as one focused job.
  The full run follows once the whole change is done.

Every tier is defined by what a file tests, not by its run time.
The times follow from it: assembling a kinetic system is what compiles.

## Optimisation level

`julia -O0` compiles the tests several times faster and runs them slower.
On 2026-10-04 the whole suite took 7 minutes as an array at `-O0`, and the component tests 4.6 minutes as one process.

- A run of a few files and the component tests run at `-O0`.
- The full run uses the default level, the level of a production run.
- Two kinds of assertion hold for optimised code alone.
  Below `-O2` an assertion on allocations is skipped and reported as broken (`skip = !OPTIMIZED` in the test files).
  The stored values of `test_regression_values.jl` are compared to 1e-12 in place of bit for bit, since the sums run in another order.
- No other test of the suite failed at `-O0`.

## Rules for a test file

- A test file may use Runes2P, Test, `Diagonal`, `example_cases.jl`, `example_disruption.jl` and `TestHelpers`, and nothing that another test file defines.
  The runner includes every file into a module of its own, so a missing definition fails at once.
- A file loads the packages it uses itself (`using HDF5`, `using Random: Xoshiro`).
  A package that only the tests use goes into `test/Project.toml`.
- A helper that a second file needs moves into `helpers.jl`, with a comment naming the file it came from, and its name into the export list.
- A new test file goes into a group of its tier in `groups.toml`.
  A file that runs longer than about ten minutes alone gets a group of its own.
  `runtests.jl` checks that every `test_*.jl` is in exactly one group.
- A test of a record (a settings record, a type name in an output file) does not depend on the module the test runs in.
  `ext/Runes2PInterpolationsExt.jl` recorded type names relative to `Main` until the module per file showed it.

## After a merge of `dev` into a branch that changes the tests

Each round of new test files from `dev` needs the same steps:

1. Resolve `groups.toml` and the test environment: new files into groups, new packages into `test/Project.toml`, and `test/Manifest.toml` regenerated offline from the root manifest.
2. Put each new file into a group of its tier, or into the group `component` when it tests one building block in isolation.
   The runner includes every file into a module of its own, so a definition taken from another file fails at once.
3. Run the full suite on the final commit, with a passing report.
4. Pack the groups again from the times of that run when a group has grown beyond about 15 minutes.
   The next run then uses the packed groups.

## Why the files are slow

Compilation, not the size of the problems.
Run twice in one process, `test_fluid_schur_preconditioner.jl` took 1 210 s and then 10 s, and `test_populations.jl` 940 s and then 9 s (before the fixed tags of the Jacobians, below).
The grids of the tests have 120 to 1 540 unknowns.

The measurement, on `test_field_unknown.jl` cold in a process of its own:

- Of 1 337 s, 1 328 s were compilation, and 1 096 s of it ten instances of the automatic Jacobian of a system, about 120 s each.
- At `-O0` the file took 338 s, so about three quarters of the time was the optimisation by LLVM.
- ForwardDiff's default tag gave every system dual numbers of a type of its own, and the code of every term was compiled again for each system.
- Since then the Jacobians use the fixed tags `JacobianTag` and `OperatorJacobianTag`, and the file takes 441 s.
  The summed time of the files of the suite fell from 327 to 242 minutes.

`RUNES2P_JACOBIAN_CHUNK_SIZE=8` is recommended for the test suite.
ForwardDiff chooses the chunk size of the dual numbers as the smaller of 12 and the number of colors of a system.
Systems with 9, 10 and 11 colors then have three types of dual numbers, and the code of the terms is compiled for each, about 110 s per type.
With the limit every kinetic system has the chunk size 8, and that code is compiled once.
The values of the Jacobian do not depend on the chunk size, and a run without the variable keeps ForwardDiff's choice.
A local run of the tests gets the same with `env RUNES2P_JACOBIAN_CHUNK_SIZE=8 julia ...`.
An empty value switches the limit off: `env RUNES2P_JACOBIAN_CHUNK_SIZE= julia ...`.

What is still compiled per case:

- `integrate`, `rhs!`, the Jacobian operator and keyword calls, once per type of `EquationSystem`;
- a Jacobian that a test file takes itself with `ForwardDiff.jacobian` of a closure, once per closure.
