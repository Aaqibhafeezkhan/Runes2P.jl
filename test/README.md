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
julia --project=test test/runtests.jl --files test_grids.jl,test_layout.jl     # named files
```

- Without arguments every group runs, in the order of `groups.toml`.
- `--results <path>` writes a summary of the run: the commit, local changes, the counts and the time per file.
- On a cluster the full run is the array job, about 20 minutes against 70 to 160 minutes as one process.
  As one process the suite runs only after a change of `runtests.jl` or `helpers.jl`, where files of one process could affect each other.

## Tiers

- `quick`: the unit tests, the tests of each building block of the code (grids, operators, the equation system, solvers, time stepping, output, knock-on kernels).
  They run while one works on a module.
- `full`: the physics validation against independent references (analytic limits, the literature, converged results) and the comparisons with DREAM.
  The full run is both tiers, and it passes before a change is merged.

The quick tier is defined by what a file tests, not by its run time.
Several unit-test files take 10 to 20 minutes alone, most of it compilation.

## Rules for a test file

- A test file may use Runes2P, Test, `Diagonal`, `example_cases.jl` and `TestHelpers`, and nothing that another test file defines.
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
2. Run every file alone, with `groups.toml` set to one group per file, to find definitions that a new file takes from another file.
   This probe also gives the time of each file.
3. Pack the groups again from those times, about 16 minutes each on a node.
4. Run the full suite on the final commit, with a passing report.

## Why the files are slow

Compilation, not the size of the problems.
Run twice in one process, `test_fluid_schur_preconditioner.jl` takes 1 210 s and then 10 s, and `test_populations.jl` 940 s and then 9 s.
Every new combination of terms, closures and solvers is a new type of `EquationSystem`, which compiles `integrate` and the Jacobian again.
The grids of the tests have 120 to 1 540 unknowns.
