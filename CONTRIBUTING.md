# Contributing to Runes2P

These rules hold for every contribution, by people and by coding agents alike.

## Branches and pull requests

- Development happens on the branch `dev`.
  Pull requests go to `dev`.
- `main` and `dev` change only through pull requests, and only the maintainer merges them.
  `main` is updated from `dev`.
- Open a pull request as a draft while it is in progress.
  The lint, the format check, the component tests and the docs build run on every push; the full suite runs once the pull request is marked ready, and again on every push after that.
- Five checks are required for a merge: Markdown lint, Format check, Component tests, Documentation build and Full suite.
- A pull request into `dev` must be up to date with `dev`.
- `dev` takes merge commits or squash merges, no rebase merges.
  `main` takes merge commits only.
- A pull request whose commits are mostly review rounds may suggest a squash merge, with a proposed squash message.
  Its subject has at most 72 characters, including the ` (#n)` that GitHub appends; its body has at most 6 lines, or about 12 above the size target below.
  GitHub pre-fills a `Co-authored-by` line for each co-author of the squashed commits; they stay below the proposed message.
- A pull request aims at about 500 changed lines in `src/` and `ext/`; tests, reference data and documentation do not count.
  Above that, its description says why the change cannot be split.
  The usual split is a series of pull requests that each pass the full suite on their own, for example the data layout, then the operator, then the solver wiring with its tests.
- A pull request whose checks pass needs no comment about them, since GitHub shows them.
  After a failed check, the author comments once the fixes are in: what failed and what fixed it.

## Running the tests

From the root of a checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                                   # every group
julia -O0 --project=test test/runtests.jl component                            # the component tests
julia --project=test test/runtests.jl --files test_grids.jl,test_layout.jl     # named files
```

[`test/README.md`](test/README.md) describes the tiers and groups, the optimisation level, the environment variables and the rules for a new test file.
The tests on OpenADAS data need the cache that `scripts/fetch_adas.py` fills.
Without it they are skipped and reported as broken.
The high-density files of hydrogen are not served by OpenADAS and are copied from the `tools/cache` of a DREAM checkout, given with `--dream-cache`.
A cache without them makes the tests of the cache and of deuterium fail; the docstring of `scripts/fetch_adas.py` has the details.

## Building the documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=docs docs/make.jl
```

The build is strict: an exported name without a docstring or an unresolved `@ref` link fails it.
A new source file goes into the table `CATEGORIES` of `docs/make.jl`.
The header of `docs/make.jl` gives the command of a live local preview.

## Code and comments

- The code is formatted with JuliaFormatter 2.14.0 and the `.JuliaFormatter.toml` of the root (SciML style): `julia -e 'using JuliaFormatter; format(".")'` at the root of a checkout, and CI checks it.
- The code explains itself.
  Comments and docstrings state what the code does and why, not how it came to be.
- No references to issues, pull requests or commits in the code.
- Longer reasons go on the pages of the documentation, not into comments.
- A value or default that neither DREAM's examples nor a cited reference fixes is marked "a choice of this package" where it is set.
  It gets a row on the docs page [Choices of this package](docs/src/choices.md).
- Every exported name has a docstring of its own.
  A docstring is a string literal: a LaTeX backslash is written twice (`\\partial`), formulas go into math (```` ```math ```` blocks, ``` ``…`` ``` inline), and `@ref` links point to documented names only.

## Physics values and their sources

- Every physics-relevant value in the code is cited in the docstring next to it.
  The citation is the full literature reference, or DREAM's file for a value ported from DREAM.
- Agreement with DREAM shows that both codes use the same method, not that the method is right.
  So each new piece of physics gets a check against a reference independent of DREAM: an analytic limit, the literature or a manufactured solution.
- Where a default of Runes2P differs from DREAM, the behaviour of DREAM stays available as an option, and the comparison tests run with it.
  A new departure gets such an option where it can.
  The departures, with the options that exist, are listed on the docs page [Comparison with DREAM](docs/src/dream-comparison.md#departures-from-dream).

## Reference data

- The reference data of the comparison tests come from the case scripts of `reference/`.
  [`reference/dream/README.md`](reference/dream/README.md) describes the build of DREAM and the regeneration.
- Each set is published as a tarball and read by the tests as a lazy artifact of `test/Artifacts.toml`.
  The data themselves are not in the repository.
- The sets and their tests are on the docs page [Reference data](docs/src/reference-data.md).

## Data

- Ask in an issue or a pull request before adding new data.
- By default new data goes into a TOML file of `data/`, with the header that [`data/README.md`](data/README.md) describes.
- Data too large for the repository is published as an artifact instead, as the reference sets are.

## Markdown

- One sentence per line, with no line break inside a sentence.
- A sentence has at most 200 characters.
  A longer one is split or turned into a list.
- `python3 scripts/lint_markdown.py <file> ...` checks these rules.
