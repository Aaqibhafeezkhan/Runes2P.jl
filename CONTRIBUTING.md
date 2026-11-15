# Contributing to Runes2P

These rules hold for every contribution, by people and by coding agents alike.

## Branches and pull requests

The rules below cover the branches, the checks of a pull request, its review, its merge, its size and its entry in the changelog.

- Development happens on the branch `dev`.
  Pull requests go to `dev`, or to an integration branch (below).
- `main` and `dev` change only through pull requests, and only the maintainer merges them.
  `main` is updated from `dev`.
- A suggested branch name is `<type>/<topic>`, with the type one of `fix`, `docs`, `ci`, `test`, `perf` or `chore`.
  `feat/<topic>` names an integration branch only (below).
  A new feature in one pull request takes the type of its main change.
  The topic is a few lowercase words joined by hyphens, for example `fix/pmax-boundary`.
- Open a pull request as a draft while it is in progress.
  The lint (below), the code style check, the format check, the component tests and the docs build run on every push to a pull request.
  The full suite runs once a pull request into `dev` or `main` is marked ready, and again on every push after that.
  A merge into `dev` or `main` runs the lint, the code style check, the format check and the component tests again, on `dev` with the deployment of the docs in place of the docs build.
  The full suite runs on `dev` once a week.
- A pull request that changes no code skips the component tests and the full suite.
  The skipped check of the component tests counts as passed.
  The check of the full suite passes without tests.
- No code means markdown files, files under `docs/src/`, and Julia files outside `test/` whose code stays the same without docstrings and comments.
  `scripts/code_changes.jl` decides.
  Its header lists the exceptions.
- Six checks are required for a merge into `dev` or `main`: Markdown lint, Code style check, Format check, Component tests, Documentation build and Full suite.
  A step into an integration branch (below) requires these checks except the Full suite.
- A pull request into `dev` must be up to date with `dev`.
- A pull request into `dev` needs the approval of its latest push by the maintainer.
  A later push dismisses it.
  Coding agents never approve a pull request, and merge one only into an integration branch (below).
- GitHub deletes the branch of a merged pull request.
- `dev` takes merge commits or squash merges, no rebase merges.
  `main` takes merge commits only.
- A pull request merges with a merge commit by default, which keeps its signed commits and suits stacked pull requests.
- A small fix that needed several commits, with no pull request stacked on it, may suggest a squash merge, with a proposed squash message.
  Its subject has at most 72 characters, including the ` (#n)` that GitHub appends.
  The body of the message has at most 6 lines.
  GitHub pre-fills a `Co-authored-by` line for each co-author of the squashed commits.
  These lines stay below the proposed message.
- A pull request aims at about 500 changed lines in `src/` and `ext/`.
  Tests, reference data and documentation do not count.
  Above that size, the description of the pull request says why the change cannot be split.
  The usual split is a series of pull requests into `dev` that each pass the full suite on their own, or the steps of an integration branch.
  An example of such a series is the data layout, then the operator, then the solver wiring with its tests.
- A larger change may collect its steps on an integration branch `feat/<topic>` off `dev`.
  The steps are branches of the other types (`fix/…`, `perf/…`, `test/…`, `docs/…`, `chore/…`), with pull requests into the integration branch.
- A step of an integration branch runs the checks of `.github/workflows/CI.yml`.
  When it changes code, as `scripts/code_changes.jl` decides, its author also runs `test/test_regression_values.jl` at `-O2` and the test files of the code it changes.
  The step merges with a merge commit once these pass.
  The full suite runs on the integration branch by hand every few steps, and on its pull request into `dev`, which the maintainer reviews and merges.
- A pull request with a change that users see adds an entry under "Unreleased" in [`CHANGELOG.md`](CHANGELOG.md), with a link to the pull request.
  Such changes are those of the public API (a rename, a removed export, a changed keyword or order of arguments), new features, and changed numerical results.
  Refactors, tests, CI and documentation get no entry.
  At the first release, "Unreleased" becomes the section of that version.
- A pull request whose checks pass needs no comment about them, since GitHub shows them.
  After a failed check, the author comments once the fixes are in, saying what failed and what fixed it.

## Running the tests

The commands below run the tests from the root of a checkout.
The paragraphs after them name the data that some tests need.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                                   # every group
julia -O0 --project=test test/runtests.jl component                            # the component tests
julia --project=test test/runtests.jl --files test_grids.jl,test_layout.jl     # named files
```

[`test/README.md`](test/README.md) describes the tiers and groups, the optimisation level, the environment variables and the rules for a new test file.

The tests on OpenADAS data need the cache that `scripts/fetch_adas.py` fills.
Without it they are skipped and reported as broken.
The high-density files of hydrogen are not served by OpenADAS and are copied from the `tools/cache` of a DREAM checkout, given with `--dream-cache`.
A cache without them makes the tests of the cache and of deuterium fail.
The docstring of `scripts/fetch_adas.py` has the details.

## Building the documentation

The commands below build the documentation.
The paragraph after them says what fails the build.

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=docs docs/make.jl
```

The build is strict.
A public name (exported or declared `public`) without a docstring fails it, and so does an unresolved `@ref` link.
A new source file goes into the table `CATEGORIES` of `docs/make.jl`.
The header of `docs/make.jl` gives the command of a live local preview.

## Code and comments

- The code is formatted with JuliaFormatter 2.14.0 and the `.JuliaFormatter.toml` of the root (SciML style).
  The command is `julia -e 'using JuliaFormatter; format(".")'` at the root of a checkout.
  CI checks the format.
- The code explains itself.
  Comments and docstrings state what the code does and why, not how it came to be (section "Writing", below).
- Code, comments and docstrings do not refer to issues, pull requests or commits.
- Longer reasons go on the pages of the documentation, not into comments.
- A value or default that neither DREAM's examples nor a cited reference fixes is marked "a choice of this package" where it is set.
  It gets a row on the docs page [Choices of this package](docs/src/choices.md).
- Performance and accuracy are weighed together.
  A change that affects either one states what it costs and what it gains, from a measurement.
  A change that makes a run significantly slower gives its reason in its pull request and is optimized there as far as it can be.
  An issue takes the rest, with suggestions.
  The docs page [Performance and accuracy](docs/src/performance.md) has the guidelines and the details of the hot paths.
- Every public name (exported or declared `public`) has a docstring of its own.
  A docstring is a string literal, so a LaTeX backslash is written twice (`\\partial`).
  Formulas go into math (```` ```math ```` blocks, ``` ``…`` ``` inline).
  `@ref` links point to documented names only.
- The docstring of a public name gives the unit of every argument, keyword, field and result that is a physical quantity.
  A dimensionless quantity needs no unit.
  The units are SI, except temperatures and particle energies in eV and momenta in units of mₑc, and the distribution is in m⁻³.
  A generic term whose quantity takes the unit of its row or of its unknown says so instead of naming a unit.

## Names

The rules below cover the names, keywords and fields that users see, the short forms and the order of arguments.

- A public name (exported or declared `public`) has one meaning.
  A concept has one spelling in the names, keywords and fields that users see.
  Settled spellings include `coulomb_logarithm`, `kinetic_ionization`, `ionization_cross_section_form`, `free_electron_density` and `temperature`.
- A function that builds a term or another type has a lowercase name, such as `ohmic_heating_term`.
  A name in CamelCase is a type.
- What users call is exported, also the functions of the term interface that a new term adds methods to and that callers use, such as `rhs!` and `dependencies`.
  A function that only a new term or operator implements or calls is declared `public`, such as `validate`, `add_local_jacobian!` and `lags_in_ode_form`.
  The rest stays internal.
- Public names, the keywords of their methods and the fields of their types are ASCII, which `test/test_names.jl` checks.
  A local variable may use the Greek symbol of a cited formula.
  A name that becomes exported, public, a keyword or a field users read is renamed to ASCII in every use.
- The same kind of unknown has the same keyword in every constructor.
  A switch of type `Bool` among the keywords users see never shares a name with an unknown.
- A key of a settings record changes only with the output format, so a keyword that is such a key keeps its name even where the rule above would rename it.
  A field may be named differently from its key.
- Established short forms stay, such as `dt`, `p_max`, `p_cutoff` and `xi`, and `p` for a momentum in argument names.
- A function of the package that takes a density and a temperature takes the density first, such as `thermal_coulomb_logarithm(free_electron_density, temperature)`.
  The exception is a positional constructor, which follows the order of the fields of its type.
- When a choice is in doubt, the clearer name wins, even if it is longer.

## Code style

These rules hold for `src/` and `ext/`, where the physics and the numerics are read and checked.
Tests, examples and scripts may be denser.
New and changed code follows these rules.
The formatter sets the layout only, so the reviews check these rules, helped by the script of the last item.

- Control flow is written with `if`.
  `cond && action` and `cond || action` are not used for a side effect, an assignment or a `throw`.
  An early return, such as `cond && return x`, is allowed anywhere, and so are `cond && continue` and `cond && break`.
  The one exception is a check of an argument at the start of a function, in either form, such as `length(x) == n || throw(DimensionMismatch("..."))`.
  A check is at the start of a function when only other checks and the assignments it reads come before it.
- A check rewritten from `cond || throw(...)` into an `if` keeps its condition.
  An ordered or compound condition is negated as `if !(cond)`, since a simplified comparison can differ for `NaN`.
  A negated equality is written `a != b` or `a !== b`.
- An `if` that gives one variable its value is written `x = if ... else ... end`, unless a conditional expression `a ? b : c` of one line fits.
  An `if` that gives the value of a function has a `return` in each branch.
- A line holds one statement, so no `;` joins two statements.
  No assignment sits inside an expression.
- A conditional expression `a ? b : c` chooses between two simple values, short enough for one line, and is never nested.
- An intermediate quantity gets a name, the name of the physical quantity where there is one, instead of a long chain of calls or broadcasts.
  A line that transcribes a cited formula may stay one expression, so that it can be compared with its source.
- A comprehension or generator has one level.
  A nested one, or a `map` or `foreach` whose function takes more than one line, is written as a loop.
- Functions are not composed with `∘` as an operator in code.
  Calls are not chained with `|>`.
- `python3 scripts/check_code_style.py <file> ...` finds the simplest breaks of these rules.
  It reports an assignment or a call ending in `!` after `&&` or `||`, and a `throw` or `error` after them outside the checks at the start of a function.
  It also reports a nested or two-line `a ? b : c`, a `;` between two statements, a chain of `|>` and `∘` as an operator.
  CI runs it on the lines a pull request adds, and on every line of `src/` and `ext/` after a push to `dev` or `main`.
  The reviews check the rest.

## Physics values and their sources

- Every physics-relevant value in the code is cited in the docstring next to it.
  The citation is the full literature reference, or DREAM's file for a value ported from DREAM.
- Agreement with DREAM shows that both codes use the same method, not that the method is right.
  So each new piece of physics gets a check against a reference independent of DREAM: an analytic limit, the literature or a manufactured solution.
- Where a default of Runes2P differs from DREAM, the behaviour of DREAM stays available as an option.
  The comparison tests run with that option.
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

## Writing

These rules hold for the pages of `docs/src/`, the README files, docstrings and comments where they hold prose, the descriptions of pull requests, and the text of issues and of their comments.
They hold for this file and for new entries of `CHANGELOG.md` too.
New and changed text follows these rules.
The rules serve the reader.
Where following one would make a passage harder to read, the passage wins.

- A sentence makes one statement.
  A second clause may join it where it gives the reason, the consequence or the exception of the first, or where it shares its subject.
  In prose, outside code, math and citations, no semicolon joins two clauses.
  In a table, a semicolon may separate alternatives in a cell.
- A value is a number, with its unit where it has one, measured or set.
  Names, equation numbers, citations and numbers inside a formula are not values.
  A condition, such as "at 120 eV", is not a value either.
- A sentence holds at most two values of a kind.
  Three or more values of a kind go into a table.
  In a comment or a docstring, where no table fits, a list takes the place of the table.
- A list item is one or more full sentences, each ending with a period.
  An item may be a label, a colon and a phrase, as in a list of definitions, or a label and a colon above a nested list.
  A list of bare names, files or links is a list of names, and its items take no periods.
  A sentence is not cut into list items.
  An enumeration of other things than values and names becomes a list only when each item is a sentence.
- A paragraph holds one topic in at most five sentences.
- A page starts with one or two sentences on what it holds.
  So does a section of more than about twenty lines of text and tables.
- A sentence names its subject instead of reaching it through a chain of three or more "of".
- A sentence that opens with "it", "this" or "that" refers to the sentence before it.
- A name, a symbol or an abbreviation is defined where it is first used on a page, in the same sentence or the next, or it is linked.
  An API name linked with `@ref` counts as linked.
  A reference forward says "(below)".
  The names of codes and common abbreviations, such as DREAM, CI and API, need no definition.
- A difference names its kind, relative or absolute, and what it is divided by, once per section.
  An example is "|Runes2P − DREAM| divided by DREAM's largest value at the radius".
  A signed deviation is written (value − reference) / reference.
- A table has one convention for its differences, or a column "Reference".
- A table cell holds what its column header says, as a value, a name or a short phrase.
  A column may hold reasons, sources or tests when its header says so and its cells stay short.
  What is too long for a short phrase goes into a note below the table.
  When prose becomes a table, each cell is checked against its header.
- A count says how it counts when that is not obvious.
- A result, such as a measured difference or a timing, has one owning page of `docs/src/`.
  Other pages link to that page.
  Where the reader needs the result on the spot, it is stated once next to the link.
  The comments of the tests may repeat it.
- The pages of `docs/src/`, the README files, docstrings and comments describe the code as it is.
  How the code came to be stays out of them, or goes into [`CHANGELOG.md`](CHANGELOG.md) when the change is one that users see.
  The section "History of the repository" of `README.md` is the one exception.
  It stays short and takes no new developments, which the pull requests, the commits and `CHANGELOG.md` record.
- A page explains what its reader needs and does not reproduce the comments of the tests.
- GitHub renders headings and tables in pull requests, issues and comments.
  The description of a pull request has a heading for each part, as the template `.github/PULL_REQUEST_TEMPLATE.md` gives.
  A part is not one long paragraph.
  GitHub renders a newline as a line break, so a paragraph of such text is one line.

## Markdown

These rules hold for the markdown files of the repository, except those of `.github/` and `THIRD_PARTY_NOTICES.md`.
The files of `.github/` keep one paragraph per line, as the text of pull requests and issues does.

- A line holds one sentence.
- No line break falls inside a sentence.
- A sentence has at most 200 characters.
  A longer one is split into shorter sentences or into a list of sentences.
- No second sentence stands inside parentheses.
- `python3 scripts/lint_markdown.py <file> ...` checks these rules.
  It skips fenced code blocks, tables, headings, quotes, comments and horizontal rules.
