# Performance and accuracy

```@meta
CurrentModule = Runes2P
```

This page has two parts.
The first gives the guidelines that hold for every change: how accuracy and cost are weighed, and how a cost is measured.
The second holds the details of the hot paths and the linear solvers, with the measurements of the coupled disruption case.

## Guidelines

### Accuracy first, cost with it

- A result is trusted because a test holds it: against DREAM with the options that reproduce it, to the tolerance of its solver or closer.
  Each piece of physics also has a test against a reference independent of DREAM.
  A faster code path keeps these tests, and the regression values at `-O2` bit for bit where it can.
- A change that moves results at round-off says so, with what was compared and on which case.
  A sum taken in another order changes the last digits, and a check of single terms does not see an entry that several terms share.
- A numerical choice is recorded with its reason, and with its cost where it trades accuracy for cost ([Choices of this package](choices.md)).
  A departure from DREAM is judged by its accuracy for its cost.
  DREAM's behaviour stays an option where it can ([Departures from DREAM](dream-departures.md)).
- A change that makes a run significantly slower gives its reason in the pull request and is optimized there as far as it can be.
  Where that is too large for the pull request, an issue takes the rest, with suggestions for the improvement.

### Measure, then change

- The time of a step is in the [`SolverStatistics`](@ref) of a run: `wall_time`, and within it `jacobian_time`, `residual_time` and `linear_solve_time`.
  `factorization_time` is the part of `linear_solve_time` spent in the factorizations.
  They have one entry per attempted step.
  Their sums over the steps, divided by the sum of `newton_iterations`, give the cost of a Newton iteration, the unit of this page.
- A measurement comes after a warm-up of a few steps, so that it leaves out the compilation, and at `-O2`.
- Every number names its threads: those of Julia, of BLAS (`OPENBLAS_NUM_THREADS` or `BLAS.set_num_threads`) and of the solver (`threads` of [`MKLPardiso`](@ref)).
- Two versions are compared in one sitting on an idle machine, alternating: base, head, head, base.
  Repeated runs of one version differ by a few milliseconds per iteration on the case below, and a gain below that spread is not a gain.
- Counts do not vary between machines, and they show a change of the cost before a timer does.
  They are the stored nonzeros (`nnz` of [`jacobian_sparsity`](@ref)), the colors of the automatic Jacobian ([`jacobian_colors`](@ref)) and the Newton iterations per step.
  The bytes of a second call (`@allocated`) are such a count too.

### Habits of the code

- A term that knows the entries of its block assembles them itself ([`jacobian_assembly`](@ref) `:direct`), above all a row that reads many cells.
  The colored forward-mode differentiation serves the other terms, the stencil of the kinetic operator among them.
  A test compares a direct block with the automatic differentiation of its `rhs!` or with a central-difference Jacobian.
- The pattern of the stored Jacobian holds the entries that the terms write and no others.
  Every declared entry costs fill in the factorization, also when it holds a zero.
- An inner loop has concrete types.
  A loop over values whose type is known only at run time allocates per entry, so it sits behind a function barrier.
- A buffer is kept between calls, not allocated per call.
  A residual or a Jacobian evaluated a second time at one state allocates little, and tests bound it at `-O2`.
  With more than one Julia thread the threaded loops allocate their tasks, so the tests that expect no allocation there are skipped above one thread.
- A cache is keyed on every input that its values read, such as the plasma values per radius of a [`PlasmaKey`](@ref).
  Its test changes an input alone and counts one rebuild, and none at an unchanged state.
- The terms are assembled one after the other, never two at once.
  A loop that runs on threads splits the radii of one term, and each radius writes only its own entries.
  The caches it reads are filled before the loop, so the caches and buffers of the terms need no copy per task.

## The disruption case

The measurements of this page are of DREAM's disruption case with a runaway grid ([The disruption case](disruption-case.md)).
It is scenario 1 of DREAM's `examples/Disruptions/generate.py`, fully kinetic in a cylinder, with the parameters of `ASDEXU.py` of that example.

| Quantity | Value |
|---|---|
| minor radius, wall radius | 0.5 m, 0.55 m |
| magnetic field | 2.5 T, uniform |
| plasma current | 800 kA |
| electron density | 2.6e19 m⁻³, uniform |
| central temperature | 5.8 keV, a Gaussian profile |
| injected material | deuterium and argon, each at the density of the background, at 1 eV |
| heat diffusivity | 4000 m²/s |
| hot-tail grid | 140 momentum cells to `p_max = 0.8`, 15 pitch cells |
| runaway grid | the default momentum grid of `--runaway-grid`, 32 cells from 0.8 to 20, and 15 pitch cells (`--re-nxi 15 --re-nxi-sep 10` of the case script) |
| size of the measurements | 15 radii, 39303 unknowns, 2350635 stored nonzeros |

The momenta are in units of ``m_e c``.
The case has two phases, the ionization of the injected material and the thermal and current quench.
The timed window is 10 steps from step 170 of phase 2, at full size ([Choices of this package](choices.md#The-window-of-the-test)).
Runes2P starts it from DREAM's state with DREAM's settings (`compat = :dream` of the builder of the case).
A run of 2 steps warms up first, and the timed run starts with a new solver.
The tests run the same window at a reduced size, with 2 radii, coarser grids and 835 unknowns (`test_disruption_runaway_dream.jl`).

The table gives the set-up of the two codes in the measurements.

| Setting | Runes2P | DREAM |
|---|---|---|
| version | this package | v26.5 |
| threads outside the linear solver | one Julia thread, one BLAS thread | one |
| linear solver | [`SparseLU`](@ref) or [`MKLPardiso`](@ref) | MKL PARDISO |
| threads of PARDISO | 1 or 8 | 1, as v26.5 pins it, or 8 in a build without that pin |
| refinement steps of PARDISO | none, the default of [`MKLPardiso`](@ref) | allows up to ten per solve |
| Julia optimisation level | `-O2` | - |
| timers | per attempted step of the window | sums over the quench |
| figures | per Newton iteration of the window | means over the quench, per Newton iteration |
| MKL | 2025.2, through Pardiso.jl 1.1.2 | - |
| machine | one idle machine with 8 cores, one run at a time | the same |

- MKL PARDISO is not DREAM's default solver, which is PETSc's LU.
- The case at this size is not among the reference data: the tests hold the window at the reduced size.
- The timings and agreements of this page are measurements, and no test repeats them.

## A Newton iteration

In milliseconds per Newton iteration, without radial transport of the distributions.
DREAM's rows are means over its whole quench, and Runes2P's rows cover the window.
The column "Against DREAM" is the time of an iteration divided by DREAM's at the same number of PARDISO threads, so a value below 1 means Runes2P is faster.

| Code and solver | Jacobian | Residual | Factorization | Solve | Iteration | Newton per step | Against DREAM |
|---|---|---|---|---|---|---|---|
| DREAM, PARDISO at 1 thread | 93.8 | 3.4 | 88.2, with the solve | | 186.0 | 4.51 (quench mean), 5 in the window | 1.00 |
| DREAM, PARDISO at 8 threads | 97.4 | 3.6 | 45.4, with the solve | | 147.0 | 4.51 (quench mean), 5 in the window | 1.00 |
| Runes2P, [`SparseLU`](@ref) | 35.1 | 6.7 | 106.7 | 5.4 | 158.7 | 4.50 | 0.85 |
| Runes2P, [`MKLPardiso`](@ref) at 1 thread | 35.3 | 9.0 | 70.6 | 7.0 | 126.8 | 4.50 | 0.68 |
| Runes2P, [`MKLPardiso`](@ref) at 8 threads | 35.2 | 8.9 | 20.5 | 2.6 | 72.0 | 4.50 | 0.49 |

- [`SparseLU`](@ref) is compared with DREAM at one thread.
- DREAM's Jacobian holds the rebuild of its coefficients, about 31 ms.
  Runes2P rebuilds them inside its residual and its Jacobian.
- About 5 ms of a Runes2P iteration are outside the four timers.
- The factorization of [`MKLPardiso`](@ref) holds its one analysis, in the first timed step.
  In the later steps the factorization takes about 66 ms per iteration at one thread.
- The run at eight threads holds one slow residual in one step.
  Without it the iteration takes about 70 ms.
- At this size, the largest relative difference between the states of the two codes over the steps of the window is 4.1e-9, in the hot-electron density.
  It is relative to DREAM's largest hot-electron density over the radii at the same step, the measure of `phase_differences` of `test/example_disruption.jl`.

The factorization and the solve take this share of an iteration:

| Solver | Factorization and solve, share of an iteration |
|---|---|
| [`SparseLU`](@ref) | 71 % |
| [`MKLPardiso`](@ref) at 1 thread | 61 % |
| [`MKLPardiso`](@ref) at 8 threads | 32 % |

At one thread the linear solver is therefore the first place to look in a slow run, and the stored pattern decides its cost.

## The hot paths

### The stored pattern

The kinetic operator declares the stencil of the coefficients it holds ([`stencil_pattern`](@ref)).
An operator without radial transport stores no radial entries, and one without cross diffusion no corners, the corner entries of the stencil that the cross diffusion fills.
A coefficient outside the declared stencil is refused where the operator enters a system and when its coefficients are rebuilt.
[`check_stencil_pattern`](@ref) checks in a test that the entries of the matrix lie inside the declared pattern.
With the radial entries and corners declared for this case, which fills none of them, UMFPACK would take about seven times as long to factorize.

### Colored differentiation and direct rows

The terms with automatic assembly share one coloring, and each color is one directional derivative of their residuals ([`jacobian_colors`](@ref)).
An evaluation in dual numbers carries up to 12 colors.
A row that reads many cells of a radius would take a color per cell.
Such rows are assembled directly, among them the moments of the density, the current and the energy, and the flux through the interface of two momentum grids.
The case has 17 colors, which take two evaluations.
With the interface rows colored it would have 33 colors and take three evaluations.
A moment keeps the positions and weights of its row for the pattern of the matrix, so a later assembly only adds values.

### Plasma columns

The derivatives with respect to the temperature, the density and the charge states are columns of their own ([Plasma columns of the Jacobian](model/unknowns.md#Plasma-columns-of-the-Jacobian)).
The coefficients in dual numbers are built once per state and shared by the term and the terms that read its coefficients.
The loops that insert their partials run behind function barriers, with one search of the pattern per radius and column.

### Coefficients

The coefficients of a [`FokkerPlanckTerm`](@ref) are rebuilt only when the plasma values of some radius change ([Caches and rebuilds](model/unknowns.md#Caches-and-rebuilds)).
The slowing down and the energy diffusion come from one sweep over the momentum grid.
The factors of the bremsstrahlung and of the bound electrons depend on the momentum and the charge state, not on the plasma, and are tables per momentum grid.

### Fluid terms

A term that reads every charge state of a radius assembles its block from one evaluation in dual numbers, seeded at every radius at once ([`add_local_jacobian!`](@ref)).
It keeps its dual buffers between calls, and the tables of the atomic rates have concrete types.
The [`FrictionEnergyTransferTerm`](@ref) keeps its weights under a [`PlasmaKey`](@ref).
The test of the case bounds the bytes of a second call of the Jacobian, of these blocks and of the residuals of the ion rates.

## The linear solvers

- [`SparseLU`](@ref), UMFPACK, is the default and needs no further package.
  Its symbolic analysis is kept, so each Newton iteration pays the numeric factorization alone.
- [`MKLPardiso`](@ref) keeps its ordering and matching in the same way until a solve fails ([Choices of this package](choices.md#Linear-solvers)).
  Each analysis is counted in the factorization time of its solve.
  It runs threaded: from one thread to eight its factorization and solve go from 70.6 + 7.0 = 77.6 ms to 20.5 + 2.6 = 23.1 ms in the table above.
  DREAM's factorization and solve go from 88.2 to 45.4 ms.
- More BLAS threads do not help UMFPACK on this matrix, and ten refinement steps of PARDISO cost more than its factorization (the notes below the next table).
- Both solvers take the field-particle term by a low-rank update of the factorized matrix ([`low_rank_jacobian`](@ref)), which adds solves and no fill.
- [`GMRES`](@ref) with a block preconditioner factorizes blocks in place of the whole matrix ([Linear solvers and preconditioners](model/time-stepping.md#Linear-solvers-and-preconditioners)).

The table gives a factorization and a solve of the Newton matrix of the case outside a run.
Each figure is the median of five factorizations and five solves.

| Solver | Threads | Factorization and solve |
|---|---|---|
| [`SparseLU`](@ref), UMFPACK | 1 | 108 ms |
| [`SparseLU`](@ref), UMFPACK | 8 BLAS threads | 109 ms |
| [`MKLPardiso`](@ref) | 1 | 72 ms |
| [`MKLPardiso`](@ref) | 8 | 17 ms |

- Ten refinement steps of PARDISO take 80 to 90 ms per solve on that matrix, more than the factorization.
- The figures of the tables of iterations are per Newton iteration of a run, so they differ from these.

## Threads over radii

This section gives the gain of several Julia threads on the disruption case, whose kinetic terms then split the loops over their radii into tasks ([Threads](model/time-stepping.md#Threads)).
The table gives the window without radial transport, in milliseconds per Newton iteration, at the number of Julia threads in the first column.
Each run has one garbage-collector thread and one BLAS thread, and `taskset` binds it to one logical CPU per physical core.
The runs are a sitting of their own, so the rows at one thread differ from the table [A Newton iteration](#A-Newton-iteration) by up to 2 ms, apart from its one slow residual.
The last column is the iteration at one Julia thread with the same solver divided by the iteration of the row.

| Julia threads | Linear solver | Jacobian | Residual | Factorization | Solve | Iteration | Speedup |
|---|---|---|---|---|---|---|---|
| 1 | [`SparseLU`](@ref) | 33.8 | 6.8 | 108.5 | 5.4 | 159.3 | 1.00 |
| 2 | [`SparseLU`](@ref) | 27.6 | 5.9 | 108.9 | 5.4 | 153.1 | 1.04 |
| 4 | [`SparseLU`](@ref) | 18.8 | 4.4 | 108.7 | 5.4 | 142.3 | 1.12 |
| 8 | [`SparseLU`](@ref) | 15.1 | 3.3 | 109.0 | 5.4 | 137.7 | 1.16 |
| 1 | [`MKLPardiso`](@ref) at 8 threads | 34.8 | 6.7 | 20.6 | 2.6 | 70.0 | 1.00 |
| 8 | [`MKLPardiso`](@ref) at 8 threads | 15.0 | 4.2 | 20.1 | 2.5 | 46.6 | 1.50 |

- A row of [`SparseLU`](@ref) is the mean of two runs, whose iterations differ by at most 2.0 ms.
  A row of [`MKLPardiso`](@ref) is the median of four runs in each column, since in each row one run took 3.6 to 6.4 ms longer in the Jacobian than the other three.
- About 5 ms of an iteration are outside the four timers, as in the table [A Newton iteration](#A-Newton-iteration).
- With [`SparseLU`](@ref) the states of all runs are equal bit for bit, at every number of threads.
  [`MKLPardiso`](@ref) at 8 threads does not repeat its own states bit for bit from one run to the next.
- With [`SparseLU`](@ref) at 8 threads, the Jacobian takes 0.45 of its time at one thread, and the residual 0.49.
  The factorization does not change, so it is 79 % of the iteration at 8 threads.
- From 4 threads to 8 the gain is small.
  At 8 threads seven tasks hold two of the 15 radii and one holds one, so the threaded loops could be at most 7.5 times faster than at one thread.
  Their own time was not measured apart from the serial work in the same timers, so the cause of the smaller gain is open.
- With [`MKLPardiso`](@ref) at 8 threads and 8 Julia threads, the iteration takes 0.32 of DREAM's at 8 PARDISO threads (147.0 ms, [above](#A-Newton-iteration)).
- With [`SparseLU`](@ref) the run allocates 851 MB per iteration at every number of threads, most of them in UMFPACK, and the garbage collector takes 3.8 % to 4.5 % of the window.
  With [`MKLPardiso`](@ref) the run allocates 5 MB per iteration.

## Radial transport of the distributions

Radial transport couples the cells of neighbouring radii, and the fill of the factorization grows with it.
The same window with Rechester–Rosenbluth diffusion of both distributions at ``\delta B/B = 10^{-3}``, in milliseconds per Newton iteration:

| Code and solver | Jacobian | Residual | Factorization | Solve | Iteration | Newton per step | Against DREAM |
|---|---|---|---|---|---|---|---|
| DREAM, PARDISO at 1 thread | 129.9 | 4.1 | 3096.8, with the solve | | 3231.5 | 4.61 (mean over 238 steps), 4 in the window | 1.00 |
| DREAM, PARDISO at 8 threads | 138.8 | 4.5 | 864.9, with the solve | | 1008.9 | 4.61 (mean over 238 steps), 4 in the window | 1.00 |
| Runes2P, [`SparseLU`](@ref) | 59.3 | 10.8 | 836.5 | 21.1 | 931.9 | 4.80 | 0.29 |
| Runes2P, [`MKLPardiso`](@ref) at 1 thread | 56.2 | 11.3 | 1029.7 | 18.4 | 1119.8 | 4.80 | 0.35 |
| Runes2P, [`MKLPardiso`](@ref) at 8 threads | 57.5 | 11.9 | 232.4 | 8.9 | 314.7 | 4.80 | 0.31 |

- DREAM's figures are means over the first 238 of its 429 quench steps, the part of its run with transport that was timed.
- At this size, the largest relative difference between the states of the two codes over the steps of the window is 2.4e-9, in `j_ohm`.
  It is relative to DREAM's largest `j_ohm` over the radii at the same step, the measure of `phase_differences` of `test/example_disruption.jl`.
- The factorization and the solve are 92 % of an iteration with [`SparseLU`](@ref), and 77 % with [`MKLPardiso`](@ref) at eight threads.
- At one thread [`SparseLU`](@ref) is the faster solver with transport, and [`MKLPardiso`](@ref) without it.
  With eight threads [`MKLPardiso`](@ref) is the faster in both.

The transport changes the counts of the system and the cost of a factorization:

| Quantity | Without transport | With transport |
|---|---|---|
| stored nonzeros | 2350635 | 2670370 |
| colors | 17 | 28 |

| Factorization at one thread | With transport / without |
|---|---|
| [`SparseLU`](@ref) | 7.8 |
| [`MKLPardiso`](@ref) | 14.6 |
| DREAM's factorization and solve | 35 |

- DREAM's ratio is between means over different spans of its quench.
- The transport adds 319735 stored nonzeros.
  142800 of them are the corners of the declared cross diffusion, which hold zeros in this case.
- An operator of radial transport alone built with `cross_diffusion = false` stores no corners ([`AdvectionDiffusion`](@ref)).
  The measurement declared them, and the cost without them is not measured at this size.

## What the tests hold

- The comparison tests with DREAM and the regression values hold the results.
- `test_disruption_runaway_dream.jl` bounds the bytes of a second call of `jacobian!` on the window, of the plasma columns of each term and of the kept blocks.
  The bounds hold at `-O2` and are skipped below it.
  The bound of the plasma columns is also skipped with more than one Julia thread.
- The blocks of the terms that keep buffers, weights or positions are compared bit for bit with blocks built afresh, at a state and at a changed state.
- Tests of single systems bound their colors ([`jacobian_colors`](@ref)), and tests of operators the stored nonzeros of their matrices.
  No test holds the colors or the stored nonzeros of the disruption case, so a change there is checked by hand.

A change that moves one of these bounds or counts updates it in the same pull request and says so in its description.
