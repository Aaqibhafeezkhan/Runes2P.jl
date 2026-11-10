# Performance and accuracy

```@meta
CurrentModule = Runes2P
```

This page has two parts.
The first gives the guidelines that hold for every change: how accuracy and cost are weighed, and how a cost is measured.
The second holds the details of the hot paths and the linear solvers, with the measurements of the coupled disruption case.

## Guidelines

### Accuracy first, cost with it

- A result is trusted because a test holds it: against DREAM to round-off in DREAM's mode, and against a reference independent of DREAM for each piece of physics.
  A faster code path keeps these tests, and the regression values at `-O2` bit for bit where it can.
- A change that moves results at round-off says so, with what was compared and on which case.
  A sum taken in another order changes the last digits, and a check of single terms does not see an entry that several terms share.
- A numerical choice is recorded with its cost as well as its reason ([Choices of this package](choices.md)).
  A departure from DREAM is judged by its accuracy for its cost, and DREAM's behaviour stays an option ([Comparison with DREAM](dream-comparison.md#Departures-from-DREAM)).
- A change that makes a run significantly slower gives its reason in the pull request and is optimized there as far as it can be.
  Where that is too large for the pull request, an issue takes the rest, with suggestions for the improvement.

### Measure, then change

- The time of a step is in the [`SolverStatistics`](@ref) of a run: `wall_time`, and within it `jacobian_time`, `residual_time` and `linear_solve_time`.
  `factorization_time` is the part of `linear_solve_time` spent in the factorizations.
  Divided by `newton_iterations` they give the cost of a Newton iteration, the unit of this page.
- A measurement comes after a warm-up of a few steps, so that it leaves out the compilation, and at `-O2`.
- Every number names its threads: those of Julia, of BLAS (`OPENBLAS_NUM_THREADS`) and of the solver (`threads` of [`MKLPardiso`](@ref)).
- Two versions are compared in one sitting on an idle machine, alternating: base, head, head, base.
  Repeated runs of one version differ by a few milliseconds per iteration on the case below, and a gain below that spread is not a gain.
- Counts do not vary between machines, and they show a change of the cost before a timer does.
  They are the stored nonzeros (`nnz` of [`jacobian_sparsity`](@ref)), the passes of the colored differentiation ([`jacobian_colors`](@ref)) and the Newton iterations per step.
  The bytes of a second call (`@allocated`) are such a count too.

### Habits of the code

- An operator that holds its coefficients assembles its Jacobian from them ([`jacobian_assembly`](@ref) `:direct`).
  The colored forward-mode differentiation serves the terms without an assembled form, and a central-difference Jacobian checks every assembled block in the tests.
- The pattern of the stored Jacobian holds the entries that the terms write and no others.
  Every declared entry costs fill in the factorization, also when it holds a zero.
- An inner loop has concrete types.
  A loop over values whose type is known only at run time allocates per entry, so it sits behind a function barrier.
- A buffer is kept between calls, not allocated per call.
  A residual or a Jacobian evaluated a second time at one state allocates little, and tests bound it at `-O2`.
- A cache is keyed on every input that its values read, such as the plasma values per radius of a [`PlasmaKey`](@ref).
  Its test changes each input alone and counts one rebuild, and none for an unchanged input.
- The caches and buffers of the terms assume that one task assembles the system.

## The disruption case

The measurements of this page are of DREAM's disruption case with a runaway grid ([The disruption case](dream-comparison.md#The-disruption-case)).
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
| runaway grid | 32 momentum cells from 0.8 to 20, 15 pitch cells |
| size of the measurements | 15 radii, 39303 unknowns, 2350635 stored nonzeros |

The momenta are in units of ``m_e c``.
The case has two phases, the ionization of the injected material and the thermal and current quench.
The timed window is 10 steps of 6.53e-7 s from step 170 of DREAM's quench.
Runes2P starts it from DREAM's state, in DREAM's mode, and takes 2 steps of warm-up first.
The tests run the same window with 2 radii and 835 unknowns (`test_disruption_runaway_dream.jl`).

Runes2P ran with one Julia thread, one BLAS thread and `-O2`, DREAM v26.5 with MKL PARDISO; in both codes only PARDISO ran threaded.
The machine was one idle desktop computer with 8 cores, one run at a time.
DREAM's timers are sums over a phase, so its figures are means over its quench, per Newton iteration.

## A Newton iteration

In milliseconds per Newton iteration, without radial transport of the distributions:

| Code and solver | Jacobian | Residual | Factorization | Solve | Iteration | Newton per step | Against DREAM |
|---|---|---|---|---|---|---|---|
| DREAM, PARDISO at 1 thread | 93.8 | 3.4 | 88.2, with the solve | | 186.0 | 4.51 | 1.00 |
| DREAM, PARDISO at 8 threads | 97.4 | 3.6 | 45.4, with the solve | | 147.0 | 4.51 | 1.00 |
| Runes2P, [`SparseLU`](@ref) | 35.1 | 6.7 | 106.7 | 5.4 | 158.7 | 4.50 | 0.85 |
| Runes2P, [`MKLPardiso`](@ref) at 1 thread | 35.3 | 9.0 | 70.6 | 7.0 | 126.8 | 4.50 | 0.68 |
| Runes2P, [`MKLPardiso`](@ref) at 8 threads | 35.2 | 8.9 | 20.5 | 2.6 | 72.0 | 4.50 | 0.49 |

The last column compares with DREAM at the same number of PARDISO threads, and [`SparseLU`](@ref) with DREAM at one.
DREAM's Jacobian holds the rebuild of its coefficients, about 31 ms, which Runes2P does inside its residual and its Jacobian.
About 5 ms of a Runes2P iteration are outside the four timers.
At the end of the window the states of the two codes agree to 4.1e-9, the largest difference in the hot-electron density.

With [`SparseLU`](@ref) the factorization and the solve are 71 % of an iteration, with [`MKLPardiso`](@ref) 61 % at one thread and 32 % at eight.
The linear solver is therefore the first place to look in a slow run, and the stored pattern decides its cost.

## The hot paths

### The stored pattern

The kinetic operator declares the stencil of the coefficients it holds ([`stencil_pattern`](@ref)).
An operator without radial transport stores no radial entries, and one without cross diffusion no corners.
[`check_stencil_pattern`](@ref) throws when a coefficient lies outside the declared stencil, so an entry is never dropped silently.
With the radial entries and corners declared for this case, which fills none of them, UMFPACK factorizes in 861 ms in place of 107 ms.

### Colored differentiation and direct rows

The terms with automatic assembly share one coloring, and each color is one evaluation of their residuals in dual numbers ([`jacobian_colors`](@ref)).
A row that reads many cells of a radius would take a color per cell.
Such rows are assembled directly: the moments of the density and the current, and the flux through the interface of two momentum grids.
With the interface rows direct the case has 17 colors in place of 33.
A moment keeps the positions and weights of its row for the pattern of the matrix, so a later assembly only adds values.

### Plasma columns

The derivatives with respect to the temperature, the density and the charge states are columns of their own ([Plasma columns of the Jacobian](model/unknowns.md#Plasma-columns-of-the-Jacobian)).
The coefficients in dual numbers are built once per state and shared by the term and the terms that read its coefficients.
The loops that insert their partials run behind function barriers, with one search of the pattern per radius and column.

### Coefficients

The coefficients of a [`FokkerPlanckTerm`](@ref) are rebuilt only when the plasma values of some radius change ([Caches and rebuilds](model/unknowns.md#Caches-and-rebuilds)).
The slowing down and the energy diffusion come from one sweep over the momentum grid.
The factors that depend on the grid alone, those of the bremsstrahlung and of the bound electrons, are tables per momentum grid.

### Fluid terms

A term that reads every charge state of a radius assembles its block from one evaluation in dual numbers per radius ([`add_local_jacobian!`](@ref)).
It keeps its dual buffers between calls, and the tables of the atomic rates have concrete types.
The [`FrictionEnergyTransferTerm`](@ref) keeps its weights under a [`PlasmaKey`](@ref).
The test of the case bounds the bytes of a second call of the Jacobian, of these blocks and of the residuals of the ion rates.

## The linear solvers

- [`SparseLU`](@ref), UMFPACK, is the default and needs no further package.
  Its symbolic analysis is kept, so each Newton iteration pays the numeric factorization alone.
- [`MKLPardiso`](@ref) keeps its ordering and matching in the same way, and its one analysis is counted in the factorization time of the first solve.
  It runs threaded: from one thread to eight its factorization and solve go from 77.6 to 23.1 ms in the table above.
  DREAM's inversion goes from 88.2 to 45.4 ms.
- More BLAS threads do not help UMFPACK on this matrix, and refinement steps of PARDISO cost more than its factorization.
  The timings of the factorizations alone are on the page [Choices of this package](choices.md#Linear-solvers).
- Both solvers take the field-particle term by a low-rank update of the factorized matrix ([`low_rank_jacobian`](@ref)), which adds solves and no fill.
- [`GMRES`](@ref) with a block preconditioner factorizes blocks in place of the whole matrix ([Linear solvers and preconditioners](model/time-stepping.md#Linear-solvers-and-preconditioners)).

## Radial transport of the distributions

Radial transport couples the cells of neighbouring radii, and the fill of the factorization grows with it.
The same window with Rechester–Rosenbluth diffusion of both distributions at ``\delta B/B = 10^{-3}``, in milliseconds per Newton iteration:

| Code and solver | Jacobian | Residual | Factorization | Solve | Iteration | Newton per step | Against DREAM |
|---|---|---|---|---|---|---|---|
| DREAM, PARDISO at 1 thread | 129.9 | 4.1 | 3096.8, with the solve | | 3231.5 | 4.61 | 1.00 |
| DREAM, PARDISO at 8 threads | 138.8 | 4.5 | 864.9, with the solve | | 1008.9 | 4.61 | 1.00 |
| Runes2P, [`SparseLU`](@ref) | 59.3 | 10.8 | 836.5 | 21.1 | 931.9 | 4.80 | 0.29 |
| Runes2P, [`MKLPardiso`](@ref) at 1 thread | 56.2 | 11.3 | 1029.7 | 18.4 | 1119.8 | 4.80 | 0.35 |
| Runes2P, [`MKLPardiso`](@ref) at 8 threads | 57.5 | 11.9 | 232.4 | 8.9 | 314.7 | 4.80 | 0.31 |

DREAM's figures are means over the first 238 steps of its quench with transport.
In the window itself DREAM takes 4 Newton iterations per step and Runes2P 4.8; without transport DREAM takes 5 and Runes2P 4.5.
The states of the two codes agree to 2.4e-9 at the end of the window.

- The factorization and the solve are 92 % of an iteration with [`SparseLU`](@ref), and 77 % with [`MKLPardiso`](@ref) at eight threads.
- The stored pattern has 2670370 nonzeros and 28 colors, against 2350635 and 17 without transport.
- At one thread [`SparseLU`](@ref) is the faster solver with transport, and [`MKLPardiso`](@ref) without it.
  With eight threads [`MKLPardiso`](@ref) is the faster in both.
- The transport multiplies the factorization of [`SparseLU`](@ref) by 7.8 and that of [`MKLPardiso`](@ref) by 14.6 at one thread.
  DREAM's inversion grows by 35.
- Of the entries that the transport adds outside the kinetic stencil, two thirds are the corners of the declared cross diffusion, which hold zeros in this case.

## What the tests hold

- The comparison tests with DREAM and the regression values hold the results.
- `test_disruption_runaway_dream.jl` bounds the bytes of a second call of `jacobian!` on the window, of the plasma columns of each term and of the kept blocks.
  The bounds hold at `-O2` and are skipped below it.
- The blocks of the terms that keep buffers, weights or positions are compared bit for bit with those of a new term, at a state and at a changed state.
- Tests of single systems bound their colors ([`jacobian_colors`](@ref)) and their stored nonzeros.

A change that moves one of these bounds or counts updates it in the same pull request and says so in its description.
