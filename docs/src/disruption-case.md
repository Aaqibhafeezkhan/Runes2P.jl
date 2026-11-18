# The disruption case

```@meta
CurrentModule = Runes2P
```

This page compares Runes2P with DREAM in DREAM's disruption case.
The tests compare with DREAM release v26.5, whose outputs of the case are reference sets of the page [Reference data](reference-data.md).

The reference case is DREAM's disruption of `examples/Disruptions/generate.py`, scenario 1, in a cylinder.
Its case script is `reference/dream/cases/disruption_kinetic.py` ([Reference data](reference-data.md)).
The tests run the case at a reduced size, fully kinetic and in DREAM's superthermal mode.
The whole quenches of scenarios 1 and 4 at full size are measurements that no test repeats.

## The runs of the case script

The case script runs DREAM's chain of `generate.py` as six runs, 0 to 5.
Run 5 needs the option `--window-start`.

| Run | What it does |
|---|---|
| 0 | one step: the conductivity of the initial plasma and the field of the prescribed current profile |
| 1 | 1 s in 3 steps from the Maxwellian with that field |
| 2 | run 1 again, with the field rescaled to the plasma current |
| 3 | the ionization phase (phase 1): the ionization of the injected D and Ar |
| 4 | the thermal and current quench (phase 2) |
| 5 | a window of phase 2: run 4 stops at a saved step, and run 5 takes the next steps from DREAM's state there |

Runs 0 to 2 are the initialization.

## Parts of the comparison

The builder of the case is `build` of `test/example_disruption.jl`.
Its keyword `compat = :dream` takes DREAM's settings of the case where the codes differ, and `compat = :runes2p` takes Runes2P's defaults ([Defaults that differ](dream-departures.md#Defaults-that-differ)).
The synchrotron loss is off at both values.

| Part | DREAM runs | Runes2P starts from or builds | Test |
|---|---|---|---|
| initialization | runs 0 to 2 | the same runs (notes below) | `test_disruption_initialization_dream.jl` |
| ionization phase | run 3 | DREAM's state at the start of the phase | `test_disruption_dream.jl` |
| thermal and current quench | run 4 | DREAM's state at the start of the phase | `test_disruption_dream.jl` |
| the chain end to end | runs 0 to 4 | Runes2P's own initialization, then each phase from Runes2P's end state of the run before | `test_disruption_chain_dream.jl` |
| the chain end to end in the superthermal mode | runs 0 to 4 with the case script's `--mode superthermal` (DREAM's `generate.py --superthermal`) | the same with `mode = :superthermal`, and, separately, each phase from DREAM's state | `test_disruption_superthermal_dream.jl` |
| quench with a runaway grid | runs 1 to 4 with the runaway grid to step 170 of 429 of phase 2, then 10 steps (run 5) | DREAM's state at the start of run 5 | `test_disruption_runaway_dream.jl` |
| the same from `f_re` zero | not compared with DREAM | the start of phase 2, with `f_re` zero | `test_disruption_runaway.jl` |
| whole quench of scenario 1 at full size | runs 0 to 4, the runaway grid in runs 1 to 4 | DREAM's state at the start of phase 2, 429 steps in one run | none: a measurement |
| the same in scenario 4 | runs 0 to 4 of scenario 4, the runaway grid in runs 1 to 4 | DREAM's state at the start of phase 2, 1600 steps in one run | none: a measurement |

DREAM's run 4 without the runaway grid holds no `f_re`, so the test from `f_re` zero has nothing to compare with.

The reduced sizes of the tests, as momentum × pitch cells:

| Test | Radii | Hot-tail grid | Runaway grid | Steps |
|---|---|---|---|---|
| initialization | 2 | 30 × 4 | none | runs 1 and 2: 3 steps over 1 s each |
| phases, chain | 2 | 30 × 4 | none | phase 1: 10 of 4e-8 s, phase 2: 10 of 1e-7 s |
| superthermal mode | 2 | 20 × 4, uniform, without the fine cells below p = 0.05 | none | as in the two rows above |
| quench with a runaway grid | 2 | 30 × 4 | 32 × 8, from p = 0.8 to 20 | 10 of 6.53e-7 s, from step 170 of 429 of phase 2 |
| the same from `f_re` zero | 2 | 30 × 4 | 32 × 8 | 3 of 1e-7 s |

The momenta are in units of ``m_e c``.
The hot-tail grid ends at `p_max = 0.8`, where the runaway grid starts.

The initialization in Runes2P:

- Runes2P runs the initialization itself, as runs 0 to 2.
- The field and the cold temperature stay at their prescribed values in runs 1 and 2.
- Runes2P builds the state at the start of phase 1 from the end of run 2 and the ions of the settings.

## The runaway grid

The runaway grid is the case script's `--runaway-grid` ([Choices](choices.md#Reference-cases)).
`build` adds it with `runaway = true`.

- The flux of `f_hot` through the hot-tail `p_max` enters `f_re` and `n_re` ([`MomentumInterfaceTerm`](@ref), [`InterfaceParticleFluxTerm`](@ref)).
- The kinetic avalanche source acts on `f_re`, on the runaway grid.
- The fluid avalanche acts in `n_re`, which counts every electron above the hot-tail `p_max`.
  On the runaway grid the fluid avalanche equals the kinetic source of `f_re`, and in `n_re` it adds the tail beyond the runaway grid's `p_max`.
- `f_re` carries the runaway current, ionizes the ions and heats the cold electrons.
- In the window of run 5, the runaway current is 5.9 % of the plasma current at the start.

## Balances checked without DREAM

The tests of the phases, of the chains and of the runaway grid also check balances of the assembled system that do not need DREAM.
They check them at both `compat` values.
The test of the initialization checks none.
`field` is one value over the whole run, from the first state to the last.
Every other value is the worst over the steps.

| Balance | What is checked | Relative to | Bound |
|---|---|---|---|
| `species` | the density of each ion species stays fixed, since the case has no ion transport | the largest density | 1e-13 |
| `exchange` | the energy that the electrons give the ions is the energy the ions take | the largest electron side | 1e-13 |
| `diffusion` | the heat diffusion has no net flux, since no heat leaves at the edge | its absolute sum | 1e-13 |
| `closure` | the change of `W_cold` equals the sum of its energy channels | dW/dt | 1e-12 |
| `field` | the change of the poloidal field energy equals the work of the field on the currents | the change | 1e-8 |
| field energy | the magnetic energy falls over the run | | its change below 0 |
| `drift` | the density of `f_hot` less n_cold and n_hot stays fixed, fully kinetic mode only | n_cold | 1e-6 in phase 1, 1e-13 in phase 2 |
| `interface` | the electrons the interface takes from `f_hot` are those it gives to `f_re` and to `n_re` | that influx at the radius in the step | 2e-14 |
| `avalanche` | n_re − ∫f_re grows by the avalanche beyond the runaway grid and the outflow there | the larger of dn_re/dt and d∫f_re/dt of the step at that radius | 1e-11 in the window, 1e-9 from `f_re` zero |
| influx | electrons cross from `f_hot` into `f_re` at every step and radius | | the smallest influx above 0 |

- In the fully kinetic mode `f_hot` covers the cold bulk as well, and the particle source keeps the balance of the `drift` row.
- In phase 1 the drift follows the convergence of Newton's method on the particle source, 2.7e-8 measured, so its bound is larger there.
- The rows `interface`, `avalanche` and influx hold for the tests with the runaway grid.
- The test from `f_re` zero also checks that the steps are accepted.
  It checks that `build` without the runaway grid has no `f_re`, and that the runaway grid adds `f_re` alone to the unknowns.
  It checks that the start state takes `S_particle` from a DREAM output that holds it.

## Agreement at the size of the tests

This section gives the largest differences against DREAM in the tests, all at `compat = :dream`.
Each difference is relative: |Runes2P − DREAM| divided by a reference value of DREAM.
Each number is the largest such difference over the saved steps after the first.
`phase_differences` of `test/example_disruption.jl` takes the reference by the kind of quantity:

| Short name | Reference | Used for |
|---|---|---|
| over the radii | DREAM's largest value over the radii at the same step | the radial quantities, such as `E_field`, the currents, `n_hot` and `psi_p` |
| per radius and step | DREAM's largest value over the cells, charge states or species at the same radius and step | `f_hot`, `f_re`, `n_i`, `W_i` |
| `I_p` | DREAM's `I_p` at the same step | `I_p` |
| per radius over the run | DREAM's largest value at the same radius over the run | `n_re` and `j_re`, where a test compares them per radius |
| per unknown | DREAM's largest entry of the unknown in its first state of phase 1 | the hand-over from the initialization to phase 1 |

The initialization compares `n_re` and `j_re` over the radii only.
The chain also compares `n_re` and `j_re` at the first radius, per radius over the run.
The values are measured at Julia's default optimisation level, unless marked, and change slightly with it.
The functions `phase_differences`, `read_dream_state` and `run_window` belong to `test/example_disruption.jl`, not to the package's API.

### The initialization

`test_disruption_initialization_dream.jl` compares runs 1 and 2, and the state that they hand to phase 1.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `j_hot` | 7.9e-10 | over the radii |
| `I_p` | 7.8e-10 | `I_p` |
| `E_field` | 7.7e-10 | over the radii |
| `j_ohm` | 7.5e-10 | over the radii |
| `j_tot` | 7.5e-10 | over the radii |
| `psi_p` | 4.9e-10 | over the radii |
| `f_hot` | 2.7e-11 | per radius and step |
| `n_hot` | 1.7e-11 | over the radii |
| every unknown, at the hand-over to phase 1 | 7.7e-10 | per unknown |

DREAM's Newton tolerance of these runs is 1e-7, so the differences lie far below it.

### The phases

`test_disruption_dream.jl` starts each phase from DREAM's state.
`test_disruption_chain_dream.jl` runs the chain end to end from Runes2P's own initialization.

| Quantity | From DREAM's state | The chain end to end | Reference |
|---|---|---|---|
| `f_hot` | 4.4e-5 | 4.4e-5 | per radius and step |
| `E_field` | 4.1e-7 | 4.1e-7 | over the radii |
| `psi_p` | 6.8e-13 | 2.8e-10 | over the radii |
| `I_p` | 1.3e-12 | 1.3e-11 | `I_p` |
| `n_re` at the first radius | 5.5e-12, at `-O0` | 3.6e-11 | per radius over the run |

- In the chain, `psi_p`, `I_p` and `n_re` carry the differences of the initialization.
- The table gives `n_re` at the first radius only.
  At the second radius `n_re` is 2e-15 of the first.
- From DREAM's state the test bounds `n_re` per radius at the second radius too, where it differs by 1.5e-11 at `-O0`.
- At the start of phase 1, DREAM's `n_re` at the second radius is 6.8e-3 m⁻³.
  DREAM's output holds it with too few significant bits for a comparison.
  So the chain compares the growth of `n_re` over phase 2 there, which differs by 1.9e-3 of DREAM's largest growth.

### The superthermal mode

`test_disruption_superthermal_dream.jl` runs the chain in the superthermal mode, and each phase from DREAM's state.
In the first four rows the value is the largest difference over all compared quantities, each in its own reference.
These rows take `n_re` and `j_re` of the chain over the radii, and the last row takes them per radius.

| Comparison | Largest in | Largest relative difference | Reference |
|---|---|---|---|
| initialization, runs 1 and 2 | `j_tot`, `j_ohm`, `j_hot` | 1.5e-9 | over the radii |
| hand-over to phase 1 | `n_cold` | 1.3e-8 | per unknown |
| phases of the chain | `n_i` | 3.5e-6 | per radius and step |
| phases from DREAM's state | `n_i` | 3.2e-6 | per radius and step |
| phases of the chain, `n_re` and `j_re` at the second radius | | 3.8e-4 | per radius over the run |

- `n_cold` and `W_cold` come next to `n_i`, at 2.9e-6 and 2.8e-6 in both kinds of run.
- The 3.8e-4 comes from the initialization.
  At the second radius `n_re` is 4e-7 of the first radius, fed by the far tail of f.

### The window with the runaway grid

`test_disruption_runaway_dream.jl` starts from DREAM's state at the start of run 5 and takes its 10 steps.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `f_re` | 1.1e-9 | per radius and step |
| `n_re` | 1.1e-9 | per radius over the run |
| `j_re` | 1.2e-9 | per radius over the run |
| `n_i`, the largest of the other quantities | 2.2e-8 | per radius and step |

- Each value is the larger of those at `-O0` and at the default optimisation level.
- Every other quantity differs by less than 2.2e-8 in its own reference.
- At the start, DREAM's `j_re` equals the moment of its `f_re` on Runes2P's grid to 1.1e-15.
  This is a check of DREAM's state, not a difference between the codes.

## The whole quench of scenario 1

This measurement runs phase 2 of scenario 1 at the full size of DREAM's example, in 429 steps of 6.53e-7 s.

### The run

The table gives the set-up of the measurement.
The page [Performance and accuracy](performance.md#The-disruption-case) gives the same case in more detail.

| | |
|---|---|
| Case | scenario 1 of DREAM's `examples/Disruptions/generate.py`, thermal and current quench |
| DREAM | runs 0 to 4 of the case script (the command line below) |
| Size | 15 radii, hot-tail grid 140 × 15, runaway grid 32 × 15 (`--re-nxi 15 --re-nxi-sep 10`) |
| Start | DREAM's state at the start of phase 2, not Runes2P's own chain |
| Steps | DREAM's 429 fixed steps of 6.53e-7 s |
| Settings of Runes2P | `runaway = true` and `compat = :dream` of `build` |
| Solver | Runes2P's default direct solver, UMFPACK ([`SparseLU`](@ref)), at Julia's default optimisation level, DREAM with `--linear-solver mkl` |

DREAM runs the case script with its default sizes, which are the full size of DREAM's example:

```sh
disruption_kinetic.py --nt-ioniz 500 --t-ioniz 2e-5 --nt-quench 429 --t-quench 3e-4 --linear-solver mkl --runaway-grid --re-nxi 15 --re-nxi-sep 10
```

`--t-quench` is the end time of phase 2, counted from the start of phase 1.
So phase 2 lasts 3e-4 s − 2e-5 s, and each of its 429 steps is 6.53e-7 s.

### Newton iterations

The table counts the Newton iterations of each code over the whole quench.

| Code | Accepted steps | Newton iterations, total over the 429 steps |
|---|---|---|
| Runes2P | 429 of 429 | 1822 |
| DREAM | 429 fixed steps | 1935 |

### Agreement with DREAM

Each difference is relative: |Runes2P − DREAM| divided by the reference of its row.
Each number is the largest such difference over the 429 steps of the quench.
1e5 m⁻³ is DREAM's absolute tolerance of `n_re`.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `n_re` | 4.4e-6 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re` | 3.4e-6 | DREAM's largest `j_re` at the same radius over the quench |
| `j_hot` | 2.0e-6 | DREAM's largest value over the radii at the same step |
| `E_field` | 1.1e-6 | DREAM's largest value over the radii at the same step |
| `j_ohm` | 1.1e-6 | DREAM's largest value over the radii at the same step |
| `j_tot` | 2.2e-7 | DREAM's largest value over the radii at the same step |
| `f_hot` | 5.3e-8 | DREAM's largest `f_hot` at the same radius and step |
| `I_p` | 5.0e-9 | DREAM's `I_p` at the same step |
| `f_re`, where DREAM's `n_re` is above 1e5 m⁻³ | at most 1.3e-5 | DREAM's largest `f_re` at the same radius and step |
| every other quantity | 5e-8 | the reference of its kind (table of references above) |

Over all radii and steps, `phase_differences` gives a larger value for `f_re`.
The larger differences of `f_re` are at the two outermost radii:

| Radius | Steps | Largest relative difference of `f_re` |
|---|---|---|
| 0.45 m | 1 to 4 | 0.51 |
| 0.48 m | 15 to 25 | 4.6e-2 |

In these steps DREAM's `n_re` at the two radii is at most 6.1e-7 m⁻³.
These cells hold almost no runaways: DREAM's `n_re` there lies far below its absolute tolerance.

At the end of the quench the runaway current is 67.8 % of the plasma current in both codes.

### How to repeat it

1. DREAM runs the case script with the command line of "The run" above.
2. `read_dream_state` reads DREAM's output of run 4.
3. `run_window` runs Runes2P over the saved times of a DREAM run, here all 429 steps of the quench.
4. `phase_differences` returns the largest difference of each quantity.

## The whole quench of scenario 4

Scenario 4 of `generate.py` has a quench of 4.99e-3 s in 1600 steps.
Its heat diffusivity is 1000 m²/s, against the 4000 m²/s of scenario 1.
The quench is a measurement at full size.

### The run

| | |
|---|---|
| Case | scenario 4 of DREAM's `examples/Disruptions/generate.py`, thermal and current quench |
| DREAM | v26.5, runs 0 to 4 of the case script (the command line below) |
| Size | 15 radii, hot-tail grid 140 × 15, runaway grid 52 × 15 to `p_max = 200` (12 fine and 40 coarse momentum cells), 43803 unknowns of Runes2P |
| Start | DREAM's state at the start of the quench |
| Steps | DREAM's 1600 fixed steps of 3.11875e-6 s, to 4.99e-3 s |
| Settings of Runes2P | `runaway = true`, `compat = :dream` and `diffusivity = 1000.0` of the builder of the case |
| Solver | MKL PARDISO on one thread in both codes, Runes2P at `-O2` with one Julia thread |

DREAM runs the case script with its default sizes and these options:

```sh
disruption_kinetic.py --scenario 4 --linear-solver mkl --runaway-grid --re-pmax 200 --re-np-coarse 40 --re-nxi 15 --re-nxi-sep 10
```

The runaway grid is a choice of this package ([Choices](choices.md#Reference-cases)).
The runs of Runes2P with fixed steps started with the particle source at zero, in place of DREAM's `S_particle`.
`initial_state` of `test/example_disruption.jl` takes `S_particle` where DREAM's output holds it.
With fixed steps the first step solves the source.
With DREAM's `S_particle` at the start, the differences against DREAM in the first steps are the same to the digits of the tables below.

### Newton iterations

| Code | Accepted steps | Newton iterations | Most in one step |
|---|---|---|---|
| Runes2P | 1600 of 1600 | 6356 | 6 |
| DREAM v26.5 | 1600 | 7899 | 14 |

### Agreement with DREAM

Each difference is relative: |Runes2P − DREAM| divided by the reference of its row.
Each number is the largest such difference over the 1600 saved steps.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `I_p` | 6.67e-8 | DREAM's `I_p` at the same step |
| `j_hot` | 2.80e-6 | DREAM's largest value over the radii at the same step |
| `j_ohm` | 1.55e-6 | DREAM's largest value over the radii at the same step |
| `E_field` | 9.31e-7 | DREAM's largest value over the radii at the same step |
| `n_hot` | 4.77e-7 | DREAM's largest value over the radii at the same step |
| `T_cold` | 3.48e-7 | DREAM's largest value over the radii at the same step |
| `j_tot` | 2.95e-7 | DREAM's largest value over the radii at the same step |
| `n_cold` | 3.44e-8 | DREAM's largest value over the radii at the same step |
| `n_re`, radii 1 to 11 | 4.3e-7, at radius 4 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re`, radii 1 to 11 | 4.4e-7, at radius 4 | DREAM's largest `j_re` at the same radius over the quench |
| `n_re` and `j_re`, radii 12 and 13 | 5.6e-7 | DREAM's largest value at the same radius over the quench |
| `n_re`, radii 14 and 15 | 2.2e-2 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re`, radii 14 and 15 | 3.4e-2 | DREAM's largest `j_re` at the same radius over the quench |
| `f_re` | 1.91e-6 | DREAM's largest `f_re` at the same radius and step |
| `f_hot` | 1.12e-5 | DREAM's largest `f_hot` at the same radius and step |

- At radii 1 to 11 DREAM's `n_re` is above its absolute tolerance of 1e5 m⁻³.
- At radii 12 and 13 it is below that tolerance.
  Its largest values there are 4.9e3 and 8.8 m⁻³.
- At radii 14 and 15 it stays below 2e-3 m⁻³.
- The `f_re` row counts only the radii and steps where DREAM's largest `f_re` is above 1e5, its absolute tolerance in DREAM's runs of the case.
- At 4.99e-3 s `I_p` is 2.543120e5 A in both codes.
- At 4.99e-3 s the share of `n_re` above the runaway grid is 4.0e-5 to 4.1e-5.
  This holds at radii 1 to 11, and the two codes give it equal to four digits.

### The difference of `f_hot`

- The largest difference, 1.12e-5, is in the first step, in the lowest momentum cell.
- It falls to 1.5e-8 by step 20.
- At the time of the first step, the runs of 3200 steps differ by 1.10e-5 in the same measure.
- The 2-norm of the difference is 3.8e-7 of the 2-norm of `f_hot`.
- Its cause is not established.

### Negative cells

The cells are counted over the 1601 saved states, at the radii where DREAM's largest value is above its absolute tolerance.
A cell that is negative in 10 states counts 10 times.

| Distribution | Negative in Runes2P | Negative in DREAM | Negative in both | Most negative value, relative to the largest value at its radius |
|---|---|---|---|---|
| `f_re` | 77146 | 77762 | 77146 | −3.9e-6 in both codes |
| `f_hot` | 947945 | 1034261 | 947620 | −1.2e-16 in both codes |

- Every negative cell of `f_re` in Runes2P is negative in DREAM too.
- The most negative cell of `f_re` is in the first momentum cell of the runaway grid.
- The negative cells of `f_hot` are at round-off.
  The most negative one is the same cell in both codes.
- A cell of `f_hot` negative in one code only holds at most 3.3e-26 of DREAM's largest value at its radius and step.
- DREAM's start state holds 2411 negative cells, in `f_hot` and `f_re` together and at all radii.
- `polish_negatives = :auto` of [`NewtonSolver`](@ref) stops polishing after 1.56e-5 s.
  By then five polished solves in a row have left negative entries.

### The time-step error of the reference

The agreement above is at equal steps, so it does not show the error of the step length.
Runes2P was run with the three step counts of the table and the runaway grid to `p_max = 200`.
The limit is the Richardson extrapolation of these three runs.

| Run | `I_p` at 4.99e-3 s (A) | (`I_p` − limit) / limit | `E_field` at the first radius at 4.00135625e-3 s (V/m) | (`E_field` − limit) / limit |
|---|---|---|---|---|
| 1600 steps (the reference) | 2.5431201e5 | +1.00e-2 | 1.1793628 | −7.2e-2 |
| 3200 steps | 2.5306491e5 | +5.1e-3 | 1.2218632 | −3.9e-2 |
| 6400 steps | 2.5243459e5 | +2.6e-3 | 1.2447340 | −2.1e-2 |
| Richardson limit | 2.5179047e5 | | 1.2713809 | |

- The observed order is 0.98 in `I_p` and 0.89 in the field.
- An extrapolation of order 1 gives 1.268 V/m for the field.

DREAM v26.5 was run with the same three step counts and `p_max = 200`.
The limits of the two codes agree:

| Quantity | Runes2P limit | DREAM limit | Observed order (both) | (DREAM − Runes2P) / Runes2P limit |
|---|---|---|---|---|
| `I_p` at 4.99e-3 s | 2.5179047e5 A | 2.5179052e5 A | 0.98 | +1.7e-7 |
| `E_field` at the first radius at 4.00135625e-3 s | 1.2713809 V/m | 1.2713786 V/m | 0.89 | −1.8e-6 |

The table of the runs above uses Runes2P's limit.

At 6400 steps and `p_max = 200`, DREAM's values at the times of that table are close to Runes2P's:

| Quantity | DREAM | (DREAM − Runes2P) / Runes2P |
|---|---|---|
| `I_p` at 4.99e-3 s | 2.5243461e5 A | +6.6e-8 |
| `E_field` at the first radius at 4.00135625e-3 s | 1.2447333 V/m | −5.5e-7 |

The agreement at the other step counts takes the references of the agreement table.

| Steps | `p_max` | Compared times | Radial quantities | `I_p` | `n_re` and `j_re`, radii 1 to 11 |
|---|---|---|---|---|---|
| 3200 | 200 in both codes | 1613 of DREAM's 3201 | 6.8e-7 | 4.0e-8 | 2.5e-7, at radius 4 |
| 6400 | 200 in both codes | 1639 of DREAM's 6401 | 1.1e-6, largest in `j_ohm` | 6.6e-8 | 3.9e-7, at radius 4 |
| 6400 | 60 in both codes | all 6400 | 2.4e-6 | 6.6e-8 | 3.9e-7, at radius 4 |

- The run of Runes2P with 3200 steps saved every second step and the steps around 4e-3 s.
  That gives the 1613 compared times.
- The run of Runes2P with 6400 steps and `p_max = 200` saved every fourth step and the steps around 4e-3 s.
  That gives the 1639 compared times.

### The step controller against fixed steps

[`StepController`](@ref) starts from the same state as the fixed steps.
Its start takes the particle source from DREAM's `S_particle` through `initial_state`.
Its `relative_tolerance` is 1e-3 (the default) or 1e-4, and its other settings are the defaults.
The runs save only at 4.00135625e-3 s and 4.99e-3 s, where a step ends.
The Newton iterations count every attempt.

| Run | Accepted steps | Newton iterations | (`I_p` − limit) / limit | (`E_field` − limit) / limit | (`T_cold` − value at 6400 fixed steps) / that value |
|---|---|---|---|---|---|
| fixed, 1600 steps | 1600 | 6356 | +1.00e-2 | −7.2e-2 | +2.1e-5 |
| fixed, 6400 steps | 6400 | 20743 | +2.6e-3 | −2.1e-2 | 0 |
| `StepController`, `relative_tolerance = 1e-3` | 1107 | 4681 | +6.3e-3 | −2.8e-2 | +1.3e-3 |
| `StepController`, `relative_tolerance = 1e-4` | 3239 | 11898 | +2.2e-3 | −1.2e-2 | +4.6e-4 |

- `T_cold` is that of the first radius at 4.99e-3 s.
- At 1e-3 the controller is closer to the limit in `I_p` and the field than 1600 fixed steps, with fewer Newton iterations.
- At 1e-4 it is closer than 6400 fixed steps, with fewer Newton iterations.
- In `T_cold` it is further off than the fixed steps.
  The runs with fixed steps differ there by at most 2.1e-5 of the value at 6400 fixed steps.
- At 6400 fixed steps with `p_max = 200`, DREAM v26.5 takes 25809 Newton iterations, at most 14 in a step.
  Runes2P takes 20743, at most 5 in a step.

The start of the controller:

| | `relative_tolerance = 1e-3` | `relative_tolerance = 1e-4` |
|---|---|---|
| Initial step | 3.11875e-6 s, rejected | 3.11875e-6 s, rejected |
| Steps rejected on the error estimate, the initial step among them | 3 | 3 |
| Steps repeated after a Newton failure | 1, at 1.03e-3 s | 0 |
| First accepted step | 4.05e-7 s | 9.4e-8 s |
| Smallest accepted step | 2.7e-7 s | 7.2e-8 s |

### How to repeat it

The default `diffusivity` of `build` and `run_window` of `test/example_disruption.jl` is that of scenario 1.
So the run of scenario 4 needs `diffusivity = 1000.0`.

1. `read_dream_state` reads DREAM's output of run 4.
2. `run_window(start, :dream; diffusivity = 1000.0, solver)` runs the quench, with a [`NewtonSolver`](@ref) on [`MKLPardiso`](@ref).
3. `phase_differences` returns the largest difference of each quantity.

`phase_differences` gives larger values for three quantities of this run:

| Quantity | Value of `phase_differences` |
|---|---|
| `n_re` | 2.23e-2 |
| `j_re` | 3.35e-2 |
| `f_re` | 18.0 |

These come from the outer radii, below DREAM's tolerances.
The table above takes `n_re` and `j_re` per group of radii.
It takes `f_re` only where DREAM's largest value at the radius and step is above 1e5.
