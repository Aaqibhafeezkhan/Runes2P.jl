# Examples

This folder holds basic runs of the methods in the code, one script each.
Each script takes `name=value` switches on the command line, and its defaults run as they are.
The environment is `examples/Project.toml`, with Runes2P from the parent directory and HDF5 for the output:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

## The scripts

The three scripts, what each runs and what it reports:

| Script | Run | Report |
|---|---|---|
| `relaxation.jl` | a Maxwell-Jüttner distribution under a fixed plasma and field (Dreicer regime by default) | with `p_star=`, the fraction above that momentum |
| `cooling.jl` | the temperature decays from `T=` to `T_final=` over `t_end=` | where a scheme goes negative |
| `avalanche.jl` | knock-on multiplication of a 100 keV seed on DREAM's avalanche example | the growth rate |

- `relaxation.jl`: `T0=` sets a seed temperature different from the plasma's.
- `cooling.jl`: the collision coefficients follow the temperature.
  With `nr=`, that many radii are coupled by radial diffusion of strength `dbb=`.
- `avalanche.jl`: `treatment=` takes `explicit_primary` (the default), `explicit` or `implicit`.
  `cutoff=` sets the knock-on cutoff.
  The script builds its own grid and plasma and takes no checkpoint, stop or log switches.

## What a script prints and writes

Each script prints the solver statistics, the negative entries of the saved states and the density at every save.
The statistics are the counts of `SolverStatistics`, described with the convergence tests on the docs page [Time stepping and solvers](../docs/src/model/time-stepping.md).

`output=run.h5` writes the solution with `write_output`.
The file holds the layout, the saved states per unknown, the solver statistics, and the settings of the run with the commit of Runes2P.

## Switches

The switches are shared through `common.jl`.
A script's own defaults are in its header.

| Switch | Values | What it sets |
|---|---|---|
| `grid=` | `uniform` (default), `biuniform`, `tanh`, `biuniform_270` | momentum grid type |
| `np=`, `p_max=` | integer, number | momentum cells and extent |
| `np_low=`, `p_sep=` | integer, number | cells below and position of the split of the biuniform and tanh grids |
| `pitch=`, `nxi=` | `uniform` (default) or `theta`; integer | pitch grid type and cells |
| `scheme=` | `exponential_fitting` (default), `central`, `upwind`, `quick`, `upwind_second_order`, `van_albada`, `tcdf`, `muscl`, `smart`, `ospre` | interpolation scheme in both directions |
| `p_scheme=`, `xi_scheme=` | as `scheme=` | one direction |
| `jacobian=` | `exact` (default), `frozen_limiter`, `upwind`, `central`, `exponential_fitting` | the stored Jacobian of a limiter |
| `pmax_boundary=` | `f_zero` (default), `constant_flux`, `extrapolated_flux`, `extrapolated_flux_consistent`, `outflow` | boundary at `p_max` |
| `mode=` | `full` (default of `relaxation.jl` and `cooling.jl`), `superthermal`, `ultra_relativistic` | collision frequency mode |
| `screening=`, `lnlambda=`, `ion_size_table=` | `partial` (default), `complete`, `none`; `thermal` (default), `energy_dependent`, `relativistic`; `hesslow` (default), `dream` | collision settings (`avalanche.jl` reads them with its example's values as defaults) |
| `T=`, `n=`, `B=` | eV, m⁻³, T | plasma |
| `E=` or `e_over_ec=` | V/m, or multiples of the critical field | electric field |
| `n_ar=` | m⁻³ | neutral argon added to the deuterium plasma |
| `synchrotron=` | `true`, `false` (default) | synchrotron losses (`relaxation.jl`) |
| `nonnegative=` | `true`, `false` (default) | lower limit f ≥ 0 declared on the unknown |
| `solver=` | `lu` (default), `gmres` | monolithic sparse LU, or GMRES on the Newton operator |
| `preconditioner=`, `ilu_drop=` | `lu` (default), `ilu`; number | block factorization per radius of the stored matrix for GMRES, exact or incomplete |
| `refresh=` | `every_step` (default here), `every_iteration`, `on_demand` | when the preconditioner is factorized again |
| `operator=` | `exact` (default), `stored` | GMRES's Newton operator over an approximate stored Jacobian |
| `restart=`, `max_iterations=`, `residual_tolerance=`, `line_search=` | integer, integer, number, boolean | GMRES restart length, Newton iteration limit and residual tolerance, backtracking line search |
| `dt=`, `t_end=`, `saves=` | s, s, integer | time step, end time, number of saved states |
| `output=` | file name | HDF5 output of the solution |
| `checkpoint=`, `checkpoint_dir=`, `checkpoint_interval=` | name, directory, s | checkpoints and segment outputs of a run of that name (`relaxation.jl`, `cooling.jl`) |
| `resume=`, `overwrite=` | `true`, `false` (default) | continuation and replacement of the segment files |
| `stop_file=`, `wall_budget=`, `slack=` | file name, s, s | the stop of a segment |
| `log=`, `request_file=` | `stdout` or a file name; file name | progress log of the run (`relaxation.jl`, `cooling.jl`), and the file whose appearance asks for an entry |

### Notes on the grids and schemes

- `grid=biuniform_270` is a fully kinetic hot-tail grid of 270 momentum cells to p = 2.5.
  Of these, 70 lie below p = 0.07.
  It has 15 pitch cells, and 10 of them lie within 0.5 rad of ξ = 1.
  It is a choice of this package (see the docs page *Choices of this package*).
- `pitch=theta` is biuniform in the pitch angle, with `nxi_low=` cells within `theta_sep=` radians of ξ = 1.
- The schemes up to `upwind_second_order` are linear in f.
  The rest are flux limiters, which are nonlinear.
- `jacobian=exact` takes the exact Jacobian by automatic differentiation.
- `jacobian=frozen_limiter` freezes the limiter at the iterate (DREAM `AD_INTERP_JACOBIAN_LINEAR`).
- `jacobian=upwind`, `central` and `exponential_fitting` take the Jacobian of that linear scheme.
  `upwind` is DREAM `AD_INTERP_JACOBIAN_UPWIND`.

The first three values of `pmax_boundary=` are DREAM's boundary conditions at `p_max`, and the fourth is the consistent form of the third.

| `pmax_boundary=` | DREAM |
|---|---|
| `f_zero` | `BC_F_0` |
| `constant_flux` | `BC_PHI_CONST` |
| `extrapolated_flux` | `BC_DPHI_CONST` |
| `extrapolated_flux_consistent` | the consistent form of `BC_DPHI_CONST` |

### Notes on the solver and the run

- `operator=exact` uses the approximate stored Jacobian only as a preconditioner.
- `operator=stored` iterates with the stored matrix, a quasi-Newton iteration as in DREAM.
- `checkpoint_interval=` is 1800 s by default.
- `resume=true` continues from the last checkpoint when there is one.
- `overwrite=true` lets a run replace the segment files of an earlier one.
- The segment stops at a time point when the file of `stop_file=` appears or the budget of `wall_budget=` runs out.
- In a Slurm job with checkpoints, the end of the job (`SLURM_JOB_END_TIME`) is the budget.
- `log=` with a file name appends the log to that file.

### The lower limit

`nonnegative=true` follows DREAM's rule.

- A Newton step may remove at most 90% of an entry's distance to zero.
- A step that would push an entry already at zero below it fails and is retried with half the time step.

A Maxwell-Jüttner tail underflows to exact zeros, so with this switch a run rejects and halves many steps.
This is the rule at work, not a fault.

## Parameter values and schemes

Parameter values come from DREAM's examples or are choices of this package (see the docs page *Choices of this package*), cited in each script.

The default scheme, exponential fitting, keeps f non-negative at every resolution.
It is first order where a smooth tail is driven by the field.
So field-driven quantities need a check against a finer `np=`: the runaway fraction of `relaxation.jl` at large `e_over_ec`, and the loss through `p_max` in `cooling.jl`.
The limiters converge faster there and cost more per step.
DREAM's default is the central value (`AD_INTERP_CENTRED` in DREAM's `py/DREAM/Settings/Equations/DistributionFunction.py`), and its other choices include the flux limiters.
