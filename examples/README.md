# Examples

Basic runs of the methods in the code, each a script with `name=value` switches on the command line and defaults that run as they are.
The environment is `examples/Project.toml` (Runes2P from the parent directory, HDF5 for the output):

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

Each script prints the solver statistics (steps, Newton and Krylov iterations, factorizations, steps accepted at the residual floor or as a linear system), the negative entries of the saved states and the density at every save.
`output=run.h5` writes the solution with `write_output` (layout, saved states per unknown, solver statistics, and the settings of the run with the commit of Runes2P).

| Script | Run |
|---|---|
| `relaxation.jl` | a Maxwellian under a fixed plasma and field (Dreicer regime by default); `p_star=` adds the fraction above a momentum, `T0=` a seed temperature different from the plasma's |
| `cooling.jl` | the temperature decays from `T=` to `T_final=` over `t_end=`, the collision coefficients follow it; `nr=` radii coupled by radial diffusion `dbb=`; reports where a scheme goes negative |
| `avalanche.jl` | knock-on multiplication of a 100 keV seed on DREAM's avalanche example; `treatment=explicit|implicit`, `cutoff=`; reports the growth rate |

The switches, shared through `common.jl` (a script's own defaults are in its header):

| Switch | Values | What it sets |
|---|---|---|
| `grid=` | `uniform` (default), `biuniform`, `tanh`, `production` | momentum grid type; `production` is a fully kinetic hot-tail grid at production resolution (270 cells to 2.5, 15 pitch cells) |
| `np=`, `p_max=` | integer, number | momentum cells and extent |
| `np_low=`, `p_sep=` | integer, number | cells below and position of the split of the biuniform and tanh grids |
| `pitch=`, `nxi=` | `uniform` (default) or `theta`; integer | pitch grid type and cells; `theta` is biuniform in the pitch angle with `nxi_low=` cells within `theta_sep=` radians of ξ = 1 |
| `scheme=` | `exponential_fitting` (default), `central`, `upwind`, `quick`, `upwind_second_order`, `van_albada`, `tcdf`, `muscl`, `smart`, `ospre` | interpolation scheme in both directions; linear in f up to the second-order upwind, the rest are flux limiters (nonlinear) |
| `p_scheme=`, `xi_scheme=` | as `scheme=` | one direction |
| `jacobian=` | `exact` (default), `frozen_limiter`, `upwind`, `central`, `exponential_fitting` | the stored Jacobian of a limiter: exact AD, the limiter frozen at the iterate (DREAM `AD_INTERP_JACOBIAN_LINEAR`), or a linear scheme's (upwind is DREAM `AD_INTERP_JACOBIAN_UPWIND`) |
| `pmax_boundary=` | `f_zero` (default), `constant_flux`, `extrapolated_flux`, `extrapolated_flux_consistent`, `outflow` | boundary at `p_max` (DREAM `BC_F_0`, `BC_PHI_CONST`, `BC_DPHI_CONST`, its consistent form) |
| `mode=` | `full` (default of `relaxation.jl` and `cooling.jl`), `superthermal`, `ultra_relativistic` | collision frequency mode |
| `screening=`, `lnlambda=`, `ion_size_table=` | `partial` (default), `complete`, `none`; `thermal` (default), `energy_dependent`, `relativistic`; `hesslow` (default), `dream` | collision settings (`avalanche.jl` uses its example's values) |
| `T=`, `n=`, `B=` | eV, m⁻³, T | plasma |
| `E=` or `e_over_ec=` | V/m, or multiples of the critical field | electric field |
| `n_ar=` | m⁻³ | neutral argon added to the deuterium plasma |
| `synchrotron=` | `true`, `false` (default) | synchrotron losses (`relaxation.jl`) |
| `nonnegative=` | `true`, `false` (default) | lower limit f ≥ 0 declared on the unknown (DREAM's rule: a Newton step may remove at most 90% of an entry's distance to zero, and a step that would push an entry already at zero below it fails and is retried with half the time step). A Maxwellian tail underflows to exact zeros, so with this switch a run rejects and halves many steps (334 accepted of 355 attempted for 10 nominal steps in a test run); it is the rule at work, not a fault |
| `solver=` | `lu` (default), `gmres` | monolithic sparse LU, or GMRES on the Newton operator |
| `preconditioner=`, `ilu_drop=` | `lu` (default), `ilu`; number | block factorization per radius of the stored matrix for GMRES, exact or incomplete |
| `refresh=` | `every_step` (default here), `every_iteration`, `on_demand` | when the preconditioner is factorized again |
| `operator=` | `exact` (default), `stored` | GMRES's Newton operator over an approximate stored Jacobian: exact (the approximation only preconditions) or the stored matrix (quasi-Newton, DREAM's iteration) |
| `restart=`, `max_iterations=`, `residual_tolerance=`, `line_search=` | integer, integer, number, boolean | GMRES restart length, Newton iteration limit and residual tolerance, backtracking line search |
| `dt=`, `t_end=`, `saves=` | s, s, integer | time step, end time, number of saved states |
| `output=` | file name | HDF5 output of the solution |

Parameter values come from DREAM's examples or, for the cooling plasma, a generic test case; each script names its source.
The default scheme, exponential fitting, keeps f non-negative at every resolution and is first order where a smooth tail is driven by the field, so field-driven quantities (the runaway fraction of `relaxation.jl` at large `e_over_ec`, the loss through `p_max` in `cooling.jl`) need a check against a finer `np=` (see the README of the repository, "DREAM compatibility and departures").
The limiters converge faster there and cost about twice per step.
