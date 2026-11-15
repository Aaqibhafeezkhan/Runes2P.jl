# Getting started

```@meta
CurrentModule = Runes2P
```

This page covers the installation, a first run with the examples, the units of the inputs, the settings of a run and its output files.

## Installation

Runes2P needs Julia 1.13.
It is not registered in the General registry and is added from its repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/osrep/Runes2P.jl")
```

The HDF5 output is a package extension.
It loads with `using HDF5` next to `using Runes2P` and defines [`write_output`](@ref) and [`read_output`](@ref).
Checkpoints need it loaded.

The examples run from a checkout of the repository.
Their environment, `examples/Project.toml`, takes Runes2P from the parent directory and adds HDF5:

```sh
git clone https://github.com/osrep/Runes2P.jl
cd Runes2P.jl
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

The scripts are described in [The examples](#The-examples), and the arguments such as `scheme=` in [Switches](#Switches).

## The examples

Each script solves one kinetic equation for the electron distribution ``f(p, \xi, r)`` and runs without arguments.
Its header lists its default values, the sources of these values and further command lines.

| Script | What it integrates | What it reports |
|---|---|---|
| `relaxation.jl` | a Maxwell-Jüttner distribution in a fixed plasma and field | the density fraction above `p_star=`, when given |
| `cooling.jl` | a thermal quench that starts in the slide-away regime | the saves at which ``f`` has negative cells, and at each the lowest negative momentum cell |
| `avalanche.jl` | knock-on multiplication of a Maxwell-Jüttner seed | the mean growth rate of the density over the run |

### `relaxation.jl`

- The initial state is a Maxwell-Jüttner distribution at the plasma temperature, or at `T0=`.
- The plasma is deuterium, and the plasma and the field stay fixed.
- The operator is that of [`fokker_planck_operator`](@ref).
- `p_star=` adds the fraction of the density above that momentum at every save.
- The plasma values are choices of this package ([Choices of this package](choices.md)).
- By default the field lies between the critical and the Dreicer field ([Choices of this package](choices.md)).

### `cooling.jl`

- The temperature decays exponentially from the initial temperature `T=` to `T_final=` over `t_end=`.
- [`prescribed_fokker_planck_term`](@ref) rebuilds the collision coefficients from the temperature.
- The field stays at `e_over_ec=` times the critical field at the initial temperature.
- With `nr=` above one, radial diffusion along stochastic field lines couples the radii ([`add_rechester_rosenbluth_diffusion!`](@ref)).
  Its strength is `dbb=` (``\delta B/B``), and ``f = 0`` at the wall.
  The diffusion is that of A. B. Rechester and M. N. Rosenbluth, Phys. Rev. Lett. 40, 38 (1978), doi:10.1103/PhysRevLett.40.38.
- The plasma is that of `reference/dream/cases/cooling_positivity.py`, cooled over 2 ms where that case takes 1 ms.
- The radial values are those of DREAM's SPI example.
- The grid and the time step are a choice of this package.
- With the default scheme, check the loss through `p_max` against a finer `np=` ([Exponential fitting](model/positivity.md#Exponential-fitting)).

### `avalanche.jl`

- The seed is a Maxwell-Jüttner distribution of temperature 100 keV.
- The knock-on collisions are those of [`KnockOnOperator`](@ref) and [`KnockOnTerm`](@ref).
- The setup is that of `examples/avalanche/generateCylindrical.py` of DREAM's `boltzmann-eqterm` branch.
- It takes that example's values, among them superthermal collisions, the energy-dependent Coulomb logarithm and DREAM's ion-size table.
- The interpolation scheme is the default of this package, where the example uses TCDF (`scheme=tcdf`).

The switch `treatment=` of [`KnockOnTerm`](@ref) sets the time level of the knock-on term.

| `treatment=` | The knock-on term |
|---|---|
| `explicit_primary` (default) | the primary distribution at the previous time level, as in DREAM |
| `explicit` | the whole term at the previous time level |
| `implicit` | the whole term at the new time level |

`implicit` applies the term matrix-free in each Krylov iteration of [`GMRES`](@ref).
The direct solvers do not take it, so with `treatment=implicit` the solver defaults to GMRES, and `solver=gmres` in the command above only repeats that default.

### What every script prints

A run is one segment, or a chain of segments when it is split over several allocations ([Checkpoints, segments and the stop](model/time-stepping.md#Checkpoints,-segments-and-the-stop)).
Every script prints the segment and its status, the solver statistics ([`SolverStatistics`](@ref)) and the negative entries of the saved states.
It then prints the density per radius at every save.
In a GMRES run the statistics include the Krylov iterations and factorizations.
Before the timed run a script integrates over the first three time points, so that the timing leaves out the compilation.
`relaxation.jl` and `cooling.jl` end with [`exit_status`](@ref) as the exit status of the process.

## The parts of a run

The three scripts build a run in the same order.

1. A [`MomentumGrid`](@ref) from momentum faces ([`uniform_p_faces`](@ref), [`biuniform_p_faces`](@ref), [`tanh_p_faces`](@ref)) and pitch faces ([`uniform_xi_faces`](@ref), [`biuniform_theta_xi_faces`](@ref)), combined with a [`RadialGrid`](@ref) by [`cylindrical_grid`](@ref) ([Grids](model/grids.md)).
2. A [`PlasmaState`](@ref) with its [`IonSpecies`](@ref), and the [`CollisionSettings`](@ref) of the collision frequencies ([Collision models](model/collisions.md)).
3. The kinetic operator, an [`AdvectionDiffusion`](@ref) with its interpolation schemes and boundaries ([Positivity and flux limiters](model/positivity.md)).
4. An [`UnknownLayout`](@ref) with one [`KineticUnknown`](@ref) `:f`, and an [`EquationSystem`](@ref) of one [`Equation`](@ref): a [`TimeDerivative`](@ref) and the terms, among them [`FokkerPlanckTerm`](@ref) or [`prescribed_fokker_planck_term`](@ref) ([Unknowns and the equation system](model/unknowns.md)).
5. A [`NewtonSolver`](@ref) with its linear solver, and [`integrate`](@ref) over the time points, which returns a [`TimeSolution`](@ref) ([Time stepping and solvers](model/time-stepping.md)).

The shared helpers of the scripts are in `examples/common.jl`: the switches, a Maxwell-Jüttner initial state normalized with [`density`](@ref), the report and the output.

## Units

The inputs of a run are in SI units, except temperatures and particle energies in eV and momenta in units of ``m_e c``, with ``m_e`` the electron mass and ``c`` the speed of light.
One setting is in minutes, `guard_minutes` of [`ProgressLog`](@ref).
The code checks almost no unit, so a value in another unit usually gives a wrong result without an error.
The table lists the units by group, for the inputs and for the results of the functions it names.

| Group | Quantity | Unit |
|---|---|---|
| plasma | temperatures of [`PlasmaState`](@ref) and of the temperature unknown | eV |
| plasma | electron and ion densities | m⁻³ |
| plasma | ion masses: the result of [`ion_mass`](@ref), `masses` of [`IonEnergyExchangeTerm`](@ref) | kg |
| plasma | charge numbers of [`IonSpecies`](@ref), effective charge, Coulomb logarithm | dimensionless |
| grids | momentum faces, limits of a [`MomentumRegion`](@ref) | ``m_e c`` |
| grids | `p_cutoff` of [`KnockOnOperator`](@ref) and [`RosenbluthPutvinskiSourceTerm`](@ref) | ``m_e c`` |
| grids | pitch ``\xi`` | dimensionless |
| grids | pitch-angle separation of [`biuniform_theta_xi_faces`](@ref) | rad |
| grids | radial faces, minor radius, wall radius, major radius | m |
| time | times, time steps, save times | s |
| time | wall times of [`CheckpointSettings`](@ref) and [`StopCondition`](@ref) | s |
| time | `guard_minutes` of [`ProgressLog`](@ref) | min |
| distribution and moments | distribution ``f``, per momentum volume with ``p`` in ``m_e c`` ([notes](#Notes-on-the-units)) | m⁻³ |
| distribution and moments | density moment ([`density`](@ref)) | m⁻³ |
| distribution and moments | current moment ([`current_density`](@ref)) | A/m² |
| fields and currents | electric field | V/m |
| fields and currents | magnetic field | T |
| fields and currents | current density | A/m² |
| fields and currents | plasma current | A |
| fields and currents | poloidal flux divided by the major radius ([as in DREAM](model/unknowns.md#Pairing-of-rows-and-unknowns)) | V s/m |
| fields and currents | conductivity | S/m |
| energies | particle energies, ionization energies | eV |
| energies | widths of the secondary spectrum ([`SECONDARY_SPECTRUM_WIDTHS`](@ref)) | eV |
| energies | thermal energy densities ([notes](#Notes-on-the-units)) | J/m³ |
| energies | power densities | W/m³ |
| rates | collision frequencies, such as [`slowing_down_frequency`](@ref) | s⁻¹ |
| rates | avalanche growth rate ([`avalanche_growth_rate`](@ref)) | s⁻¹ |
| rates | growth rate in the report of `avalanche.jl` | s⁻¹ |
| rates | Dreicer rate ([`connor_hastie_rate`](@ref)) | m⁻³ s⁻¹ |
| rates | `rate` of [`MaxwellianSourceTerm`](@ref) and [`LossRateTerm`](@ref) | s⁻¹ |
| transport | radial diffusivities | m²/s |
| transport | relative perturbation ``\delta B/B`` of the magnetic field | dimensionless |
| atomic data | rate coefficients of ionization, recombination and charge exchange | m³/s |
| atomic data | coefficients of the radiated power | W m³ |

The units of the unknowns that the package names by default are on the page [Unknowns and the equation system](model/unknowns.md#Units-of-the-unknowns).
The output files store the unit of every unknown, of the times and of the grid faces ([Unknowns](model/output.md#Unknowns)).
The settings record stores no unit with its values ([Records](model/output.md#Records)).

### Notes on the units

These notes explain the distribution, the ion masses, the thermal energy densities, the radial totals and the atomic data that the package reads.

The distribution ``f`` is a density per unit of momentum volume, with the momentum ``p`` in units of ``m_e c``.
So ``f`` is in m⁻³.
Its density moment at a radius is ``n = \int 2\pi p^2 f \, \mathrm{d}p \, \mathrm{d}\xi`` ([`density`](@ref)).
The tests against DREAM read DREAM's `f_hot` and `f_re` without a conversion factor and compare them with ``f`` value by value.

[`IonEnergyExchangeTerm`](@ref) takes the masses of the species from [`ion_mass`](@ref).
Its keyword `masses` replaces them all, with one mass per species in kg.

A thermal energy density is ``W = \tfrac32 e\, n T`` in J/m³, with ``e`` the elementary charge ([The energy of the cold electrons](model/energy-current.md#The-energy-of-the-cold-electrons)).
The factor ``e`` converts the temperature ``T`` from eV to J ([`THERMAL_ENERGY_FACTOR`](@ref)).

[`radial_particle_content`](@ref) gives ``n V' \Delta r`` at each radius, in m⁻¹.
Its terms are the density ``n`` in m⁻³, the spatial Jacobian ``V' = r`` of a cylinder in m and the radial cell width ``\Delta r`` in m.
Its sum over the radii is [`total_particles`](@ref).
The number of electrons of ``f`` per metre of the cylinder is ``2\pi`` times `total_particles`.

The package converts the atomic data that come in other units.
[`read_adf11`](@ref) converts the densities of an ADAS file from cm⁻³ to m⁻³ and its coefficients from cm³ to m³ ([ADAS rate coefficients](model/atomic-data.md#ADAS-rate-coefficients)).
The fits of CHIANTI ([`CHIANTI_IONIZATION_FITS`](@ref)) keep the units of their files.
These are eV for the shell energies and 10⁻¹⁴ cm² eV² for the scaled cross sections ([`ChiantiDirectShell`](@ref)), and Ry for the excitation energies ([`ChiantiAutoionization`](@ref)).
[`chianti_cross_section`](@ref) gives the cross sections computed from them in m².

## Switches

A script takes `name=value` arguments, and every switch has a default.
The switches of `common.jl` are shared, and a script's header names the defaults it changes.

| Switches | Values | Sets |
|---|---|---|
| `grid=` | `uniform` (default), `biuniform`, `tanh`, `biuniform_270` | the momentum faces ([notes](#Notes-on-the-grids-and-schemes)) |
| `np=`, `p_max=` | integer, ``m_e c`` | the momentum cells and the extent |
| `np_low=`, `p_sep=` | integer, ``m_e c`` | the cells below the split of the biuniform and tanh grids, and its position |
| `pitch=` | `uniform` (default), `theta` (biuniform in the pitch angle) | the pitch faces |
| `nxi=` | integer | the pitch cells |
| `nxi_low=`, `theta_sep=` | integer, rad | the cells of `pitch=theta` within `theta_sep=` of ``\xi = 1`` |
| `T=`, `n=`, `B=` | eV, m⁻³, T | the deuterium plasma, with `n=` the density of its electrons and ions |
| `n_ar=` | m⁻³ | neutral argon added to the plasma (`relaxation.jl`) |
| `T0=` | eV | the temperature of the initial Maxwell-Jüttner distribution of `relaxation.jl`, the plasma temperature by default |
| `p_star=` | ``m_e c`` | the momentum above which `relaxation.jl` reports the density fraction |
| `T_final=` | eV | the final temperature of `cooling.jl`, which starts at `T=` |
| `nr=` | integer | the radii of `cooling.jl` |
| `dbb=` | dimensionless | ``\delta B/B`` of the radial diffusion of `cooling.jl` with `nr=` above one |
| `E=` or `e_over_ec=` | V/m, or multiples of [`critical_field`](@ref) of `n=` at the [`thermal_coulomb_logarithm`](@ref) | the electric field |
| `mode=` | `full` (default of `relaxation.jl` and `cooling.jl`), `superthermal`, `ultra_relativistic` | the collision frequency mode ([`CollisionSettings`](@ref)) |
| `screening=` | `partial` (default), `complete`, `none` | the screening |
| `lnlambda=` | `thermal` (default), `energy_dependent`, `relativistic` | the Coulomb logarithm |
| `ion_size_table=` | `hesslow` (default), `dream` | the table of ion sizes |
| `synchrotron=` | `true`, `false` (default) | the synchrotron loss of `relaxation.jl` |
| `scheme=`, `p_scheme=`, `xi_scheme=` | `exponential_fitting` (default), `central`, `upwind`, `quick`, `upwind_second_order`, `van_albada`, `tcdf`, `muscl`, `smart`, `ospre` | the interpolation scheme in both directions or in one ([`AdvectionDiffusion`](@ref), [notes](#Notes-on-the-grids-and-schemes)) |
| `jacobian=` | `exact` (default), `frozen_limiter`, `upwind`, `central`, `exponential_fitting` | the stored Jacobian of a limiter ([`FokkerPlanckTerm`](@ref), [notes](#Notes-on-the-grids-and-schemes)) |
| `pmax_boundary=` | `f_zero` (default), `constant_flux`, `extrapolated_flux` (needs a `p_scheme=` other than the default), `extrapolated_flux_consistent`, `outflow` | the boundary at `p_max` ([`AdvectionDiffusion`](@ref), [notes](#Notes-on-the-grids-and-schemes)) |
| `nonnegative=` | `true`, `false` (default) | a [`LowerLimit`](@ref) ``f \ge 0`` on the unknown (`relaxation.jl`, `cooling.jl`, [Lower limits](#Lower-limits)) |
| `solver=` | `lu` (default), `gmres` | the monolithic sparse LU of [`SparseLU`](@ref), or [`GMRES`](@ref) on the Newton operator |
| `preconditioner=` | `lu` (default), `ilu` | the factorization of the stored matrix per radius for GMRES, [`RadiusBlockLU`](@ref) or [`RadiusBlockILU`](@ref) ([notes](#Notes-on-the-solver)) |
| `ilu_drop=` | number | the absolute drop tolerance of [`RadiusBlockILU`](@ref), on the factors of the scaled matrix |
| `refresh=` | `every_step` (default), `every_iteration`, `on_demand` | when the preconditioner is factorized again ([notes](#Notes-on-the-solver)) |
| `operator=` | `exact` (default), `stored` | the Newton operator of GMRES ([notes](#Notes-on-the-solver)) |
| `restart=` | integer | the restart length of GMRES ([notes](#Notes-on-the-solver)) |
| `max_iterations=`, `residual_tolerance=` | integer, number | the Newton iterations and the residual tolerance, relative to the scale of each unknown ([`NewtonSolver`](@ref)) |
| `line_search=` | boolean | the backtracking line search of [`NewtonSolver`](@ref) |
| `dt=`, `t_end=` | s, s | the time points of [`integrate`](@ref) |
| `saves=` | integer | the number of save times |
| `output=` | file name | the HDF5 output, written with [`write_output`](@ref) |
| `checkpoint=`, `checkpoint_dir=` | name, directory | the [`CheckpointSettings`](@ref) of a run of that name |
| `checkpoint_interval=` | s | the wall time between checkpoints, 1800 s by default |
| `resume=` | `true`, `false` (default) | the continuation from the last checkpoint, when there is one |
| `overwrite=` | `true`, `false` (default) | whether a run may replace the segment files of an earlier one |
| `stop_file=`, `wall_budget=`, `slack=` | file name, s, s | the [`StopCondition`](@ref) of a segment ([The stop of a segment](#The-stop-of-a-segment)) |
| `log=` | `stdout` or a file name | the [`ProgressLog`](@ref), appended to the named file |
| `request_file=` | file name | the file whose appearance asks the log for an entry |

### Notes on the grids and schemes

These notes belong to the rows `grid=`, `scheme=`, `jacobian=` and `pmax_boundary=` of the table above.
Momenta are in units of ``m_e c`` ([Units](#Units)).

`grid=biuniform_270` is a fully kinetic hot-tail grid and a choice of this package ([Choices of this package](choices.md)).
It ignores the other grid switches.

| `grid=biuniform_270` | Value |
|---|---|
| momentum cells | 270 |
| of them below ``p = 0.07`` | 70 |
| upper end | ``p = 2.5`` |
| pitch cells | 15 |
| of them within 0.5 rad of ``\xi = 1`` | 10 |

The schemes up to `upwind_second_order` are linear in ``f``.
The rest are flux limiters, which are nonlinear.
The default scheme, exponential fitting, is a choice of this package.
Its properties and the other schemes are on the page [Positivity and flux limiters](model/positivity.md).

DREAM's default is `AD_INTERP_CENTRED` (DREAM's `py/DREAM/Settings/Equations/DistributionFunction.py`).
`scheme=central` matches it ([Defaults that differ](dream-comparison.md#Defaults-that-differ)).
DREAM's other choices include the flux limiters ([Comparison with DREAM](dream-comparison.md#Kinetic-options)).

The stored Jacobian, or stored matrix, is the Jacobian of the implicit terms that have a stored Jacobian, which the solver assembles ([The step](model/time-stepping.md#The-step)).
[`SparseLU`](@ref) factorizes the Newton matrix built from it, and the preconditioners of [`GMRES`](@ref) factorize the stored matrix in blocks ([Linear solvers and preconditioners](model/time-stepping.md#Linear-solvers-and-preconditioners)).

- `jacobian=exact` takes the exact Jacobian by automatic differentiation.
- `jacobian=frozen_limiter` freezes the limiter at the iterate (DREAM `AD_INTERP_JACOBIAN_LINEAR`).
- `jacobian=upwind`, `central` and `exponential_fitting` take the Jacobian of that linear scheme.
  `upwind` is DREAM `AD_INTERP_JACOBIAN_UPWIND`.

| `pmax_boundary=` | DREAM |
|---|---|
| `f_zero` | `BC_F_0` |
| `constant_flux` | `BC_PHI_CONST` |
| `extrapolated_flux` | `BC_DPHI_CONST` as DREAM's code has it |
| `extrapolated_flux_consistent` | `BC_DPHI_CONST` as DREAM defines it |
| `outflow` | — |

`extrapolated_flux_consistent` extrapolates the flux with the ``\mathcal{V}'`` and the flux of each face, as DREAM's definition of `BC_DPHI_CONST` states ([Comparison with DREAM](dream-comparison.md#Extrapolated-flux-at-the-highest-momentum)).
`outflow` lets only the advection leave the grid, and [`AdvectionDiffusion`](@ref) names no DREAM condition for it.

### Notes on the solver

These notes belong to the rows `preconditioner=`, `refresh=`, `restart=` and `operator=` of the table above.
The examples set three keywords of [`GMRES`](@ref) differently from its defaults.

| Switch | Keyword of [`GMRES`](@ref) | Default of the examples | Default of [`GMRES`](@ref) |
|---|---|---|---|
| `preconditioner=` | `preconditioner` | [`RadiusBlockLU`](@ref) | [`FluidSchurLU`](@ref) |
| `refresh=` | `preconditioner_refresh` | `:every_step` | `:every_iteration` |
| `restart=` | `restart_length` | 30 | 50 |

`operator=` matters only for a term whose block of the stored Jacobian approximates its Jacobian ([`jacobian_kind`](@ref)), such as a limiter with a `jacobian=` other than `exact`.
Without such terms the two values are the same.
With `operator=exact` the approximate block only preconditions the exact Newton operator.
`operator=stored` iterates with the stored matrix, a quasi-Newton iteration as in DREAM.

### The switches of `avalanche.jl`

`avalanche.jl` builds its own grid and plasma, and it takes no checkpoint, stop or log switches.
Of the shared switches it reads the schemes, `pmax_boundary=`, `jacobian=`, the solver switches, `dt=`, `t_end=`, `saves=` and `output=`.
It reads `mode=`, `screening=`, `lnlambda=` and `ion_size_table=` with the values of its DREAM example as defaults.
Its own switches are these.

| Switch | Sets |
|---|---|
| `treatment=` | the time level of the knock-on term ([`avalanche.jl`](#avalanche.jl)) |
| `cutoff=` | the knock-on cutoff momentum, in units of ``m_e c`` |
| `np=`, `np_low=` | the momentum cells, and those below the fixed split at ``p = 0.7`` |
| `nxi=` | the cells of its own pitch grid, finer towards ``\xi = 1`` |
| `e_over_ec=` | the field in multiples of the critical field of the total electron density |

### Lower limits

`nonnegative=true` declares a [`LowerLimit`](@ref) of value 0 and tolerance 0 on ``f``.
[`NewtonSolver`](@ref) shortens the Newton update so that no limited entry loses more than the fraction `max_reduction` of its distance to its limit in one iteration.
The default `max_reduction = 0.9` is DREAM's value ([Lower limits of the Newton step](model/positivity.md#Lower-limits-of-the-Newton-step)).

A step that would push an entry already at its limit further down fails with `:below_lower_limit`.
[`FixedSteps`](@ref) then retries the step with half the time step.
DREAM skips such entries instead.

At a low temperature on a wide grid, the Maxwell-Jüttner tail underflows to exact zeros, so such a run rejects and halves many steps.
Shorter steps do not remove a round-off negative update of an entry at zero.
A positive tolerance of the [`LowerLimit`](@ref) does.
No switch of the examples sets the tolerance or `max_reduction`.

### The stop of a segment

A segment stops after an accepted step when the file of `stop_file=` exists ([`StopCondition`](@ref), [Checkpoints, segments and the stop](model/time-stepping.md#Checkpoints,-segments-and-the-stop)).
It also stops when the wall-clock time left until its end is less than a margin.
That end is the end of `wall_budget=` or the end of the Slurm job.
The margin is the longest accepted step of the segment so far, plus the longest write of a checkpoint with its output, plus `slack=` seconds.
`wall_budget=` counts its seconds from the start of the timed run, after the warm-up over the first three time points ([What every script prints](#What-every-script-prints)).

In a Slurm job, [`StopCondition`](@ref) takes `SLURM_JOB_END_TIME` as the end of the job.
A run with `checkpoint=` stops before the end of the job without further switches.
Without `checkpoint=`, the end of the job counts only when `stop_file=` or `wall_budget=` is given.
With `wall_budget=` in a Slurm job, the earlier of the end of `wall_budget=` and the end of the job counts.

## The settings of a run

[`integrate`](@ref) records the settings of the run with [`run_settings`](@ref) before the first step and returns them in `settings` of the [`TimeSolution`](@ref).
Each object of the package the run is built from has a [`describe`](@ref) record, and the record collects them by section.

- The sections of the setup are `grids`, `layout`, `system`, `updates`, `prescribed`, `solver`, `time` and `run`.
  Each has a [`settings_hash`](@ref), and the record has a hash of them together.
- `run` holds the `description` that a script passes to [`integrate`](@ref), such as the name of the example and the origin of its initial state.
- The sections `code`, `environment`, `checkpoint`, `stop` and `log` are outside the hashes.
- `code` holds the [`code_version`](@ref): the commit of Runes2P when it runs from a git checkout, as the examples do, and `"unknown"` for an installed package.
- `segment` holds the number, the start, the status and the reason of a stop of this segment.
- `complete` is false when some object of the run has no record of its own, such as a term defined outside the package.

The record is taken before a stored state is restored, so restoring does not change it.
A resumed segment whose setup differs lists the changed sections in its `segment` record.

## Output files

`output=run.h5` writes the solution with [`write_output`](@ref): the saved states per unknown, the layout, the solver statistics and the settings record.
[`read_output`](@ref) reads it back as a [`TimeSolution`](@ref), and [`read_settings`](@ref) reads the settings record alone.

`checkpoint=name` writes, in `checkpoint_dir=`, the files that [`CheckpointSettings`](@ref) names.

- `name.checkpoint.h5`, the last checkpoint, and `name.checkpoint.prev.h5`, the one before it.
- `name.segment<k>.h5` (`k` with four digits), the output of segment `k`.

A run that stops writes its checkpoint and exits with [`exit_status`](@ref), so that a batch script can start the next segment with `resume=true`.
[`read_output`](@ref) with the [`CheckpointSettings`](@ref) of the run joins its segments into one solution:

```julia
using Runes2P, HDF5
solution = read_output(CheckpointSettings("cooling"; directory = "run"))
f_end = unknown_view(solution.states[:, end], solution.layout, :f)   # f at the last save
```

The format of these files is on the page [Output files](model/output.md).
