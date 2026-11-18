# Departures from DREAM

```@meta
CurrentModule = Runes2P
```

This page lists the defaults that differ from DREAM's, with the options that give DREAM's behaviour, and the behaviour without an option for DREAM's.
The last three sections give the details of the extrapolated flux at the highest momentum, the approximate Jacobians and the line search.
The page [Comparison with DREAM](dream-comparison.md) gives the features of DREAM and their state in Runes2P, and the version of DREAM that the file and line citations below refer to.

## Defaults that differ

Each row names the default of Runes2P and the option that gives DREAM's behaviour.
The reasons of the defaults that are choices of this package are on [Choices of this package](choices.md).
A cell that ends with "(note)" has a note under its table, which gives the detail.

| Topic | Runes2P default | DREAM | Option for DREAM's behaviour |
|---|---|---|---|
| interpolation of ``f`` in ``p`` and ``\xi`` | exponential fitting in [`fokker_planck_operator`](@ref) | centred interpolation | `scheme = :central` |
| boundary at ``p_\mathrm{max}`` | `p_upper_boundary = :f_zero` in [`fokker_planck_operator`](@ref) | `BC_PHI_CONST` | `p_upper_boundary = :constant_flux` |
| synchrotron loss | on | off | `synchrotron = false` |
| bremsstrahlung stopping power | on | off | `bremsstrahlung = :neglect` |
| bremsstrahlung in the energy diffusion | a pure drag | the whole ``ν_s`` diffused | `bremsstrahlung_diffusion = :dream` |
| constant of the stopping power | ``4/3`` of the formula 4BN of Koch and Motz | ``1`` | `bremsstrahlung_constant = :dream` |
| ion sizes of partial screening | Table 1 of Hesslow et al. (note) | a lookup that shifts the sizes from Ar¹⁷⁺ on | `ion_size_table = :dream` |
| momenta of partial screening | above the excitation energy of each charge state | all momenta | `screening_window = :dream` |
| knock-on pitch | integrated over the cell pair, primary averaged over its cell | both at the cell centres | `dream_compatible = true` of [`KnockOnOperator`](@ref) |
| current of a distribution | ``f`` linear across a pitch cell | ``f`` constant over a pitch cell | `pitch_rule = :cell_center` |
| flux between the two grids | by the overlap cells of the pitch grids | interpolated in pitch | `pitch_mapping = :dream` |
| flux through ``p = 0`` on a runaway grid from ``p = 0`` | `p_zero_boundary = :outflow` of [`fokker_planck_operator`](@ref) (note) | none | `p_zero_boundary = :zero_flux`, set by the caller |
| Rechester–Rosenbluth coefficient ``D = \pi q R_0 v_\parallel (\delta B/B)^2`` | safety factor and major radius as inputs | ``q = 1`` and ``R_0 = 1`` m in a cylinder | `safety_factor = 1`, `major_radius = 1` |
| heating of the cold electrons by a runaway grid | [`free_electron_energy_transfer_term`](@ref) (note) | the moment ``m_e c\, v p\, \nu_s f`` at the cell centres | [`FrictionEnergyTransferTerm`](@ref) in place of the default |
| radiated power of the ions | line and recombination radiation, the binding energy of ionization and recombination | line radiation, the ionization loss and Gould's bremsstrahlung | `radiation = :dream` of [`RadiatedPowerTerm`](@ref), with `bremsstrahlung_corrections = :dream` for DREAM's corrections |
| relativistic corrections of Gould's bremsstrahlung, the coefficients of ``T/m_ec^2`` | ``11/24`` and ``5(44 - 3\pi^2)/(24\sqrt{2})`` ([`GOULD_ION_CORRECTION`](@ref), [`GOULD_ELECTRON_CORRECTION`](@ref)) (note) | ``19/24`` and ``5(44 - 3\pi^2)/(8\sqrt{2})`` | `bremsstrahlung_corrections = :dream` |
| cross sections of the kinetic ionization | CHIANTI's for H, D, T, Ne and Ar, DREAM's for the other elements | DREAM's fits | `form = :dream` |
| ionization at the hot threshold | the fluid coefficient below the threshold, `kinetic_ionization` of [`IonRateTerm`](@ref) | the whole ADAS rate next to the kinetic term | [`IonRateTerm`](@ref) without `kinetic_ionization` |
| Coulomb logarithm among the ions | the ion-ion logarithm of the NRL Plasma Formulary, with the temperature of each species | one logarithm with ``T_e`` | `ion_coulomb_logarithm = :dream` of [`IonEnergyExchangeTerm`](@ref) |
| ion masses | nuclear masses of H, D and T, standard atomic weights otherwise | standard atomic weights up to ``Z = 40``, ``2.3\,Z`` u beyond | `mass_table = :dream` |
| charge states in the collisions of ``f`` | those of the start of the step, `lagged = :ions` | those of the iterate with their Jacobian columns | `lagged = ()` |
| Newton matrix of a linear system | assembled and factorized once per step | every iteration | `reuse_linear_matrix = false` of [`NewtonSolver`](@ref) |
| negative entries of a kinetic unknown at convergence | one more Newton iteration for the schemes of [`POLISHED_SCHEMES`](@ref) | none | `polish_negatives = false` of [`NewtonSolver`](@ref) |
| iterative refinement of MKL PARDISO | none, `refinement_steps = 0` of [`MKLPardiso`](@ref) | up to 10 steps per solve (`fvm/Solvers/MIMKL.cpp`) | `MKLPardiso(refinement_steps = 10)` |
| floor of a temperature unknown | [`DEFAULT_TEMPERATURE_FLOOR`](@ref) ([`fluid_temperature_floors`](@ref)) | damped towards zero | a lower floor of [`PlasmaFromUnknowns`](@ref) or a [`LowerLimit`](@ref) |

The sources of the values of Runes2P in the table are these.

| Row | Source |
|---|---|
| constant of the stopping power | the formula 4BN of H. W. Koch and J. W. Motz, Rev. Mod. Phys. 31, 920 (1959), [doi:10.1103/RevModPhys.31.920](https://doi.org/10.1103/RevModPhys.31.920) |
| ion sizes of partial screening | Table 1 of L. Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), [doi:10.1017/S0022377818001113](https://doi.org/10.1017/S0022377818001113) |
| relativistic corrections of Gould's bremsstrahlung, electron-electron | the ratio of eqs. (22) and (18) of R. J. Gould, Astrophys. J. 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068) |
| relativistic corrections of Gould's bremsstrahlung, electron-ion | the erratum of the paper of 1980, Astrophys. J. 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634) |
| Coulomb logarithm among the ions | the NRL Plasma Formulary, A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019 |

**Ion sizes of partial screening.**
DREAM's lookup has a list of nuclear charges with 17 argon entries for 18 charge states.
This shifts the sizes from Ar¹⁷⁺ on.

**Flux through ``p = 0`` on a runaway grid from ``p = 0``.**
Runes2P takes the operator's default, `p_zero_boundary = :outflow` of [`fokker_planck_operator`](@ref), as DREAM's hot-tail grid.
The friction carries electrons out by ``\lim_{p \to 0} p^3 ν_s`` ([`p3_slowing_down_frequency_at_zero`](@ref)).
With it the runaway grid of the case `heating` of `reference/dream/cases/initial_state.py` loses 1.55e-4 of the density of ``f_\mathrm{re}`` in the first step, where DREAM's stays constant to round-off.

The caller sets the option `p_zero_boundary = :zero_flux` for a runaway grid from ``p = 0``, as both `compat` values of `runaway_heating_example` do (`test_runaway_heating_dream.jl`).
In DREAM's source:

- ``f_\mathrm{re}`` has no internal boundary condition (`src/Settings/Equations/f_re.cpp:87`).
- ``f_\mathrm{re}`` keeps the Dirichlet lower boundary of its advection in ``p`` (`fvm/Equation/AdvectionTerm.cpp:36`).
- Unlike ``f_\mathrm{re}``, ``f_\mathrm{hot}`` has an internal boundary condition (`src/Settings/Equations/f_hot.cpp:96`).
  With it, the lower boundary of its advection in ``p`` is mirrored (`AD_BC_MIRRORED`) in place of Dirichlet (`src/Settings/Equations/distribution.cpp:234-237`).

**Heating of the cold electrons by a runaway grid.**
Runes2P's default is the energy change of the electron-electron operator of the grid ([`KineticEnergyLossTerm`](@ref)), whose momentum faces take the upwind side of the whole advection under a scheme with a side, such as `:upwind`.
That operator takes no flux through ``p_\mathrm{max}``, so the electrons that cross it give no energy to the cold electrons.
DREAM's moment, [`FrictionEnergyTransferTerm`](@ref), also counts the energy that the friction carries out of the grid through ``p = 0`` where that flux is applied (`p_zero_boundary = :outflow`).
The test case `runaway_heating_example` (`test/example_cases.jl`) takes the option at `compat = :dream`.
It is compared with DREAM's run in `test_runaway_heating_dream.jl`.

The case sets `:upwind` at both `compat` values, from DREAM's script, and the side matters at `compat = :runes2p` alone.
With cells 0.5 ``m_e c`` wide and nearly all of ``f_\mathrm{re}`` in the first, the field beats the friction at the face above it.
The energy change is then 1.39e5 W/m³ at the start, 1.772 to 1.779 times DREAM's moment on the same states.
The page [The side on a coarse momentum grid](model/energy-current.md#The-side-on-a-coarse-momentum-grid) gives 2.127, as it divides by the friction moment with the collision settings of the grid, which is 0.837 of DREAM's moment.

At the end of the 5 µs, (Runes2P − DREAM) / DREAM of ``T_\mathrm{cold}`` is +2.8e-4 (`test_runaway_heating_dream.jl`).
The extra heating of about 0.30 J/m³ over the heat capacity of 12 J/(m³ eV) gives 2.5e-4 of it.
The timing of the ohmic heating gives 3e-5.

**Relativistic corrections of Gould's bremsstrahlung.**
The coefficients of ``T/m_ec^2`` are these.

| Coefficient | Runes2P | DREAM |
|---|---|---|
| electron-ion | ``11/24`` ([`GOULD_ION_CORRECTION`](@ref)) | ``19/24`` |
| electron-electron | ``5(44 - 3\pi^2)/(24\sqrt{2})`` ([`GOULD_ELECTRON_CORRECTION`](@ref)) | ``5(44 - 3\pi^2)/(8\sqrt{2})`` |

## Behaviour without an option for DREAM's

In these points Runes2P differs from DREAM, and no option gives DREAM's behaviour.

- Newton's method converges on both the update and the scaled residual, except for the linear steps that `linear_acceptance` accepts on the update ([`NewtonSolver`](@ref)).
  A stiff step at the round-off floor of its residual is accepted on the update too, within the round-off bound of its rows or where the residual stopped decreasing.
  DREAM converges on the update.
  With `setCheckResidual` on, its default, DREAM warns when the residual is not converged (`src/Solver/SolverNonLinear.cpp:305-310`, `py/DREAM/Settings/Solver.py:28`).
- A failed Newton step is reported as a status of [`NewtonResult`](@ref).
  The time stepper retries it with a shorter step.
- Each charge state has a lower limit relative to the density of its species ([`ion_densities_unknown`](@ref)).
  In DREAM the charge states have no limit of their own (`src/Solver/SolverNonLinear.cpp:441-508`).
  DREAM's `MaximalPhysicalStepLength` damps these quantities (`src/Solver/SolverNonLinear.cpp:441-508`):
  - ``T_\mathrm{cold}``, ``n_\mathrm{tot}`` and ``n_\mathrm{cold}``.
  - ``W_\mathrm{cold}``, ``W_i`` and the species densities ``N_i``, where they are unknowns.
  - ``\sum Z_0^2 n`` per radius.
- A lower limit shortens only steps towards the limit, never growth ([`LowerLimit`](@ref)).
- A doubled step can fail on an unknown at zero with no absolute tolerance while every other unknown has no error.
  Such a step is retried with half its length ([`StepDoubling`](@ref)).
  DREAM's next step is unbounded there (`src/TimeStepper/TimeStepperAdaptive.cpp:458-482`).
- The ion rows take the sign convention of every other row, so [`IonDiffusionTerm`](@ref) takes a positive coefficient (DREAM: `src/Equations/Fluid/IonTransientTerm.cpp:78`).
- [`braams_karney_conductivity`](@ref) takes the last column of the table where ``\Theta = T/m_ec^2`` exceeds it by at most 1e-8 relative ([Choices of this package](choices.md#Conductivity)).
  DREAM throws for every ``\Theta`` above the table, from the interpolation of GSL (`src/Equations/RunawayFluid.cpp:751`).
  DREAM itself throws at ``T = 100`` times its own rest energy in eV, since it forms ``\Theta`` in the conductivity from ``m_e``, ``c`` and ``e``.
- [`coronal_equilibrium`](@ref) multiplies the ratios of the rates in log space, where DREAM multiplies them directly (`src/Settings/IonEquilibrium.cpp:100-137`).
- The derivatives of the ADAS coefficients are exact, without DREAM's floors in the Jacobian ([`adas_coefficient`](@ref), [`IonRateTerm`](@ref)).
- The Jacobian of [`ExternalAvalancheTerm`](@ref) holds every column of the rate, where DREAM's holds those of the field and of ``n_\mathrm{tot}``.
- The temperature column of [`ParticleSourceTerm`](@ref) holds the derivative of the normalization of its shape, which DREAM's column omits (`src/Equations/Kinetic/ParticleSourceTerm.cpp`).
- The recombination gain of `radiation = :recombination` takes the ionization energy of the recombined state at every radius.
  DREAM's takes that of the recombining state from its second radial cell on (`src/Equations/Fluid/RadiatedPowerTerm.cpp`).
  `test/test_adas_cache.jl` checks the radiated power in coronal equilibrium, which detects this mismatch.
- [`RechesterRosenbluthUpdate`](@ref) takes the level of the perturbation at the start of each step.
  DREAM interpolates its table linearly in time and takes the value at the end of the step.
- [`StepDoubling`](@ref) reads the tolerances by unknown, for the acceptance and for the next step.
  DREAM's step update reads them by position in its list of non-trivial unknowns (`GetErrorScale`, `src/ConvergenceChecker.cpp`).
- `check_every` of [`StepDoubling`](@ref) takes the steps between two checked steps at the current length, the documented intent of DREAM's `checkevery`.
  DREAM's code takes them from a stale start time, with other lengths and one step more (`src/TimeStepper/TimeStepperAdaptive.cpp`).
- The `:smooth` envelope of a [`MomentumRegion`](@ref) whose limit lies at or above the last cell centre takes the width of the last cell.
  DREAM takes the limit minus the last centre (`FindThresholdStep`, `fvm/Equation/MomentQuantity.cpp`).
  That gives zero width and a NaN at that centre, and a jump of the envelope as the limit crosses it.

## Extrapolated flux at the highest momentum

DREAM's `BC_DPHI_CONST` is defined by a constant ``\mathrm{d}Φ/\mathrm{d}p`` at ``p_\mathrm{max}``, with ``Φ = \mathcal{V}' Γ^p``, the phase-space Jacobian ``\mathcal{V}'`` times the momentum flux ``Γ^p``.
Its code (`fvm/Equation/BoundaryConditions/PXiExternalLoss.cpp`) shares the ``\mathcal{V}'`` of the last interior face between the two terms of the extrapolation.
It multiplies both terms by the advection coefficient ``A^p`` of that face.
The interpolation and the diffusion of the second term are those of the face before it.
`p_upper_boundary = :extrapolated_flux` of [`AdvectionDiffusion`](@ref) ports this code literally, and the comparisons with DREAM use it.
`:extrapolated_flux_consistent` extrapolates with the ``\mathcal{V}'`` and the ``Γ^p`` of each face, as the definition states.
`test/test_pmax_boundary.jl` checks both.
`test/test_pmax_boundary_dream.jl` compares the literal port with DREAM.

## Approximate Jacobians

The residual of a [`FokkerPlanckTerm`](@ref) is always that of its scheme.
`jacobian` selects the stored block alone:

- `:exact` differentiates the limiter.
- `:frozen_limiter` is DREAM's `AD_INTERP_JACOBIAN_LINEAR`.
- A linear scheme name gives that scheme's Jacobian ([`with_scheme`](@ref)).

`jacobian = :upwind` with a central residual is DREAM's `AD_INTERP_JACOBIAN_UPWIND`, which makes some of DREAM's runs at high momentum resolution converge (DREAM issue [#126](https://github.com/chalmersplasmatheory/DREAM/issues/126)).
A term with an approximate block declares [`jacobian_kind`](@ref)`(term) === :approximate`.
Newton's method on the stored matrix is then a quasi-Newton iteration.
With [`GMRES`](@ref)`(operator = :exact)` the stored block only preconditions the exact Newton operator ([`jacobian_correction_product!`](@ref)).
`test/test_jacobian_strategies.jl` checks the modes.

## Line search

DREAM's nonlinear solver (`src/Solver/SolverNonLinear.cpp`) damps the Newton step.
It has no line search.
With `line_search = true`, [`NewtonSolver`](@ref) backtracks on ``\tfrac{1}{2}\lVert F \rVert^2`` of the scaled residual ``F`` with the Armijo condition.
It starts from the step length that the lower limits allow.
The default is off, DREAM's behaviour.
`test/test_time_stepping.jl` checks it.
