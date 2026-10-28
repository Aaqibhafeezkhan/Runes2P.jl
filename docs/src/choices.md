# Choices of this package

```@meta
CurrentModule = Runes2P
```

A choice of this package is a value or a default that neither DREAM's examples nor a cited reference fixes.
The code marks each one where it is set and points to this page.
Where a reason is stated, the column Notes gives it.
Where DREAM behaves differently, the option that reproduces DREAM is named below the table.

## Kinetic operator

| Choice | Value | Notes |
|---|---|---|
| synchrotron loss of the Fokker–Planck operator | `synchrotron = true` ([`add_synchrotron!`](@ref)) | chosen for accuracy |
| interpolation scheme of [`fokker_planck_operator`](@ref) in `p` and `ξ` | `scheme = :exponential_fitting` ([`AdvectionDiffusion`](@ref)) | chosen for performance and positivity |
| lag of the plasma of a [`FokkerPlanckTerm`](@ref) that reads unknowns | `lagged = :ions` | - |
| boundary of [`fokker_planck_operator`](@ref) at `p_max` | `p_upper_boundary = :f_zero`, DREAM's `BC_F_0`; DREAM's default is `BC_PHI_CONST` (`:constant_flux`) | - |

The synchrotron default holds for [`fokker_planck_operator`](@ref), [`set_fokker_planck_coefficients!`](@ref) and a [`FokkerPlanckTerm`](@ref) that rebuilds an operator built otherwise.
DREAM leaves the synchrotron loss out by default, which `synchrotron = false` reproduces.
The properties of exponential fitting are on the page [Positivity and flux limiters](model/positivity.md#Exponential-fitting).
DREAM's default is the central interpolation, `scheme = :central`.
With `lagged = :ions` the charge-state densities enter the collisions of `f` at the start of the step, and the stored Jacobian has no columns of them.
The lag makes the term first order in time under BDF2, and [`ode_function`](@ref) passes no Jacobian for a system with such a term ([`Runes2P.lags_in_ode_form`](@ref)).

## Collision frequencies

The rows are fields of [`CollisionSettings`](@ref).

| Choice | Value |
|---|---|
| window of partial screening, in units of the mean excitation energy ``I_j`` | `screening_window_start = 1`, `screening_window_end = 4` (with `screening_window = :excitation_energy`) |
| radiative stopping on the nuclei | `bremsstrahlung = :stopping_power` ([`bremsstrahlung_cross_section`](@ref)) |
| energy diffusion of the radiative stopping | `bremsstrahlung_diffusion = :none` |
| width of the threshold of the kinetic ionization in the friction | `ionization_threshold_width = 0.1` |

The partial screening is that of L. Hesslow et al., Journal of Plasma Physics 84, 905840605 (2018), [doi:10.1017/S0022377818001113](https://doi.org/10.1017/S0022377818001113).
The bound-electron parts of charge state ``j`` enter through a smooth step in the kinetic energy from ``I_j`` to ``4 I_j``, and below it the screening is complete.
`screening_window = :dream` applies partial screening at all momenta, as DREAM does.
`bremsstrahlung = :neglect` leaves the radiative stopping out, as DREAM does by default.
With `bremsstrahlung_diffusion = :none` the radiative stopping is a pure drag, as in the model of M. Hoppe et al., Computer Physics Communications 268, 108098 (2021), [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098), eqs. 8, 9, 14 and 15.
`bremsstrahlung_diffusion = :dream` gives the radiative drag a thermal energy diffusion ``\Theta \gamma \nu_s``, as DREAM does.
With a threshold factor `ionization_threshold` above zero, the kinetic ionization starts at ``p_0 =`` `ionization_threshold` ``\sqrt{2T/m_e c^2}``.
Its weight in the friction rises in a cubic smooth step from ``(1 - w) p_0`` to ``(1 + w) p_0``, with ``w =`` `ionization_threshold_width`, a numerical width without a counterpart in DREAM.

## Kinetic ionization

| Choice | Value |
|---|---|
| cross sections of [`ionization_cross_section`](@ref), [`KineticIonizationTerm`](@ref) and the friction of [`CollisionSettings`](@ref) | `form = :chianti`, `cross_section_form = :chianti`, `ionization_cross_section_form = :chianti` ([`chianti_cross_section`](@ref)) |
| energy above which CHIANTI's cross section follows the relativistic Bethe form | [`CHIANTI_MATCHING_ENERGY`](@ref) = 100 keV |
| factor of the window of the weight between the continued fit and the Bethe form | [`CHIANTI_MATCHING_WIDTH`](@ref) = 2 |
| weight of the relativistic rise of each shell and transition of [`chianti_cross_section`](@ref) | ``1 - 1/u``, ``u = \beta^2/\beta_I^2`` |
| spectrum of the secondaries of [`SecondaryHeatingTerm`](@ref), [`ionization_energy_flows`](@ref) and the friction of [`CollisionSettings`](@ref) | `spectrum = :beb`, `secondary_spectrum = :beb` ([`beb_mean_secondary_energy`](@ref)) |
| smallest threshold factor of the split of the ionization with a fluid term | [`MINIMUM_SPLIT_THRESHOLD_FACTORS`](@ref): 7 for `:chianti`, 5 for `:dream` and `:bethe` |

The cross sections are the fits of the CHIANTI database, cited on [Home](index.md#Citing-and-acknowledgements), whose ionization data are those of K. P. Dere, Astronomy and Astrophysics 466, 771 (2007), [doi:10.1051/0004-6361:20066728](https://doi.org/10.1051/0004-6361:20066728).
These fits are nonrelativistic.
[`chianti_cross_section`](@ref) evaluates them at the speed of the electron and adds the relativistic rise.
Its weight starts at zero at the threshold of each shell, with ``\beta_I`` the speed of an electron at the threshold energy.
Above the matching energy ``E_m =`` [`CHIANTI_MATCHING_ENERGY`](@ref) the cross section is the relativistic Bethe form, joined to the continued fit by a cubic weight in ``\ln E`` from ``E_m/2`` to ``2 E_m``.
The elements without CHIANTI data in [`CHIANTI_IONIZATION_FITS`](@ref) keep DREAM's form, and `form = :dream` takes DREAM's form for every element.
The binary-encounter-Bethe spectrum of Y.-K. Kim and M. E. Rudd, Physical Review A 50, 3954 (1994), [doi:10.1103/PhysRevA.50.3954](https://doi.org/10.1103/PhysRevA.50.3954), takes the ionization energy of a charge state as its binding energy.
`spectrum = :opal` takes the spectrum of C. B. Opal, W. K. Peterson and E. C. Beaty, The Journal of Chemical Physics 55, 4100 (1971), [doi:10.1063/1.1676707](https://doi.org/10.1063/1.1676707).
It takes the width of the neutral element in [`SECONDARY_SPECTRUM_WIDTHS`](@ref) for every charge state.
The fluid coefficient of the split ([`fluid_ionization_coefficient`](@ref)) has no floor, so with a Maxwellian `f` the kinetic and fluid rates add up to the ADAS rate.
At a threshold factor below that of its form, [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref) refuse the split.
The factor 5 is the hot-region threshold of DREAM's disruption example, and the default hot region of [`ThermalMomentum`](@ref) takes 7 for every form.

## Atomic data and radiation

| Choice | Value | Notes |
|---|---|---|
| continuation of the ADAS coefficients outside the tables, in [`adas_coefficient`](@ref), [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref) | `extrapolation = :dream` | the default behaviour matches DREAM, a code used by researchers in the field |
| radiation mode of [`RadiatedPowerTerm`](@ref) | `radiation = :recombination` | - |

Inside the tables the interpolation is bicubic in ``\log_{10}`` of the coefficient over ``\log_{10} n`` and ``\log_{10} T``.
`extrapolation = :dream` continues the bilinear interpolation of ``\log_{10}`` of the coefficient in ``\log_{10} n`` and ``\log_{10} T`` from the edge cell, as DREAM's `ADASRateInterpolator` does.
`extrapolation = :clamp_density` clamps the density to the table and extrapolates in ``T`` alone; at low densities the coefficients are those of the lowest tabulated density, near the coronal limit.
The radiation takes the ADAS classes PLT (line radiation) and PRB (recombination radiation and bremsstrahlung).
`radiation = :recombination` takes PLT + PRB and accounts for the binding energy of every ionization and recombination.
`radiation = :dream` is DREAM's default: PLT, the ionization loss and the bremsstrahlung of R. J. Gould, The Astrophysical Journal 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068).
The erratum is The Astrophysical Journal 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634).
It reproduces DREAM's radiation with `bremsstrahlung_corrections = :dream`, which takes DREAM's coefficients of Gould's corrections.

## Plasma from the unknowns

| Choice | Value |
|---|---|
| floor of a temperature unknown | [`DEFAULT_TEMPERATURE_FLOOR`](@ref) = 0.5 eV |

The floor is the default of [`PlasmaFromUnknowns`](@ref) and of [`fluid_temperature_floors`](@ref).
DREAM has no floor, so a comparison that follows DREAM below 0.5 eV sets a lower value.

## Grids

| Choice | Value |
|---|---|
| pitch grid of [`runaway_grid`](@ref) | [`biuniform_theta_xi_faces`](@ref)`(20, 12, 0.9)`: 12 cells uniform in ``\theta \in [0, 0.9]`` rad and 8 up to ``\pi`` |

The cells are finest near ``\xi = 1``, the direction of the runaway beam.
The pitch grid of the runaway grid is the same at all radii and independent of that of the hot-tail grid ([Grids](model/grids.md)).

## Knock-on cutoff

| Choice | Value |
|---|---|
| rule of [`cutoff_target`](@ref) | ``W_\mathrm{cut} = \sqrt{T W_c}`` where the critical momentum is finite and ``W_c > T``, else `no_ceiling_factor` times ``T`` |
| factor of the cutoff without a threshold | `no_ceiling_factor = 10` |
| placement of [`KnockOnCutoffUpdate`](@ref) | `placement = :energy_midpoint` |

``W_\mathrm{cut}`` is the kinetic energy at the cutoff and ``W_c`` that at the critical momentum ``p_c``, by default from [`total_critical_momentum`](@ref).
[`KnockOnCutoffUpdate`](@ref) applies the rule at the start of the run and after every accepted step, at every radius (`per_radius = true`).
In DREAM the cutoff (`pCutAvalanche`) is fixed within a run and changes only between restarts.
The cutoff is a face of the knock-on grid, kept while the target lies between its neighbouring faces ([`choose_cutoff_face`](@ref)).

## Linear solvers

| Choice | Value |
|---|---|
| restart length of [`GMRES`](@ref) | `restart_length = 50` |

When a step needs more Krylov iterations than the restart length, restarted GMRES can stagnate, and the step ends with `:linear_solve_failed`.
The Krylov basis holds `restart_length` vectors.

## Examples

The parameter values of the scripts of `examples/` come from DREAM's examples or are choices of this package, cited in each script.

| Choice | Value |
|---|---|
| hot-tail grid of the switch `grid=biuniform_270` (`examples/common.jl`) | 270 momentum cells to ``p = 2.5`` with 70 below ``p = 0.07`` ([`biuniform_p_faces`](@ref)); 15 pitch cells with 10 within 0.5 rad of ``\xi = 1`` ([`biuniform_theta_xi_faces`](@ref)) |
| plasma, grid and step of `examples/relaxation.jl` | deuterium at ``T = 1`` keV and ``n = 10^{20}`` m⁻³, ``E = 5 E_c``; 200 momentum cells to ``p = 2``, 15 pitch cells; ``\Delta t = 10^{-5}`` s up to ``t = 1`` ms |
| grid and step of `examples/cooling.jl` | 200 momentum cells to ``p = 2``, 15 pitch cells; ``\Delta t = 10^{-5}`` s |

The `grid=biuniform_270` preset is a fully kinetic hot-tail grid.
In `examples/relaxation.jl` ``E_c`` is the Connor–Hastie critical field ([`critical_field`](@ref)), and the field lies between the critical and the Dreicer field.
The plasma of `examples/cooling.jl` is that of the reference case `reference/dream/cases/cooling_positivity.py`, and its radial values come from DREAM's SPI example.

## Reference cases

The reference case `reference/dream/cases/disruption_kinetic.py` runs the disruption chain of DREAM's `examples/Disruptions/generate.py` with the parameters of `examples/Disruptions/ASDEXU.py`.

| Choice | Value |
|---|---|
| scenario of the disruption chain | `--scenario 1`, the full-conversion scenario |
| elongation of the radial grid | `--elongation 1`, a cylinder without shaping |

The DREAM paper runs scenarios 1 and 4, and `generate.py` runs scenario 0 by default, which `--scenario 0` reproduces.
The cylinder of Runes2P has no shaping, so the case departs from the elongation of `ASDEXU.py`, linear from 1.0 to 1.15 in the radius, which `--elongation asdexu` reproduces.
