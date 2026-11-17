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
| boundary of [`fokker_planck_operator`](@ref) at `p_max` | `p_upper_boundary = :f_zero`, DREAM's `BC_F_0` | - |

The synchrotron default holds for [`fokker_planck_operator`](@ref), [`set_fokker_planck_coefficients!`](@ref) and a [`FokkerPlanckTerm`](@ref) that rebuilds an operator built otherwise.
DREAM leaves the synchrotron loss out by default, which `synchrotron = false` reproduces.
The properties of exponential fitting are on the page [Positivity and flux limiters](model/positivity.md#Exponential-fitting).
DREAM's default is the central interpolation, `scheme = :central`.
With `lagged = :ions` the charge-state densities enter the collisions of `f` at the start of the step, and the stored Jacobian has no columns of them.
The lag makes the term first order in time under BDF2, and [`ode_function`](@ref) passes no Jacobian for a system with such a term ([`Runes2P.lags_in_ode_form`](@ref)).
DREAM's default boundary at `p_max` for the hot-tail grid is `BC_PHI_CONST`, which `p_upper_boundary = :constant_flux` reproduces.

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
| cross sections of [`ionization_cross_section`](@ref), [`KineticIonizationTerm`](@ref) and the friction of [`CollisionSettings`](@ref) | `form = :chianti` of the forms `:chianti`, `:dream` and `:bethe`, and `ionization_cross_section_form = :chianti` of the term and of the settings ([`chianti_cross_section`](@ref)) |
| energy above which CHIANTI's cross section follows the relativistic Bethe form | [`CHIANTI_MATCHING_ENERGY`](@ref) = 100 keV |
| factor by which the weight passes from the CHIANTI fit with its relativistic rise to the Bethe form, below and above the matching energy | [`CHIANTI_MATCHING_WIDTH`](@ref) = 2 |
| weight of the relativistic rise of each shell and transition of [`chianti_cross_section`](@ref) | ``1 - 1/u``, ``u = \beta^2/\beta_I^2`` |
| spectrum of the secondaries of [`SecondaryHeatingTerm`](@ref), [`ionization_energy_flows`](@ref) and the friction of [`CollisionSettings`](@ref) | `spectrum = :beb`, `secondary_spectrum = :beb` ([`beb_mean_secondary_energy`](@ref)) |
| smallest threshold factor of the split of the ionization with a fluid term | [`MINIMUM_SPLIT_THRESHOLD_FACTORS`](@ref): 7 for `:chianti`, 5 for `:dream` and `:bethe` |
| part of the Maxwellian below a threshold that lies in the first momentum cell, in the split of [`Runes2P.threshold_split`](@ref) | the share of that cell in the density of the Maxwellian, ``M_1 w_1 / \sum M w`` |

The cross sections are the fits of the CHIANTI database, cited on [Home](index.md#Citing-and-acknowledgements), whose ionization data are those of K. P. Dere, Astronomy and Astrophysics 466, 771 (2007), [doi:10.1051/0004-6361:20066728](https://doi.org/10.1051/0004-6361:20066728).
These fits are nonrelativistic.
[`chianti_cross_section`](@ref) evaluates them at the speed of the electron and adds the relativistic rise.
The weight of the rise starts at zero at the threshold of each shell, with ``\beta_I`` the speed of an electron at the threshold energy.

Above the matching energy ``E_m =`` [`CHIANTI_MATCHING_ENERGY`](@ref) the cross section is the relativistic Bethe form.
A cubic weight in ``\ln E`` from ``E_m/2`` to ``2 E_m`` joins it to the fit with its relativistic rise.
The elements without CHIANTI data in [`CHIANTI_IONIZATION_FITS`](@ref) keep DREAM's form.
`form = :dream` takes DREAM's form for every element.
`form = :bethe` takes DREAM's form with the factor ``1/\beta^2`` on its relativistic Bethe term, and refitted parameters ([Atomic data](model/atomic-data.md)).

The spectrum of the secondaries is the binary-encounter-Bethe spectrum of Y.-K. Kim and M. E. Rudd, Physical Review A 50, 3954 (1994), [doi:10.1103/PhysRevA.50.3954](https://doi.org/10.1103/PhysRevA.50.3954).
It takes the ionization energy of a charge state as its binding energy.
`spectrum = :opal` takes the spectrum of C. B. Opal, W. K. Peterson and E. C. Beaty, The Journal of Chemical Physics 55, 4100 (1971), [doi:10.1063/1.1676707](https://doi.org/10.1063/1.1676707).
It takes the width of the neutral element in [`SECONDARY_SPECTRUM_WIDTHS`](@ref) for every charge state.

The split at the threshold divides the ionization between the kinetic term and a fluid term.
The fluid coefficient of the split ([`fluid_ionization_coefficient`](@ref)) has no floor, so with a Maxwellian `f` the kinetic and fluid rates add up to the ADAS rate.
In the split, ``f_>`` is the fraction of the density of the Maxwellian above the threshold.
``M`` is the relativistic Maxwellian at the cell centres, and ``w`` are the cell weights of a density moment.

Where the threshold lies in the first momentum cell, [`Runes2P.threshold_split`](@ref) takes the part of the Maxwellian below it, ``1 - f_>``, as the weight of that cell.
At the threshold factors of the split, such a threshold leaves nearly the whole Maxwellian in the first cell, below the threshold.
The weight keeps the fluid coefficient finite.
With it the kinetic and fluid rates add up to the ADAS rate only to within that weight.
A warning names the radius and the temperature.

At a threshold factor below that of its form, [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref) refuse the split.
The factor 5 is the hot-region threshold of DREAM's disruption example.
The default hot region of [`ThermalMomentum`](@ref) takes 7 for every form.

## Atomic data and radiation

| Choice | Value | Notes |
|---|---|---|
| continuation of the ADAS coefficients outside the tables, in [`adas_coefficient`](@ref), [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref) | `extrapolation = :dream` | the default behaviour matches DREAM, a code used by researchers in the field |
| radiation mode of [`RadiatedPowerTerm`](@ref) | `radiation = :recombination` | - |
| range of the ion masses that [`IonEnergyExchangeTerm`](@ref) accepts | `MINIMUM_ION_MASS = 1e-27` to `MAXIMUM_ION_MASS = 1e-24`, in kg | a guard against masses in other units |

Inside the tables the interpolation is bicubic in ``\log_{10}`` of the coefficient over ``\log_{10} n`` and ``\log_{10} T``.
`extrapolation = :dream` continues the bilinear interpolation of ``\log_{10}`` of the coefficient in ``\log_{10} n`` and ``\log_{10} T`` from the edge cell, as DREAM's `ADASRateInterpolator` does.
`extrapolation = :clamp_density` clamps the density to the table and extrapolates in ``T`` alone.
At low densities the coefficients are those of the lowest tabulated density, near the coronal limit.
The radiation takes the ADAS classes PLT (line radiation) and PRB (recombination radiation and bremsstrahlung).
`radiation = :recombination` takes PLT + PRB and accounts for the binding energy of every ionization and recombination.
`radiation = :dream` is DREAM's default: PLT, the ionization loss and the bremsstrahlung of R. J. Gould, The Astrophysical Journal 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068).
The erratum is The Astrophysical Journal 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634).
It reproduces DREAM's radiation with `bremsstrahlung_corrections = :dream`, which takes DREAM's coefficients of Gould's corrections.

[`IonEnergyExchangeTerm`](@ref) refuses a mass of `masses` outside the range of the ion masses.
The masses of [`ion_mass`](@ref) lie inside it for every element up to uranium, in both tables.
A mass in atomic mass units or in grams lies above it.

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
| pitch rule of the moments, in [`pitch_moment_weights`](@ref) and [`current_density`](@ref) | `pitch_rule = :linear`: ``f`` linear across a pitch cell |

The cells are finest near ``\xi = 1``, the direction of the runaway beam.
The pitch grid of the runaway grid is the same at all radii and independent of that of the hot-tail grid ([Grids](model/grids.md)).

With `:linear` the mean pitch of a beam narrower than the last pitch cell can exceed 1 ([Mean pitch of a beam on a coarse pitch grid](model/grids.md#Mean-pitch-of-a-beam-on-a-coarse-pitch-grid)).
In the disruption case with a runaway grid the beams stay far from that regime.
In scenario 4 the largest mean pitch of `f_re` is 0.9908 with `:linear` and 0.9880 with `:cell_center`.
Its narrowest beam ``f \propto e^{\kappa\xi}`` has ``\kappa`` of at most about 540, and the mean pitch exceeds 1 on the grid of 15 pitch cells of that comparison from ``\kappa`` of about 1700.
DREAM's rule, ``f`` constant over a pitch cell, is `pitch_rule = :cell_center` ([Defaults that differ](dream-comparison.md#Defaults-that-differ)).

## Knock-on cutoff

| Choice | Value |
|---|---|
| rule of [`cutoff_target`](@ref) | ``W_\mathrm{cut} = \sqrt{T W_c}`` where the critical momentum is finite and ``W_c > T``, else `no_ceiling_factor` times ``T`` |
| factor of the cutoff without a threshold | `no_ceiling_factor = 10` |
| placement of [`KnockOnCutoffUpdate`](@ref) | `placement = :energy_midpoint` |
| lower bound of the window of [`cutoff_window`](@ref) and of `placement = :window`, in units of the thermal momentum | `thermal_factor = 4` |

``W_\mathrm{cut}`` is the kinetic energy at the cutoff and ``W_c`` that at the critical momentum ``p_c``, by default from [`total_critical_momentum`](@ref).
[`KnockOnCutoffUpdate`](@ref) applies the rule at the start of the run and after every accepted step, at every radius (`per_radius = true`).
In DREAM the cutoff (`pCutAvalanche`) is fixed within a run and changes only between restarts.
The cutoff is a face of the knock-on grid, kept while the target lies between its neighbouring faces ([`choose_cutoff_face`](@ref)).
The lower bound keeps the cutoff above the thermal momentum, so that the momentum and energy that the knock-on source creates without a matching loss stay small.
DREAM's documentation gives an upper bound for `pCutAvalanche`, the critical momentum, and no lower bound.

## Total energy of the electrons

| Choice | Value |
|---|---|
| who pays the energy of the knock-ons that [`RosenbluthPutvinskiSourceTerm`](@ref) creates, under [`total_energy_equation`](@ref) | no population: [`KineticEnergyChangeTerm`](@ref) creates it in the system |
| who pays the work of the electron-electron collisions on the electrons that cross the interface to a runaway grid, under [`total_energy_equation`](@ref) | the cold electrons: [`InterfaceCollisionWorkTerm`](@ref) with `coefficient = -1` |
| upwind side of the momentum face values of the parts of [`KineticEnergyLossTerm`](@ref) under a scheme with a side | the side of the whole advection of the Fokker-Planck term, field included, for every part |
| the energy of radial transport of the hot-tail distribution that the row counts, under [`total_energy_equation`](@ref) | that of the whole grid: [`KineticEnergyChangeTerm`](@ref) without a region |

The primaries of the source are the fluid runaways above the grid, which have no energy unknown.
The rejected alternative charges the energy of the knock-ons to the cold electrons, which a row of the total energy without the term would do.
The term takes no region, since every knock-on is in the total energy.
DREAM's source also creates the knock-ons without taking their energy from a population, and DREAM has no total energy of the electrons ([Energy and current balance](model/energy-current.md#What-the-system-enforces)).

The work of the collisions at the interface is the energy step between the two cells at the face times the part of the flux that the electron-electron collisions drive.
No operator holds it, since neither grid takes a flux through the face in its energy moments.
The cold electrons pay it because they are the targets of these collisions, which give them the energy that the electrons lose inside each grid.
The rejected alternative leaves the work to no population, as a row without the term would, and the energy of the two grids is then not conserved across the face ([Energy and current balance](model/energy-current.md#The-energy-step-at-the-interface)).

The parts of the energy loss take the side of the whole advection under `:upwind`, `:quick`, `:upwind_second_order` and the flux limiters, so that they sum to the momentum flux of the Fokker-Planck term face by face.
The rejected alternative takes each operator of a part on the side of its own advection.
The parts then miss the flux at every face where the advection of an operator and the whole advection have opposite signs.
That happens where the field beats the friction, and at ``p = 0`` under a four-point scheme with `p_zero_boundary = :outflow`, where the field alone sets the side.
In a test at 2 V/m their sum differs there from the flux of the whole by up to 0.44 of the largest face flux of the whole (`test/test_energy_transfer.jl`).
The shared side makes each part jump where a face changes its side, and an implicit step across that change may then need a smaller step ([Energy and current balance](model/energy-current.md#Energy-transfer-between-the-distributions)).

The energy term of radial transport takes the whole grid, for two reasons.
The transport then conserves the total energy exactly at any thresholds of the hot region.
With a closed edge, the volume integral of the term stays below 1e-13 of that of the absolute values of its cells (`test/test_energy_transfer.jl`).
The thermal energy of the electrons below the threshold, the bulk, then moves in radius with these electrons, into the cold electrons where they arrive ([Energy and current balance](model/energy-current.md#Radial-transport-of-the-kinetic-electrons)).

The rejected alternative takes the hot region.
The threshold of the hot region follows the temperature, so under a temperature profile it differs between the radii.
An electron that the transport carries across it then leaves or joins the total energy, and no population pays its energy.
`test/test_energy_transfer.jl` shows that the moment over the hot region does not integrate to zero there.

## Linear solvers

| Choice | Value |
|---|---|
| restart length of [`GMRES`](@ref) | `restart_length = 50` |
| linear solver of [`NewtonSolver`](@ref) | `linear_solver = SparseLU()`, UMFPACK ([`SparseLU`](@ref)) |
| refinement steps of [`MKLPardiso`](@ref) | `refinement_steps = 0` |
| largest relative residual of a solve of [`MKLPardiso`](@ref) | `MAX_RELATIVE_RESIDUAL = 1e-6` of `Runes2PMKLPardisoExt` |

When a step needs more Krylov iterations than the restart length, restarted GMRES can stagnate, and the step ends with `:linear_solve_failed`.
The Krylov basis holds `restart_length` vectors.

UMFPACK comes with Julia's standard library SparseArrays, so the default needs no MKL.
Its timings against MKL PARDISO ([`MKLPardiso`](@ref)) are on the page [Performance and accuracy](performance.md#The-linear-solvers).
DREAM's default solver is PETSc's own LU (`LINEAR_SOLVER_LU`).

On the disruption case, ten refinement steps of PARDISO cost more than a factorization ([Performance and accuracy](performance.md#The-linear-solvers)).
Without them the relative residual of a solve is 1.0e-9, below the `floor_linear_tolerance` of 1e-8 of [`NewtonSolver`](@ref).
DREAM's MKL PARDISO solver allows 10 refinement steps (`fvm/Solvers/MIMKL.cpp`), which `MKLPardiso(refinement_steps = 10)` reproduces.

PARDISO perturbs a small pivot instead of failing, so a singular or nearly singular matrix gives a finite solution whose residual is large.
A solve of [`MKLPardiso`](@ref) whose relative residual exceeds `MAX_RELATIVE_RESIDUAL` fails with `:singular_matrix`, as a zero pivot does with [`SparseLU`](@ref).
The step is then retried shorter at once.
The table gives the relative residuals of PARDISO's solves on both sides of the bound of 1e-6.

| Matrices | Relative residual of a solve |
|---|---|
| the Newton matrices of the disruption case | 1e-13 to 1e-9 |
| singular matrices | 1e-2 to 10 |
| a matrix of the pattern, solved with the ordering and the matching of another | just above the bound, or far above 1 |

The bound lies three orders of magnitude above the residuals of the disruption case, and four or more below those of singular matrices.
A false failure costs only a shorter step.
The solve after a failure computes a new ordering and matching.

## Time stepping

| Choice | Value |
|---|---|
| next step of [`StepDoubling`](@ref) after a step that an unknown at zero fails while every other unknown has no error | half the step |
| round-off bound of a residual row of [`NewtonSolver`](@ref) on a stiff step | `c = 1` in the bound `c eps Σ_j abs((∂F/∂u)_ij u_j)` |
| Newton iteration after an iterate that passes both tests with negative entries in a kinetic unknown | one more for the schemes of [`POLISHED_SCHEMES`](@ref), turned off after [`POLISH_MISSES`](@ref) steps in a row that it does not clean, `polish_negatives = :auto` of [`NewtonSolver`](@ref) |

DREAM rejects a step in which an unknown with no absolute tolerance is zero after the full step and not after the two half steps.
It takes the next step from the largest error of the other unknowns, as [`StepDoubling`](@ref) does.
When that error is zero, DREAM's next step is unbounded.
Halving the step tried keeps the retry finite.
DREAM has no finite behaviour to reproduce there, so no option gives DREAM's.

The round-off bound counts each term of a residual row at its own size, one rounding error per term.
A row whose terms cancel to many digits cannot fall below it, whatever the iterate.
With `c = 1` the bound lies about 2.5 times above the largest residual measured at the floor, in the quench of the disruption case at DREAM's resolution.
DREAM converges on the update alone and only warns about the residual ([Comparison with DREAM](dream-comparison.md)), so it has no residual test to waive.

A positivity-preserving limiter keeps the converged discrete solution non-negative, and the extra iteration converges it further where an accepted iterate still has negative entries ([Positivity](model/positivity.md#Convergence-of-the-limiter)).
The schemes of [`POLISHED_SCHEMES`](@ref) are those whose converged solution stayed non-negative in the tests.
On the others the extra iteration is wasted.
An unconverged limiter is cleaned by one extra iteration, so [`POLISH_MISSES`](@ref), five misses in a row, shows a scheme that does not keep the run non-negative and bounds the cost of finding out.
DREAM takes no such iteration: `polish_negatives = false`.

## Examples

The parameter values of the scripts of `examples/` come from DREAM's examples or are choices of this package, cited in each script.

| Script | Momentum cells, `p_max` | Pitch cells | Plasma | Time step |
|---|---|---|---|---|
| `examples/common.jl`, switch `grid=biuniform_270` | 270 to ``p = 2.5``, 70 of them below ``p = 0.07`` ([`biuniform_p_faces`](@ref)) | 15, 10 of them within 0.5 rad of ``\xi = 1`` ([`biuniform_theta_xi_faces`](@ref)) | - | - |
| `examples/relaxation.jl` | 200 to ``p = 2`` | 15 | deuterium at ``T = 1`` keV and ``n = 10^{20}`` m⁻³, ``E = 5 E_c`` | ``\Delta t = 10^{-5}`` s up to ``t = 1`` ms |
| `examples/cooling.jl` | 200 to ``p = 2`` | 15 | that of `reference/dream/cases/cooling_positivity.py` | ``\Delta t = 10^{-5}`` s |

The `grid=biuniform_270` preset is a fully kinetic hot-tail grid.
In `examples/relaxation.jl` ``E_c`` is the Connor–Hastie critical field ([`critical_field`](@ref)), and the field lies between the critical and the Dreicer field.
The radial values of the plasma of `examples/cooling.jl` come from DREAM's SPI example.

## Reference cases

This section holds the choices of the disruption chain, of the window of its test and of the case `heating`.

### The disruption chain

The reference case `reference/dream/cases/disruption_kinetic.py` runs the disruption chain of DREAM's `examples/Disruptions/generate.py` with the parameters of `examples/Disruptions/ASDEXU.py`.
Its options and their values are in the table, with one note per row below it.

| Choice | Value |
|---|---|
| scenario of the disruption chain | `--scenario 1`, the full-conversion scenario |
| elongation of the radial grid | `--elongation 1`, a cylinder without shaping |
| steps of phase 1 | `--nt-ioniz 500`, steps of 4e-8 s |
| runaway grid (`--runaway-grid`) | the default grid of the table of runaway grids below |
| runaway grid of the measurement of scenario 4 at full size ([The disruption case](disruption-case.md#The-whole-quench-of-scenario-4)) | `--re-pmax 200 --re-np-coarse 40 --re-nxi 15 --re-nxi-sep 10` |
| advection scheme of `f_re` | that of `f_hot` (`--scheme`) in both phases |
| absolute tolerance of `f_re` | 1e5 m⁻³ in every run after the conductivity, the absolute tolerance of `f_hot` in the ionization and the quench |
| window of phase 2 of the test | `--window-start 170 --window-steps 10`, the default of `--window-steps` ([The window of the test](#The-window-of-the-test)) |

- Scenario: the DREAM paper runs scenarios 1 and 4.
  `generate.py` runs scenario 0 by default, which `--scenario 0` reproduces.
- Elongation: the cylinder of Runes2P has no shaping.
  `ASDEXU.py` takes an elongation linear from 1.0 to 1.15 in the radius, which `--elongation asdexu` reproduces.
- Steps of phase 1: the case passes `--nt-ioniz 500` explicitly, at the default `--elongation 1`.
  The default of the script stays the 200 steps of scenario 1.
  Each step of phase 1 is then 4e-8 s instead of 1e-7 s.
  With 200 steps DREAM's first step fails in runs with `--linear-solver mkl`, also when retried, since its Newton iteration meets a non-finite value (`brent.c:69: computed function value is infinite or NaN`).
  With 500 steps phase 1 converges.
  The step of 4e-8 s is also that of the case at the size of a test in `test/test_disruption_dream.jl`.
- Runaway grid: DREAM's example runs no runaway grid.
  The defaults of `--runaway-grid` serve scenario 1 at the reduced size of the tests.
  At 300 µs of scenario 1, a grid to `p_max = 60` holds 3e-5 of the density of `f_re` above `p = 20`.
  The avalanche source keeps a small population up to the last cell, so a few cells more do not change the result.
  The 8 pitch cells, fine near `ξ = 1`, differ from the 4 uniform cells of the reduced hot-tail grid of the tests, so that a test crosses DREAM's mapping between the two pitch grids.
- Runaway grid of scenario 4: the measurement takes the grid to `p_max = 200`, for the reason given below the grids.
- Advection scheme: the scheme of `f_hot` on `f_re`, with its switch to upwind in phase 2 where `generate.py` makes it, keeps one scheme across the interface.
  DREAM's own default for `f_re` is the centred scheme.
- Absolute tolerance: `generate.py` sets no absolute tolerance for `f_re`, and DREAM sets none either (`ToleranceSettings.py`).
  The case gives `f_re` the 1e5 m⁻³ of `f_hot`.
  DREAM's output gives both distributions in m⁻³, since the momentum is in units of ``m_e c`` and the volume of momentum space has then no unit.

The runaway grids have a fine segment above the hot-tail `p_max` of 0.8 and a geometric segment above the fine one.
The momenta are in units of ``m_e c``.

| Grid | Segment | Momentum cells | From `p` | To `p` | Cell width |
|---|---|---|---|---|---|
| default | fine | 12 | 0.8 | 2 | 0.1 |
| default | geometric | 20 | 2 | 20 | growing, the first 0.1 (from 2.0 to 2.1) |
| scenario 4 at full size | fine | 12 | 0.8 | 2 | 0.1 |
| scenario 4 at full size | geometric | 40 | 2 | 200 | growing |

- The default grid has 8 pitch cells, biuniform in θ.
  5 of them lie below θ = 0.5.
- The grid of scenario 4 has 15 pitch cells (`--re-nxi 15 --re-nxi-sep 10`).

Two measures from runs of DREAM v26.5 in scenario 4 compare the values of `p_max` of the runaway grid.
The first is how much of `n_re` the grid holds.

| `p_max` | Share of `n_re` | Where and when |
|---|---|---|
| 20, the default of the case script | 0.44 of `n_re` is held in `f_re` | at the centre, at the end of the quench |
| 200 | 4.0e-5 to 4.1e-5 of `n_re` lies above the grid | at radii 1 to 11, at 4.99e-3 s ([The disruption case](disruption-case.md#The-whole-quench-of-scenario-4)) |

The second is the plasma current `I_p` at 4.99e-3 s, in runs of 3200 steps.

| `p_max` | `I_p` | Relative difference from `p_max = 200` |
|---|---|---|
| 60 | 2.5305347e5 A | −4.5e-5 |
| 200 | 2.5306492e5 A | - |

- The relative difference is (`I_p` − `I_p` at `p_max = 200`) / `I_p` at `p_max = 200`.
- The default grid holds too little of `n_re` in scenario 4, so the measurement takes the grid to `p_max = 200`.

### The window of the test

The window is the part of phase 2 that the test of the runaway grid runs from DREAM's state.
The test runs it on the reduced grid of the tests with the step of the scenario.
The page [Performance and accuracy](performance.md#The-disruption-case) times the same window at full size.
The table compares phase 2 of the case with that of scenario 1 and with the short phases of `DISRUPTION_ARGS` in `reference/dream/run_all.sh`.

| Run | Options | Duration of phase 2 | Steps | Step |
|---|---|---|---|---|
| the case at the reduced size of the tests | `--t-ioniz 2e-5 --nt-quench 429 --t-quench 3e-4` | 2.8e-4 s | 429 | 6.527e-7 s |
| scenario 1 of `generate.py` | `t_ioniz` 2e-5, `t_TQ` 1e-3, `nt_TQ` 1500 | 9.8e-4 s | 1500 | 6.533e-7 s |
| `DISRUPTION_ARGS` of `reference/dream/run_all.sh` | `--t-ioniz 4e-7 --nt-quench 10 --t-quench 1.4e-6` | 1.0e-6 s | 10 | - |

- `--t-quench` is the end time of phase 2, counted from the start of phase 1, so it includes the length of phase 1 (`--t-ioniz`).
- The step of the case equals that of scenario 1 within 1e-3, relative.
- Phase 2 of the case ends at 300 µs of the scenario, the end of the benchmark of scenario 1.

The table gives the window of the test at the reduced size.

| Quantity | Window | Criterion |
|---|---|---|
| step whose state the window starts from | 170 | - |
| time of the scenario at that step | 131.0 µs | 130 to 150 µs |
| runaway share of the plasma current at that step | 5.9 % | 1 to 10 % |
| steps | 10 | - |
| run time of the whole chain of the case script at the reduced size | 10.5 s | - |
| test data of the window | 95.9 kB | - |

- The window starts where `f_re` carries part of the current.
- The column Criterion gives the ranges that the start of the window is chosen to lie in.
- Ten steps keep the reference small.

### The case `heating`

The reference case `reference/dream/cases/initial_state.py` has a case `heating`, which runs the runaway grid of DREAM's `examples/init_f_re/generate.py`.
The test case `runaway_heating_example` (`test/example_cases.jl`) builds it in Runes2P.
The table holds the choices of that case.

| Choice | Value |
|---|---|
| length of the run | 5e-6 s in the 20 steps of the example, in place of its 1e-3 s |
| start of ``f_\mathrm{re}`` | `INIT_FORWARD` from ``n_\mathrm{re} = 10^{16}`` m⁻³, in place of the analytic avalanche shape of the example |
| cold temperature | self-consistent from 100 eV, `TYPE_SELFCONSISTENT`, in place of the prescribed temperature of the example |

The run of 5e-6 s keeps ``T_\mathrm{cold}`` near the 100 eV of the example: the ohmic heating raises it to 101.4 eV.
With the self-consistent temperature DREAM's equation of ``T_\mathrm{cold}`` holds the collisional heating by ``f_\mathrm{re}`` (`src/Settings/Equations/T_cold.cpp:341-354`), which the case saves with the states it is evaluated from.
