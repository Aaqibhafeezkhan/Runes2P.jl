# Changelog

The changes of Runes2P that users see: the public API, new features and changed numerical results.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until the first release, every change is listed under "Unreleased", with the subsections in the order of that format.

## Unreleased

### Added

- `InterfaceCollisionWorkTerm(interface; coefficient = -1)` is the work of the electron-electron collisions on the electrons that cross a `MomentumInterfaceTerm`, for the row of the temperature ([#297](https://github.com/osrep/Runes2P.jl/pull/297)).
  It is the part of the flux through the face that these collisions drive, at the face value of the whole flux, times an energy step.
  The step is the energy of the first cell of the runaway grid less that of the last cell of the hot-tail grid.
  The settings record of a run holds the record of the term, in the format 0.0.6.
- `KineticEnergyChangeTerm(source; region, coefficient)` is the power that a source term gives its distribution, for the row of the temperature, with a `RosenbluthPutvinskiSourceTerm` as `source` ([#296](https://github.com/osrep/Runes2P.jl/pull/296)).
  It is the energy moment of the knock-ons of the fluid runaways above the grid, `n_re n_tot` times a sum over the cells that the source fills.
  A region takes a fixed limit, and a limit that follows the temperature or time is refused.
  The settings record of a run holds the record of the term, and `OUTPUT_FORMAT_VERSION` is 0.0.6 for it and for the key `knockon_term` of a `SecondaryHeatingTerm` with `cut = :knockon`.
  A file of the format 0.0.5 reads as before.
- `SecondaryHeatingTerm(...; cut = :knockon, knockon_term = term)` gives the cold electrons only the energy of the secondaries below the cutoff of the `KnockOnTerm` `term` at each radius ([#294](https://github.com/osrep/Runes2P.jl/pull/294)).
  The secondaries above the cutoff, which the knock-on term puts into a distribution, are no longer counted twice.
  The heating reads the cutoff at each evaluation, so it follows a `KnockOnCutoffUpdate`.
  The settings record of the term holds the knock-on term under the key `knockon_term` with this cut.
- With `cut = :knockon`, `EquationSystem` requires the knock-on term `term` in the system with the kinetic unknown of the ionization as its primary ([#294](https://github.com/osrep/Runes2P.jl/pull/294)).
  It refuses a `term` on `:cold_electron_density` or `:runaway_density`, which count free electrons only.
  Next to a knock-on term of the same primary on `:total_electron_density`, `term` must be such a term.
- `secondary_energy_below` and `beb_secondary_energy_below` give the energy per ionization of the secondaries below a cut, for the spectra of Opal et al. and of Kim and Rudd ([#294](https://github.com/osrep/Runes2P.jl/pull/294)).
- Every unknown carries a unit string, set by the keyword `unit` of `ScalarUnknown`, `RadialUnknown` and `KineticUnknown` and read by `unknown_unit` ([#288](https://github.com/osrep/Runes2P.jl/pull/288)).
  A scalar or radial unknown of a default name takes the unit of the table "Units of the unknowns", a `KineticUnknown` `"m^-3"`, and any other unknown the empty string.
  The output file stores the unit as the attribute `unit` of every entry of `unknowns`, of `time` and of the grid faces in `layout`, and `OUTPUT_FORMAT_VERSION` is 0.0.5 for it.
  A file of the format 0.0.4 or older reads with the default units, and the setup hash does not change.
- A unit of an unknown is checked where it is declared, at a resume and at a join of segments ([#288](https://github.com/osrep/Runes2P.jl/pull/288)).
  A unit is ASCII without control characters, and the constructors refuse another one with an `ArgumentError`.
  A resume warns for an unknown whose unit stored in the file and declared unit are both nonempty and differ, and `StoredState` carries the stored units as `units`.
  Joined segments take the unit of each unknown from the latest segment that stores one.
  The join warns for an unknown for which two segments store different nonempty units.
  A reader drops a stored unit that is not valid with a warning.
- `MKLPardiso` solves the Newton systems with Intel MKL PARDISO; its solves come with the extension `Runes2PMKLPardisoExt`, loaded by `using Pardiso` ([#243](https://github.com/osrep/Runes2P.jl/pull/243)).
  It takes the systems of `SparseLU`, the field-particle term by the same Woodbury update, and runs threaded with `threads`; `refinement_steps` sets PARDISO's refinement steps.
  A solve whose relative residual exceeds 1e-6 fails with `:singular_matrix`, as a zero pivot does with `SparseLU`, and the next factorization computes a new ordering and matching ([#255](https://github.com/osrep/Runes2P.jl/pull/255)).
  `SparseLU` stays the default.
  Its settings record is new in the format 0.0.4.
- `InterfaceParticleFluxTerm` puts the particle flux through the interface into a radial row, the influx of DREAM's `n_re` from the hot-tail grid.
  Its settings record is new in the format 0.0.4 ([#214](https://github.com/osrep/Runes2P.jl/pull/214)).
- `p_zero_boundary` of `AdvectionDiffusion`, `fokker_planck_operator` and `electron_electron_operator` sets the flux through `p = 0` on a grid from `p = 0`.
  The default `:outflow` keeps the flux of the friction there, as DREAM's hot-tail grid; `:zero_flux` takes none, as DREAM's runaway grid.
  The settings record of an operator holds it, and `OUTPUT_FORMAT_VERSION` is 0.0.4 for the new key ([#203](https://github.com/osrep/Runes2P.jl/pull/203)).
- `fill_coronal_equilibrium!` starts the ions of a run in coronal equilibrium at the free-electron density they give, DREAM's `init_equil`, with the rates of an `IonRateTerm` or of a composition ([#191](https://github.com/osrep/Runes2P.jl/pull/191)).
  Its iterations stop at `MAXIMUM_EQUILIBRIUM_ITERATIONS`.
- `fill_runaway_distribution!` fills a distribution from a density in DREAM's four shapes of `f_re`, `RUNAWAY_SHAPES` ([#191](https://github.com/osrep/Runes2P.jl/pull/191)).
- `FrictionEnergyTransferTerm` and `FRICTION_TRANSFER_SETTINGS`: the power that a distribution gives to the cold electrons by friction, as DREAM's `CollisionalEnergyTransferKineticTerm` ([#185](https://github.com/osrep/Runes2P.jl/pull/185)).
  It is the heating in the row of the temperature of a superthermal run, and takes any scheme of the operator.
- `NewtonSolver(polish_negatives = :auto)`: one more Newton iteration when an iterate that passes both tests has negative entries in a kinetic unknown ([#163](https://github.com/osrep/Runes2P.jl/pull/163)).
  It removes the negative entries that an unconverged positivity-preserving limiter leaves near zero in the tail.
  `:auto` polishes the unknowns whose schemes are in `POLISHED_SCHEMES` and stops, with a warning in the log, after `POLISH_MISSES` steps in a row that it does not clean; `true` polishes every kinetic unknown, `false` none.
  The step statistics count such steps in `polished_for_negatives`, and `OUTPUT_FORMAT_VERSION` is 0.0.3 for the new statistic and setting.
- `fill_maxwellian!` fills a distribution of the state with a Maxwellian from profiles of the density and the temperature, rescaled to the density on the grid as DREAM starts a fully kinetic run ([#162](https://github.com/osrep/Runes2P.jl/pull/162)).
- `validate`, `has_plasma_columns`, `plasma_column_sparsity!` and `add_plasma_columns!`, the functions that a new term implements and the system calls, are declared `public` ([#126](https://github.com/osrep/Runes2P.jl/pull/126)).

### Changed

- Under the total energy of the electrons (`total_energy_equation`), an `EquationSystem` refuses a row of the temperature with an `InterfaceEnergyFluxTerm` but without the `InterfaceCollisionWorkTerm` of the same interface, with a negative coefficient ([#297](https://github.com/osrep/Runes2P.jl/pull/297)).
  Until now no term carried the work of the electron-electron collisions on the electrons that cross the interface, and the energy of the two grids was not conserved across the face.
  The cold electrons now pay that work.
  The system refuses a second such term of one interface, which would count the work twice, and the term of an interface whose lower distribution is not in the total energy.
- Under the total energy of the electrons (`total_energy_equation`), an `EquationSystem` accepts a `RosenbluthPutvinskiSourceTerm` in the equation of a distribution whose energy is in the total, which it refused until now ([#296](https://github.com/osrep/Runes2P.jl/pull/296)).
  The row of the temperature then needs one `KineticEnergyChangeTerm` of that very source, with a positive coefficient and without a region, since every knock-on is in the total energy.
  The term creates the energy of the knock-ons in the system, and no population pays it.
  The row refuses a `KineticEnergyChangeTerm` whose source is not in the equation of a distribution in the total energy.
- `EquationSystem` refuses `SecondaryHeatingTerm` with `cut = :none` or `cut = :threshold` next to a `KnockOnTerm` of the same primary on `:total_electron_density`, which counts the energy above the knock-on cutoff twice ([#294](https://github.com/osrep/Runes2P.jl/pull/294)).
  `bound_friction = :ionization_split` accepts the heating with `cut = :knockon` as well as with `cut = :none`.
- Under the total energy of the electrons (`total_energy_equation`), an `EquationSystem` refuses a `LossRateTerm`, a `MaxwellianSourceTerm` or a `LinearTerm` with a nonzero coefficient in the equation of a distribution whose energy is in the total ([#295](https://github.com/osrep/Runes2P.jl/pull/295)).
  Until now the cold electrons gained or paid the energy of the electrons that these terms remove or add, since no term of the row of the temperature carries it.
- A run refuses a `RechesterRosenbluthUpdate` whose operator moves a distribution whose energy is in the total energy of the electrons ([#295](https://github.com/osrep/Runes2P.jl/pull/295)).
  The system refuses radial transport of such a distribution when it is built, but the update sets the coefficients in the run, after that check.
  A run also repeats the checks of the total energy before its first step, so coefficients of radial transport set by hand after the build are refused too.
  Both refusals come before the first step, with an `ArgumentError`.
- A `KineticEnergyLossTerm` whose source lags the whole plasma (`lagged = :all`) throws an `ArgumentError` naming `prepare_step!` when it is evaluated on its own before any preparation ([#264](https://github.com/osrep/Runes2P.jl/pull/264)).
  Until now it read the plasma of the state.
  An `EquationSystem` prepares such a term at its initial state when it is built, unless the term was prepared before.
  The results of a run do not change.
- `InterfaceParticleFluxTerm` and `InterfaceEnergyFluxTerm` assemble their row of the Jacobian directly, from the flux in dual numbers seeded on the cells at the face and the field ([#239](https://github.com/osrep/Runes2P.jl/pull/239)).
  The stored pattern and the partials of the term are unchanged.
  An entry that the term shares with two or more other terms is summed in another order and changes at round-off, and the results with it.
  Such an entry is the column of the field in a row of the total energy that also holds the work of the field (`ProductTerm`).
- The stencil of an `AdvectionDiffusion` holds the radial neighbours only when the operator declares radial transport, and the corners of the 3×3 block in p and ξ only when it declares cross diffusion ([#228](https://github.com/osrep/Runes2P.jl/pull/228)).
  The keywords `radial_transport` and `cross_diffusion` declare them, both `true` for an operator built by hand.
  `stencil_pattern(grid; ...)` takes the same two keywords.
  `fokker_planck_operator` and `electron_electron_operator` declare neither, and a `FokkerPlanckTerm` with a plasma source clears both, since the rebuild of its coefficients writes neither.
  The radii of such an operator are then separate blocks of the direct factorization, unless radial transport acts on the same unknown.
  The values of the Jacobian are unchanged, and with a direct solver the factorization has another pivot order.
  The results then change within the Newton tolerance, and at round-off where the Newton iterations of the steps are the same.
  With `GMRES` the factors of the preconditioner have another pivot order too, so its Krylov iterations and its refreshes on demand can change, and the states within the Newton tolerance.
  With either solver the step lengths of a `StepController` may differ.
  A nonzero coefficient that an operator does not declare is refused, and the radial transport writers require `radial_transport`.
  `ParticleSourceTerm` with `shape = :lowest_cell` stores its first momentum cells alone and no column of the temperature.
  The settings record of an operator holds both declarations, new in the format 0.0.4.
  A run resumed from a checkpoint written before this change records the section of the operator as changed and warns of it.
- `MomentumInterfaceTerm` takes a hot-tail Fokker-Planck term whose coefficients follow the plasma unknowns (`lagged = ()` or `:ions`).
  Its Jacobian holds the columns of the temperature, the density, and the ion densities unless the lower term lags them, as DREAM's `PXiExternalKineticKinetic`.
  Until now it asked for `lagged = :all` ([#214](https://github.com/osrep/Runes2P.jl/pull/214)).
- `NewtonSolver` accepts a stiff step at the round-off floor of its residual at the first iterate that reaches it ([#179](https://github.com/osrep/Runes2P.jl/pull/179)).
  It is accepted when every row above `residual_tolerance` is within its round-off bound `eps Σ_j |(∂F/∂u)_ij u_j|`, computed from the stored Jacobian when that holds every implicit term exactly.
  Until now such a step waited for the residual to stop decreasing, which at the floor wanders for a few iterations; that stall test stays where the bound is not available or a row is above it.
  In the disruption case at DREAM's resolution the accepted states change at round-off, 1e-14 of the bulk entries of `f`.
  The Newton count of its quench falls by 9 % (4968 iterations in 1500 steps, from 5455).
- `radial_values(u, layout, id)`, the view of a radial unknown over the radii, is renamed `unknown_values(u, layout, id)` ([#123](https://github.com/osrep/Runes2P.jl/pull/123)).
  `radial_values(profile, points)`, the evaluation of a radial profile, keeps its name.
- `FreeElectronEnergyTransferTerm` is renamed `free_electron_energy_transfer_term`, and `PrescribedFokkerPlanckTerm` is renamed `prescribed_fokker_planck_term`, since both are functions that build a term ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- The argument `coulomb_log` of `critical_field`, `dreicer_field` and `thermal_collision_time` is renamed `coulomb_logarithm` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- The keywords `thermal_coulomb_log` and `relativistic_coulomb_log` of `connor_hastie_rate` are renamed `thermal_coulomb_logarithm` and `relativistic_coulomb_logarithm` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- The keyword `kinetic` of `IonRateTerm` and `RadiatedPowerTerm`, which takes a `KineticIonizationTerm`, is renamed `kinetic_ionization` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
  So are the positional argument of `SecondaryHeatingTerm` and `ionization_energy_flows` and the field of the three types.
  The settings records keep the key `"kinetic_ionization"`.
- The keyword `cross_section_form` of `KineticIonizationTerm` is renamed `ionization_cross_section_form`, the name of the setting of `CollisionSettings` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
  The key `"cross_sections"` of its settings record is unchanged.
- The keyword `electric_field` of `fokker_planck_operator` and `set_fokker_planck_coefficients!`, a switch of type `Bool`, is renamed `field_in_coefficients` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
  The settings record keeps the key `"electric_field"` of the coefficient source.
- The keyword `total` of `total_energy_equation` is renamed `total_energy`, and its argument `distribution` is renamed `kinetic` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- The keyword `current` of `ohmic_heating_term` is renamed `current_density` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- The keyword `density` of `coronal_equilibrium`, the total density of the species, is renamed `species_density` ([#124](https://github.com/osrep/Runes2P.jl/pull/124)).
- `add_local_jacobian!`, `plasma_columns`, `lags_in_ode_form` and `is_local` are declared `public` instead of exported ([#122](https://github.com/osrep/Runes2P.jl/pull/122)).
  A term or an operator that adds methods to them imports them with `using Runes2P: ...` or qualifies them.
- `thermal_coulomb_logarithm`, `relativistic_coulomb_logarithm`, `coulomb_logarithm` and `braams_karney_conductivity` take the density first; until now the temperature came first ([#134](https://github.com/osrep/Runes2P.jl/pull/134)).
  The new orders are `thermal_coulomb_logarithm(free_electron_density, temperature)` and `relativistic_coulomb_logarithm(free_electron_density, temperature)`.
  `coulomb_logarithm(collision, model, p, free_electron_density, temperature)` and `braams_karney_conductivity(free_electron_density, temperature, effective_charge)` follow.
- `free_critical_momentum` and `total_critical_momentum` take the thermal Coulomb logarithm by default, as DREAM's default run does; until now they took the relativistic one ([#158](https://github.com/osrep/Runes2P.jl/pull/158)).
  So the default ceiling of the cutoff of `KnockOnCutoffUpdate` moves.
  Their new keyword `coulomb_logarithm` takes the Coulomb logarithm of the collisions: `:thermal`, or `:energy_dependent` and `:relativistic`, which give the former values.
  `KnockOnCutoffUpdate` takes the same keyword for its default ceiling, and its settings record has the new key `"coulomb_logarithm"`.
  A run refuses its default `:thermal` next to a Fokker-Planck term whose collisions take another model.
  `OUTPUT_FORMAT_VERSION` is 0.0.2 for the new key.
- The residual test of `NewtonSolver` weights the differential rows of a kinetic unknown by the larger of the row scale and the entry at the start of the step ([#163](https://github.com/osrep/Runes2P.jl/pull/163)).
  The collision fluxes of the bulk cancel in these rows, and the round-off of that cancellation held a step of the disruption case at the residual floor, with one or two extra iterations.
  The accepted states move by at most 2e-8 of the scale of the unknown in that case.
- A `RadialUnknown` given `groups`, even a single group, declares them: it takes the tolerances per group, and the layout of the settings record and of the output files writes its `group_starts` ([#181](https://github.com/osrep/Runes2P.jl/pull/181)).
  Until now `group_starts` was written only for more than one group.
- `time_points` of a `Leg` places the save points as DREAM does, a running sum of `steps / saves` rounded half away from zero: `Leg(T; steps = 5, saves = 2)` saves step 3 (step 2 until now) ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).
- An update of the run whose changes are continuous in time (`changes_continuously`, a `RechesterRosenbluthUpdate` without forced time points) keeps the history of `StepController`; it still discards the stored Jacobian ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).
  Before, every change restarted the step controller with a doubling.
  A level given as a function counts as continuous; a level that jumps is given as a `PiecewiseConstant`.
- `NewtonWorkspace` warns when the `relative_tolerance` of `GMRES` is above the `floor_linear_tolerance` of `NewtonSolver`, where neither waiver of the Newton convergence test applies ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).

### Removed

- `with_state_ions`, `threshold_split`, `local_layout`, `resolved_pitch_options`, `write_header`, `write_entry` and `log_event!` are no longer exported ([#122](https://github.com/osrep/Runes2P.jl/pull/122)).
  They remain reachable as `Runes2P.<name>`, but they are internal and may change without an entry here.

### Fixed

- The like-species ratio N²²/M²² of the three-mode `FieldParticleOperator` is −2625/8385, the value of eq. (5.22) of Hirshman and Sigmar (1981) ([#156](https://github.com/osrep/Runes2P.jl/pull/156)).
  The former −2125/8385 came from a misprint of N²² in Sugama et al. (2019), their equation (C5).
  The conductivity of the default three-mode model moves by up to 0.03 %, depending on the effective charge.
- `StepDoubling` rejects a step in which an unknown with no absolute tolerance is zero after the full step and not after the two half steps, as DREAM's acceptance does; before, it accepted it ([#157](https://github.com/osrep/Runes2P.jl/pull/157)).
  The next step follows from the largest error of the other unknowns, as in DREAM, and is half the step tried when that error is zero.
- `ProgressLog` writes a message with `maxlog = n` at most `n` times and leaves `maxlog` out of the printed keywords, as Julia's loggers do ([#165](https://github.com/osrep/Runes2P.jl/pull/165)).
- An ion composition of a single species takes the tolerances relative to its own density, in Newton and in the step controller, as a composition of several species does ([#181](https://github.com/osrep/Runes2P.jl/pull/181)).
  Until now a single species was measured against the scale of the unknown, so a trace impurity alone converged loosely.
- The split of the ionization at the threshold of a `KineticIonizationTerm` stays finite where the threshold lies in the first momentum cell ([#181](https://github.com/osrep/Runes2P.jl/pull/181)).
  The part of the Maxwellian below such a threshold is taken as the weight of that cell, with a warning.
  Until now it was the part the envelope gives within one cell, which tends to 0 as the threshold nears the lower face.
- The `:smooth` envelope of a `MomentumRegion` takes the width of the last cell when no cell centre lies above its limit ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).
  Before, a limit at the last centre gave NaN, and the envelope jumped as the limit crossed it; the envelope of every limit above the last centre changes.
- `RadiusBlockILU` refuses a factorization with a zero diagonal of `U`; before, a zero last pivot of a block passed and the solve divided by zero ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).
- `fill_initial_values!` refuses a row that couples the entries of its own unknown with a message that says so; before, the message named units or a cycle ([#205](https://github.com/osrep/Runes2P.jl/pull/205)).
- The `reason` of a settings record whose prescribed function fails holds the type of the exception without its module and the first line of its message ([#265](https://github.com/osrep/Runes2P.jl/pull/265)).
  Absolute file paths in it are replaced by `<path>`; before, it held the full message, so the record depended on the machine, the checkout and the packages loaded.
  A run resumed from a checkpoint with the old reason records its section as changed and warns of it.
- `IonEnergyExchangeTerm` refuses a mass of `masses` outside the range 1e-27 kg to 1e-24 kg, with an `ArgumentError` that names the species ([#286](https://github.com/osrep/Runes2P.jl/pull/286)).
  Before, a mass in atomic mass units or in grams gave a wrong exchange without an error.
  The masses of `ion_mass` lie inside the range, and the results of a run do not change.
- `KineticEnergyLossTerm` with `part = :free_electrons`, and with it `free_electron_energy_transfer_term`, takes no flux through `p_max`, as the part `:sinks` ([#295](https://github.com/osrep/Runes2P.jl/pull/295)).
  Before, it took the electron-electron flux through `p_max` of every upper boundary that carries one: `:f_zero`, `:constant_flux`, `:extrapolated_flux` and `:extrapolated_flux_consistent`.
  `:interface` carries none, and `:outflow` carries one only where the electron-electron advection at `p_max` points outward.
  With `:f_zero`, on the grid of the row checks of `test/test_energy_budget.jl` with its tail at `p_max`, the old transfer was larger than the new one by 1.0e-5 of the new one with exponential fitting and by 2.5e-1 with upwind.
- With the change above, the heating of the cold electrons by `f_re` in the disruption case with a runaway grid at `compat = :runes2p` rises ([#295](https://github.com/osrep/Runes2P.jl/pull/295)).
  Its runaway grid takes `:constant_flux`, DREAM's default, which continues the flux of the last interior face through `p_max`, and that flux points into the grid.
  In the window of that case that is compared with DREAM, a heating of about 1e4 W/m³ rises by up to 2.2e-5 of its former value.
  In the state after the third step of the quench, where `f_re` started from zero, a heating of about 1e-4 W/m³ rises by up to 1.7 % of its former value.
  The case `heating` of the runaway grid does not change, since its `f_re` vanishes at `p_max`.
