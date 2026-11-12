# Changelog

The changes of Runes2P that users see: the public API, new features and changed numerical results.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until the first release, every change is listed under "Unreleased", with the subsections in the order of that format.

## Unreleased

### Added

- `MKLPardiso` solves the Newton systems with Intel MKL PARDISO; its solves come with the extension `Runes2PMKLPardisoExt`, loaded by `using Pardiso` ([#243](https://github.com/osrep/Runes2P.jl/pull/243)).
  It takes the systems of `SparseLU`, the field-particle term by the same Woodbury update, and runs threaded with `threads`; `refinement_steps` sets PARDISO's refinement steps.
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
- The `reason` of a settings record whose prescribed function fails keeps only the type of the exception and the first line of its message ([#265](https://github.com/osrep/Runes2P.jl/pull/265)).
  A path in it of two or more parts from the root, `~` or a drive letter is replaced by `<path>`.
  Until now it held the full message, which for a `MethodError` lists the closest candidates with their source files, so the record depended on the machine, the checkout and the packages loaded.
  A run with such a function resumed from a checkpoint written before this change records its section as changed and warns of it.
