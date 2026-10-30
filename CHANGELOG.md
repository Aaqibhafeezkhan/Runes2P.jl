# Changelog

The changes of Runes2P that users see: the public API, new features and changed numerical results.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until the first release, every change is listed under "Unreleased", with the subsections in the order of that format.

## Unreleased

### Added

- `validate`, `has_plasma_columns`, `plasma_column_sparsity!` and `add_plasma_columns!`, the functions that a new term implements and the system calls, are declared `public` ([#126](https://github.com/osrep/Runes2P.jl/pull/126)).

### Changed

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

### Removed

- `with_state_ions`, `threshold_split`, `local_layout`, `resolved_pitch_options`, `write_header`, `write_entry` and `log_event!` are no longer exported ([#122](https://github.com/osrep/Runes2P.jl/pull/122)).
  They remain reachable as `Runes2P.<name>`, but they are internal and may change without an entry here.

### Fixed

- The like-species ratio N²²/M²² of the three-mode `FieldParticleOperator` is −2625/8385, the value of eq. (5.22) of Hirshman and Sigmar (1981) ([#156](https://github.com/osrep/Runes2P.jl/pull/156)).
  The former −2125/8385 came from a misprint of N²² in Sugama et al. (2019), their equation (C5).
  The conductivity of the default three-mode model moves by up to 0.03 %, depending on the effective charge.
- `StepDoubling` rejects a step in which an unknown with no absolute tolerance is zero after the full step and not after the two half steps, as DREAM's acceptance does; before, it accepted it ([#157](https://github.com/osrep/Runes2P.jl/pull/157)).
  The retry takes half the step tried.
