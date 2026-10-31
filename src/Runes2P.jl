"""
    Runes2P

A relativistic electron kinetic solver for runaway electrons in the style of DREAM: the
Fokker-Planck equation of the electrons on momentum grids per radius, coupled to the fluid
quantities of a cylindrical plasma and solved implicitly.
"""
module Runes2P

include("physics/constants.jl")
include("physics/data_files.jl")
include("physics/atomic_data.jl")
include("physics/plasma.jl")
include("physics/adas.jl")
include("physics/ionization_energies.jl")
include("physics/ionization_chianti.jl")
include("physics/ionization_chianti_fits.jl")
include("physics/ionization_bethe_fits.jl")
include("physics/ionization_cross_sections.jl")
include("physics/coulomb_logarithm.jl")
include("physics/conductivity.jl")
include("physics/collision_frequencies.jl")
include("grids/momentum.jl")
include("grids/pitch_overlap.jl")
include("grids/radial.jl")
include("grids/momentum_region.jl")
include("grids/kinetic.jl")
include("grids/radial_profiles.jl")
include("operators/flux_limiters.jl")
include("operators/advection_diffusion.jl")
include("operators/kinetic_terms.jl")
include("operators/radial_transport.jl")
include("operators/knockon/kinematics.jl")
include("operators/knockon/energy_kernel.jl")
include("operators/knockon/pitch_kernel.jl")
include("operators/knockon/operator.jl")
include("operators/knockon/cutoff.jl")
include("physics/runaway_rates.jl")
include("physics/effective_critical_field.jl")
include("system/jacobian.jl")
include("system/layout.jl")
include("physics/plasma_source.jl")
include("system/terms.jl")
include("system/equations.jl")
include("system/algebraic_terms.jl")
include("system/moment_terms.jl")
include("system/fokker_planck_terms.jl")
include("system/knockon_terms.jl")
include("system/interface_term.jl")
include("system/transport_terms.jl")
include("system/population_terms.jl")
include("system/avalanche_terms.jl")
include("system/energy_terms.jl")
include("system/kinetic_ionization.jl")
include("system/radiation_terms.jl")
include("system/heat_transport_terms.jl")
include("system/ion_energy_terms.jl")
include("operators/field_particle.jl")
include("system/energy_transfer_terms.jl")
include("system/conductivity_terms.jl")
include("system/ion_terms.jl")
include("system/equation_system.jl")
include("system/induction_terms.jl")
include("solvers/linear_solver.jl")
include("solvers/newton.jl")
include("solvers/block_preconditioners.jl")
include("solvers/fluid_schur_preconditioner.jl")
include("solvers/krylov.jl")
include("settings.jl")
include("solvers/interval_updates.jl")
include("solvers/transport_update.jl")
include("solvers/checkpoint.jl")
include("solvers/progress_log.jl")
include("solvers/legs.jl")
include("solvers/time_stepping.jl")

export MomentumGrid, uniform_p_faces, biuniform_p_faces, tanh_p_faces, uniform_xi_faces
export biuniform_theta_xi_faces
export PitchOverlap, overlap_cell_count, has_identical_grids, remap_to_right!,
       remap_to_left!
export has_same_cell_counts, has_same_p_faces, radial_face_overlap,
       radial_face_pitch_cell_count
export radial_neighbour_pitch_cells, runaway_grid
export MomentumInterfaceTerm, interface_flux
export add_rechester_rosenbluth_diffusion!, radial_values, radial_face_values,
       radial_particle_flux, radial_particle_content
export RadialEdgeLossTerm, LossRateTerm, PiecewiseConstant, RechesterRosenbluthUpdate,
       forced_time_points
export RadialGrid, uniform_r_faces
export KineticGrid, cylindrical_grid, density, current_density, pitch_moment_weights,
       total_particles, has_shared_momentum_grid, momentum_grid
export phase_space_jacobian, phase_space_jacobian_p_faces, phase_space_jacobian_xi_faces
export phase_space_jacobian_r_faces, phase_space_jacobian_over_p2_at_zero
export momentum_cell_count, pitch_cell_count, radial_cell_count, cell_count
export AdvectionDiffusion, apply!, momentum_flux, with_scheme
export Constants, IonSpecies, PlasmaState, free_electron_density, bound_electron_density
export IonComposition, component_count, species_components, species_index, component_groups
export WeightedSumTerm, free_electron_weights, ion_densities_unknown, set_ion_densities!,
       ion_densities
export deposit_neutrals!, population_equations, IonRateTerm, ion_rate_matrix,
       coronal_equilibrium
export RosenbluthPutvinskiSourceTerm, rosenbluth_putvinski_rate, ExternalAvalancheTerm
export KineticIonizationTerm, fluid_ionization_coefficient, kinetic_ionization_rates,
       MINIMUM_SPLIT_THRESHOLD_FACTORS
export IonDiffusionTerm
export SECONDARY_SPECTRUM_WIDTHS, mean_secondary_energy, beb_mean_secondary_energy,
       SecondaryHeatingTerm, ionization_energy_flows
export ADF11Table, read_adf11, adas_coefficient, adf11_file_name, adas_cache_directory,
       read_adas
export IONIZATION_ENERGIES, ionization_energy, KINETIC_IONIZATION_FITS,
       KINETIC_IONIZATION_BETHE_FITS, ionization_cross_section
export CHIANTI_IONIZATION_FITS, ChiantiIonization, chianti_cross_section
export PlasmaSource, PrescribedPlasma, PlasmaFromUnknowns, plasma_values, is_state_dependent
export coulomb_logarithm, thermal_coulomb_logarithm, relativistic_coulomb_logarithm
export CollisionSettings, CollisionModel, slowing_down_frequency, deflection_frequency
export parallel_diffusion_frequency, p3_slowing_down_frequency_at_zero,
       bremsstrahlung_cross_section
export mean_excitation_energy, ion_size
export add_electric_field!, add_slowing_down!, add_pitch_scattering!, add_energy_diffusion!
export add_synchrotron!, fokker_planck_operator, zero_coefficients!,
       set_fokker_planck_coefficients!
export set_unit_field_coefficients!, LayoutRadialValues, unknown_values, with_electric_field
export SparseJacobian, jacobian, jacobian!, operator_matrix, stencil_pattern,
       sparse_ad_backend
export check_stencil_pattern
export xi_star, moller_cross_section, moller_flux, max_knockon_momentum, moller_total_rate
export MollerEnergyKernel, MollerPitchKernel, pitch_cell_fraction
export pitch_kernel_column!, pitch_kernel_column_split!, pitch_kernel_matrix!
export KnockOnOperator, KnockOnKernels, knockon_linear_operator, set_cutoff!, energy_kernel,
       pitch_kernel
export knockon_kernels, momentum_at_xi_star
export apply_at_radius!
export electron_electron_operator, FieldParticleOperator, FieldParticleTerm,
       set_field_particle_coefficients!
export parallel_momentum, LowRankBlock, low_rank_jacobian, has_low_rank_jacobian
export thermal_momentum, critical_field, critical_momentum, cutoff_window, choose_cutoff
export dreicer_field, thermal_collision_time, connor_hastie_rate, avalanche_growth_rate
export effective_critical_field, avalanche_critical_momentum, fluid_avalanche_growth_rate,
       critical_field_advection,
       runaway_fluid_settings, SYNCHROTRON_RATE_PER_FIELD_SQUARED
export ScalarUnknown, RadialUnknown, KineticUnknown, LowerLimit, UnknownLayout
export unknown_id, unknown_names, unknown, unknown_range, radius_block_range,
       scalar_block_range
export radial_view, kinetic_view, scalar_view, unknown_view, unknown_scales, describe,
       copy_unknowns!
export kinetic_grid
export EquationTerm, dependencies, coupling, treatment, sparsity!, rhs!,
       jacobian_vector_product!
export jacobian_kind, jacobian_correction_product!, is_linear, jacobian_assembly,
       add_jacobian!, prepare_step!
export PrescribedValueTerm
export MomentumRegion, ThermalMomentum, momentum_limit, momentum_envelope,
       momentum_envelope!, is_time_dependent
export PoloidalFluxDiffusionTerm, CrossSectionIntegralTerm, cross_section_weights,
       edge_to_wall_inductance
export InductionNames, induction_unknowns, induction_equations, initial_poloidal_flux
export ProductTerm, THERMAL_ENERGY_FACTOR, thermal_energy_equation, set_thermal_energy!,
       EnergyMomentTerm
export ohmic_heating_term, RadiatedPowerTerm, radiated_power
export HeatTransportTerm, heat_flux_through_edge
export IonEnergyExchangeTerm, set_ion_thermal_energy!, ion_temperatures, ion_mass
export total_energy_equation, RunawayFluidEnergyTransferTerm,
       free_electron_energy_transfer_term, KineticEnergyLossTerm, InterfaceEnergyFluxTerm
export FokkerPlanckTerm, prescribed_fokker_planck_term, KnockOnTerm, LinearTerm,
       DensityMomentTerm, CurrentMomentTerm, MaxwellianSourceTerm, LayoutKineticArray,
       kinetic_array
export Equation, TimeDerivative, Algebraic, Constraint, EquationSystem, Tolerance, tolerance
export ParticleSourceTerm, source_shape, MomentumBoundaryFluxTerm
export normalized_conductivity, effective_charge, braams_karney_conductivity,
       test_particle_conductivity_fraction
export kinetic_conductivity, ConductivityModel, ConductivityTerm, conductivity_at
export residual!, fill_initial_values!, fill_maxwellian!, mass_matrix
export jacobian_sparsity, jacobian_prototype, jacobian_operator, jacobian_colors,
       residual_scales, ode_function
export FluidSchurLU
export LinearSolver, SparseLU, GMRES, RadiusBlockLU, RadiusBlockILU, NewtonSolver,
       NewtonWorkspace, NewtonResult, newton_solve!, invalidate_jacobian!
export SolverStatistics, TimeSolution, TimeStepFailure, integrate, write_output
export RunOptions, TimeStepper, FixedSteps, StepDoubling, StepController, Leg, time_points
export OUTPUT_FORMAT_VERSION, DescriptionContext, run_settings, read_settings,
       settings_hash, settings_fields
export code_version, run_environment, describe_prescribed_plasma
export IntervalUpdate, KnockOnCutoffUpdate, free_critical_momentum, total_critical_momentum
export kinetic_energy, momentum_at_energy, cutoff_target, choose_cutoff_face
export CheckpointSettings, StoredState, read_checkpoint, write_checkpoint, read_state,
       read_output
export StopCondition, stop_reason, run_status, exit_status
export ProgressLog

public add_local_jacobian!, plasma_columns, lags_in_ode_form, is_local, validate,
       has_plasma_columns, plasma_column_sparsity!, add_plasma_columns!

end # module Runes2P
