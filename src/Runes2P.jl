"""
    Runes2P

A prototype of a DREAM-style relativistic electron kinetic solver, written to test whether a
rewrite focused on kinetic accuracy and parallel CPU scaling is feasible.
"""
module Runes2P

include("physics/constants.jl")
include("physics/atomic_data.jl")
include("physics/plasma.jl")
include("physics/coulomb_logarithm.jl")
include("physics/collision_frequencies.jl")
include("grids/momentum.jl")
include("grids/radial.jl")
include("grids/kinetic.jl")
include("operators/flux_limiters.jl")
include("operators/advection_diffusion.jl")
include("operators/kinetic_terms.jl")
include("operators/knockon/kinematics.jl")
include("operators/knockon/energy_kernel.jl")
include("operators/knockon/pitch_kernel.jl")
include("operators/knockon/operator.jl")
include("operators/knockon/cutoff.jl")
include("physics/runaway_rates.jl")
include("system/jacobian.jl")
include("system/layout.jl")
include("system/terms.jl")
include("system/equation_system.jl")
include("solvers/linear_solver.jl")
include("solvers/newton.jl")
include("solvers/krylov.jl")
include("settings.jl")
include("solvers/interval_updates.jl")
include("solvers/checkpoint.jl")
include("solvers/time_stepping.jl")

export MomentumGrid, uniform_p_faces, biuniform_p_faces, tanh_p_faces, uniform_xi_faces
export biuniform_theta_xi_faces
export RadialGrid, uniform_r_faces
export KineticGrid, cylindrical_grid, density, current_density, pitch_moment_weights, total_particles, has_shared_momentum_grid, momentum_grid
export phase_space_jacobian, phase_space_jacobian_p_faces, phase_space_jacobian_xi_faces
export phase_space_jacobian_r_faces, phase_space_jacobian_over_p2_at_zero
export momentum_cell_count, pitch_cell_count, radial_cell_count, cell_count
export AdvectionDiffusion, apply!, momentum_flux, with_scheme
export Constants, IonSpecies, PlasmaState, free_electron_density, bound_electron_density
export coulomb_logarithm, thermal_coulomb_logarithm, relativistic_coulomb_logarithm
export CollisionSettings, CollisionModel, slowing_down_frequency, deflection_frequency
export parallel_diffusion_frequency, p3_slowing_down_frequency_at_zero
export mean_excitation_energy, ion_size
export add_electric_field!, add_slowing_down!, add_pitch_scattering!, add_energy_diffusion!
export add_synchrotron!, fokker_planck_operator, zero_coefficients!, set_fokker_planck_coefficients!
export SparseJacobian, jacobian, jacobian!, operator_matrix, stencil_pattern, sparse_ad_backend
export is_local, check_stencil_pattern
export xi_star, moller_cross_section, moller_flux, max_knockon_momentum, moller_total_rate
export MollerEnergyKernel, MollerPitchKernel, pitch_cell_fraction
export pitch_kernel_column!, pitch_kernel_column_split!, pitch_kernel_matrix!
export KnockOnOperator, KnockOnKernels, knockon_linear_operator, set_cutoff!, energy_kernel, pitch_kernel
export knockon_kernels, momentum_at_xi_star
export apply_at_radius!
export thermal_momentum, critical_field, critical_momentum, cutoff_window, choose_cutoff
export dreicer_field, thermal_collision_time, connor_hastie_rate, avalanche_growth_rate
export ScalarUnknown, RadialUnknown, KineticUnknown, LowerLimit, UnknownLayout
export unknown_id, unknown_names, unknown, unknown_range, radius_block_range, scalar_block_range
export radial_view, kinetic_view, scalar_view, unknown_view, unknown_scales, describe, copy_unknowns!
export kinetic_grid
export EquationTerm, dependencies, coupling, treatment, sparsity!, rhs!, jacobian_vector_product!
export jacobian_kind, jacobian_correction_product!, is_linear
export FokkerPlanckTerm, PrescribedFokkerPlanckTerm, KnockOnTerm, LinearTerm, DensityMomentTerm, CurrentMomentTerm, MaxwellianSourceTerm, LayoutKineticArray, kinetic_array
export Equation, TimeDerivative, Algebraic, EquationSystem, Tolerance, tolerance
export residual!, mass_matrix, jacobian_sparsity, jacobian_prototype, jacobian_operator, jacobian_colors, residual_scales, ode_function
export LinearSolver, SparseLU, GMRES, RadiusBlockLU, RadiusBlockILU, NewtonSolver, NewtonWorkspace, NewtonResult, newton_solve!
export SolverStatistics, TimeSolution, TimeStepFailure, integrate, write_output
export OUTPUT_FORMAT_VERSION, DescriptionContext, run_settings, read_settings, settings_hash, settings_fields
export code_version, run_environment, describe_prescribed_plasma, resolved_pitch_options
export IntervalUpdate, KnockOnCutoffUpdate, free_critical_momentum
export CheckpointSettings, StoredState, read_checkpoint, write_checkpoint, read_state, read_output
export StopCondition, stop_reason, run_status, exit_status

end # module Runes2P
