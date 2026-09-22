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
include("operators/advection_diffusion.jl")
include("operators/kinetic_terms.jl")
include("operators/knockon/kinematics.jl")
include("operators/knockon/energy_kernel.jl")
include("operators/knockon/pitch_kernel.jl")
include("operators/knockon/operator.jl")
include("operators/knockon/cutoff.jl")
include("system/jacobian.jl")
include("system/layout.jl")
include("system/terms.jl")
include("system/equation_system.jl")
include("solvers/linear_solver.jl")
include("solvers/newton.jl")
include("solvers/time_stepping.jl")

export MomentumGrid, uniform_p_faces, biuniform_p_faces, uniform_xi_faces
export RadialGrid, uniform_r_faces
export KineticGrid, cylindrical_grid, density, total_particles, has_shared_momentum_grid, momentum_grid
export phase_space_jacobian, phase_space_jacobian_p_faces, phase_space_jacobian_xi_faces
export phase_space_jacobian_r_faces, phase_space_jacobian_over_p2_at_zero
export momentum_cell_count, pitch_cell_count, radial_cell_count, cell_count
export AdvectionDiffusion, apply!
export Constants, IonSpecies, PlasmaState, free_electron_density, bound_electron_density
export coulomb_logarithm, thermal_coulomb_logarithm, relativistic_coulomb_logarithm
export CollisionSettings, CollisionModel, slowing_down_frequency, deflection_frequency
export parallel_diffusion_frequency, p3_slowing_down_frequency_at_zero
export mean_excitation_energy, ion_size
export add_electric_field!, add_slowing_down!, add_pitch_scattering!, add_energy_diffusion!
export add_synchrotron!, fokker_planck_operator
export SparseJacobian, jacobian, jacobian!, operator_matrix, stencil_pattern, sparse_ad_backend
export is_local, check_stencil_pattern
export xi_star, moller_cross_section, moller_flux, max_knockon_momentum, moller_total_rate
export MollerEnergyKernel, MollerPitchKernel, pitch_cell_fraction
export pitch_kernel_column!, pitch_kernel_column_split!, pitch_kernel_matrix!
export KnockOnOperator, KnockOnKernels, knockon_linear_operator, set_cutoff!, energy_kernel, pitch_kernel
export knockon_kernels, momentum_at_xi_star
export apply_at_radius!
export thermal_momentum, critical_field, critical_momentum, cutoff_window, choose_cutoff
export ScalarUnknown, RadialUnknown, KineticUnknown, LowerLimit, UnknownLayout
export unknown_id, unknown_names, unknown, unknown_range, radius_block_range, scalar_block_range
export radial_view, kinetic_view, scalar_view, unknown_view, unknown_scales, describe, copy_unknowns!
export EquationTerm, dependencies, coupling, treatment, sparsity!, rhs!, jacobian_vector_product!
export FokkerPlanckTerm, KnockOnTerm, LinearTerm, DensityMomentTerm, LayoutKineticArray, kinetic_array
export Equation, TimeDerivative, Algebraic, EquationSystem, Tolerance, tolerance
export residual!, mass_matrix, jacobian_sparsity, jacobian_prototype, jacobian_operator, residual_scales, ode_function
export LinearSolver, SparseLU, NewtonSolver, NewtonWorkspace, NewtonResult, newton_solve!
export SolverStatistics, TimeSolution, TimeStepFailure, integrate, write_output

end # module Runes2P
