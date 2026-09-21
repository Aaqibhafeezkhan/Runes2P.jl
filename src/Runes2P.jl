"""
    Runes2P

A prototype of a DREAM-style relativistic electron kinetic solver, written to test whether a
rewrite focused on kinetic accuracy and parallel CPU scaling is feasible.
"""
module Runes2P

include("physics/constants.jl")
include("physics/atomic_data.jl")
include("physics/plasma.jl")
include("physics/coulomb_log.jl")
include("physics/collision_frequencies.jl")
include("grids/momentum.jl")
include("grids/radial.jl")
include("grids/kinetic.jl")
include("operators/advection_diffusion.jl")
include("operators/kinetic_terms.jl")
include("operators/knockon/kinematics.jl")
include("operators/knockon/energy_kernel.jl")
include("operators/knockon/angle_kernel.jl")
include("operators/knockon/operator.jl")
include("operators/knockon/cutoff.jl")
include("system/jacobian.jl")

export PXiGrid, uniform_p_faces, biuniform_p_faces, uniform_xi_faces
export RadialGrid, uniform_r_faces
export KineticGrid, cylindrical_grid, density, total_particles
export n_p, n_xi, n_r, ncells
export AdvectionDiffusion, apply!
export Constants, IonSpecies, PlasmaState, free_electron_density, bound_electron_density
export lnlambda, lnlambda_thermal, lnlambda_relativistic
export CollisionSettings, CollisionModel, nu_s, nu_D, nu_par, p3nu_s_at_zero
export mean_excitation_energy, ion_size
export add_electric_field!, add_slowing_down!, add_pitch_scattering!, add_energy_diffusion!
export add_synchrotron!, fokker_planck_operator
export SparseJacobian, jacobian, jacobian!, operator_matrix, stencil_pattern, sparse_ad_backend
export is_local, check_stencil_pattern
export xi_star, moller_cross_section, moller_flux, max_knockon_momentum, moller_total_rate
export MollerEnergyKernel, MollerAngleKernel, pitch_cell_fraction, delta_column!, delta_column_split!, delta_matrix!
export KnockOnOperator, KnockOnKernels, knockon_linear_operator, set_cutoff!, energy_kernel, angle_kernel
export thermal_momentum, critical_field, critical_momentum, cutoff_window, choose_cutoff

end # module Runes2P
