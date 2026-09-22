using Runes2P
using Test
using LinearAlgebra: Diagonal

include("example_cases.jl")

@testset "Runes2P" begin
    include("test_grids.jl")
    include("test_per_radius_grids.jl")
    include("test_advection_diffusion.jl")
    include("test_jacobian.jl")
    include("test_layout.jl")
    include("test_equation_system.jl")
    include("test_sciml_extension.jl")
    include("test_time_stepping.jl")
    include("test_newton_krylov.jl")
    include("test_discretization.jl")
    include("test_physics.jl")
    include("test_runaway_rates.jl")
    include("test_runaway_physics.jl")
    include("test_collisions_dream.jl")
    include("test_kinetic_operator_dream.jl")
    include("test_time_stepping_dream.jl")
    include("test_example_runaway.jl")
    include("test_knockon_kinematics.jl")
    include("test_knockon_energy_kernel.jl")
    include("test_knockon_pitch_kernel.jl")
    include("test_knockon_operator.jl")
    include("test_example_avalanche.jl")
    include("test_knockon_physics.jl")
    include("test_knockon_implicit.jl")
    include("test_knockon_cutoff.jl")
    include("test_knockon_reference.jl")
    include("test_regression_values.jl")
end
