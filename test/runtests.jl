using Runes2P
using Test
using LinearAlgebra: Diagonal

@testset "Runes2P" begin
    include("test_grids.jl")
    include("test_advection_diffusion.jl")
    include("test_jacobian.jl")
    include("test_physics.jl")
    include("test_collisions_dream.jl")
    include("test_kinetic_operator_dream.jl")
end
