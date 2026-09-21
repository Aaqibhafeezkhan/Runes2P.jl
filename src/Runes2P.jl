"""
    Runes2P

Prototype of a DREAM-style relativistic electron kinetic solver, written to test
whether a rewrite focused on kinetic accuracy and parallel CPU scaling is feasible.
"""
module Runes2P

include("grids/momentum.jl")
include("grids/radial.jl")
include("grids/kinetic.jl")
include("operators/advection_diffusion.jl")
include("system/jacobian.jl")

export PXiGrid, uniform_p_faces, biuniform_p_faces, uniform_xi_faces
export RadialGrid, uniform_r_faces
export KineticGrid, cylindrical_grid, density, total_particles
export np, nxi, nr, ncells
export AdvectionDiffusion, apply!
export SparseJacobian, jacobian, jacobian!, operator_matrix

end # module Runes2P
