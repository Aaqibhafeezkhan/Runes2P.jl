"""
    KineticGrid{T}

Phase-space grid `(p, ξ, r)` with a momentum grid per radius, and the phase-space
Jacobian `𝒱'` (DREAM `Vp`) at cell centres and at the faces in each direction.

Callers use the accessors, not the fields: [`momentum_grid`](@ref),
[`phase_space_jacobian`](@ref) and its face variants, the cell counts, and
[`has_shared_momentum_grid`](@ref). When all radii share one momentum grid, a
distribution can be stored as a 3D array `f[i, j, ir]` with `p` fastest, and the
accessors without a radius argument (`momentum_grid(grid)`, `size(grid)`, …) are
available; they throw an `ArgumentError` otherwise. Radial faces have a single momentum
grid only in the shared case, so the radial-face Jacobian exists only there (face
quantities for different neighbouring grids are not defined yet).

In cylindrical geometry `𝒱' = 2π p² V'`, so that the density is
`n(r) = (1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j f_ij`.
"""
struct KineticGrid{T<:AbstractFloat}
    radial_grid::RadialGrid{T}
    momentum_grids::Vector{MomentumGrid{T}}
    jacobian_centers::Vector{Matrix{T}}            # per radius, (p cells, ξ cells)
    jacobian_p_faces::Vector{Matrix{T}}            # per radius, (p cells + 1, ξ cells)
    jacobian_xi_faces::Vector{Matrix{T}}           # per radius, (p cells, ξ cells + 1)
    jacobian_r_faces::Union{Nothing,Array{T,3}}    # (p cells, ξ cells, radii + 1); shared grid only
    jacobian_over_p2_at_zero::Vector{Vector{T}}    # per radius, lim_{p→0} 𝒱'/p² per ξ cell
    shared::Bool
end

"""
    cylindrical_grid(momentum_grid, radial_grid)
    cylindrical_grid(momentum_grids::AbstractVector, radial_grid)

Kinetic grid with a homogeneous magnetic field (no trapping, no bounce averaging), with
one momentum grid for all radii or one per radius. Per-radius grids with identical faces
count as shared.
"""
function cylindrical_grid(momentum_grid::MomentumGrid{T}, radial_grid::RadialGrid{T}) where {T}
    return cylindrical_grid(fill(momentum_grid, radial_cell_count(radial_grid)), radial_grid)
end

function cylindrical_grid(momentum_grids::AbstractVector{MomentumGrid{T}},
                          radial_grid::RadialGrid{T}) where {T}
    nr = radial_cell_count(radial_grid)
    length(momentum_grids) == nr ||
        throw(DimensionMismatch("need one momentum grid per radius"))
    first_grid = first(momentum_grids)
    shared = all(m -> m === first_grid ||
                      (m.p_faces == first_grid.p_faces && m.xi_faces == first_grid.xi_faces),
                 momentum_grids)
    grids = shared ? fill(first_grid, nr) : collect(momentum_grids)
    Vr, Vr_faces = radial_grid.spatial_jacobian, radial_grid.spatial_jacobian_faces
    # 2π p² V'(r) on (p positions, ξ count) for each radius
    per_radius(p_positions, ir, nxi) = [2pi * p^2 * Vr[ir] for p in p_positions, _ in 1:nxi]
    centers = [per_radius(m.p_centers, ir, pitch_cell_count(m)) for (ir, m) in enumerate(grids)]
    on_p_faces = [per_radius(m.p_faces, ir, pitch_cell_count(m)) for (ir, m) in enumerate(grids)]
    on_xi_faces = [per_radius(m.p_centers, ir, pitch_cell_count(m) + 1) for (ir, m) in enumerate(grids)]
    on_r_faces = if shared
        p = first_grid.p_centers
        [2pi * p[i]^2 * Vr_faces[ir] for i in 1:momentum_cell_count(first_grid),
                                          _ in 1:pitch_cell_count(first_grid), ir in 1:nr+1]
    else
        nothing
    end
    over_p2_at_zero = [fill(2pi * Vr[ir], pitch_cell_count(m)) for (ir, m) in enumerate(grids)]
    return KineticGrid{T}(radial_grid, grids, centers, on_p_faces, on_xi_faces, on_r_faces,
                          over_p2_at_zero, shared)
end

"""
    has_shared_momentum_grid(grid)

Whether all radii use the same momentum grid, so that a distribution fits a 3D array.
"""
has_shared_momentum_grid(grid::KineticGrid) = grid.shared

function _require_shared(grid::KineticGrid)
    grid.shared || throw(ArgumentError("the kinetic grid has different momentum grids at " *
                                       "different radii; pass a radius index"))
    return nothing
end

"""
    momentum_grid(grid, ir)
    momentum_grid(grid)

Momentum grid at radius index `ir`; without `ir`, the grid shared by all radii.
"""
momentum_grid(grid::KineticGrid, ir::Integer) = grid.momentum_grids[ir]
momentum_grid(grid::KineticGrid) = (_require_shared(grid); first(grid.momentum_grids))

"""
    phase_space_jacobian(grid, ir)
    phase_space_jacobian_p_faces(grid, ir)
    phase_space_jacobian_xi_faces(grid, ir)
    phase_space_jacobian_over_p2_at_zero(grid, ir)

Phase-space Jacobian `𝒱'` at radius index `ir`: at cell centres `(p cells, ξ cells)`, on
the p faces `(p cells + 1, ξ cells)`, on the ξ faces `(p cells, ξ cells + 1)`, and the limit
`𝒱'/p²` at `p = 0` for each ξ cell.
"""
phase_space_jacobian(grid::KineticGrid, ir::Integer) = grid.jacobian_centers[ir]
phase_space_jacobian_p_faces(grid::KineticGrid, ir::Integer) = grid.jacobian_p_faces[ir]
phase_space_jacobian_xi_faces(grid::KineticGrid, ir::Integer) = grid.jacobian_xi_faces[ir]
phase_space_jacobian_over_p2_at_zero(grid::KineticGrid, ir::Integer) = grid.jacobian_over_p2_at_zero[ir]

"""
    phase_space_jacobian_r_faces(grid)

Phase-space Jacobian on the radial faces, `(p cells, ξ cells, radii + 1)`; defined only
for a shared momentum grid.
"""
phase_space_jacobian_r_faces(grid::KineticGrid) = (_require_shared(grid); grid.jacobian_r_faces)

momentum_cell_count(grid::KineticGrid, ir::Integer) = momentum_cell_count(momentum_grid(grid, ir))
pitch_cell_count(grid::KineticGrid, ir::Integer) = pitch_cell_count(momentum_grid(grid, ir))
cell_count(grid::KineticGrid, ir::Integer) = cell_count(momentum_grid(grid, ir))
momentum_cell_count(grid::KineticGrid) = momentum_cell_count(momentum_grid(grid))
pitch_cell_count(grid::KineticGrid) = pitch_cell_count(momentum_grid(grid))
radial_cell_count(grid::KineticGrid) = radial_cell_count(grid.radial_grid)
cell_count(grid::KineticGrid) = sum(ir -> cell_count(grid, ir), 1:radial_cell_count(grid))

"""
    size(grid::KineticGrid)

`(p cells, ξ cells, radii)` of a grid with a shared momentum grid, the shape of a
distribution stored as a 3D array.
"""
Base.size(grid::KineticGrid) = (momentum_cell_count(grid), pitch_cell_count(grid), radial_cell_count(grid))

"""
    density(grid, f)

Density moment `n(r) = (1/V') ∫ 𝒱' f dp dξ` at each radius, for `f` stored as a 3D array
(shared momentum grid).
"""
function density(grid::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    size(f) == size(grid) || throw(DimensionMismatch("f must have the grid size"))
    n = zeros(promote_type(T, S), radial_cell_count(grid))
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            n[ir] += V[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j, ir]
        end
    end
    n ./= grid.radial_grid.spatial_jacobian
    return n
end

"""
    total_particles(grid, f)

Total particle content `Σ 𝒱' Δp Δξ Δr f`, conserved by a flux-form discretization, for `f`
stored as a 3D array (shared momentum grid).
"""
function total_particles(grid::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    size(f) == size(grid) || throw(DimensionMismatch("f must have the grid size"))
    dr = grid.radial_grid.r_widths
    total = zero(promote_type(T, S))
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            total += V[i, j] * m.p_widths[i] * m.xi_widths[j] * dr[ir] * f[i, j, ir]
        end
    end
    return total
end
