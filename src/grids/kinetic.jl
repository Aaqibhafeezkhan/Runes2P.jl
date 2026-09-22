"""
    KineticGrid{T}

Phase-space grid `(p, ξ, r)` with a momentum grid shared by all radii, and the
phase-space Jacobian `𝒱'` (DREAM `Vp`) at cell centres and at the faces in each
direction. The distribution function is stored as `f[i, j, ir]` with `p` fastest.

In cylindrical geometry `𝒱' = 2π p² V'`, so that the density is
`n(r) = (1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j f_ij`.
"""
struct KineticGrid{T<:AbstractFloat}
    momentum_grid::MomentumGrid{T}
    radial_grid::RadialGrid{T}
    phase_space_jacobian::Array{T,3}           # (p cells, ξ cells, radii)
    phase_space_jacobian_p_faces::Array{T,3}   # (p cells + 1, ξ cells, radii)
    phase_space_jacobian_xi_faces::Array{T,3}  # (p cells, ξ cells + 1, radii)
    phase_space_jacobian_r_faces::Array{T,3}   # (p cells, ξ cells, radii + 1)
    phase_space_jacobian_over_p2_at_zero::Matrix{T}  # lim_{p→0} 𝒱'/p², (ξ cells, radii)
end

"""
    cylindrical_grid(momentum_grid, radial_grid)

Kinetic grid with a homogeneous magnetic field (no trapping, no bounce averaging).
"""
function cylindrical_grid(momentum_grid::MomentumGrid{T}, radial_grid::RadialGrid{T}) where {T}
    np, nxi, nr = momentum_cell_count(momentum_grid), pitch_cell_count(momentum_grid),
                  radial_cell_count(radial_grid)
    p, p_faces = momentum_grid.p_centers, momentum_grid.p_faces
    Vr, Vr_faces = radial_grid.spatial_jacobian, radial_grid.spatial_jacobian_faces
    centers = [2pi * p[i]^2 * Vr[ir] for i in 1:np, _ in 1:nxi, ir in 1:nr]
    on_p_faces = [2pi * p_faces[i]^2 * Vr[ir] for i in 1:np+1, _ in 1:nxi, ir in 1:nr]
    on_xi_faces = [2pi * p[i]^2 * Vr[ir] for i in 1:np, _ in 1:nxi+1, ir in 1:nr]
    on_r_faces = [2pi * p[i]^2 * Vr_faces[ir] for i in 1:np, _ in 1:nxi, ir in 1:nr+1]
    over_p2_at_zero = [2pi * Vr[ir] for _ in 1:nxi, ir in 1:nr]
    return KineticGrid{T}(momentum_grid, radial_grid, centers, on_p_faces, on_xi_faces,
                          on_r_faces, over_p2_at_zero)
end

momentum_cell_count(grid::KineticGrid) = momentum_cell_count(grid.momentum_grid)
pitch_cell_count(grid::KineticGrid) = pitch_cell_count(grid.momentum_grid)
radial_cell_count(grid::KineticGrid) = radial_cell_count(grid.radial_grid)
Base.size(grid::KineticGrid) = (momentum_cell_count(grid), pitch_cell_count(grid), radial_cell_count(grid))
cell_count(grid::KineticGrid) = prod(size(grid))

"""
    density(grid, f)

Density moment `n(r) = (1/V') ∫ 𝒱' f dp dξ` at each radius.
"""
function density(grid::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    m = grid.momentum_grid
    n = zeros(promote_type(T, S), radial_cell_count(grid))
    for ir in 1:radial_cell_count(grid), j in 1:pitch_cell_count(grid), i in 1:momentum_cell_count(grid)
        n[ir] += grid.phase_space_jacobian[i, j, ir] * m.p_widths[i] * m.xi_widths[j] * f[i, j, ir]
    end
    n ./= grid.radial_grid.spatial_jacobian
    return n
end

"""
    total_particles(grid, f)

Total particle content `Σ 𝒱' Δp Δξ Δr f`, conserved by a flux-form discretization.
"""
function total_particles(grid::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    m = grid.momentum_grid
    dr = grid.radial_grid.r_widths
    total = zero(promote_type(T, S))
    for ir in 1:radial_cell_count(grid), j in 1:pitch_cell_count(grid), i in 1:momentum_cell_count(grid)
        total += grid.phase_space_jacobian[i, j, ir] * m.p_widths[i] * m.xi_widths[j] * dr[ir] * f[i, j, ir]
    end
    return total
end
