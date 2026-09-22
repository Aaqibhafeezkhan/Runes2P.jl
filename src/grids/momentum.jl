"""
    MomentumGrid{T}

Momentum grid in spherical coordinates `(p, ξ)`, with `p` the momentum normalized to
`mₑc` and `ξ = cos θ` the pitch. Cells are defined by their faces; cell centres are
face midpoints.
"""
struct MomentumGrid{T<:AbstractFloat}
    p_centers::Vector{T}   # length momentum_cell_count
    p_faces::Vector{T}     # length momentum_cell_count + 1
    p_widths::Vector{T}    # length momentum_cell_count
    xi_centers::Vector{T}  # length pitch_cell_count
    xi_faces::Vector{T}    # length pitch_cell_count + 1
    xi_widths::Vector{T}   # length pitch_cell_count
end

function MomentumGrid(p_faces::AbstractVector{T}, xi_faces::AbstractVector{T}) where {T<:AbstractFloat}
    length(p_faces) >= 2 || throw(ArgumentError("p grid needs at least one cell"))
    length(xi_faces) >= 2 || throw(ArgumentError("ξ grid needs at least one cell"))
    issorted(p_faces; lt=<=) || throw(ArgumentError("p faces must be strictly increasing"))
    issorted(xi_faces; lt=<=) || throw(ArgumentError("ξ faces must be strictly increasing"))
    first(p_faces) >= 0 || throw(ArgumentError("p faces must be non-negative"))
    (first(xi_faces) >= -1 && last(xi_faces) <= 1) || throw(ArgumentError("ξ faces must lie in [-1, 1]"))
    p_faces = collect(p_faces)
    xi_faces = collect(xi_faces)
    minimum(diff(p_faces)) > 1e-10 * (last(p_faces) - first(p_faces)) ||
        throw(ArgumentError("p grid has a degenerate cell"))
    minimum(diff(xi_faces)) > 1e-10 * (last(xi_faces) - first(xi_faces)) ||
        throw(ArgumentError("ξ grid has a degenerate cell"))
    p_centers = (p_faces[1:end-1] .+ p_faces[2:end]) ./ 2
    xi_centers = (xi_faces[1:end-1] .+ xi_faces[2:end]) ./ 2
    return MomentumGrid{T}(p_centers, p_faces, diff(p_faces), xi_centers, xi_faces, diff(xi_faces))
end

momentum_cell_count(grid::MomentumGrid) = length(grid.p_centers)
pitch_cell_count(grid::MomentumGrid) = length(grid.xi_centers)
cell_count(grid::MomentumGrid) = momentum_cell_count(grid) * pitch_cell_count(grid)

"""
    uniform_p_faces(cell_count, p_min, p_max)

Faces of a uniform momentum grid with `cell_count` cells.
"""
uniform_p_faces(cell_count::Integer, p_min::Real, p_max::Real) =
    collect(range(float(p_min), float(p_max); length=cell_count + 1))

"""
    biuniform_p_faces(cell_count, lower_cell_count, p_min, p_separation, p_max)

Faces of a two-region uniform momentum grid: `lower_cell_count` cells on
`[p_min, p_separation]` and `cell_count - lower_cell_count` cells on `[p_separation, p_max]`
(DREAM `PBiUniformGridGenerator`).
"""
function biuniform_p_faces(cell_count::Integer, lower_cell_count::Integer, p_min::Real,
                           p_separation::Real, p_max::Real)
    (1 <= lower_cell_count < cell_count) || throw(ArgumentError("each region needs at least one cell"))
    (p_min < p_separation < p_max) || throw(ArgumentError("require p_min < p_separation < p_max"))
    lower = range(float(p_min), float(p_separation); length=lower_cell_count + 1)
    upper = range(float(p_separation), float(p_max); length=cell_count - lower_cell_count + 1)
    return vcat(collect(lower), collect(upper)[2:end])
end

"""
    uniform_xi_faces(cell_count)

Faces of a uniform pitch grid on `[-1, 1]` with `cell_count` cells.
"""
uniform_xi_faces(cell_count::Integer) = collect(range(-1.0, 1.0; length=cell_count + 1))
