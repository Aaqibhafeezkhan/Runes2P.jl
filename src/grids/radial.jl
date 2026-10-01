"""
    RadialGrid{T}

Radial grid on `[0, a]` defined by its faces, with the spatial Jacobian `V'(r)` at cell
centres and faces (DREAM `VpVol`). For a cylinder `V'(r) = r`, i.e. the volume per unit
length divided by 2π.
"""
struct RadialGrid{T<:AbstractFloat}
    r_centers::Vector{T}
    r_faces::Vector{T}
    r_widths::Vector{T}
    spatial_jacobian::Vector{T}        # at cell centres
    spatial_jacobian_faces::Vector{T}  # at cell faces
end

function RadialGrid(r_faces::AbstractVector{T}) where {T<:AbstractFloat}
    length(r_faces) >= 2 || throw(ArgumentError("radial grid needs at least one cell"))
    issorted(r_faces; lt=<=) || throw(ArgumentError("radial faces must be strictly increasing"))
    first(r_faces) >= 0 || throw(ArgumentError("radial faces must be non-negative"))
    r_faces = collect(r_faces)
    r_centers = (r_faces[1:end-1] .+ r_faces[2:end]) ./ 2
    return RadialGrid{T}(r_centers, r_faces, diff(r_faces), copy(r_centers), copy(r_faces))
end

"""
    radial_cell_count(grid)

Number of radial cells of a [`RadialGrid`](@ref), or of the radial grid of a
[`KineticGrid`](@ref) or [`UnknownLayout`](@ref), or the radius count of a
[`PlasmaState`](@ref).
"""
radial_cell_count(grid::RadialGrid) = length(grid.r_centers)

"""
    uniform_r_faces(cell_count, minor_radius)

Faces of a uniform radial grid on `[0, minor_radius]` with `cell_count` cells.
"""
uniform_r_faces(cell_count::Integer, minor_radius::Real) =
    collect(range(0.0, float(minor_radius); length=cell_count + 1))
