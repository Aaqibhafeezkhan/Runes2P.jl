"""
    RadialGrid{T}

Radial grid on `[0, a]` defined by its faces, with the spatial volume factor `V'(r)`
at cell centres and faces (DREAM `VpVol`). For a cylinder `V'(r) = r`, i.e. the volume
per unit length divided by 2π.
"""
struct RadialGrid{T<:AbstractFloat}
    r::Vector{T}
    r_f::Vector{T}
    dr::Vector{T}
    VpVol::Vector{T}    # at cell centres
    VpVol_f::Vector{T}  # at cell faces
end

function RadialGrid(r_f::AbstractVector{T}) where {T<:AbstractFloat}
    length(r_f) >= 2 || throw(ArgumentError("radial grid needs at least one cell"))
    issorted(r_f; lt=<=) || throw(ArgumentError("radial faces must be strictly increasing"))
    first(r_f) >= 0 || throw(ArgumentError("radial faces must be non-negative"))
    r_f = collect(r_f)
    r = (r_f[1:end-1] .+ r_f[2:end]) ./ 2
    return RadialGrid{T}(r, r_f, diff(r_f), copy(r), copy(r_f))
end

nr(g::RadialGrid) = length(g.r)

"""
    uniform_r_faces(nr, a)

Faces of a uniform radial grid on `[0, a]` with `nr` cells.
"""
uniform_r_faces(nr::Integer, a::Real) = collect(range(0.0, float(a); length=nr + 1))
