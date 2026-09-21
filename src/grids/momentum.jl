"""
    PXiGrid{T}

Momentum grid in spherical coordinates `(p, ξ)`, with `p` the momentum normalized to
`m_e c` and `ξ = cos θ` the pitch. Cells are defined by their faces; cell centres are
face midpoints. The same momentum grid is used at all radii.
"""
struct PXiGrid{T<:AbstractFloat}
    p::Vector{T}     # cell centres, length np
    p_f::Vector{T}   # cell faces, length np + 1
    dp::Vector{T}    # cell widths, length np
    xi::Vector{T}    # cell centres, length nxi
    xi_f::Vector{T}  # cell faces, length nxi + 1
    dxi::Vector{T}   # cell widths, length nxi
end

function PXiGrid(p_f::AbstractVector{T}, xi_f::AbstractVector{T}) where {T<:AbstractFloat}
    length(p_f) >= 2 || throw(ArgumentError("p grid needs at least one cell"))
    length(xi_f) >= 2 || throw(ArgumentError("ξ grid needs at least one cell"))
    issorted(p_f; lt=<=) || throw(ArgumentError("p faces must be strictly increasing"))
    issorted(xi_f; lt=<=) || throw(ArgumentError("ξ faces must be strictly increasing"))
    first(p_f) >= 0 || throw(ArgumentError("p faces must be non-negative"))
    (first(xi_f) >= -1 && last(xi_f) <= 1) || throw(ArgumentError("ξ faces must lie in [-1, 1]"))
    p_f = collect(p_f)
    xi_f = collect(xi_f)
    p = (p_f[1:end-1] .+ p_f[2:end]) ./ 2
    xi = (xi_f[1:end-1] .+ xi_f[2:end]) ./ 2
    return PXiGrid{T}(p, p_f, diff(p_f), xi, xi_f, diff(xi_f))
end

np(g::PXiGrid) = length(g.p)
nxi(g::PXiGrid) = length(g.xi)
ncells(g::PXiGrid) = np(g) * nxi(g)

"""
    uniform_p_faces(np, pmin, pmax)

Faces of a uniform momentum grid with `np` cells.
"""
uniform_p_faces(np::Integer, pmin::Real, pmax::Real) =
    collect(range(float(pmin), float(pmax); length=np + 1))

"""
    biuniform_p_faces(np, np_sep, pmin, psep, pmax)

Faces of a two-region uniform momentum grid: `np_sep` cells on `[pmin, psep]` and
`np - np_sep` cells on `[psep, pmax]` (DREAM `PBiUniformGridGenerator`).
"""
function biuniform_p_faces(np::Integer, np_sep::Integer, pmin::Real, psep::Real, pmax::Real)
    (1 <= np_sep < np) || throw(ArgumentError("each region needs at least one cell"))
    (pmin < psep < pmax) || throw(ArgumentError("require pmin < psep < pmax"))
    lo = range(float(pmin), float(psep); length=np_sep + 1)
    hi = range(float(psep), float(pmax); length=np - np_sep + 1)
    return vcat(collect(lo), collect(hi)[2:end])
end

"""
    uniform_xi_faces(nxi)

Faces of a uniform pitch grid on `[-1, 1]` with `nxi` cells.
"""
uniform_xi_faces(nxi::Integer) = collect(range(-1.0, 1.0; length=nxi + 1))
