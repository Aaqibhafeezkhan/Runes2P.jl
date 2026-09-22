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
    tanh_p_faces(cell_count, lower_cell_count, p_min, p_separation, p_max;
                 transition_cells=cell_count/20)

Faces of a smoothly stretched momentum grid: the biuniform grid of
[`biuniform_p_faces`](@ref) with its spacing jump at `p_separation` replaced by a tanh
transition over about `transition_cells` cells above it. In the continuous cell index `x`
the spacing is

    h(x) = h₁ + (h₂ - h₁) (1 + tanh((x - n₁ - 3w)/w))/2,   n₁ = lower_cell_count, w = transition_cells,

and face `k` sits at `p_min + ∫₀ᵏ h`, with `h₁` and `h₂` chosen so that face `n₁` is at
`p_separation` and the last face at `p_max`. The transition is centred three widths above
the separation, so up to `p_separation` the spacing exceeds `h₁` by at most 0.25% of
`h₂ - h₁` (the bulk keeps its resolution) and it grows over the next few `w` cells; the
upper region needs at least `5w` cells. `transition_cells = 0` gives the biuniform grid
exactly, and the faces approach it linearly in `w`. The default width, a twentieth of the
cells, grows with the cell count, which keeps the grid map smooth under refinement; with a
fixed width in cells the max-norm order stalls near 1.5 in the manufactured-solution test.
Not available in DREAM, which has uniform, biuniform and custom momentum grids; a DREAM
comparison passes these faces as a custom grid.

The biuniform grid converges at second order despite its jump: the local truncation
error is O(1) at the jump cell only, and the global error stays second order
(`test/test_manufactured_solutions.jl`). This grid keeps the truncation error second order
everywhere when `transition_cells` grows with the cell count. Where the solution still
varies above `p_separation`, its transition cells, finer than the biuniform grid's upper
cells, make it more accurate at equal cell count: 3.5 times on the manufactured solution,
2 times on the avalanche example with upwind, and no gain on DREAM's `runaway/basic.py`
split at p = 0.1 with the central scheme.
"""
function tanh_p_faces(cell_count::Integer, lower_cell_count::Integer, p_min::Real, p_separation::Real,
                      p_max::Real; transition_cells::Real=cell_count / 20)
    (1 <= lower_cell_count < cell_count) || throw(ArgumentError("each region needs at least one cell"))
    (p_min < p_separation < p_max) || throw(ArgumentError("require p_min < p_separation < p_max"))
    transition_cells >= 0 || throw(ArgumentError("transition_cells must be nonnegative"))
    transition_cells == 0 && return biuniform_p_faces(cell_count, lower_cell_count, p_min, p_separation, p_max)
    n1, w = float(lower_cell_count), float(transition_cells)
    center = n1 + 3w
    center + 2w <= cell_count ||
        throw(ArgumentError("transition_cells = $transition_cells does not fit in the $(cell_count - lower_cell_count) " *
                            "upper cells (5 transition_cells needed)"))
    # log cosh without overflow
    logcosh(x) = abs(x) + log1p(exp(-2abs(x))) - log(2.0)
    # integral of (1 + tanh((x - center)/w))/2 from 0 to x
    upper_share(x) = (x + w * (logcosh((x - center) / w) - logcosh(center / w))) / 2
    # p(x) - p_min = h₁ (x - S(x)) + h₂ S(x)
    s1, sN = upper_share(n1), upper_share(float(cell_count))
    determinant = (n1 - s1) * sN - s1 * (cell_count - sN)
    lower_span, total_span = p_separation - p_min, p_max - p_min
    h1 = (lower_span * sN - s1 * total_span) / determinant
    h2 = ((n1 - s1) * total_span - (cell_count - sN) * lower_span) / determinant
    (h1 > 0 && h2 > 0) ||
        throw(ArgumentError("transition_cells = $transition_cells is too wide for this grid; the spacings are not positive"))
    faces = [p_min + h1 * (k - upper_share(float(k))) + h2 * upper_share(float(k)) for k in 0:cell_count]
    # land exactly on the prescribed separation and ends
    faces[1], faces[lower_cell_count+1], faces[end] = p_min, p_separation, p_max
    return faces
end

"""
    uniform_xi_faces(cell_count)

Faces of a uniform pitch grid on `[-1, 1]` with `cell_count` cells.
"""
uniform_xi_faces(cell_count::Integer) = collect(range(-1.0, 1.0; length=cell_count + 1))
