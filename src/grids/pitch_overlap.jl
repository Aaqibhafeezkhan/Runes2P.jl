# Overlap of two pitch grids: the common refinement of their faces, which
# carries fluxes and cell contents between grids without loss. Used for the radial faces
# between radii with different pitch grids, for the momentum face between the hot-tail and
# the runaway grid, and for a remap onto a changed grid.

"""
    PitchOverlap{T}

Overlap of two pitch grids on the same interval, called the left and the right grid (the
two sides of a face: the inner and the outer radius, or the lower and the upper momentum
grid). The faces of both grids together cut the interval into overlap cells. Overlap cell
`s` lies in exactly one cell of each grid, `left_cells[s]` and `right_cells[s]`, and the
overlap cells of a grid cell `j` are the contiguous range `left_ranges[j]` (or
`right_ranges[j]`).

`left_fractions[s]` is the share of the left cell's measure that lies in overlap cell `s`,
and `right_fractions[s]` the share of the right cell's. The measure of an interval is its
width by default, which is the phase-space volume where `𝒱'` does not depend on ξ
(cylindrical geometry). `measure(a, b)` replaces it for a weight that varies with ξ
(`∫ 𝒱' dξ` in toroidal geometry).

A quantity that is one number per overlap cell (a flux through the face, the content moved
by a remap) and is given to both sides conserves the total: the left cell `j` receives
`Σ_s left_fractions[s] q_s` over `left_ranges[j]` per unit of its measure, the right cell
the same with its fractions.

Faces of the two grids closer than `1e-12` of the interval count as one face (the left
grid's value is kept), so that faces equal up to round-off, such as `cos(π/2)` and `0`,
leave no sliver. The fractions of a cell then sum to 1 within that tolerance, and to
round-off otherwise.
"""
struct PitchOverlap{T<:AbstractFloat}
    xi_faces::Vector{T}                    # faces of the overlap cells
    xi_centers::Vector{T}
    xi_widths::Vector{T}
    measures::Vector{T}                    # measure of each overlap cell
    left_cells::Vector{Int}                # per overlap cell: the cell of the left grid
    right_cells::Vector{Int}
    left_ranges::Vector{UnitRange{Int}}    # per cell of the left grid: its overlap cells
    right_ranges::Vector{UnitRange{Int}}
    left_fractions::Vector{T}              # per overlap cell: measure / measure of the left cell
    right_fractions::Vector{T}
    identical::Bool                        # the two grids have the same faces
end

_interval_width(a, b) = b - a

"""
    PitchOverlap(left_xi_faces, right_xi_faces; measure=(a, b) -> b - a)
    PitchOverlap(left::MomentumGrid, right::MomentumGrid; measure)

Overlap of the pitch grids with the given faces. Both must cover the same interval.
"""
function PitchOverlap(left_xi_faces::AbstractVector{T}, right_xi_faces::AbstractVector{T};
                      measure=_interval_width) where {T<:AbstractFloat}
    (length(left_xi_faces) >= 2 && length(right_xi_faces) >= 2) ||
        throw(ArgumentError("a pitch grid needs at least one cell"))
    (issorted(left_xi_faces; lt=<=) && issorted(right_xi_faces; lt=<=)) ||
        throw(ArgumentError("ξ faces must be strictly increasing"))
    tolerance = T(1e-12) * (last(left_xi_faces) - first(left_xi_faces))
    (abs(first(left_xi_faces) - first(right_xi_faces)) <= tolerance &&
     abs(last(left_xi_faces) - last(right_xi_faces)) <= tolerance) ||
        throw(ArgumentError("the two pitch grids must cover the same interval, got " *
                            "[$(first(left_xi_faces)), $(last(left_xi_faces))] and " *
                            "[$(first(right_xi_faces)), $(last(right_xi_faces))]"))
    left_count, right_count = length(left_xi_faces) - 1, length(right_xi_faces) - 1
    faces = T[first(left_xi_faces)]
    left_cells, right_cells = Int[], Int[]
    j_left, j_right = 1, 1
    while j_left <= left_count && j_right <= right_count
        push!(left_cells, j_left)
        push!(right_cells, j_right)
        upper_left, upper_right = left_xi_faces[j_left+1], right_xi_faces[j_right+1]
        if abs(upper_left - upper_right) <= tolerance
            push!(faces, upper_left)
            j_left += 1
            j_right += 1
        elseif upper_left < upper_right
            push!(faces, upper_left)
            j_left += 1
        else
            push!(faces, upper_right)
            j_right += 1
        end
    end
    (j_left == left_count + 1 && j_right == right_count + 1) ||
        throw(ArgumentError("the ξ faces of the two grids could not be merged (a cell narrower than " *
                            "1e-12 of the interval)"))
    measures = T[measure(faces[s], faces[s+1]) for s in eachindex(left_cells)]
    all(>(0), measures) || throw(ArgumentError("the measure of every overlap cell must be positive"))
    left_ranges = _cell_ranges(left_cells, left_count)
    right_ranges = _cell_ranges(right_cells, right_count)
    left_measures = T[measure(left_xi_faces[j], left_xi_faces[j+1]) for j in 1:left_count]
    right_measures = T[measure(right_xi_faces[j], right_xi_faces[j+1]) for j in 1:right_count]
    left_fractions = T[measures[s] / left_measures[left_cells[s]] for s in eachindex(measures)]
    right_fractions = T[measures[s] / right_measures[right_cells[s]] for s in eachindex(measures)]
    identical = left_xi_faces == right_xi_faces
    if identical
        # one overlap cell per grid cell: the fractions are exactly 1, so that fluxes between
        # equal grids keep the arithmetic of a shared grid
        fill!(left_fractions, one(T))
        fill!(right_fractions, one(T))
    end
    centers = (faces[1:end-1] .+ faces[2:end]) ./ 2
    return PitchOverlap{T}(faces, centers, diff(faces), measures, left_cells, right_cells, left_ranges,
                           right_ranges, left_fractions, right_fractions, identical)
end

PitchOverlap(left::MomentumGrid, right::MomentumGrid; measure=_interval_width) =
    PitchOverlap(left.xi_faces, right.xi_faces; measure)

# contiguous ranges of the overlap cells of each grid cell, from the sorted cell indices
function _cell_ranges(cells::Vector{Int}, cell_count::Int)
    ranges = Vector{UnitRange{Int}}(undef, cell_count)
    start = 1
    for j in 1:cell_count
        stop = start - 1
        while stop < length(cells) && cells[stop+1] == j
            stop += 1
        end
        ranges[j] = start:stop
        start = stop + 1
    end
    return ranges
end

"""
    overlap_cell_count(overlap)

Number of overlap cells: between the larger of the two pitch cell counts (nested grids)
and their sum minus one (no common interior face).
"""
overlap_cell_count(overlap::PitchOverlap) = length(overlap.left_cells)

"""
    has_identical_grids(overlap)

Whether the two pitch grids have the same faces, so that every overlap cell is a cell of
both.
"""
has_identical_grids(overlap::PitchOverlap) = overlap.identical

"""
    remap_to_right!(f_right, overlap, f_left)

Conservative remap of cell averages from the left pitch grid of `overlap` to the right one
([`remap_to_left!`](@ref) is the reverse), for a vector over the pitch cells or a
`(p cells, ξ cells)` matrix (each momentum cell on its own). The source is taken as constant in each of its cells, and the new cell average is
the measure-weighted mean of the source over the cell:

```math
f^\\mathrm{right}_j = \\sum_{s \\in \\mathtt{right\\_ranges}[j]} \\mathtt{right\\_fractions}[s]\\, f^\\mathrm{left}_{\\mathtt{left\\_cells}[s]}.
```

The content `Σ_j measure_j f_j` is conserved to round-off, a constant stays a constant, the
result lies between the smallest and the largest source value (so a non-negative source
stays non-negative), and the remap between identical grids is a copy. First order in the
cell width: a remap to another grid and back smooths the source. Allocation free. Does
not add to the target, it overwrites it.
"""
function remap_to_right!(f_right::AbstractVecOrMat, overlap::PitchOverlap, f_left::AbstractVecOrMat)
    return _remap!(f_right, f_left, overlap.right_ranges, overlap.right_fractions, overlap.left_cells,
                   length(overlap.left_ranges))
end

"""
    remap_to_left!(f_left, overlap, f_right)

Conservative remap of cell averages from the right pitch grid of `overlap` to the left one,
the reverse of [`remap_to_right!`](@ref), with the same properties.
"""
function remap_to_left!(f_left::AbstractVecOrMat, overlap::PitchOverlap, f_right::AbstractVecOrMat)
    return _remap!(f_left, f_right, overlap.left_ranges, overlap.left_fractions, overlap.right_cells,
                   length(overlap.right_ranges))
end

function _remap!(target::AbstractVector, source::AbstractVector, ranges, fractions, source_cells,
                 source_count)
    length(target) == length(ranges) || throw(DimensionMismatch("the target does not match its pitch grid"))
    length(source) == source_count || throw(DimensionMismatch("the source does not match its pitch grid"))
    @inbounds for j in eachindex(ranges)
        value = zero(eltype(target))
        for s in ranges[j]
            value += fractions[s] * source[source_cells[s]]
        end
        target[j] = value
    end
    return target
end

function _remap!(target::AbstractMatrix, source::AbstractMatrix, ranges, fractions, source_cells,
                 source_count)
    size(target, 2) == length(ranges) || throw(DimensionMismatch("the target does not match its pitch grid"))
    size(source, 2) == source_count || throw(DimensionMismatch("the source does not match its pitch grid"))
    size(target, 1) == size(source, 1) ||
        throw(DimensionMismatch("the target and the source need the same momentum cells"))
    @inbounds for j in eachindex(ranges), i in axes(target, 1)
        value = zero(eltype(target))
        for s in ranges[j]
            value += fractions[s] * source[i, source_cells[s]]
        end
        target[i, j] = value
    end
    return target
end
