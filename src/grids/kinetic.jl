"""
    KineticGrid{T}

Phase-space grid `(p, ξ, r)` with a momentum grid per radius, and the phase-space
Jacobian `𝒱'` (DREAM `Vp`) at cell centres and at the faces in each direction.

Callers use the accessors, not the fields: [`momentum_grid`](@ref),
[`phase_space_jacobian`](@ref) and its face variants, the cell counts, and
[`has_shared_momentum_grid`](@ref). When all radii have the same cell counts
([`has_same_cell_counts`](@ref)), a distribution can be stored as a 3D array `f[i, j, ir]`
with `p` fastest, and `size(grid)` and the cell counts without a radius argument are
available. `momentum_grid(grid)` needs one momentum grid shared by all radii. They throw
an `ArgumentError` otherwise.

A radial face lies between two radii, whose pitch grids may differ. Its
pitch cells are the overlap cells of the two grids ([`radial_face_overlap`](@ref),
[`PitchOverlap`](@ref)), and quantities on the face are given per overlap cell. The
radial-face Jacobian exists when the p faces are the same at all radii
([`has_same_p_faces`](@ref)).

In cylindrical geometry `𝒱' = 2π p² V'`, so that the density is
`n(r) = (1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j f_ij`.
"""
struct KineticGrid{T<:AbstractFloat}
    radial_grid::RadialGrid{T}
    momentum_grids::Vector{MomentumGrid{T}}
    jacobian_centers::Vector{Matrix{T}}            # per radius, (p cells, ξ cells)
    jacobian_p_faces::Vector{Matrix{T}}            # per radius, (p cells + 1, ξ cells)
    jacobian_xi_faces::Vector{Matrix{T}}           # per radius, (p cells, ξ cells + 1)
    jacobian_r_faces::Union{Nothing,Array{T,3}}    # (p cells, face pitch cells, radii + 1); same p faces only
    jacobian_over_p2_at_zero::Vector{Vector{T}}    # per radius, lim_{p→0} 𝒱'/p² per ξ cell
    shared::Bool
    same_p_faces::Bool
    same_cell_counts::Bool
    # per radial face (radii + 1): overlap of the pitch grids of the two radii, of the
    # boundary radius with itself on the first and the last face; `nothing` when the pitch
    # grids do not cover one interval
    radial_overlaps::Union{Nothing,Vector{PitchOverlap{T}}}
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
    same_p_faces = all(m -> m.p_faces == first_grid.p_faces, grids)
    same_cell_counts = all(m -> momentum_cell_count(m) == momentum_cell_count(first_grid) &&
                                pitch_cell_count(m) == pitch_cell_count(first_grid), grids)
    radial_overlaps = _radial_overlaps(grids)
    on_r_faces = if same_p_faces && radial_overlaps !== nothing
        # zero beyond the pitch cells of a face
        p = first_grid.p_centers
        face_cells = maximum(overlap_cell_count, radial_overlaps)
        [s <= overlap_cell_count(radial_overlaps[k]) ? 2pi * p[i]^2 * Vr_faces[k] : zero(T)
         for i in 1:momentum_cell_count(first_grid), s in 1:face_cells, k in 1:nr+1]
    else
        nothing
    end
    over_p2_at_zero = [fill(2pi * Vr[ir], pitch_cell_count(m)) for (ir, m) in enumerate(grids)]
    return KineticGrid{T}(radial_grid, grids, centers, on_p_faces, on_xi_faces, on_r_faces,
                          over_p2_at_zero, shared, same_p_faces, same_cell_counts, radial_overlaps)
end

# overlaps of the pitch grids across every radial face; one object per pair of grids, so a
# shared grid has a single overlap
function _radial_overlaps(grids::Vector{MomentumGrid{T}}) where {T}
    nr = length(grids)
    interval(m) = (first(m.xi_faces), last(m.xi_faces))
    tolerance = 1e-12 * (last(first(grids).xi_faces) - first(first(grids).xi_faces))
    all(m -> all(abs.(interval(m) .- interval(first(grids))) .<= tolerance), grids) || return nothing
    built = Dict{Tuple{UInt,UInt},PitchOverlap{T}}()
    overlap(left, right) = get!(() -> PitchOverlap(left, right), built, (objectid(left), objectid(right)))
    return PitchOverlap{T}[overlap(grids[max(k - 1, 1)], grids[min(k, nr)]) for k in 1:nr+1]
end

"""
    runaway_grid(hot_tail_grid, p_faces_above, xi_faces=biuniform_theta_xi_faces(20, 12, 0.9))

Cylindrical kinetic grid of the runaway electrons above `hot_tail_grid`, on the same radii:
at every radius its first p face is the last p face of the hot-tail grid there, followed
by `p_faces_above` (increasing, all above that face, the last one the `p_max` of the
runaway grid), e.g. `uniform_p_faces(n, p, p_max)[2:end]` with `p` the hot-tail `p_max`.
Taking the first face from the hot-tail grid keeps the two grids on one face to the bit,
which the [`MomentumInterfaceTerm`](@ref) between them requires.

The default pitch grid is the one chosen for the runaway grid:
uniform in θ in two regions, 12 cells for θ ∈ [0, 0.9 rad] and 8 cells up to π. Fully
kinetic DREAM runs at production resolution use 10 cells to 0.5 rad and 5 to π, `biuniform_theta_xi_faces(15, 10, 0.5)`. The pitch
grid is the same at all radii and independent of the pitch grid of the hot-tail grid.
"""
function runaway_grid(hot_tail_grid::KineticGrid{T}, p_faces_above::AbstractVector{<:Real},
                      xi_faces::AbstractVector{<:Real}=biuniform_theta_xi_faces(20, 12, 0.9)) where {T}
    isempty(p_faces_above) && throw(ArgumentError("the runaway grid needs at least one momentum cell"))
    pitch = collect(T, xi_faces)
    built = Dict{T,MomentumGrid{T}}()
    grids = map(1:radial_cell_count(hot_tail_grid)) do ir
        p_interface = last(momentum_grid(hot_tail_grid, ir).p_faces)
        first(p_faces_above) > p_interface ||
            throw(ArgumentError("the faces of the runaway grid must lie above the hot-tail p_max = " *
                                "$p_interface, the first is $(first(p_faces_above))"))
        get!(() -> MomentumGrid(vcat(p_interface, collect(T, p_faces_above)), pitch), built, p_interface)
    end
    return cylindrical_grid(grids, hot_tail_grid.radial_grid)
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
    has_same_cell_counts(grid)

Whether all radii have the same momentum and pitch cell counts, so that a distribution
fits a 3D array. The faces may differ between radii (per-radius pitch grids).
"""
has_same_cell_counts(grid::KineticGrid) = grid.same_cell_counts

"""
    has_same_p_faces(grid)

Whether the momentum faces are the same at all radii. The pitch faces may differ.
"""
has_same_p_faces(grid::KineticGrid) = grid.same_p_faces

function _require_same_cell_counts(grid::KineticGrid)
    grid.same_cell_counts || throw(ArgumentError("the kinetic grid has different cell counts at " *
                                                 "different radii; pass a radius index"))
    return nothing
end

"""
    radial_face_overlap(grid, face)

[`PitchOverlap`](@ref) of the pitch grids on the two sides of radial face `face` (between
the radii `face - 1`, the left grid, and `face`, the right grid). Its overlap cells are the
pitch cells of the face. On the first and the last face both sides are the grid of the
boundary radius.
"""
function radial_face_overlap(grid::KineticGrid, face::Integer)
    grid.radial_overlaps === nothing &&
        throw(ArgumentError("the pitch grids of the radii do not cover one interval"))
    return grid.radial_overlaps[face]
end

"""
    radial_face_pitch_cell_count(grid, face)
    radial_face_pitch_cell_count(grid)

Number of pitch cells of radial face `face` (the overlap cells of the two neighbouring
pitch grids); without `face`, the largest count over the faces, which is the second
dimension of the arrays on the radial faces.
"""
radial_face_pitch_cell_count(grid::KineticGrid, face::Integer) =
    overlap_cell_count(radial_face_overlap(grid, face))
radial_face_pitch_cell_count(grid::KineticGrid) =
    maximum(face -> radial_face_pitch_cell_count(grid, face), 1:radial_cell_count(grid)+1)

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
    radial_neighbour_pitch_cells(grid, ir, j, direction)

Pitch cells at the radius `ir + direction` (`direction` is -1 or 1) that overlap pitch cell
`j` of radius `ir`, as a range; empty beyond the first and the last radius. These are the
cells a radial flux couples cell `j` to.
"""
function radial_neighbour_pitch_cells(grid::KineticGrid, ir::Integer, j::Integer, direction::Integer)
    direction in (-1, 1) || throw(ArgumentError("direction must be -1 or 1"))
    1 <= ir + direction <= radial_cell_count(grid) || return 1:0
    if direction == 1
        overlap = radial_face_overlap(grid, ir + 1)
        range = overlap.left_ranges[j]
        return overlap.right_cells[first(range)]:overlap.right_cells[last(range)]
    end
    overlap = radial_face_overlap(grid, ir)
    range = overlap.right_ranges[j]
    return overlap.left_cells[first(range)]:overlap.left_cells[last(range)]
end

"""
    phase_space_jacobian_r_faces(grid)

Phase-space Jacobian on the radial faces, `(p cells, face pitch cells, radii + 1)`, with
the pitch cells of a face those of [`radial_face_overlap`](@ref) (the pitch cells of the
grid when the radii share their pitch grid) and zeros beyond the count of a face. Defined
when the p faces are the same at all radii.
"""
function phase_space_jacobian_r_faces(grid::KineticGrid)
    grid.jacobian_r_faces === nothing &&
        throw(ArgumentError("the kinetic grid has different p faces at different radii, or pitch grids " *
                            "that do not cover one interval; its radial faces carry no Jacobian"))
    return grid.jacobian_r_faces
end

momentum_cell_count(grid::KineticGrid, ir::Integer) = momentum_cell_count(momentum_grid(grid, ir))
pitch_cell_count(grid::KineticGrid, ir::Integer) = pitch_cell_count(momentum_grid(grid, ir))
cell_count(grid::KineticGrid, ir::Integer) = cell_count(momentum_grid(grid, ir))
momentum_cell_count(grid::KineticGrid) = (_require_same_cell_counts(grid); momentum_cell_count(grid, 1))
pitch_cell_count(grid::KineticGrid) = (_require_same_cell_counts(grid); pitch_cell_count(grid, 1))
radial_cell_count(grid::KineticGrid) = radial_cell_count(grid.radial_grid)
cell_count(grid::KineticGrid) = sum(ir -> cell_count(grid, ir), 1:radial_cell_count(grid))

"""
    size(grid::KineticGrid)

`(p cells, ξ cells, radii)` of a grid with the same cell counts at all radii, the shape of
a distribution stored as a 3D array.
"""
Base.size(grid::KineticGrid) = (momentum_cell_count(grid), pitch_cell_count(grid), radial_cell_count(grid))

"""
    density(grid, f)

Density moment `n(r) = (1/V') ∫ 𝒱' f dp dξ` at each radius, for `f` stored as a 3D array
(the same cell counts at all radii).
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
    pitch_moment_weights(momentum_grid; pitch_rule=:linear)

Weights `w_j` with `∫ ξ f dξ ≈ Σ_j w_j f_j` for a distribution given by its values `f_j` at
the pitch cell centres of one momentum grid. The density moment (weight 1) is the same
for both rules; the rules differ for a weight that varies across the cell.

- `:cell_center`: `w_j = ξ_j Δξ_j`, the midpoint rule, DREAM's moment
  (`CurrentDensityFromDistributionFunction`). Exact for `f` constant in ξ; for `f` linear
  in ξ it underestimates `∫ ξ² dξ` by the relative amount `Δξ²/4` on a uniform grid
  (25% at 2 pitch cells, 0.1% at 32).
- `:linear`: `f` reconstructed in each cell as `f_j + s_j (ξ − ξ_j)`, with `s_j` the slope
  at `ξ_j` of the parabola through the three nearest centres (the central difference on a
  uniform grid, one-sided at the ends; the two-point slope with two cells, zero with one).
  The reconstruction keeps the density of every cell (`∫ (ξ − ξ_j) dξ = 0`) and adds
  `s_j Δξ_j³/12` to its ξ moment. Exact for `f` linear in ξ on any grid, which removes the
  `Δξ²/4` term; the remaining error is second order and proportional to the curvature of
  `f`, so it vanishes for the Spitzer-like `f₁ ∝ ξ` of the conductivity.
"""
function pitch_moment_weights(m::MomentumGrid{T}; pitch_rule::Symbol=:linear) where {T}
    xi, dxi = m.xi_centers, m.xi_widths
    w = xi .* dxi
    pitch_rule === :cell_center && return w
    pitch_rule === :linear ||
        throw(ArgumentError("pitch_rule must be :linear or :cell_center, got $pitch_rule"))
    nxi = length(xi)
    nxi == 1 && return w
    for j in 1:nxi
        c = dxi[j]^3 / 12
        if nxi == 2
            s = 1 / (xi[2] - xi[1])
            w[1] -= c * s
            w[2] += c * s
        else
            k0 = clamp(j - 1, 1, nxi - 2)   # stencil (k0, k0+1, k0+2) evaluated at ξ_j
            x0, x1, x2 = xi[k0], xi[k0 + 1], xi[k0 + 2]
            x = xi[j]
            w[k0] += c * ((x - x1) + (x - x2)) / ((x0 - x1) * (x0 - x2))
            w[k0 + 1] += c * ((x - x0) + (x - x2)) / ((x1 - x0) * (x1 - x2))
            w[k0 + 2] += c * ((x - x0) + (x - x1)) / ((x2 - x0) * (x2 - x1))
        end
    end
    return w
end

"""
    current_density(grid, f; pitch_rule=:linear)

Current density `j(r) = e c (1/V') Σ_ij 𝒱'_ij Δp_i (p_i/γ_i) w_j f_ij` in A/m² carried by
the distribution `f` (3D array, the same cell counts at all radii), with the pitch weights `w_j` of
[`pitch_moment_weights`](@ref); `:cell_center` is DREAM's `j_hot` moment. Sign as in
DREAM: positive when the electrons move towards ξ = 1, the direction in which
[`add_electric_field!`](@ref) accelerates them for `E > 0`, so `j` and `E` share their
sign and the electron charge is folded into that convention. `𝒱'` and `p/γ` are taken at
the cell centre; a pitch-dependent Jacobian (toroidal geometry) would enter the
pitch weights.
"""
function current_density(grid::KineticGrid{T}, f::AbstractArray{S,3}; pitch_rule::Symbol=:linear) where {T,S}
    size(f) == size(grid) || throw(DimensionMismatch("f must have the grid size"))
    j = zeros(promote_type(T, S), radial_cell_count(grid))
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        w = pitch_moment_weights(m; pitch_rule)
        j[ir] = _current_moment(m, w, phase_space_jacobian(grid, ir), view(f, :, :, ir)) /
                grid.radial_grid.spatial_jacobian[ir]
    end
    return j
end

# e c Σ_ij 𝒱'_ij Δp_i (p_i/γ_i) w_j f_ij on one momentum grid (without the 1/V')
function _current_moment(m::MomentumGrid, w::AbstractVector, jacobian::AbstractMatrix, f::AbstractMatrix)
    j = zero(promote_type(eltype(w), eltype(f)))
    @inbounds for k in 1:pitch_cell_count(m)
        column = zero(j)
        for i in 1:momentum_cell_count(m)
            p = m.p_centers[i]
            column += jacobian[i, k] * m.p_widths[i] * p / sqrt(1 + p^2) * f[i, k]
        end
        j += w[k] * column
    end
    return Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT * j
end

"""
    total_particles(grid, f)

Total particle content `Σ 𝒱' Δp Δξ Δr f`, conserved by a flux-form discretization, for `f`
stored as a 3D array (the same cell counts at all radii).
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
