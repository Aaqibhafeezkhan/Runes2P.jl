# Pitch part of the knock-on operator in cylindrical geometry (B constant, ξ = ξ₀),
# following DREAM `KnockOnUtilities.cpp` (namespace `AnalyticDelta`,
# `SetDeltaMatrixColumnOnGrid`) and `MollerDeltaAngleKernel.cpp`.
#
# A primary with pitch ξ₁ produces a knock-on at angle arccos ξ* from its direction.
# Averaged over the gyrophase φ, the knock-on pitch ξ = a + b cos φ with a = ξ₁ξ*,
# b = √((1 - ξ₁²)(1 - ξ*²)) is distributed with density 1/(π√(b² - (ξ - a)²)) on
# [a - b, a + b]; its antiderivative is asin((ξ - a)/b)/π.
#
# Local names: xi1 is ξ₁ (primary pitch), xi_s is ξ* (scattering cosine).

using QuadGK: gauss

"""
    pitch_cell_fraction(lower_face, upper_face, xi1, xi_s)

Fraction of knock-ons with pitch in `[lower_face, upper_face]` produced by a primary with
pitch `xi1` (ξ₁) at scattering-angle cosine `xi_s` (ξ*), averaged over the gyrophase
(DREAM `D`, `AnalyticDelta::EvaluateXiIntegral`). The argument of `asin` is clamped to
`[-1, 1]`, so no separate reachability test is needed. When the distribution degenerates
to a point (`b² ≤ 5ε`), the point is assigned to the half-open cell
`(lower_face, upper_face]`, as in DREAM.
"""
function pitch_cell_fraction(lower_face, upper_face, xi1, xi_s)
    a = xi_s * xi1
    b2 = (1 - xi_s^2) * (1 - xi1^2)
    if b2 <= 5 * eps(typeof(b2))
        return (lower_face < a && a <= upper_face) ? one(a) : zero(a)
    end
    b = sqrt(b2)
    antiderivative(y) = asin(clamp((y - a) / b, -one(a), one(a))) / pi
    return antiderivative(upper_face) - antiderivative(lower_face)
end

"""
    pitch_kernel_column!(column, xi_faces, xi_s; primary_cell, nodes=(0.0,), weights=(2.0,))

Fill `column[j] = δ_j` for knock-on pitch cells with faces `xi_faces`, normalized so that
`Σ_j Δξ_j δ_j = 1`, and return the norm before normalization. The primary pitch is
averaged over `primary_cell = (xi1_lower, xi1_upper)` with Gauss nodes and weights on
`[-1, 1]`; with `primary_cell = (xi1, xi1)` and the default single node the primary sits
at `xi1` (DREAM). A primary at `xi1 = -1` whose knock-on pitch is exactly `-1` is assigned
to the first cell.
"""
function pitch_kernel_column!(column::AbstractVector, xi_faces::AbstractVector, xi_s;
        primary_cell, nodes = (0.0,), weights = (2.0,))
    nxi = length(column)
    lower, upper = primary_cell
    fill!(column, 0)
    for (x, w) in zip(nodes, weights)
        xi1 = (lower + upper) / 2 + x * (upper - lower) / 2
        for j in 1:nxi
            column[j] += w / 2 *
                         pitch_cell_fraction(xi_faces[j], xi_faces[j + 1], xi1, xi_s)
        end
        # point mass at ξ = -1 (half-open cells exclude it)
        if (1 - xi_s^2) * (1 - xi1^2) <= 5 * eps(xi_s) && xi_s * xi1 == xi_faces[1]
            column[1] += w / 2
        end
    end
    norm = sum(column)
    for j in 1:nxi
        column[j] /= (xi_faces[j + 1] - xi_faces[j])
    end
    if norm > 0
        column ./= norm
    end
    return norm
end

"""
    pitch_kernel_column_split!(column, xi_faces, xi_s; primary_cell, nodes, weights)

Like [`pitch_kernel_column!`](@ref) with the primary averaged over
`primary_cell = (xi1_lower, xi1_upper)`, but exact to high order: as a function of the
primary pitch ξ₁, the fraction `D_j(ξ₁)` reaching knock-on cell `j` has square-root kinks
where an edge `a ± b` of the gyro-averaged distribution crosses a face `t` of the
knock-on grid, i.e. at `ξ₁ = cos(arccos t ± θ*)` with `cos θ* = ξ*` (and jumps at the
faces when `ξ* = 1`). The cell is split at these points, and each piece is integrated
with the Gauss `nodes` and `weights` (on `[-1, 1]`) in the variable `s` of
`ξ₁ = mid + h sin(πs/2)`, which turns the square-root behaviour at the piece ends into
a smooth one.
"""
function pitch_kernel_column_split!(column::AbstractVector, xi_faces::AbstractVector, xi_s;
        primary_cell, nodes, weights)
    nxi = length(column)
    lower, upper = primary_cell
    if !(upper > lower)
        throw(ArgumentError("the primary cell must have positive width"))
    end
    theta_s = acos(clamp(xi_s, -one(xi_s), one(xi_s)))
    cuts = [lower, upper]
    for face in xi_faces
        theta_face = acos(clamp(face, -one(face), one(face)))
        for x in (cos(theta_face + theta_s), cos(theta_face - theta_s))
            if lower < x < upper
                push!(cuts, x)
            end
        end
    end
    sort!(cuts)
    fill!(column, 0)
    for q in 1:(length(cuts) - 1)
        u, v = cuts[q], cuts[q + 1]
        v > u || continue
        mid, h = (u + v) / 2, (v - u) / 2
        for (x, w) in zip(nodes, weights)
            xi1 = mid + h * sin(pi * x / 2)
            weight = w * h * (pi / 2) * cos(pi * x / 2)
            for j in 1:nxi
                column[j] += weight *
                             pitch_cell_fraction(xi_faces[j], xi_faces[j + 1], xi1, xi_s)
            end
        end
    end
    column ./= (upper - lower)
    norm = sum(column)
    for j in 1:nxi
        column[j] /= (xi_faces[j + 1] - xi_faces[j])
    end
    if norm > 0
        column ./= norm
    end
    return norm
end

"""
    MollerPitchKernel{T}

Knock-on pitch kernel `δ[j, l](ξ*)` between primary pitch cells `l` and knock-on pitch
cells `j`, tabulated on a uniform grid of `ξ*` values and interpolated linearly to the
`ξ*` of each momentum pair `(i, k)`. Every tabulated column satisfies
`Σ_j Δξ_j δ[j, l] = 1`, and so does every interpolated one.

- `xi_star_values`: tabulated ξ*, one per table slice.
- `table[j, l, m]`: δ of slice `m`.
- `lower_slice[i, k]`, `upper_weight[i, k]`: the single-ξ* interpolation of pair `(i, k)`,
  which uses slices `lower_slice` and `lower_slice + 1` with weights `1 - upper_weight`
  and `upper_weight`. With `xi_star_mode = :interval` the operator instead spreads the
  pair over every slice its ξ* range covers ([`KnockOnKernels`](@ref)); these two fields
  then describe the midpoint interpolation, which serves as a reference.
- `xi_star_mode`: how ξ* of a pair enters (`:interval`, `:midpoint` or `:centre`).
- `no_contributing_pairs`: no `(i, k)` pair produces knock-ons (cutoff above the grid).

In cylindrical geometry the kernel does not depend on radius; one table serves all
radii (DREAM stores one per radius).
"""
struct MollerPitchKernel{T <: AbstractFloat}
    xi_star_values::Vector{T}   # length table_size
    table::Array{T, 3}           # (knock-on ξ cells, primary ξ cells, table_size)
    lower_slice::Matrix{Int}    # (knock-on p cells, primary p cells)
    upper_weight::Matrix{T}     # (knock-on p cells, primary p cells)
    xi_star_mode::Symbol
    no_contributing_pairs::Bool
end

"""
    MollerPitchKernel(knockon, primary, p_cutoff; table_size=80, dream_compatible=false,
                      xi_star_mode=(dream_compatible ? :centre : :interval),
                      primary_pitch=(dream_compatible ? :centre : :split),
                      primary_pitch_points=4)

- The part of knock-on cell `i` that receives knock-ons from primary cell `k` is
  `[max(p_{i-1/2}, p_cutoff), min(p_{i+1/2}, p_max(p1_k))]`, the interval the energy
  kernel integrates over. `ξ*` varies over it, fastest at low `p`.
- `xi_star_mode` selects how that is treated. `:interval` (the default) spreads each pair
  over the table slices its `ξ*` range covers, with the weights of
  [`KnockOnKernels`](@ref); the table then covers the `ξ*` of the interval ends of all
  contributing pairs, so that no pair is clamped and the production stays exact.
  `:midpoint` evaluates `ξ*` once, at the midpoint of the interval, and `:centre` once at
  the knock-on cell centre (DREAM). `:midpoint` keeps `ξ* < 1`.
  `:centre` does not on a knock-on grid coarser than the primary grid, where a cell
  centre can lie above `p1`: a pair with `ξ* > 1` in the table range is refused with an
  `ArgumentError`, since no pitch distribution exists for it. The table of the two
  single-`ξ*` modes covers the `ξ*` of all contributing pairs.
- `dream_compatible = true` reproduces DREAM (`MollerDeltaAngleKernel.cpp`): `ξ*` at the
  knock-on cell centre, and a table range from the pairs with `γ(p_{i-1/2}) < γ(p1_k) - 1`
  (a total energy compared with a kinetic energy). For primaries below γ₁ ≈ 3 this misses
  contributing pairs, whose `ξ*` is then clamped to the table edge; with a knock-on grid
  coarser than the primary grid, a cell centre above `p1` gives `ξ* > 1`, which the
  range criterion excludes for most grids and clamps.
- The knock-on pitch grid must span `[-1, 1]`; the primary pitch grid may cover part of it.
- `primary_pitch`: how the primary pitch enters. `:centre` evaluates it at the cell
  centre (DREAM), which breaks reciprocity and isotropy at O(1) in the max norm.
  `:split` averages it
  over the cell, split at the kink points ([`pitch_kernel_column_split!`](@ref)), with
  `primary_pitch_points` Gauss points per piece: isotropic primaries then give isotropic
  knock-ons to quadrature accuracy. `:gauss` averages with `primary_pitch_points` plain
  Gauss points over the whole cell (slow convergence; kept for comparison).
"""
function MollerPitchKernel(
        knockon::MomentumGrid{T}, primary::MomentumGrid{T}, p_cutoff::Real;
        table_size::Integer = 80, dream_compatible::Bool = false,
        xi_star_mode::Symbol = (dream_compatible ? :centre : :interval),
        primary_pitch::Symbol = (dream_compatible ? :centre : :split),
        primary_pitch_points::Integer = 4) where {T}
    table_size >= 2 || throw(ArgumentError("table_size must be at least 2"))
    primary_pitch in (:centre, :split, :gauss) ||
        throw(ArgumentError("unknown primary_pitch $primary_pitch"))
    xi_star_mode in (:interval, :midpoint, :centre) ||
        throw(ArgumentError("unknown xi_star_mode $xi_star_mode"))
    !(dream_compatible && xi_star_mode !== :centre) ||
        throw(ArgumentError("dream_compatible = true requires xi_star_mode = :centre"))
    primary_pitch_points >= 1 ||
        throw(ArgumentError("primary_pitch_points must be at least 1"))
    p_cutoff > 0 || throw(ArgumentError("p_cutoff must be positive"))
    # the gyro-averaged knock-on pitch covers [-1, 1]; on a partial grid the column
    # normalization would move the knock-ons outside it onto the grid
    (first(knockon.xi_faces) == -1 && last(knockon.xi_faces) == 1) ||
        throw(ArgumentError("the knock-on pitch grid must span [-1, 1]"))
    np_knockon, np_primary = momentum_cell_count(knockon), momentum_cell_count(primary)
    gamma(p) = sqrt(1 + p^2)
    # ξ* of each momentum pair, and whether it sets the table range
    pair_xi_star = zeros(T, np_knockon, np_primary)
    lowest, highest = T(Inf), T(-Inf)
    for i in 1:np_knockon, k in 1:np_primary

        p1 = primary.p_centers[k]
        p_lower = max(knockon.p_faces[i], T(p_cutoff))
        p_upper = min(knockon.p_faces[i + 1], max_knockon_momentum(p1))
        if xi_star_mode === :centre
            xi_s = xi_star(knockon.p_centers[i], p1)
            in_range = if dream_compatible
                knockon.p_faces[i + 1] > p_cutoff &&
                    gamma(knockon.p_faces[i]) < gamma(p1) - 1
            else
                p_upper > p_lower
            end
        else
            in_range = p_upper > p_lower
            xi_s = if in_range
                xi_star((p_lower + p_upper) / 2, p1)
            else
                xi_star(knockon.p_centers[i], p1)
            end
        end
        pair_xi_star[i, k] = xi_s
        (in_range && isfinite(xi_s)) || continue
        # ξ* at the cell centre of a knock-on cell wider than the primary's reach
        if !(xi_s <= 1 + 8 * eps(T))
            throw(ArgumentError("ξ* = $xi_s > 1 for knock-on cell $i (centre $(knockon.p_centers[i])) and " *
                                "primary cell $k (p1 = $p1): the knock-on grid is coarser than the " *
                                "primary grid; use xi_star_mode = :interval or :midpoint"))
        end
        if xi_star_mode === :interval
            # the whole ξ* range of the pair must be inside the table: the hat functions
            # are a partition of unity there, which keeps the production exact
            lowest = min(lowest, xi_star(p_lower, p1))
            highest = max(highest, xi_star(p_upper, p1))
        else
            lowest, highest = min(lowest, xi_s), max(highest, xi_s)
        end
    end
    no_contributing_pairs = !(lowest <= highest)
    if no_contributing_pairs
        lowest, highest = zero(T), one(T)
    end
    spacing = (highest - lowest) / (table_size - 1)
    xi_star_values = [lowest + m * spacing for m in 0:(table_size - 1)]

    nodes, weights = if primary_pitch === :centre
        ([zero(T)], [T(2)])
    else
        gauss(T, primary_pitch_points)
    end
    nxi_knockon, nxi_primary = pitch_cell_count(knockon), pitch_cell_count(primary)
    table = zeros(T, nxi_knockon, nxi_primary, table_size)
    for m in 1:table_size, l in 1:nxi_primary

        column = view(table, :, l, m)
        norm = if primary_pitch === :centre
            pitch_kernel_column!(column, knockon.xi_faces, xi_star_values[m];
                primary_cell = (primary.xi_centers[l], primary.xi_centers[l]))
        elseif primary_pitch === :gauss
            pitch_kernel_column!(column, knockon.xi_faces, xi_star_values[m];
                primary_cell = (primary.xi_faces[l], primary.xi_faces[l + 1]), nodes, weights)
        else
            pitch_kernel_column_split!(column, knockon.xi_faces, xi_star_values[m];
                primary_cell = (primary.xi_faces[l], primary.xi_faces[l + 1]),
                nodes, weights)
        end
        if !(norm > 0 || no_contributing_pairs)
            throw(ErrorException("empty knock-on pitch column at l = $l, ξ* = $(xi_star_values[m])"))
        end
    end

    lower_slice = ones(Int, np_knockon, np_primary)
    upper_weight = zeros(T, np_knockon, np_primary)
    for i in 1:np_knockon, k in 1:np_primary

        xi_s = pair_xi_star[i, k]
        if !(xi_s > lowest) || spacing == 0
            lower_slice[i, k], upper_weight[i, k] = 1, zero(T)
        elseif xi_s >= highest
            lower_slice[i, k], upper_weight[i, k] = table_size - 1, one(T)
        else
            s = (xi_s - lowest) / spacing
            m = min(floor(Int, s), table_size - 2)
            lower_slice[i, k], upper_weight[i, k] = m + 1, s - m
        end
    end
    return MollerPitchKernel{T}(
        xi_star_values, table, lower_slice, upper_weight, xi_star_mode,
        no_contributing_pairs)
end

"""
    pitch_kernel_matrix!(D, kernel::MollerPitchKernel, i, k)
    pitch_kernel_matrix!(D, kernels::KnockOnKernels, i, k)

Pitch kernel of the momentum pair `(i, k)`: from a [`MollerPitchKernel`](@ref) the kernel
interpolated at the pair's single ξ*, `D[j, l] = δ[j, l](ξ*(p_i, p1_k))`; from
[`KnockOnKernels`](@ref) the effective kernel the operator applies, the table slices of
the pair weighted by its entries and normalized by the pair's production, which for
`xi_star_mode = :interval` is the kernel integrated over ξ*(p) across the receiving
interval and otherwise the same interpolation. `rates[i, k]` times this matrix is what
[`apply!`](@ref) contributes for the pair.
"""
function pitch_kernel_matrix!(D::AbstractMatrix, kernel::MollerPitchKernel, i, k)
    m = kernel.lower_slice[i, k]
    w = kernel.upper_weight[i, k]
    @views D .= (1 - w) .* kernel.table[:, :, m] .+ w .* kernel.table[:, :, m + 1]
    return D
end
