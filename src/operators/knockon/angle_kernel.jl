# Pitch part of the knock-on operator in cylindrical geometry (B constant, ξ = ξ₀),
# following DREAM `KnockOnUtilities.cpp` (namespace `AnalyticDelta`,
# `SetDeltaMatrixColumnOnGrid`) and `MollerDeltaAngleKernel.cpp`.
#
# A primary with pitch ξ₁ produces a knock-on at angle arccos ξ* from its direction.
# Averaged over the gyrophase φ, the knock-on pitch ξ = a + b cos φ with a = ξ₁ξ*,
# b = √((1 - ξ₁²)(1 - ξ*²)) is distributed with density 1/(π√(b² - (ξ - a)²)) on
# [a - b, a + b]; its antiderivative is asin((ξ - a)/b)/π.

using QuadGK: gauss

"""
    pitch_cell_fraction(t1, t2, ξ1, ξs)

Fraction of knock-ons with pitch in `[t1, t2]` produced by a primary with pitch `ξ1`
at scattering-angle cosine `ξs`, averaged over the gyrophase (DREAM `D`,
`AnalyticDelta::EvaluateXiIntegral`). The argument of `asin` is clamped to `[-1, 1]`,
so no separate reachability test is needed. When the distribution degenerates to a
point (`b² ≤ 5ε`), the point is assigned to the half-open cell `(t1, t2]`, as in DREAM.
"""
function pitch_cell_fraction(t1, t2, ξ1, ξs)
    a = ξs * ξ1
    b2 = (1 - ξs^2) * (1 - ξ1^2)
    if b2 <= 5 * eps(typeof(b2))
        return (t1 < a && a <= t2) ? one(a) : zero(a)
    end
    b = sqrt(b2)
    F(y) = asin(clamp((y - a) / b, -one(a), one(a))) / π
    return F(t2) - F(t1)
end

"""
    delta_column!(col, xi_f, ξs; cell, nodes=(0.0,), weights=(2.0,))

Fill `col[j] = δ_j` for knock-on pitch cells with faces `xi_f`, normalized so that
`Σ_j Δξ_j δ_j = 1`, and return the norm before normalization. The primary pitch is
averaged over `cell = (ξ1_lo, ξ1_hi)` with Gauss nodes and weights on `[-1, 1]`; with
`cell = (ξ1, ξ1)` and the default single node the primary sits at `ξ1` (DREAM). A primary at
`ξ1 = -1` whose knock-on pitch is exactly `-1` is assigned to the first cell.
"""
function delta_column!(col::AbstractVector, xi_f::AbstractVector, ξs;
                       cell, nodes=(0.0,), weights=(2.0,))
    n2 = length(col)
    lo, hi = cell
    fill!(col, 0)
    for (x, w) in zip(nodes, weights)
        ξ1 = (lo + hi) / 2 + x * (hi - lo) / 2
        for j in 1:n2
            col[j] += w / 2 * pitch_cell_fraction(xi_f[j], xi_f[j+1], ξ1, ξs)
        end
        # point mass at ξ = -1 (half-open cells exclude it)
        if (1 - ξs^2) * (1 - ξ1^2) <= 5 * eps(ξs) && ξs * ξ1 == xi_f[1]
            col[1] += w / 2
        end
    end
    norm = sum(col)
    for j in 1:n2
        col[j] /= (xi_f[j+1] - xi_f[j])
    end
    if norm > 0
        col ./= norm
    end
    return norm
end

"""
    delta_column_split!(col, xi_f, ξs; cell, nodes, weights)

Like [`delta_column!`](@ref) with the primary averaged over `cell = (ξ1_lo, ξ1_hi)`, but
exact to high order: as a function of the primary pitch ξ₁, the fraction `D_j(ξ₁)`
reaching knock-on cell `j` has square-root kinks where an edge `a ± b` of the
gyro-averaged distribution crosses a face `t` of the knock-on grid, i.e. at
`ξ₁ = cos(arccos t ± θ*)` with `cos θ* = ξ*` (and jumps at the faces when `ξ* = 1`). The
cell is split at these points, and each piece is integrated with the Gauss `nodes` and
`weights` (on `[-1, 1]`) in the variable `s` of `ξ₁ = mid + h sin(πs/2)`, which turns the
square-root behaviour at the piece ends into a smooth one.
"""
function delta_column_split!(col::AbstractVector, xi_f::AbstractVector, ξs;
                             cell, nodes, weights)
    n2 = length(col)
    lo, hi = cell
    hi > lo || throw(ArgumentError("the primary cell must have positive width"))
    θs = acos(clamp(ξs, -one(ξs), one(ξs)))
    cuts = [lo, hi]
    for t in xi_f
        θt = acos(clamp(t, -one(t), one(t)))
        for x in (cos(θt + θs), cos(θt - θs))
            lo < x < hi && push!(cuts, x)
        end
    end
    sort!(cuts)
    fill!(col, 0)
    for q in 1:length(cuts)-1
        u, v = cuts[q], cuts[q+1]
        v > u || continue
        mid, h = (u + v) / 2, (v - u) / 2
        for (x, w) in zip(nodes, weights)
            ξ1 = mid + h * sin(π * x / 2)
            wt = w * h * (π / 2) * cos(π * x / 2)
            for j in 1:n2
                col[j] += wt * pitch_cell_fraction(xi_f[j], xi_f[j+1], ξ1, ξs)
            end
        end
    end
    col ./= (hi - lo)
    norm = sum(col)
    for j in 1:n2
        col[j] /= (xi_f[j+1] - xi_f[j])
    end
    if norm > 0
        col ./= norm
    end
    return norm
end

"""
    MollerAngleKernel{T}

Knock-on pitch kernel `δ[j, l](ξ*)` between primary pitch cells `l` and knock-on pitch
cells `j`, tabulated on a uniform grid of `ξ*` values and interpolated linearly to the
`ξ*` of each momentum pair `(i, k)`. Every tabulated column satisfies
`Σ_j Δξ_j δ[j, l] = 1`, and so does every interpolated one.

In cylindrical geometry the kernel does not depend on radius; one table serves all
radii (DREAM stores one per radius).
"""
struct MollerAngleKernel{T<:AbstractFloat}
    xs::Vector{T}       # tabulated ξ*, length M
    table::Array{T,3}   # (n_ξ knock-on, n_ξ primary, M)
    m0::Matrix{Int}     # lower slice for (i, k)
    w1::Matrix{T}       # weight of slice m0 + 1
    empty::Bool         # no (i, k) pair produces knock-ons
end

"""
    MollerAngleKernel(knockon, primary, p_cut; n_tab=80, dream=false,
                      primary_mode=(dream ? :centre : :split), primary_points=4)

- `ξ*` of pair `(i, k)` is evaluated at the midpoint of the part of knock-on cell `i`
  that receives knock-ons, `[max(p_{i-1/2}, p_cut), min(p_{i+1/2}, p_max(p1_k))]` (the
  interval the energy kernel integrates over), which keeps `ξ* < 1`; the table covers
  the `ξ*` of all such pairs.
- `dream = true` reproduces DREAM (`MollerDeltaAngleKernel.cpp`): `ξ*` at the knock-on
  cell centre, and a table range from the pairs with `γ(p_{i-1/2}) < γ(p1_k) - 1` (a
  total energy compared with a kinetic energy). For primaries below γ₁ ≈ 3 this misses
  contributing pairs, whose `ξ*` is then clamped to the table edge; with a knock-on grid
  coarser than the primary grid, a cell centre above `p1` gives `ξ* > 1`.
- The knock-on pitch grid must span `[-1, 1]`; the primary pitch grid may cover part of it.
- `primary_mode`: how the primary pitch enters. `:centre` evaluates it at the cell centre
  (DREAM), which breaks reciprocity and isotropy at O(1) in the max norm.
  `:split` averages it
  over the cell, split at the kink points ([`delta_column_split!`](@ref)), with
  `primary_points` Gauss points per piece: isotropic primaries then give isotropic
  knock-ons to quadrature accuracy. `:gauss` averages with `primary_points` plain Gauss
  points over the whole cell (slow convergence; kept for comparison).
"""
function MollerAngleKernel(knockon::PXiGrid{T}, prim_grid::PXiGrid{T}, p_cut::Real;
                           n_tab::Integer=80, dream::Bool=false,
                           primary_mode::Symbol=(dream ? :centre : :split),
                           primary_points::Integer=4) where {T}
    n_tab >= 2 || throw(ArgumentError("n_tab must be at least 2"))
    primary_mode in (:centre, :split, :gauss) ||
        throw(ArgumentError("unknown primary_mode $primary_mode"))
    primary_points >= 1 || throw(ArgumentError("primary_points must be at least 1"))
    p_cut > 0 || throw(ArgumentError("p_cut must be positive"))
    # the gyro-averaged knock-on pitch covers [-1, 1]; on a partial grid the column
    # normalization would move the knock-ons outside it onto the grid
    (first(knockon.xi_f) == -1 && last(knockon.xi_f) == 1) ||
        throw(ArgumentError("the knock-on pitch grid must span [-1, 1]"))
    nK, nP = n_p(knockon), n_p(prim_grid)
    γ(p) = sqrt(1 + p^2)
    # ξ* of each momentum pair, and whether it sets the table range
    xs_pair = zeros(T, nK, nP)
    lo, hi = T(Inf), T(-Inf)
    for i in 1:nK, k in 1:nP
        p1 = prim_grid.p[k]
        pL = max(knockon.p_f[i], T(p_cut))
        pU = min(knockon.p_f[i+1], max_knockon_momentum(p1))
        if dream
            xs = xi_star(knockon.p[i], p1)
            in_range = knockon.p_f[i+1] > p_cut && γ(knockon.p_f[i]) < γ(p1) - 1
        else
            in_range = pU > pL
            xs = in_range ? xi_star((pL + pU) / 2, p1) : xi_star(knockon.p[i], p1)
        end
        xs_pair[i, k] = xs
        (in_range && isfinite(xs)) || continue
        lo, hi = min(lo, xs), max(hi, xs)
    end
    empty = !(lo <= hi)
    if empty
        lo, hi = zero(T), one(T)
    end
    dxs = (hi - lo) / (n_tab - 1)
    xs_tab = [lo + m * dxs for m in 0:n_tab-1]

    nodes, weights = primary_mode === :centre ? ([zero(T)], [T(2)]) : gauss(T, primary_points)
    n2K, n2P = n_xi(knockon), n_xi(prim_grid)
    table = zeros(T, n2K, n2P, n_tab)
    for m in 1:n_tab, l in 1:n2P
        col = view(table, :, l, m)
        norm = if primary_mode === :centre
            delta_column!(col, knockon.xi_f, xs_tab[m]; cell=(prim_grid.xi[l], prim_grid.xi[l]))
        elseif primary_mode === :gauss
            delta_column!(col, knockon.xi_f, xs_tab[m];
                          cell=(prim_grid.xi_f[l], prim_grid.xi_f[l+1]), nodes, weights)
        else
            delta_column_split!(col, knockon.xi_f, xs_tab[m];
                                cell=(prim_grid.xi_f[l], prim_grid.xi_f[l+1]), nodes, weights)
        end
        norm > 0 || empty || throw(ErrorException("empty knock-on pitch column at l = $l, " *
                                                  "ξ* = $(xs_tab[m])"))
    end

    m0 = ones(Int, nK, nP)
    w1 = zeros(T, nK, nP)
    for i in 1:nK, k in 1:nP
        xs = xs_pair[i, k]
        if !(xs > lo) || dxs == 0
            m0[i, k], w1[i, k] = 1, zero(T)
        elseif xs >= hi
            m0[i, k], w1[i, k] = n_tab - 1, one(T)
        else
            s = (xs - lo) / dxs
            m = min(floor(Int, s), n_tab - 2)
            m0[i, k], w1[i, k] = m + 1, s - m
        end
    end
    return MollerAngleKernel{T}(xs_tab, table, m0, w1, empty)
end

"""
    delta_matrix!(D, K, i, k)

Interpolated pitch kernel `D[j, l] = δ[j, l](ξ*(p_i, p1_k))` for knock-on momentum cell
`i` and primary momentum cell `k`.
"""
function delta_matrix!(D::AbstractMatrix, K::MollerAngleKernel, i, k)
    m = K.m0[i, k]
    w = K.w1[i, k]
    @views D .= (1 - w) .* K.table[:, :, m] .+ w .* K.table[:, :, m+1]
    return D
end
