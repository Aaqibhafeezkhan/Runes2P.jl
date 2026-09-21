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
    MollerAngleKernel{T}

Knock-on pitch kernel `δ[j, l](ξ*)` between primary pitch cells `l` and knock-on pitch
cells `j`, tabulated on a uniform grid of `ξ*` values and interpolated linearly to
`ξ*(p_i, p1_k)` at the momentum cell centres. Every tabulated column satisfies
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
    MollerAngleKernel(knockon, primary, p_cut; n_tab=80, range=:support, primary_points=1)

- `range = :support`: tabulate `ξ*` over the momentum pairs `(i, k)` that produce
  knock-ons, i.e. where the energy kernel `S[i, k]` is nonzero.
  `range = :dream` reproduces DREAM's `EvaluateXiStarRangeOnGrids`, which selects pairs
  with `γ(p_{i-1/2}) < γ(p1_k) - 1` (a total energy compared with a kinetic energy).
  For primary grids below γ₁ ≈ 3 this misses contributing pairs, whose `ξ*` is then
  clamped to the table edge.
- `primary_points`: Gauss points averaging the primary over its pitch cell; `1`
  evaluates the primary at the cell centre (DREAM).
"""
function MollerAngleKernel(knockon::PXiGrid{T}, primary::PXiGrid{T}, p_cut::Real;
                           n_tab::Integer=80, range::Symbol=:support,
                           primary_points::Integer=1) where {T}
    n_tab >= 2 || throw(ArgumentError("n_tab must be at least 2"))
    primary_points >= 1 || throw(ArgumentError("primary_points must be at least 1"))
    range in (:support, :dream) || throw(ArgumentError("unknown ξ* range $range"))
    p_cut > 0 || throw(ArgumentError("p_cut must be positive"))
    nK, nP = n_p(knockon), n_p(primary)
    γ(p) = sqrt(1 + p^2)
    lo, hi = T(Inf), T(-Inf)
    for i in 1:nK, k in 1:nP
        p1 = primary.p[k]
        contributes = if range === :support
            min(knockon.p_f[i+1], max_knockon_momentum(p1)) > max(knockon.p_f[i], p_cut)
        else
            knockon.p_f[i+1] > p_cut && γ(knockon.p_f[i]) < γ(p1) - 1
        end
        contributes || continue
        xs = xi_star(knockon.p[i], p1)
        isfinite(xs) || continue
        lo, hi = min(lo, xs), max(hi, xs)
    end
    empty = !(lo <= hi)
    if empty
        lo, hi = zero(T), one(T)
    end
    dxs = (hi - lo) / (n_tab - 1)
    xs_tab = [lo + m * dxs for m in 0:n_tab-1]

    nodes, weights = primary_points == 1 ? ([zero(T)], [T(2)]) : gauss(T, primary_points)
    n2K, n2P = n_xi(knockon), n_xi(primary)
    table = zeros(T, n2K, n2P, n_tab)
    for m in 1:n_tab, l in 1:n2P
        cell = primary_points == 1 ? (primary.xi[l], primary.xi[l]) :
               (primary.xi_f[l], primary.xi_f[l+1])
        norm = delta_column!(view(table, :, l, m), knockon.xi_f, xs_tab[m];
                             cell, nodes, weights)
        norm > 0 || empty || throw(ErrorException("empty knock-on pitch column at l = $l, " *
                                                  "ξ* = $(xs_tab[m])"))
    end

    m0 = ones(Int, nK, nP)
    w1 = zeros(T, nK, nP)
    for i in 1:nK, k in 1:nP
        xs = xi_star(knockon.p[i], primary.p[k])
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
