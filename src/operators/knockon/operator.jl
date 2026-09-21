# Knock-on (Møller Boltzmann) gain operator: the energy kernel `S[i, k]` and the pitch
# kernel `δ[j, l](ξ*(p_i, p1_k))` applied to a primary distribution.
#
# Two equivalent evaluations:
# - `:loop`: DREAM `MollerBoltzmannOperator::SetSourceVector` and
#   `MollerDeltaAngleKernel::AccumulatePitch`, interpolating the pitch kernel per momentum
#   pair; cost O(nnz(S) · nξP · nξK) per radius.
# - `:blas`: contract each tabulated slice with the primary weights first,
#   `Y_m = D_m W[:, k ∈ K_m]` (one matrix product per slice m, restricted to the primary
#   momenta K_m whose pairs use slice m), then gather
#   `C[i, j] = Σ_k S_ik (w0 Y_{m0}[j, k] + w1 Y_{m0+1}[j, k])`.
#   Interpolation is linear, so both give the same result up to summation order.

using LinearAlgebra: mul!

"""
    KnockOnOperator{T}

Knock-on source on the grid `knockon` produced by primaries on the grid `primary`
(the same grid for in-grid knock-on) colliding with stationary targets of density
`n_tot`:

    C[i, j, r] = n_tot[r] / 𝒱'_K[i, j, r] Σ_k S[i, k] Σ_l W[k, l, r] δ[j, l](ξ*(p_i, p1_k)),
    W[k, l, r] = Δp_k Δξ_l 𝒱'_P[k, l, r] f[k, l, r].

`C` has the units of `∂f/∂t`. Its density moment is the total production rate:
`(1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j C_ij = n_tot Σ_kl (𝒱'_kl Δp_k Δξ_l f_kl / V') total[k]`, with
`total` from the energy kernel. Only the ratio `𝒱'_P / 𝒱'_K` at equal radius enters, so
any constant normalization of `𝒱'` cancels (DREAM's cylindrical `Vprime` is `4π²` times
Runes2P's).

Both grids must have the same radial grid. The operator is linear in `f` and in `n_tot`;
DREAM evaluates it explicitly, from `f` at the previous time level.
"""
struct KnockOnOperator{T<:AbstractFloat}
    knockon::KineticGrid{T}
    primary::KineticGrid{T}
    energy::MollerEnergyKernel{T}
    angle::MollerAngleKernel{T}
    method::Symbol
    W::Matrix{T}  # scratch (nξ primary, np primary)
    C::Vector{T}  # scratch (nξ knock-on)
    # :blas layout: primary momenta using slice m are slice_k[m]; their products are
    # columns slice_off[m] + 1 : slice_off[m] + length(slice_k[m]) of Y
    slice_k::Vector{Vector{Int}}
    slice_off::Vector{Int}
    col0::Matrix{Int}  # Y column of (slice m0, k) for pair (i, k); 0 if S[i, k] = 0
    col1::Matrix{Int}  # Y column of (slice m0 + 1, k); 0 if its weight is zero
    Wbuf::Matrix{T}    # gathered primary weights (nξ primary, np primary)
    Y::Matrix{T}       # (nξ knock-on, number of (slice, k) combinations)
end

"""
    KnockOnOperator(knockon, primary, p_cut; method=:blas, n_tab=80, dream=false,
                    primary_points=1)

Build the energy and pitch kernels for knock-on grid `knockon` and primary grid
`primary` (both `KineticGrid`) with cutoff momentum `p_cut`. `method` selects the
evaluation (`:blas` or `:loop`, see the source file); the other keyword arguments are
passed to [`MollerAngleKernel`](@ref). `dream = true, primary_points = 1, n_tab = 80`
reproduces DREAM's `MollerBoltzmannOperator` in cylindrical geometry.
"""
function KnockOnOperator(knockon::KineticGrid{T}, primary::KineticGrid{T}, p_cut::Real;
                         method::Symbol=:blas, kwargs...) where {T}
    knockon.rad.r_f == primary.rad.r_f ||
        throw(ArgumentError("knock-on and primary grids must share the radial grid"))
    method in (:blas, :loop) || throw(ArgumentError("unknown method $method"))
    E = MollerEnergyKernel(knockon.mom, primary.mom, p_cut)
    A = MollerAngleKernel(knockon.mom, primary.mom, p_cut; kwargs...)
    npK, npP = n_p(knockon), n_p(primary)
    nξK, nξP = n_xi(knockon), n_xi(primary)
    M = length(A.xs)

    # (slice, k) combinations that any pair with S ≠ 0 needs
    uses = falses(M, npP)
    for k in 1:npP, i in 1:npK
        E.S[i, k] == 0 && continue
        uses[A.m0[i, k], k] = true
        A.w1[i, k] != 0 && (uses[A.m0[i, k]+1, k] = true)
    end
    slice_k = [findall(view(uses, m, :)) for m in 1:M]
    slice_off = cumsum([0; length.(slice_k)[1:end-1]])
    col = zeros(Int, M, npP)
    for m in 1:M, (c, k) in enumerate(slice_k[m])
        col[m, k] = slice_off[m] + c
    end
    col0 = zeros(Int, npK, npP)
    col1 = zeros(Int, npK, npP)
    for k in 1:npP, i in 1:npK
        E.S[i, k] == 0 && continue
        col0[i, k] = col[A.m0[i, k], k]
        A.w1[i, k] != 0 && (col1[i, k] = col[A.m0[i, k]+1, k])
    end
    ncol = method == :blas ? sum(length, slice_k) : 0
    return KnockOnOperator{T}(knockon, primary, E, A, method,
                              zeros(T, nξP, npP), zeros(T, nξK),
                              slice_k, slice_off, col0, col1,
                              zeros(T, nξP, method == :blas ? npP : 0), zeros(T, nξK, ncol))
end

"""
    apply!(df, op::KnockOnOperator, f, n_tot; α=1)

Add `α` times the knock-on source produced by the primary distribution `f[k, l, r]` to
`df[i, j, r]` on the knock-on grid; `n_tot[r]` is the target (total electron) density.
Allocation free; not safe to call concurrently on the same operator (shared scratch).
"""
function apply!(df::AbstractArray{<:Any,3}, op::KnockOnOperator, f::AbstractArray{<:Any,3},
                n_tot::AbstractVector; α=true)
    gK, gP = op.knockon, op.primary
    size(df) == size(gK) || throw(DimensionMismatch("df must have the knock-on grid size"))
    size(f) == size(gP) || throw(DimensionMismatch("f must have the primary grid size"))
    length(n_tot) == n_r(gK) || throw(DimensionMismatch("n_tot must have one entry per radius"))
    for ir in 1:n_r(gK)
        primary_weights!(op.W, gP, f, ir)
        if op.method === :blas
            apply_radius_blas!(df, op, n_tot[ir] * α, ir)
        else
            apply_radius_loop!(df, op, n_tot[ir] * α, ir)
        end
    end
    return df
end

function primary_weights!(W, gP::KineticGrid, f, ir)
    @inbounds for k in 1:n_p(gP), l in 1:n_xi(gP)
        W[l, k] = gP.mom.dp[k] * gP.mom.dxi[l] * gP.Vp[k, l, ir] * f[k, l, ir]
    end
    return W
end

function apply_radius_loop!(df, op::KnockOnOperator, scale, ir)
    gK, gP = op.knockon, op.primary
    S, K, W, C = op.energy.S, op.angle, op.W, op.C
    nξK, nξP = n_xi(gK), n_xi(gP)
    table = K.table
    @inbounds for i in 1:n_p(gK)
        fill!(C, 0)
        for k in 1:n_p(gP)
            s = S[i, k]
            s == 0 && continue
            m = K.m0[i, k]
            w1 = K.w1[i, k]
            w0 = 1 - w1
            for l in 1:nξP
                a = s * W[l, k]
                a == 0 && continue
                for j in 1:nξK
                    C[j] += a * (w0 * table[j, l, m] + w1 * table[j, l, m+1])
                end
            end
        end
        for j in 1:nξK
            df[i, j, ir] += scale * C[j] / gK.Vp[i, j, ir]
        end
    end
    return df
end

function apply_radius_blas!(df, op::KnockOnOperator, scale, ir)
    gK = op.knockon
    S, K, W, C, Y, Wbuf = op.energy.S, op.angle, op.W, op.C, op.Y, op.Wbuf
    nξK = n_xi(gK)
    # Y_m = D_m W[:, K_m] for every slice
    for m in eachindex(op.slice_k)
        ks = op.slice_k[m]
        n = length(ks)
        n == 0 && continue
        @inbounds for (c, k) in enumerate(ks), l in axes(W, 1)
            Wbuf[l, c] = W[l, k]
        end
        off = op.slice_off[m]
        mul!(view(Y, :, off+1:off+n), view(K.table, :, :, m), view(Wbuf, :, 1:n))
    end
    # gather with the energy kernel and the interpolation weights
    @inbounds for i in 1:n_p(gK)
        fill!(C, 0)
        for k in axes(S, 2)
            c0 = op.col0[i, k]
            c0 == 0 && continue
            s = S[i, k]
            c1 = op.col1[i, k]
            w1 = K.w1[i, k]
            a0 = s * (1 - w1)
            if c1 == 0
                for j in 1:nξK
                    C[j] += a0 * Y[j, c0]
                end
            else
                a1 = s * w1
                for j in 1:nξK
                    C[j] += a0 * Y[j, c0] + a1 * Y[j, c1]
                end
            end
        end
        for j in 1:nξK
            df[i, j, ir] += scale * C[j] / gK.Vp[i, j, ir]
        end
    end
    return df
end
