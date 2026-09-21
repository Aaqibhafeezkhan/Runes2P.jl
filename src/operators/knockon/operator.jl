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
using LinearOperators: LinearOperator

"""
    KnockOnKernels{T}

Energy and pitch kernels for one cutoff momentum, with the column layout of the `:blas`
evaluation: the primary momenta whose pairs use slice m are `slice_k[m]`, and their
products occupy columns `slice_off[m] + 1 : slice_off[m] + length(slice_k[m])` of the
operator's `Y`.
"""
struct KnockOnKernels{T<:AbstractFloat}
    energy::MollerEnergyKernel{T}
    angle::MollerAngleKernel{T}
    slice_k::Vector{Vector{Int}}
    slice_off::Vector{Int}
    col0::Matrix{Int}  # Y column of (slice m0, k) for pair (i, k); 0 if S[i, k] = 0
    col1::Matrix{Int}  # Y column of (slice m0 + 1, k); 0 if its weight is zero
    ncol::Int
end

function KnockOnKernels(knockon::PXiGrid{T}, primary::PXiGrid{T}, p_cut::Real;
                        kwargs...) where {T}
    E = MollerEnergyKernel(knockon, primary, p_cut)
    A = MollerAngleKernel(knockon, primary, p_cut; kwargs...)
    npK, npP = n_p(knockon), n_p(primary)
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
    return KnockOnKernels{T}(E, A, slice_k, slice_off, col0, col1, sum(length, slice_k))
end

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

The cutoff momentum `p_cut[r]` may differ between radii and change in time
([`set_cutoff!`](@ref)); each radius uses the kernels of its cutoff, and radii with
equal cutoffs share them. Both grids must have the same radial grid. The operator is
linear in `f` and in `n_tot`; DREAM evaluates it explicitly, from `f` at the previous
time level, with one cutoff for all radii and times.
"""
mutable struct KnockOnOperator{T<:AbstractFloat}
    const knockon::KineticGrid{T}
    const primary::KineticGrid{T}
    const method::Symbol
    const angle_options::NamedTuple
    const p_cut::Vector{T}
    const kernels::Vector{KnockOnKernels{T}}      # per radius
    const cache::Dict{T,KnockOnKernels{T}}        # by cutoff value, current cutoffs only
    const W::Matrix{T}     # scratch (nξ primary, np primary)
    const C::Vector{T}     # scratch (nξ knock-on)
    const Wbuf::Matrix{T}  # gathered primary weights (nξ primary, np primary)
    Y::Matrix{T}           # (nξ knock-on, largest number of (slice, k) combinations)
end

"""
    KnockOnOperator(knockon, primary, p_cut; method=:blas, n_tab=80, dream=false,
                    primary_points=1)

Build the energy and pitch kernels for knock-on grid `knockon` and primary grid
`primary` (both `KineticGrid`) with cutoff momentum `p_cut` (a number, or one value per
radius). `method` selects the evaluation (`:blas` or `:loop`, see the source file); the
other keyword arguments are passed to [`MollerAngleKernel`](@ref).
`dream = true, primary_points = 1, n_tab = 80` with one cutoff reproduces DREAM's
`MollerBoltzmannOperator` in cylindrical geometry.
"""
function KnockOnOperator(knockon::KineticGrid{T}, primary::KineticGrid{T},
                         p_cut::Union{Real,AbstractVector}; method::Symbol=:blas,
                         kwargs...) where {T}
    knockon.rad.r_f == primary.rad.r_f ||
        throw(ArgumentError("knock-on and primary grids must share the radial grid"))
    method in (:blas, :loop) || throw(ArgumentError("unknown method $method"))
    nr = n_r(knockon)
    op = KnockOnOperator{T}(knockon, primary, method, NamedTuple(kwargs), zeros(T, nr),
                            Vector{KnockOnKernels{T}}(undef, nr),
                            Dict{T,KnockOnKernels{T}}(),
                            zeros(T, n_xi(primary), n_p(primary)), zeros(T, n_xi(knockon)),
                            zeros(T, n_xi(primary), method == :blas ? n_p(primary) : 0),
                            zeros(T, n_xi(knockon), 0))
    set_cutoff!(op, p_cut)
    return op
end

"""
    set_cutoff!(op::KnockOnOperator, p_cut)

Set the cutoff momentum (a number for all radii, or one value per radius) and rebuild
the kernels of the radii whose cutoff changed. Kernels are cached by cutoff value, so
radii with equal cutoffs share one set, and a value already in use costs nothing.
Returns the number of kernel sets built. A rebuild costs a few operator applications;
snapping the cutoff to momentum cell faces ([`choose_cutoff`](@ref)) keeps rebuilds rare
when the cutoff follows the plasma.

A `knockon_linear_operator` of `op` sees the new kernels.
"""
function set_cutoff!(op::KnockOnOperator{T}, p_cut::Union{Real,AbstractVector}) where {T}
    nr = n_r(op.knockon)
    p_cut isa AbstractVector && length(p_cut) != nr &&
        throw(DimensionMismatch("p_cut must be a number or have one entry per radius"))
    built = 0
    for ir in 1:nr
        pc = T(p_cut isa AbstractVector ? p_cut[ir] : p_cut)
        pc > 0 || throw(ArgumentError("p_cut must be positive"))
        kern = get(op.cache, pc, nothing)
        if kern === nothing
            kern = KnockOnKernels(op.knockon.mom, op.primary.mom, pc; op.angle_options...)
            op.cache[pc] = kern
            built += 1
        end
        op.p_cut[ir] = pc
        op.kernels[ir] = kern
    end
    # keep only the kernels in use
    for key in collect(keys(op.cache))
        key in op.p_cut || delete!(op.cache, key)
    end
    if op.method === :blas
        ncol = maximum(k -> k.ncol, op.kernels)
        ncol > size(op.Y, 2) && (op.Y = zeros(T, n_xi(op.knockon), ncol))
    end
    return built
end

"""
    energy_kernel(op::KnockOnOperator, ir=1)
    angle_kernel(op::KnockOnOperator, ir=1)

Energy and pitch kernels used at radius index `ir`.
"""
energy_kernel(op::KnockOnOperator, ir::Integer=1) = op.kernels[ir].energy
angle_kernel(op::KnockOnOperator, ir::Integer=1) = op.kernels[ir].angle

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
            apply_radius_blas!(df, op, op.kernels[ir], n_tot[ir] * α, ir)
        else
            apply_radius_loop!(df, op, op.kernels[ir], n_tot[ir] * α, ir)
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

function apply_radius_loop!(df, op::KnockOnOperator, kern::KnockOnKernels, scale, ir)
    gK, gP = op.knockon, op.primary
    S, K, W, C = kern.energy.S, kern.angle, op.W, op.C
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

function apply_radius_blas!(df, op::KnockOnOperator, kern::KnockOnKernels, scale, ir)
    gK = op.knockon
    S, K, W, C, Y, Wbuf = kern.energy.S, kern.angle, op.W, op.C, op.Y, op.Wbuf
    nξK = n_xi(gK)
    # Y_m = D_m W[:, K_m] for every slice
    for m in eachindex(kern.slice_k)
        ks = kern.slice_k[m]
        n = length(ks)
        n == 0 && continue
        @inbounds for (c, k) in enumerate(ks), l in axes(W, 1)
            Wbuf[l, c] = W[l, k]
        end
        off = kern.slice_off[m]
        mul!(view(Y, :, off+1:off+n), view(K.table, :, :, m), view(Wbuf, :, 1:n))
    end
    # gather with the energy kernel and the interpolation weights
    @inbounds for i in 1:n_p(gK)
        fill!(C, 0)
        for k in axes(S, 2)
            c0 = kern.col0[i, k]
            c0 == 0 && continue
            s = S[i, k]
            c1 = kern.col1[i, k]
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

"""
    knockon_linear_operator(op::KnockOnOperator, n_tot; α=1)

Matrix-free linear operator `α ∂C/∂f` of size `(ncells(knockon), ncells(primary))`, acting
on `vec(f)` of the primary grid and returning `vec` of the knock-on grid: the implicit
(Jacobian) form of the knock-on term, for use in Krylov solvers and in sums with other
`LinearOperator`s or matrices. `C` is linear in `f`, so the Jacobian-vector product is
[`apply!`](@ref) itself. `n_tot` is referenced, not copied: updating it in place (e.g.
between Newton iterations) changes the operator.

The derivative with respect to `n_tot` is `C(f)/n_tot` at each radius (DREAM's only
Jacobian block for this term, which it treats explicitly in `f`). Treating `f`
implicitly is a Runes2P departure; the explicit form stays available through `apply!`.

The product uses the operator's scratch arrays and is not safe to call concurrently.
No transpose product is provided.
"""
function knockon_linear_operator(op::KnockOnOperator{T}, n_tot::AbstractVector; α=one(T)) where {T}
    szK, szP = size(op.knockon), size(op.primary)
    length(n_tot) == n_r(op.knockon) ||
        throw(DimensionMismatch("n_tot must have one entry per radius"))
    function prod!(res, v, a, b)
        if iszero(b)
            fill!(res, zero(eltype(res)))
        elseif !isone(b)
            res .*= b
        end
        apply!(reshape(res, szK), op, reshape(v, szP), n_tot; α=a * α)
        return res
    end
    return LinearOperator{T,Vector{T}}(prod(szK), prod(szP), false, false, prod!,
                                       nothing, nothing)
end
