# Knock-on (Møller Boltzmann) gain operator: the energy kernel `S[i, k]` and the pitch
# kernel `δ[j, l](ξ*(p_i, p1_k))` applied to a primary distribution. Loop form of DREAM
# `MollerBoltzmannOperator::SetSourceVector` and `MollerDeltaAngleKernel::AccumulatePitch`,
# cost O(n_r · npK · npP · nξP · nξK).

"""
    KnockOnOperator{T}

Knock-on source on the grid `knockon` produced by primaries on the grid `primary`
(the same grid for in-grid knock-on) colliding with stationary targets of density
`n_tot`:

    C[i, j, r] = n_tot[r] / 𝒱'_K[i, j, r] Σ_k S[i, k] Σ_l W[k, l, r] δ[j, l](ξ*(p_i, p1_k)),
    W[k, l, r] = Δp_k Δξ_l 𝒱'_P[k, l, r] f[k, l, r].

`C` is a rate density [m⁻³ s⁻¹ per unit phase-space volume, like `∂f/∂t`]. Its density
moment is the total production rate: `(1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j C_ij =
n_tot Σ_kl (𝒱'_kl Δp_k Δξ_l f_kl / V') total[k]`, with `total` from the energy kernel.
Only the ratio `𝒱'_P / 𝒱'_K` at equal radius enters, so any constant normalization of
`𝒱'` cancels (DREAM's cylindrical `Vprime` is `4π²` times Runes2P's).

Both grids must have the same radial grid. The operator is linear in `f` and in `n_tot`;
DREAM evaluates it explicitly, from `f` at the previous time level.
"""
struct KnockOnOperator{T<:AbstractFloat}
    knockon::KineticGrid{T}
    primary::KineticGrid{T}
    energy::MollerEnergyKernel{T}
    angle::MollerAngleKernel{T}
    W::Matrix{T}  # scratch (nξ primary, np primary)
    C::Vector{T}  # scratch (nξ knock-on)
end

"""
    KnockOnOperator(knockon, primary, p_cut; n_tab=80, dream=false, primary_points=1)

Build the energy and pitch kernels for knock-on grid `knockon` and primary grid
`primary` (both `KineticGrid`) with cutoff momentum `p_cut`. The keyword arguments are
passed to [`MollerAngleKernel`](@ref). `dream = true, primary_points = 1, n_tab = 80`
reproduces DREAM's `MollerBoltzmannOperator` in cylindrical geometry.
"""
function KnockOnOperator(knockon::KineticGrid{T}, primary::KineticGrid{T}, p_cut::Real;
                         kwargs...) where {T}
    knockon.rad.r_f == primary.rad.r_f ||
        throw(ArgumentError("knock-on and primary grids must share the radial grid"))
    E = MollerEnergyKernel(knockon.mom, primary.mom, p_cut)
    A = MollerAngleKernel(knockon.mom, primary.mom, p_cut; kwargs...)
    W = zeros(T, n_xi(primary), n_p(primary))
    C = zeros(T, n_xi(knockon))
    return KnockOnOperator{T}(knockon, primary, E, A, W, C)
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
        apply_radius!(df, op, f, n_tot[ir] * α, ir)
    end
    return df
end

function apply_radius!(df, op::KnockOnOperator, f, scale, ir)
    gK, gP = op.knockon, op.primary
    S, K, W, C = op.energy.S, op.angle, op.W, op.C
    npK, nξK = n_p(gK), n_xi(gK)
    npP, nξP = n_p(gP), n_xi(gP)
    table = K.table
    @inbounds for k in 1:npP, l in 1:nξP
        W[l, k] = gP.mom.dp[k] * gP.mom.dxi[l] * gP.Vp[k, l, ir] * f[k, l, ir]
    end
    @inbounds for i in 1:npK
        fill!(C, 0)
        for k in 1:npP
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
