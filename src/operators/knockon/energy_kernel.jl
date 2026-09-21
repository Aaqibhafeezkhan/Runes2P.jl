"""
    MollerEnergyKernel{T}

Momentum part of the knock-on operator between a primary momentum grid (incident
electrons, cells `k`) and a knock-on momentum grid (cells `i`):

    S[i, k] = β₁(p1_k) [S(p_U) - S(p_L)] / Δp_i,

the Møller production rate coefficient [m³/s] averaged over knock-on cell `i`, with
`[p_L, p_U]` the part of the cell inside `[p_cut, max_knockon_momentum(p1_k)]` and the
primary evaluated at its cell centre `p1_k`. `S` is [`moller_flux`](@ref).

- `total[k] = Σ_i Δp_i S[i, k]`: production rate coefficient onto the knock-on grid.
- `above_grid[k]`: production above the top face of the knock-on grid (knock-ons that
  leave the grid, e.g. into a fluid runaway population).

Ported from DREAM `MollerEnergyKernel.cpp` and `KnockOnUtilities.cpp`
(`EvaluateMollerFluxMatrixElementOnGrid`, `EvaluateMollerFluxIntegratedOverKnockonGrid`).
"""
struct MollerEnergyKernel{T<:AbstractFloat}
    S::Matrix{T}           # (np knock-on, np primary)
    total::Vector{T}       # (np primary)
    above_grid::Vector{T}  # (np primary)
    p_cut::T
end

"""
    MollerEnergyKernel(knockon, primary, p_cut)

Build the kernel for knock-on momentum grid `knockon` and primary momentum grid
`primary` (both `PXiGrid`; only the p grids are used) with cutoff momentum `p_cut > 0`.
"""
function MollerEnergyKernel(knockon::PXiGrid{T}, primary::PXiGrid{T}, p_cut::Real) where {T}
    p_cut > 0 || throw(ArgumentError("p_cut must be positive (S is singular at p = 0)"))
    pc = T(p_cut)
    nK, nP = n_p(knockon), n_p(primary)
    S = zeros(T, nK, nP)
    total = zeros(T, nP)
    above = zeros(T, nP)
    p_top = last(knockon.p_f)
    for k in 1:nP
        p1 = primary.p[k]
        pmax = max_knockon_momentum(p1)
        β1 = p1 / sqrt(1 + p1^2)
        for i in 1:nK
            pL = max(knockon.p_f[i], pc)
            pU = min(knockon.p_f[i+1], pmax)
            pU > pL || continue
            S[i, k] = β1 * (moller_flux(pU, p1) - moller_flux(pL, p1)) / knockon.dp[i]
        end
        # O(1) evaluation of Σ_i Δp_i S[i, k], as in DREAM
        pL = max(first(knockon.p_f), pc)
        pU = min(p_top, pmax)
        if pU > pL
            total[k] = β1 * (moller_flux(pU, p1) - moller_flux(pL, p1))
        end
        above[k] = moller_total_rate(p1, max(pc, p_top))
    end
    return MollerEnergyKernel{T}(S, total, above, pc)
end
