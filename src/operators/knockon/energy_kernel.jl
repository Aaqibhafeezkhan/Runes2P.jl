"""
    MollerEnergyKernel{T}

Momentum part of the knock-on operator between a primary momentum grid (incident
electrons, cells `k`) and a knock-on momentum grid (cells `i`):

```math
\\mathtt{rates}[i, k] = \\beta_1(p_{1,k}) \\, \\frac{S(p_U) - S(p_L)}{\\Delta p_i},
```

the Møller production rate coefficient [m³/s] averaged over knock-on cell `i` (`S_ik` of
the DREAM notes), with `[p_L, p_U]` the part of the cell inside
`[p_cutoff, max_knockon_momentum(p1_k)]` and the primary evaluated at its cell centre
`p1_k`. `S` is [`moller_flux`](@ref).

- `total_rates[k] = Σ_i Δp_i rates[i, k]`: production rate coefficient onto the knock-on
  grid.
- `rates_above_grid[k]`: production above the top face of the knock-on grid (knock-ons
  that leave the grid, e.g. into a fluid runaway population).

Ported from DREAM `MollerEnergyKernel.cpp` and `KnockOnUtilities.cpp`
(`EvaluateMollerFluxMatrixElementOnGrid`, `EvaluateMollerFluxIntegratedOverKnockonGrid`).
"""
struct MollerEnergyKernel{T <: AbstractFloat}
    rates::Matrix{T}             # (knock-on p cells, primary p cells)
    total_rates::Vector{T}       # (primary p cells)
    rates_above_grid::Vector{T}  # (primary p cells)
    p_cutoff::T
end

"""
    MollerEnergyKernel(knockon, primary, p_cutoff)

Build the kernel for knock-on momentum grid `knockon` and primary momentum grid
`primary` (both `MomentumGrid`; only the p grids are used) with cutoff momentum
`p_cutoff > 0`.
"""
function MollerEnergyKernel(knockon::MomentumGrid{T}, primary::MomentumGrid{T}, p_cutoff::Real) where {T}
    p_cutoff > 0 ||
        throw(ArgumentError("p_cutoff must be positive (S is singular at p = 0)"))
    cutoff = T(p_cutoff)
    np_knockon, np_primary = momentum_cell_count(knockon), momentum_cell_count(primary)
    rates = zeros(T, np_knockon, np_primary)
    total_rates = zeros(T, np_primary)
    rates_above_grid = zeros(T, np_primary)
    p_top = last(knockon.p_faces)
    for k in 1:np_primary
        p1 = primary.p_centers[k]
        p_max = max_knockon_momentum(p1)
        beta1 = p1 / sqrt(1 + p1^2)
        for i in 1:np_knockon
            p_lower = max(knockon.p_faces[i], cutoff)
            p_upper = min(knockon.p_faces[i + 1], p_max)
            p_upper > p_lower || continue
            rates[i, k] = beta1 * (moller_flux(p_upper, p1) - moller_flux(p_lower, p1)) /
                          knockon.p_widths[i]
        end
        # O(1) evaluation of Σ_i Δp_i rates[i, k], as in DREAM
        p_lower = max(first(knockon.p_faces), cutoff)
        p_upper = min(p_top, p_max)
        if p_upper > p_lower
            total_rates[k] = beta1 * (moller_flux(p_upper, p1) - moller_flux(p_lower, p1))
        end
        rates_above_grid[k] = moller_total_rate(p1, max(cutoff, p_top))
    end
    return MollerEnergyKernel{T}(rates, total_rates, rates_above_grid, cutoff)
end
