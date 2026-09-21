"""
    AdvectionDiffusion{T}

Flux-form advection-diffusion operator on a `KineticGrid`,

    ∂f/∂t = -(1/𝒱') [∂_p(𝒱' Γᵖ) + ∂_ξ(𝒱' Γᵠ) + ∂_r(𝒱' Γʳ)],
    Γᵖ = Aᵖ f - Dᵖᵖ ∂_p f - Dᵖᵠ ∂_ξ f,
    Γᵠ = Aᵠ f - Dᵠᵠ ∂_ξ f - Dᵠᵖ ∂_p f,
    Γʳ = Aʳ f - Dʳʳ ∂_r f,

with coefficients given on the faces of each direction. Particle conservation is exact
up to the fluxes through the outer boundaries.

Boundary conditions:
- `p = p_min` (usually 0), `r = 0`: no flux.
- `ξ = ±1`: no flux.
- `p = p_max`: advective outflow only (`Aᵖ > 0`), no diffusive flux.
- `r = a`: `:zero_flux`, or `:absorbing` (`f = 0` on the wall face).

Advective face values use first-order `:upwind` or second-order `:central` interpolation.
"""
struct AdvectionDiffusion{T<:AbstractFloat}
    grid::KineticGrid{T}
    Ap::Array{T,3}; Dpp::Array{T,3}; Dpxi::Array{T,3}      # on p faces (np+1, nxi, nr)
    Axi::Array{T,3}; Dxixi::Array{T,3}; Dxip::Array{T,3}   # on ξ faces (np, nxi+1, nr)
    Ar::Array{T,3}; Drr::Array{T,3}                        # on r faces (np, nxi, nr+1)
    scheme::Symbol
    rbc::Symbol
end

"""
    AdvectionDiffusion(grid; scheme=:upwind, rbc=:zero_flux)

Operator with all coefficients zero; set them in place, e.g. `op.Dpp .= 1`.
"""
function AdvectionDiffusion(grid::KineticGrid{T}; scheme::Symbol=:upwind,
                            rbc::Symbol=:zero_flux) where {T}
    scheme in (:upwind, :central) || throw(ArgumentError("unknown scheme $scheme"))
    rbc in (:zero_flux, :absorbing) || throw(ArgumentError("unknown radial BC $rbc"))
    n1, n2, n3 = size(grid)
    fp() = zeros(T, n1 + 1, n2, n3)
    fx() = zeros(T, n1, n2 + 1, n3)
    fr() = zeros(T, n1, n2, n3 + 1)
    return AdvectionDiffusion{T}(grid, fp(), fp(), fp(), fx(), fx(), fx(), fr(), fr(),
                                 scheme, rbc)
end

@inline function _face_value(scheme, a, fl, fr, wl)
    if scheme === :upwind
        return a >= 0 ? fl : fr
    else
        return wl * fl + (1 - wl) * fr
    end
end

# Centred derivative of f in ξ at cell (i, j), one-sided at the ξ edges.
Base.@propagate_inbounds function _dxi_cell(f, g::PXiGrid, i, j, ir)
    n2 = nxi(g)
    jl = max(j - 1, 1)
    jr = min(j + 1, n2)
    jl == jr && return zero(eltype(f))
    return (f[i, jr, ir] - f[i, jl, ir]) / (g.xi[jr] - g.xi[jl])
end

# Centred derivative of f in p at cell (i, j), one-sided at the p edges.
Base.@propagate_inbounds function _dp_cell(f, g::PXiGrid, i, j, ir)
    n1 = np(g)
    il = max(i - 1, 1)
    ir_ = min(i + 1, n1)
    il == ir_ && return zero(eltype(f))
    return (f[ir_, j, ir] - f[il, j, ir]) / (g.p[ir_] - g.p[il])
end

# Flux Γᵖ through p face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _pflux(op::AdvectionDiffusion, f, k, j, ir)
    g = op.grid
    m = g.mom
    n1 = np(m)
    V = g.Vp_fp[k, j, ir]
    T = eltype(f)
    (k == 1 || V == 0) && return zero(T)
    A = op.Ap[k, j, ir]
    if k == n1 + 1
        # outer p boundary: advective outflow only
        return A > 0 ? V * A * f[n1, j, ir] : zero(T)
    end
    fl = f[k-1, j, ir]
    fr = f[k, j, ir]
    wl = (m.p[k] - m.p_f[k]) / (m.p[k] - m.p[k-1])
    F = A * _face_value(op.scheme, A, fl, fr, wl)
    F -= op.Dpp[k, j, ir] * (fr - fl) / (m.p[k] - m.p[k-1])
    D = op.Dpxi[k, j, ir]
    if D != 0
        F -= D * (_dxi_cell(f, m, k - 1, j, ir) + _dxi_cell(f, m, k, j, ir)) / 2
    end
    return V * F
end

# Flux Γᵠ through ξ face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _xiflux(op::AdvectionDiffusion, f, i, k, ir)
    g = op.grid
    m = g.mom
    T = eltype(f)
    (k == 1 || k == nxi(m) + 1) && return zero(T)
    V = g.Vp_fxi[i, k, ir]
    V == 0 && return zero(T)
    fl = f[i, k-1, ir]
    fr = f[i, k, ir]
    A = op.Axi[i, k, ir]
    wl = (m.xi[k] - m.xi_f[k]) / (m.xi[k] - m.xi[k-1])
    F = A * _face_value(op.scheme, A, fl, fr, wl)
    F -= op.Dxixi[i, k, ir] * (fr - fl) / (m.xi[k] - m.xi[k-1])
    D = op.Dxip[i, k, ir]
    if D != 0
        F -= D * (_dp_cell(f, m, i, k - 1, ir) + _dp_cell(f, m, i, k, ir)) / 2
    end
    return V * F
end

# Flux Γʳ through r face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _rflux(op::AdvectionDiffusion, f, i, j, k)
    g = op.grid
    rg = g.rad
    n3 = nr(rg)
    T = eltype(f)
    V = g.Vp_fr[i, j, k]
    (k == 1 || V == 0) && return zero(T)
    A = op.Ar[i, j, k]
    D = op.Drr[i, j, k]
    if k == n3 + 1
        op.rbc === :zero_flux && return zero(T)
        # absorbing wall, f = 0 on the face
        fl = f[i, j, n3]
        F = (A > 0 ? A * fl : zero(T)) + D * fl / (rg.r_f[k] - rg.r[n3])
        return V * F
    end
    fl = f[i, j, k-1]
    fr = f[i, j, k]
    wl = (rg.r[k] - rg.r_f[k]) / (rg.r[k] - rg.r[k-1])
    F = A * _face_value(op.scheme, A, fl, fr, wl) - D * (fr - fl) / (rg.r[k] - rg.r[k-1])
    return V * F
end

"""
    apply!(df, op, f; α=1)

Add `α` times the operator applied to `f` to `df` (in place, allocation free).
"""
function apply!(df::AbstractArray{<:Any,3}, op::AdvectionDiffusion, f::AbstractArray{<:Any,3};
                α=true)
    g = op.grid
    n1, n2, n3 = size(g)
    size(f) == size(df) == (n1, n2, n3) || throw(DimensionMismatch("f and df must match grid"))
    @inbounds for ir in 1:n3
        apply_radius!(df, op, f, ir, α)
    end
    return df
end

"""
    apply_radius!(df, op, f, ir, α)

Operator contribution to all momentum cells at radius index `ir`. Writes only to
`df[:, :, ir]`, so different radii can be processed concurrently.
"""
@inline function apply_radius!(df, op::AdvectionDiffusion, f, ir, α)
    g = op.grid
    m = g.mom
    rg = g.rad
    n1, n2, _ = size(g)
    @inbounds for j in 1:n2, i in 1:n1
        V = g.Vp[i, j, ir]
        V == 0 && continue
        div = (_pflux(op, f, i + 1, j, ir) - _pflux(op, f, i, j, ir)) / m.dp[i] +
              (_xiflux(op, f, i, j + 1, ir) - _xiflux(op, f, i, j, ir)) / m.dxi[j] +
              (_rflux(op, f, i, j, ir + 1) - _rflux(op, f, i, j, ir)) / rg.dr[ir]
        df[i, j, ir] -= α * div / V
    end
    return df
end
