"""
    AdvectionDiffusion{T}

Flux-form advection-diffusion operator on a `KineticGrid`,

    ∂f/∂t = -(1/𝒱') [∂_p(𝒱' Γᵖ) + ∂_ξ(𝒱' Γᵠ) + ∂_r(𝒱' Γʳ)],
    Γᵖ = Aᵖ f - Dᵖᵖ ∂_p f - Dᵖᵠ ∂_ξ f,
    Γᵠ = Aᵠ f - Dᵠᵠ ∂_ξ f - Dᵠᵖ ∂_p f,
    Γʳ = Aʳ f - Dʳʳ ∂_r f,

with coefficients given on the faces of each direction (`advection_p` is Aᵖ,
`diffusion_p_xi` is Dᵖᵠ, and so on). Particle conservation is exact up to the fluxes
through the outer boundaries.

Boundary conditions:
- `p = 0`: flux `(𝒱'/p²)|₀ · (p²Aᵖ)|₀ · f₁`, with the limit `p²Aᵖ` given in
  `op.advection_p_times_p2_at_zero[j, ir]` (nonzero for collisional friction, which
  removes particles through `p = 0` as in DREAM).
- `r = 0`: no flux, because 𝒱' vanishes there.
- `p = p_min > 0` (`p_lower_boundary`): `:zero_flux`, or `:prescribed` with the flux `Γᵖ`
  through the face given in `op.p_flux_at_lower_boundary[j, ir]` (positive into the
  domain, independent of `f`). This is the coupling point for a runaway grid fed from a
  hot-tail grid below it.
- `ξ = ±1`: no flux.
- `p = p_max` (`p_upper_boundary`): `:outflow`, advective outflow only (`Aᵖ > 0`); or
  `:f_zero`, outflow into a ghost cell with `f = 0` one cell spacing beyond the last
  centre, adding diffusive loss `Dᵖᵖ f_N / (p_N - p_{N-1})` (DREAM `BC_F_0`).
- `r = a` (`r_boundary`): `:zero_flux`, or `:absorbing` (`f = 0` on the wall face).

Advective face values use first-order `:upwind` or second-order `:central` interpolation.
"""
struct AdvectionDiffusion{T<:AbstractFloat}
    grid::KineticGrid{T}
    advection_p::Array{T,3}       # on p faces (p cells + 1, ξ cells, radii)
    diffusion_p_p::Array{T,3}
    diffusion_p_xi::Array{T,3}
    advection_xi::Array{T,3}      # on ξ faces (p cells, ξ cells + 1, radii)
    diffusion_xi_xi::Array{T,3}
    diffusion_xi_p::Array{T,3}
    advection_r::Array{T,3}       # on r faces (p cells, ξ cells, radii + 1)
    diffusion_r_r::Array{T,3}
    p_flux_at_lower_boundary::Matrix{T}      # Γᵖ at p_min (ξ cells, radii)
    advection_p_times_p2_at_zero::Matrix{T}  # lim p²Aᵖ at p = 0 (ξ cells, radii)
    scheme::Symbol
    p_lower_boundary::Symbol
    p_upper_boundary::Symbol
    r_boundary::Symbol
end

"""
    AdvectionDiffusion(grid; scheme=:upwind, p_lower_boundary=:zero_flux,
                       p_upper_boundary=:outflow, r_boundary=:zero_flux)

Operator with all coefficients zero; set them in place, e.g. `op.diffusion_p_p .= 1`.
"""
function AdvectionDiffusion(grid::KineticGrid{T}; scheme::Symbol=:upwind,
                            p_lower_boundary::Symbol=:zero_flux,
                            p_upper_boundary::Symbol=:outflow,
                            r_boundary::Symbol=:zero_flux) where {T}
    scheme in (:upwind, :central) || throw(ArgumentError("unknown scheme $scheme"))
    p_lower_boundary in (:zero_flux, :prescribed) ||
        throw(ArgumentError("unknown lower p boundary condition $p_lower_boundary"))
    p_upper_boundary in (:outflow, :f_zero) ||
        throw(ArgumentError("unknown upper p boundary condition $p_upper_boundary"))
    if p_lower_boundary === :prescribed && grid.momentum_grid.p_faces[1] == 0
        throw(ArgumentError("a prescribed p flux needs p_min > 0"))
    end
    r_boundary in (:zero_flux, :absorbing) ||
        throw(ArgumentError("unknown radial boundary condition $r_boundary"))
    np, nxi, nr = size(grid)
    on_p_faces() = zeros(T, np + 1, nxi, nr)
    on_xi_faces() = zeros(T, np, nxi + 1, nr)
    on_r_faces() = zeros(T, np, nxi, nr + 1)
    return AdvectionDiffusion{T}(grid, on_p_faces(), on_p_faces(), on_p_faces(),
                                 on_xi_faces(), on_xi_faces(), on_xi_faces(),
                                 on_r_faces(), on_r_faces(), zeros(T, nxi, nr), zeros(T, nxi, nr),
                                 scheme, p_lower_boundary, p_upper_boundary, r_boundary)
end

@inline function _face_value(scheme, advection, f_left, f_right, weight_left)
    if scheme === :upwind
        return advection >= 0 ? f_left : f_right
    else
        return weight_left * f_left + (1 - weight_left) * f_right
    end
end

# Centred derivative of f in ξ at cell (i, j), one-sided at the ξ edges.
Base.@propagate_inbounds function _xi_derivative(f, grid::MomentumGrid, i, j, ir)
    j_lower = max(j - 1, 1)
    j_upper = min(j + 1, pitch_cell_count(grid))
    j_lower == j_upper && return zero(eltype(f))
    return (f[i, j_upper, ir] - f[i, j_lower, ir]) / (grid.xi_centers[j_upper] - grid.xi_centers[j_lower])
end

# Centred derivative of f in p at cell (i, j), one-sided at the p edges.
Base.@propagate_inbounds function _p_derivative(f, grid::MomentumGrid, i, j, ir)
    i_lower = max(i - 1, 1)
    i_upper = min(i + 1, momentum_cell_count(grid))
    i_lower == i_upper && return zero(eltype(f))
    return (f[i_upper, j, ir] - f[i_lower, j, ir]) / (grid.p_centers[i_upper] - grid.p_centers[i_lower])
end

# Flux Γᵖ through p face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _p_flux(op::AdvectionDiffusion, f, k, j, ir)
    g = op.grid
    m = g.momentum_grid
    np = momentum_cell_count(m)
    V = g.phase_space_jacobian_p_faces[k, j, ir]
    T = eltype(f)
    if k == 1
        if m.p_faces[1] == 0
            return g.phase_space_jacobian_over_p2_at_zero[j, ir] *
                   op.advection_p_times_p2_at_zero[j, ir] * f[1, j, ir]
        end
        return op.p_lower_boundary === :prescribed ? V * op.p_flux_at_lower_boundary[j, ir] : zero(T)
    end
    V == 0 && return zero(T)
    A = op.advection_p[k, j, ir]
    if k == np + 1
        f_last = f[np, j, ir]
        F = A > 0 ? A * f_last : zero(T)
        if op.p_upper_boundary === :f_zero && np > 1
            F += op.diffusion_p_p[k, j, ir] * f_last / (m.p_centers[np] - m.p_centers[np-1])
        end
        return V * F
    end
    f_left = f[k-1, j, ir]
    f_right = f[k, j, ir]
    weight_left = (m.p_centers[k] - m.p_faces[k]) / (m.p_centers[k] - m.p_centers[k-1])
    F = A * _face_value(op.scheme, A, f_left, f_right, weight_left)
    F -= op.diffusion_p_p[k, j, ir] * (f_right - f_left) / (m.p_centers[k] - m.p_centers[k-1])
    D = op.diffusion_p_xi[k, j, ir]
    if D != 0
        F -= D * (_xi_derivative(f, m, k - 1, j, ir) + _xi_derivative(f, m, k, j, ir)) / 2
    end
    return V * F
end

# Flux Γᵠ through ξ face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _xi_flux(op::AdvectionDiffusion, f, i, k, ir)
    g = op.grid
    m = g.momentum_grid
    T = eltype(f)
    (k == 1 || k == pitch_cell_count(m) + 1) && return zero(T)
    V = g.phase_space_jacobian_xi_faces[i, k, ir]
    V == 0 && return zero(T)
    f_left = f[i, k-1, ir]
    f_right = f[i, k, ir]
    A = op.advection_xi[i, k, ir]
    weight_left = (m.xi_centers[k] - m.xi_faces[k]) / (m.xi_centers[k] - m.xi_centers[k-1])
    F = A * _face_value(op.scheme, A, f_left, f_right, weight_left)
    F -= op.diffusion_xi_xi[i, k, ir] * (f_right - f_left) / (m.xi_centers[k] - m.xi_centers[k-1])
    D = op.diffusion_xi_p[i, k, ir]
    if D != 0
        F -= D * (_p_derivative(f, m, i, k - 1, ir) + _p_derivative(f, m, i, k, ir)) / 2
    end
    return V * F
end

# Flux Γʳ through r face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _r_flux(op::AdvectionDiffusion, f, i, j, k)
    g = op.grid
    rg = g.radial_grid
    nr = radial_cell_count(rg)
    T = eltype(f)
    V = g.phase_space_jacobian_r_faces[i, j, k]
    (k == 1 || V == 0) && return zero(T)
    A = op.advection_r[i, j, k]
    D = op.diffusion_r_r[i, j, k]
    if k == nr + 1
        op.r_boundary === :zero_flux && return zero(T)
        # absorbing wall, f = 0 on the face
        f_left = f[i, j, nr]
        F = (A > 0 ? A * f_left : zero(T)) + D * f_left / (rg.r_faces[k] - rg.r_centers[nr])
        return V * F
    end
    f_left = f[i, j, k-1]
    f_right = f[i, j, k]
    weight_left = (rg.r_centers[k] - rg.r_faces[k]) / (rg.r_centers[k] - rg.r_centers[k-1])
    F = A * _face_value(op.scheme, A, f_left, f_right, weight_left) -
        D * (f_right - f_left) / (rg.r_centers[k] - rg.r_centers[k-1])
    return V * F
end

"""
    apply!(df, op, f; scale=1)

Add `scale` times the operator applied to `f` to `df` (in place, allocation free). The
operator is linear in `f` except for a prescribed lower-p flux, which adds a constant
source.
"""
function apply!(df::AbstractArray{<:Any,3}, op::AdvectionDiffusion, f::AbstractArray{<:Any,3};
                scale=true)
    size(f) == size(df) == size(op.grid) || throw(DimensionMismatch("f and df must match grid"))
    @inbounds for ir in 1:radial_cell_count(op.grid)
        apply_radius!(df, op, f, ir, scale)
    end
    return df
end

"""
    apply_radius!(df, op, f, ir, scale)

Operator contribution to all momentum cells at radius index `ir`. Writes only to
`df[:, :, ir]`, so different radii can be processed concurrently.
"""
@inline function apply_radius!(df, op::AdvectionDiffusion, f, ir, scale)
    g = op.grid
    m = g.momentum_grid
    rg = g.radial_grid
    np, nxi, _ = size(g)
    @inbounds for j in 1:nxi, i in 1:np
        V = g.phase_space_jacobian[i, j, ir]
        V == 0 && continue
        divergence = (_p_flux(op, f, i + 1, j, ir) - _p_flux(op, f, i, j, ir)) / m.p_widths[i] +
                     (_xi_flux(op, f, i, j + 1, ir) - _xi_flux(op, f, i, j, ir)) / m.xi_widths[j] +
                     (_r_flux(op, f, i, j, ir + 1) - _r_flux(op, f, i, j, ir)) / rg.r_widths[ir]
        df[i, j, ir] -= scale * divergence / V
    end
    return df
end
