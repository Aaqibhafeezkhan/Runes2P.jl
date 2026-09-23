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
  centre, adding diffusive loss `Dᵖᵖ f_N / (p_N - p_{N-1})` (DREAM `BC_F_0`); or
  `:constant_flux`, `𝒱'Γᵖ` at the boundary face held equal to `𝒱'Γᵖ` at the last interior
  face (DREAM `BC_PHI_CONST`, its default; needs at least 2 momentum cells); or
  `:extrapolated_flux`, `𝒱'Γᵖ` at the boundary linearly extrapolated in `p` from the last
  two interior faces (DREAM `BC_DPHI_CONST`, "known to cause instabilities near `p_max`";
  needs at least 3 momentum cells, no `:exponential_fitting` counterpart). Ported literally
  from DREAM `PXiExternalLoss.cpp`, including an apparent index inconsistency: both terms
  share the last interior face's `𝒱'` and are
  multiplied by that face's own `Aᵖ`, while the second term's interpolation (upstream
  direction and weights) and diffusion coefficient are the second-to-last face's own.
- `r = a` (`r_boundary`): `:zero_flux`, or `:absorbing` (`f = 0` on the wall face).

Schemes (`scheme`), for the advection and diffusion along each direction between two cell
centres; the cross terms `Dᵖᵠ`, `Dᵠᵖ` are always central:

- `:central`: face value interpolated linearly between the centres, second order. Loses
  positivity where the cell Péclet number `|A|Δ/D` exceeds 2 (the matrix gets negative
  off-diagonal couplings).
- `:upwind`: face value of the upstream cell, first order, positivity preserving.
- `:exponential_fitting`: the Scharfetter-Gummel (Chang-Cooper) flux
  `Γ = (D/Δ) [B(-w) f_L - B(w) f_R]`, `w = AΔ/D`, `B(x) = x/(eˣ - 1)`, with `Δ` the distance
  between the centres. Exact for steady solutions of the one-dimensional face problem
  with constant `A/D` between the centres (so the collisional Maxwellian is steady up to
  the variation of `A/D` over a cell), positivity preserving for every Péclet number
  (the off-diagonal couplings are `(D/Δ) B(±w) ≥ 0`), second order where the Péclet
  number is small, where it reduces to the central flux with midpoint weights, and
  upwind where it is large. Not in DREAM (Runes2P departure); used in
  CQL3D (Chang-Cooper). Falls back to upwind advection with central diffusion where
  `D ≤ 0`. At the `:f_zero` boundary the ghost cell enters the same flux.
- Four-point schemes (in p and ξ, diffusion central): DREAM's `:quick`
  and `:upwind_second_order` (linear in f) and flux limiters `:tcdf`, `:muscl`, `:smart`,
  `:ospre`, plus `:van_albada` (Runes2P), which are nonlinear in f. The face value uses two
  upstream cells and one downstream cell (`src/operators/flux_limiters.jl`), with ghost
  cells as DREAM sets them for the hot-tail distribution: mirrored at p = 0 (zero at a
  lower boundary p_min > 0), zero beyond p_max, mirrored at ξ = ±1. The fluxes through
  the boundary faces keep the rules listed above (upwind at p_max, as DREAM's `BC_F_0`).
  With fewer than three cells in a direction the face value is central, as in DREAM.

The scheme can differ per direction (`p_scheme`, `xi_scheme`, `r_scheme`; `scheme` sets
all three). Four-point schemes are not available in r, which has no radial transport
yet.
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
    p_scheme::Symbol
    xi_scheme::Symbol
    r_scheme::Symbol
    p_lower_boundary::Symbol
    p_upper_boundary::Symbol
    r_boundary::Symbol
    # shared-grid data read in the flux loops (the operator requires a shared momentum grid)
    momentum::MomentumGrid{T}
    jacobian_centers::Array{T,3}
    jacobian_p_faces::Array{T,3}
    jacobian_xi_faces::Array{T,3}
    jacobian_r_faces::Array{T,3}
    jacobian_over_p2_at_zero::Matrix{T}
end

"""
    AdvectionDiffusion(grid; scheme=:upwind, p_scheme=scheme, xi_scheme=scheme,
                       r_scheme=(scheme is four-point ? :central : scheme),
                       p_lower_boundary=:zero_flux, p_upper_boundary=:outflow,
                       r_boundary=:zero_flux)

Operator with all coefficients zero; set them in place, e.g. `op.diffusion_p_p .= 1`.
"""
function AdvectionDiffusion(grid::KineticGrid{T}; scheme::Symbol=:upwind, p_scheme::Symbol=scheme,
                            xi_scheme::Symbol=scheme,
                            r_scheme::Symbol=is_four_point_scheme(scheme) ? :central : scheme,
                            p_lower_boundary::Symbol=:zero_flux,
                            p_upper_boundary::Symbol=:outflow,
                            r_boundary::Symbol=:zero_flux) where {T}
    for s in (p_scheme, xi_scheme)
        s in (:upwind, :central, :exponential_fitting) || is_four_point_scheme(s) ||
            throw(ArgumentError("unknown scheme $s"))
    end
    r_scheme in (:upwind, :central, :exponential_fitting) ||
        throw(ArgumentError("scheme $r_scheme is not available in r (four-point schemes need radial transport)"))
    p_lower_boundary in (:zero_flux, :prescribed) ||
        throw(ArgumentError("unknown lower p boundary condition $p_lower_boundary"))
    p_upper_boundary in (:outflow, :f_zero, :constant_flux, :extrapolated_flux) ||
        throw(ArgumentError("unknown upper p boundary condition $p_upper_boundary"))
    np, nxi, nr = size(grid)
    if p_upper_boundary === :constant_flux && np < 2
        throw(ArgumentError("constant_flux p_max boundary needs at least 2 momentum cells"))
    end
    if p_upper_boundary === :extrapolated_flux
        np < 3 && throw(ArgumentError("extrapolated_flux p_max boundary needs at least 3 momentum cells"))
        p_scheme === :exponential_fitting &&
            throw(ArgumentError("extrapolated_flux has no exponential_fitting counterpart in DREAM " *
                                "(BC_DPHI_CONST); its flux does not split into Aᵖ times a face value"))
    end
    has_shared_momentum_grid(grid) ||
        throw(ArgumentError("AdvectionDiffusion needs one momentum grid for all radii (radial " *
                            "fluxes between different grids need a remap between them)"))
    if p_lower_boundary === :prescribed && momentum_grid(grid).p_faces[1] == 0
        throw(ArgumentError("a prescribed p flux needs p_min > 0"))
    end
    r_boundary in (:zero_flux, :absorbing) ||
        throw(ArgumentError("unknown radial boundary condition $r_boundary"))
    on_p_faces() = zeros(T, np + 1, nxi, nr)
    on_xi_faces() = zeros(T, np, nxi + 1, nr)
    on_r_faces() = zeros(T, np, nxi, nr + 1)
    radii = 1:nr
    return AdvectionDiffusion{T}(grid, on_p_faces(), on_p_faces(), on_p_faces(),
                                 on_xi_faces(), on_xi_faces(), on_xi_faces(),
                                 on_r_faces(), on_r_faces(), zeros(T, nxi, nr), zeros(T, nxi, nr),
                                 p_scheme, xi_scheme, r_scheme, p_lower_boundary, p_upper_boundary, r_boundary,
                                 momentum_grid(grid),
                                 stack(phase_space_jacobian(grid, ir) for ir in radii),
                                 stack(phase_space_jacobian_p_faces(grid, ir) for ir in radii),
                                 stack(phase_space_jacobian_xi_faces(grid, ir) for ir in radii),
                                 copy(phase_space_jacobian_r_faces(grid)),
                                 stack(phase_space_jacobian_over_p2_at_zero(grid, ir) for ir in radii))
end

@inline function _face_value(scheme, advection, f_left, f_right, weight_left)
    if scheme === :upwind
        return advection >= 0 ? f_left : f_right
    else
        return weight_left * f_left + (1 - weight_left) * f_right
    end
end

# Bernoulli function B(x) = x/(eˣ - 1), with its series near 0 (B(0) = 1). For x > 0 in
# the form x e⁻ˣ/(1 - e⁻ˣ), whose derivative stays finite under ForwardDiff where eˣ
# overflows (a Péclet number above ~709, reachable once the coefficients depend on
# unknowns such as the temperature)
@inline function _bernoulli(x)
    abs(x) < 1e-3 && return 1 - x / 2 + x^2 / 12 - x^4 / 720
    x > 0 && return x * exp(-x) / -expm1(-x)
    return x / expm1(x)
end

# Flux A f - D ∂f between two cell centres a distance `distance` apart, for the scheme;
# `weight_left` is the linear-interpolation weight of the left centre at the face
@inline function _face_flux(scheme, A, D, f_left, f_right, distance, weight_left)
    if scheme === :exponential_fitting
        if D > 0
            w = A * distance / D
            return D / distance * (_bernoulli(-w) * f_left - _bernoulli(w) * f_right)
        end
        return A * (A >= 0 ? f_left : f_right) - D * (f_right - f_left) / distance
    end
    return A * _face_value(scheme, A, f_left, f_right, weight_left) - D * (f_right - f_left) / distance
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

# Four-point face value on p face `k` with the ghost cells DREAM uses for the hot-tail
# distribution (mirrored at p = 0, zero at a lower boundary p_min > 0 and beyond p_max);
# with `state` given, the limiter weights come from `state` (frozen limiter)
Base.@propagate_inbounds function _p_four_point_face(op::AdvectionDiffusion, f, state, A, k, j, ir)
    m = op.momentum
    np = momentum_cell_count(m)
    T = eltype(f)
    # upstream U1, U2 and downstream D1 by the sign of A (DREAM: A > 0 is positive flow)
    i_u, i_uu, i_d = A > 0 ? (k - 1, k - 2, k) : (k, k + 1, k - 1)
    centers, faces = m.p_centers, m.p_faces
    mirrored = faces[1] == 0
    value(g, i) = i < 1 ? (mirrored ? g[1-i, j, ir] : zero(eltype(g))) : (i > np ? zero(eltype(g)) : g[i, j, ir])
    x_d, x_u, x_uu = _ghost_center(centers, faces, i_d), _ghost_center(centers, faces, i_u), _ghost_center(centers, faces, i_uu)
    if state === nothing || !is_nonlinear_scheme(op.p_scheme)
        return _four_point_face_value(op.p_scheme, faces[k], x_d, x_u, x_uu, value(f, i_d), value(f, i_u), value(f, i_uu))
    end
    return _frozen_face_value(op.p_scheme, faces[k], x_d, x_u, x_uu, value(state, i_d), value(state, i_u),
                              value(state, i_uu), value(f, i_d), value(f, i_u), value(f, i_uu))
end

# Four-point face value on ξ face `k`, mirrored at ξ = ±1 (DREAM AD_BC_MIRRORED)
Base.@propagate_inbounds function _xi_four_point_face(op::AdvectionDiffusion, f, state, A, i, k, ir)
    m = op.momentum
    nxi = pitch_cell_count(m)
    j_u, j_uu, j_d = A > 0 ? (k - 1, k - 2, k) : (k, k + 1, k - 1)
    centers, faces = m.xi_centers, m.xi_faces
    value(g, j) = g[i, _ghost_index(j, nxi), ir]
    x_d, x_u, x_uu = _ghost_center(centers, faces, j_d), _ghost_center(centers, faces, j_u), _ghost_center(centers, faces, j_uu)
    if state === nothing || !is_nonlinear_scheme(op.xi_scheme)
        return _four_point_face_value(op.xi_scheme, faces[k], x_d, x_u, x_uu, value(f, j_d), value(f, j_u), value(f, j_uu))
    end
    return _frozen_face_value(op.xi_scheme, faces[k], x_d, x_u, x_uu, value(state, j_d), value(state, j_u),
                              value(state, j_uu), value(f, j_d), value(f, j_u), value(f, j_uu))
end

# Γᵖ (not multiplied by 𝒱') through interior p face `k` (2 <= k <= momentum cell count),
# between cells k-1 and k.
Base.@propagate_inbounds function _p_interior_gamma(op::AdvectionDiffusion, f, k, j, ir, state)
    m = op.momentum
    np = momentum_cell_count(m)
    A = op.advection_p[k, j, ir]
    f_left = f[k-1, j, ir]
    f_right = f[k, j, ir]
    distance = m.p_centers[k] - m.p_centers[k-1]
    weight_left = (m.p_centers[k] - m.p_faces[k]) / distance
    F = if is_four_point_scheme(op.p_scheme) && np >= 3
        A * _p_four_point_face(op, f, state, A, k, j, ir) - op.diffusion_p_p[k, j, ir] * (f_right - f_left) / distance
    else
        _face_flux(op.p_scheme, A, op.diffusion_p_p[k, j, ir], f_left, f_right, distance, weight_left)
    end
    D = op.diffusion_p_xi[k, j, ir]
    if D != 0
        F -= D * (_xi_derivative(f, m, k - 1, j, ir) + _xi_derivative(f, m, k, j, ir)) / 2
    end
    return F
end

# Flux Γᵖ through p face `k` (between cells k-1 and k), multiplied by 𝒱'. `state` gives the
# limiter weights of a frozen limiter (`nothing`: the limiter at f itself).
Base.@propagate_inbounds function _p_flux(op::AdvectionDiffusion, f, k, j, ir, state=nothing)
    m = op.momentum
    np = momentum_cell_count(m)
    V = op.jacobian_p_faces[k, j, ir]
    T = eltype(f)
    if k == 1
        if m.p_faces[1] == 0
            # p²Aᵖ times the face value, which is f₁ for the two-point schemes and the
            # limiters (mirrored ghost cell) and the scheme's own value for QUICK and
            # second-order upwind (DREAM AdvectionTerm, F1PSqAtZero with the p = 0 deltas)
            face = is_four_point_scheme(op.p_scheme) && np >= 3 ?
                   _p_four_point_face(op, f, state, op.advection_p[1, j, ir], 1, j, ir) : f[1, j, ir]
            return op.jacobian_over_p2_at_zero[j, ir] * op.advection_p_times_p2_at_zero[j, ir] * face
        end
        return op.p_lower_boundary === :prescribed ? V * op.p_flux_at_lower_boundary[j, ir] : zero(T)
    end
    V == 0 && return zero(T)
    A = op.advection_p[k, j, ir]
    if k == np + 1
        f_last = f[np, j, ir]
        if op.p_upper_boundary === :constant_flux
            # 𝒱'Γᵖ continued unchanged from the last interior face (DREAM BC_PHI_CONST)
            return _p_flux(op, f, np, j, ir, state)
        elseif op.p_upper_boundary === :extrapolated_flux
            # linear extrapolation of 𝒱'Γᵖ from the last two interior faces (DREAM
            # BC_DPHI_CONST, PXiExternalLoss.cpp), ported literally including an apparent
            # index inconsistency: both advective terms are multiplied by the last
            # interior face's Aᵖ (not each term's own), while the
            # second term's interpolation (upstream direction and weights) and diffusion
            # are the second-to-last face's own. Both terms share the last interior face's
            # 𝒱'.
            A_last = op.advection_p[np, j, ir]
            A_prev = op.advection_p[np-1, j, ir]
            gamma_last = _p_interior_gamma(op, f, np, j, ir, state)
            face_value_prev = if is_four_point_scheme(op.p_scheme) && np >= 3
                _p_four_point_face(op, f, state, A_prev, np - 1, j, ir)
            else
                f_left = f[np-2, j, ir]
                f_right = f[np-1, j, ir]
                distance_prev = m.p_centers[np-1] - m.p_centers[np-2]
                weight_left = (m.p_centers[np-1] - m.p_faces[np-1]) / distance_prev
                _face_value(op.p_scheme, A_prev, f_left, f_right, weight_left)
            end
            distance_prev = m.p_centers[np-1] - m.p_centers[np-2]
            gamma_prev_dream = A_last * face_value_prev -
                                op.diffusion_p_p[np-1, j, ir] * (f[np-1, j, ir] - f[np-2, j, ir]) / distance_prev
            dd = m.p_widths[np] / m.p_widths[np-1]
            return op.jacobian_p_faces[np, j, ir] * ((1 + dd) * gamma_last - dd * gamma_prev_dream)
        elseif op.p_upper_boundary === :f_zero && np > 1
            distance = m.p_centers[np] - m.p_centers[np-1]
            if op.p_scheme === :exponential_fitting
                # the ghost cell with f = 0 at one spacing beyond the last centre
                return V * _face_flux(op.p_scheme, A, op.diffusion_p_p[k, j, ir], f_last, zero(T), distance, 0.5)
            end
            return V * ((A > 0 ? A * f_last : zero(T)) + op.diffusion_p_p[k, j, ir] * f_last / distance)
        end
        return V * (A > 0 ? A * f_last : zero(T))
    end
    return V * _p_interior_gamma(op, f, k, j, ir, state)
end

# Flux Γᵠ through ξ face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _xi_flux(op::AdvectionDiffusion, f, i, k, ir, state=nothing)
    m = op.momentum
    T = eltype(f)
    (k == 1 || k == pitch_cell_count(m) + 1) && return zero(T)
    V = op.jacobian_xi_faces[i, k, ir]
    V == 0 && return zero(T)
    f_left = f[i, k-1, ir]
    f_right = f[i, k, ir]
    A = op.advection_xi[i, k, ir]
    distance = m.xi_centers[k] - m.xi_centers[k-1]
    weight_left = (m.xi_centers[k] - m.xi_faces[k]) / distance
    nxi = pitch_cell_count(m)
    F = if is_four_point_scheme(op.xi_scheme) && nxi >= 3
        A * _xi_four_point_face(op, f, state, A, i, k, ir) - op.diffusion_xi_xi[i, k, ir] * (f_right - f_left) / distance
    else
        _face_flux(op.xi_scheme, A, op.diffusion_xi_xi[i, k, ir], f_left, f_right, distance, weight_left)
    end
    D = op.diffusion_xi_p[i, k, ir]
    if D != 0
        F -= D * (_p_derivative(f, m, i, k - 1, ir) + _p_derivative(f, m, i, k, ir)) / 2
    end
    return V * F
end

# Flux Γʳ through r face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _r_flux(op::AdvectionDiffusion, f, i, j, k)
    rg = op.grid.radial_grid
    nr = radial_cell_count(rg)
    T = eltype(f)
    V = op.jacobian_r_faces[i, j, k]
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
    distance = rg.r_centers[k] - rg.r_centers[k-1]
    weight_left = (rg.r_centers[k] - rg.r_faces[k]) / distance
    F = _face_flux(op.r_scheme, A, D, f_left, f_right, distance, weight_left)
    return V * F
end

"""
    apply!(df, op, f; scale=1, limiter_state=nothing)

Add `scale` times the operator applied to `f` to `df` (in place, allocation free). The
operator is linear in `f` except for a prescribed lower-p flux, which adds a constant
source, and for the flux limiters, whose face values depend on `f`. With
`limiter_state`, the limiter weights are evaluated at that state and applied linearly to
`f`: the limiter frozen at `limiter_state` (DREAM's linearly implicit solver takes them
from the previous time level; the frozen-limiter Jacobian).
"""
function apply!(df::AbstractArray{<:Any,3}, op::AdvectionDiffusion, f::AbstractArray{<:Any,3};
                scale=true, limiter_state=nothing)
    size(f) == size(df) == size(op.grid) || throw(DimensionMismatch("f and df must match grid"))
    limiter_state === nothing || size(limiter_state) == size(f) ||
        throw(DimensionMismatch("limiter_state must match the grid"))
    @inbounds for ir in 1:radial_cell_count(op.grid)
        apply_radius!(df, op, f, ir, scale, limiter_state)
    end
    return df
end

"""
    apply_radius!(df, op, f, ir, scale)

Operator contribution to all momentum cells at radius index `ir`. Writes only to
`df[:, :, ir]`, so different radii can be processed concurrently.
"""
@inline function apply_radius!(df, op::AdvectionDiffusion, f, ir, scale, state=nothing)
    m = op.momentum
    rg = op.grid.radial_grid
    np, nxi = momentum_cell_count(m), pitch_cell_count(m)
    @inbounds for j in 1:nxi, i in 1:np
        V = op.jacobian_centers[i, j, ir]
        V == 0 && continue
        divergence = (_p_flux(op, f, i + 1, j, ir, state) - _p_flux(op, f, i, j, ir, state)) / m.p_widths[i] +
                     (_xi_flux(op, f, i, j + 1, ir, state) - _xi_flux(op, f, i, j, ir, state)) / m.xi_widths[j] +
                     (_r_flux(op, f, i, j, ir + 1) - _r_flux(op, f, i, j, ir)) / rg.r_widths[ir]
        df[i, j, ir] -= scale * divergence / V
    end
    return df
end

"""
    momentum_flux(op::AdvectionDiffusion, f, face, ir=1)

Particle flux per unit volume [m⁻³ s⁻¹] through the momentum face `face` (between cells
`face - 1` and `face`) at radius index `ir`, `(1/V') Σ_j Δξ_j 𝒱'ᵖ Γᵖ`, positive towards
larger momentum. This is the rate at which particles cross the surface `p = p_faces[face]`,
counted in the same normalization as [`density`](@ref), so a runaway rate measured through
a surface above the critical momentum is directly comparable with
[`connor_hastie_rate`](@ref). Face 1 is the flux through `p = p_min` (the `p = 0` limit or
the prescribed boundary flux) and face `momentum_cell_count + 1` the flux out of the grid.
"""
function momentum_flux(op::AdvectionDiffusion, f::AbstractArray{<:Any,3}, face::Integer,
                       ir::Integer=1)
    m = momentum_grid(op.grid, ir)
    1 <= face <= momentum_cell_count(m) + 1 ||
        throw(BoundsError("momentum face $face outside the grid"))
    total = zero(eltype(f))
    @inbounds for j in 1:pitch_cell_count(m)
        total += m.xi_widths[j] * _p_flux(op, f, face, j, ir)
    end
    return total / op.grid.radial_grid.spatial_jacobian[ir]
end
