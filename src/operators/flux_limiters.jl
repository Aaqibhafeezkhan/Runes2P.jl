# Four-point advective face values: DREAM's higher-order interpolation
# schemes and flux limiters, ported from DREAM `fvm/Equation/AdvectionInterpolationCoefficient.cpp`
# (see THIRD_PARTY_NOTICES.md), plus van Albada as a Runes2P option. The face value between
# two cells uses the upstream cell U1, the cell U2 one further upstream and the downstream
# cell D1; outside the grid, cell centres are mirrored about the boundary face and the
# values are mirrored or zero (DREAM's AD_BC_MIRRORED and AD_BC_DIRICHLET).

"""
Schemes whose advective face value uses four cells: DREAM's `:quick` and
`:upwind_second_order` (linear in f) and the flux limiters `:tcdf`, `:muscl`, `:smart`,
`:ospre` (DREAM) and `:van_albada` (Runes2P), which are nonlinear in f.
"""
const FOUR_POINT_SCHEMES = (:quick, :upwind_second_order, :tcdf, :muscl, :smart, :ospre, :van_albada)
const NONLINEAR_SCHEMES = (:tcdf, :muscl, :smart, :ospre, :van_albada)

is_four_point_scheme(scheme::Symbol) = scheme in FOUR_POINT_SCHEMES
is_nonlinear_scheme(scheme::Symbol) = scheme in NONLINEAR_SCHEMES
stencil_half_width(scheme::Symbol) = is_four_point_scheme(scheme) ? 2 : 1

# DREAM's slope ratio r = y'(face)/y'(upstream), with y' the differences between the
# downstream and upstream cells and between the two upstream cells; ±1e5 where the
# upstream difference vanishes or the ratio exceeds it (DREAM GetFluxLimiterR)
# (plain values; returns r and whether it was clamped)
@inline function _slope_ratio(slope_down, slope_up)
    slope_up == 0 && return 1e5 * ((slope_down > 0) - (slope_down < 0)), true
    r = slope_down / slope_up
    abs(r) > 1e5 && return 1e5 * ((r > 0) - (r < 0)), true
    return r, false
end

# generalized piecewise-linear κ-scheme of Waterson and Deconinck (DREAM SetGPLKScheme):
# ψ = a + b r in the region of r, returned as (a, b)
@inline function _gplk(r, kappa, M)
    a0, b0, b1 = (1 - kappa) / 2, (1 + kappa) / 2, 2.0
    r <= 0 && return (0.0, 0.0)
    (b1 * r < M && b1 * r < a0 + b0 * r) && return (0.0, b1)
    a0 + b0 * r <= M && return (a0, b0)
    return (M, 0.0)
end

# TCDF (Zhang et al., J. Comput. Phys. 302, 114 (2015), C¹ and piecewise), OSPRE and
# van Albada as ψ(r) with its derivative ψ'(r)
@inline function _limiter(scheme, r)
    if scheme === :tcdf
        r < 0 && return r * (1 + r) / (1 + r * r), (1 + 2r - r * r) / (1 + r * r)^2
        r < 0.5 && return r * r * r - 2 * r * r + 2 * r, 3 * r * r - 4 * r + 2
        r < 2 && return 0.25 + 0.75 * r, 0.75
        return (2 * r * r - 2 * r - 2.25) / (r * r - r - 1), 0.25 * (2r - 1) / (r * r - r - 1)^2
    elseif scheme === :ospre
        return 1.5 * r * (r + 1) / (r * r + r + 1), 1.5 * (1 + 2r) / (r * r + r + 1)^2
    elseif scheme === :van_albada
        r > 0 || return 0.0, 0.0
        return (r * r + r) / (r * r + 1), (1 + 2r - r * r) / (r * r + 1)^2
    end
    throw(ArgumentError("not a limiter: $scheme"))
end

"""
    _four_point_weights(scheme, x_face, x_d, x_u, x_uu, v_d, v_u, v_uu)

Interpolation weights `(w_d, w_u, w_uu)` of the face value at `x_face` on the downstream
cell (`x_d`), the upstream cell (`x_u`) and the next upstream cell (`x_uu`), in DREAM's
form for nonuniform grids (DREAM's `deltas`). For the limiters they depend on the cell
values `v_d, v_u, v_uu` (plain numbers) through the slope ratio r; also returned are the
derivatives `(c_d, c_u, c_uu)` of the face value with respect to the three values (ψ' of
the limiter; zero where r is clamped). For the linear schemes both triples are equal.
"""
@inline function _four_point_weights(scheme, x_face, x_d, x_u, x_uu, v_d, v_u, v_uu)
    if scheme === :quick || scheme === :upwind_second_order
        # DREAM SetSecondOrderCoefficient: quadratic through the three cells (QUICK), or
        # the second-order upwind weights
        a, b, c = x_u - x_face, x_uu - x_face, x_d - x_face
        alpha = scheme === :quick ? 0.0 : -b / (8c)
        w_u = c * (b + 8alpha * a) / ((a - b) * (a - c))
        w_uu = -(1 + 8alpha) * a * c / ((a - b) * (b - c))
        w_d = a * (b + 8alpha * c) / ((a - c) * (b - c))
        return (w_d, w_u, w_uu), (w_d, w_u, w_uu)
    end
    e_d = 1 / (x_d - x_u)       # ∂(downstream slope)/∂v_d
    e_u = 1 / (x_u - x_uu)      # ∂(upstream slope)/∂v_u
    slope_down = (v_d - v_u) * e_d
    slope_up = (v_u - v_uu) * e_u
    r, clamped = _slope_ratio(slope_down, slope_up)
    dx = x_face - x_u
    if scheme === :muscl || scheme === :smart
        # ψ = a + b r written with the slopes, exact also where the upstream slope vanishes
        # (DREAM SetLinearFluxLimitedCoefficient); the region of r is locally constant
        a, b = scheme === :muscl ? _gplk(r, 0.0, 2.0) : _gplk(r, 0.5, 4.0)
        weights = (dx * b * e_d, 1 + dx * (a * e_u - b * e_d), -dx * a * e_u)
        return weights, weights
    end
    psi, dpsi = _limiter(scheme, r)
    clamped && (dpsi = 0.0)
    # face = v_u + ψ(r) dx s_u (DREAM SetFluxLimitedCoefficient); its derivative
    # ∂(ψ s_u) = ψ'(r) (∂s_d - r ∂s_u) + ψ ∂s_u
    weights = (0.0, 1 + psi * dx * e_u, -psi * dx * e_u)
    derivatives = (dx * dpsi * e_d, 1 + dx * (dpsi * (-e_d - r * e_u) + psi * e_u), dx * (dpsi * r * e_u - psi * e_u))
    return weights, derivatives
end

"""
    _four_point_face_value(scheme, x_face, x_d, x_u, x_uu, y_d, y_u, y_uu)

Face value from the three cells. For the limiters the weights are evaluated on the values
alone and the derivative is carried as `Σ c_k (y_k - value(y_k))` with the exact
coefficients of [`_four_point_weights`](@ref): differentiating the ratio of slopes
directly would overflow in the far tail of a Maxwellian, where the slopes are subnormal,
although the derivative is bounded (DREAM writes its full Jacobian the same way).
"""
@inline function _four_point_face_value(scheme, x_face, x_d, x_u, x_uu, y_d, y_u, y_uu)
    v_d, v_u, v_uu = _value(y_d), _value(y_u), _value(y_uu)
    (w_d, w_u, w_uu), (c_d, c_u, c_uu) = _four_point_weights(scheme, x_face, x_d, x_u, x_uu, v_d, v_u, v_uu)
    is_nonlinear_scheme(scheme) || return w_d * y_d + w_u * y_u + w_uu * y_uu
    face = w_d * v_d + w_u * v_u + w_uu * v_uu
    return face + c_d * (y_d - v_d) + c_u * (y_u - v_u) + c_uu * (y_uu - v_uu)
end

# face value with the weights of the limiter evaluated on another state (s_*), applied
# linearly to y_*: the limiter frozen at that state (DREAM's linearly implicit solver
# takes the weights from the previous time level)
@inline function _frozen_face_value(scheme, x_face, x_d, x_u, x_uu, s_d, s_u, s_uu, y_d, y_u, y_uu)
    (w_d, w_u, w_uu), _ = _four_point_weights(scheme, x_face, x_d, x_u, x_uu, _value(s_d), _value(s_u), _value(s_uu))
    return w_d * y_d + w_u * y_u + w_uu * y_uu
end

# value of a number without its derivative parts (a method for ForwardDiff's dual numbers
# is defined with the AD code)
_value(x::Real) = x

# centre and value of cell i along one direction, with ghost cells mirrored about the
# boundary faces; `lower_mirrored`/`upper_mirrored` false gives zero ghost values
@inline function _ghost_center(centers, faces, i)
    n = length(centers)
    i < 1 && return 2 * faces[1] - centers[1-i]
    i > n && return 2 * faces[end] - centers[2n+1-i]
    return centers[i]
end
@inline _ghost_index(i, n) = i < 1 ? 1 - i : (i > n ? 2n + 1 - i : i)
