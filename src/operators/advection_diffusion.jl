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

The pitch grid may differ between radii, with the same cell counts and
the same p faces at all radii. The pitch cells of a radial face are then the overlap cells
of the pitch grids of its two radii ([`radial_face_overlap`](@ref)), and the radial
coefficients are given per overlap cell: `advection_r[i, s, k]` for momentum cell `i` and
overlap cell `s` of face `k`, at the pitch `radial_face_overlap(grid, k).xi_centers[s]`.
Their second dimension is the largest count over the faces, and entries beyond the count
of a face are not used. The radial flux is computed once per overlap cell, from the values
of the two cells it lies in, and given to both with its share of each cell, so no particle
is lost or created between different pitch grids. Both values count as constant over their
cells. Where the pitch grids of two radii differ, the difference of the two values
therefore contains a pitch difference of the order of the cell width, which acts as a
diffusion in the pitch of the order `Dʳʳ (Δξ/Δr)²` times the fraction of the cells that do
not match. With one pitch grid for all radii the overlap cells are the pitch cells and
nothing changes.

Boundary conditions:
- `p = 0`: flux `(𝒱'/p²)|₀ · (p²Aᵖ)|₀ · f₁`, with the limit `p²Aᵖ` given in
  `op.advection_p_times_p2_at_zero[j, ir]` (nonzero for collisional friction, which
  removes particles through `p = 0` as in DREAM). With `:quick` and
  `:upwind_second_order` the scheme's own face value at `p = 0` replaces `f₁`, as in
  DREAM's `AdvectionTerm`; the limiters give `f₁` there (mirrored ghost cells).
- `r = 0`: no flux, because 𝒱' vanishes there.
- `p = p_min > 0` (`p_lower_boundary`): `:zero_flux`, or `:prescribed` with the flux `Γᵖ`
  through the face given in `op.p_flux_at_lower_boundary[j, ir]` (positive into the
  domain, independent of `f`), or `:interface`: the face is shared with the grid of
  another kinetic unknown below (a runaway grid above a hot-tail grid), the operator adds
  no flux through it, and a [`MomentumInterfaceTerm`](@ref) in the equation system carries
  the flux between the two grids.
- `ξ = ±1`: no flux.
- `p = p_max` (`p_upper_boundary`): `:outflow`, advective outflow only (`Aᵖ > 0`); or
  `:f_zero`, outflow into a ghost cell with `f = 0` one cell spacing beyond the last
  centre, adding diffusive loss `Dᵖᵖ f_N / (p_N - p_{N-1})` (DREAM `BC_F_0`; with one
  momentum cell there is no spacing and the boundary is `:outflow`, which the pitch-only
  tests on one momentum cell use); or
  `:constant_flux`, `𝒱'Γᵖ` at the boundary face held equal to `𝒱'Γᵖ` at the last interior
  face (DREAM `BC_PHI_CONST`, its default; needs at least 2 momentum cells); or
  `:extrapolated_flux`, `𝒱'Γᵖ` at the boundary linearly extrapolated in `p` from the last
  two interior faces (DREAM `BC_DPHI_CONST`, "known to cause instabilities near `p_max`";
  needs at least 3 momentum cells, no `:exponential_fitting` counterpart). Ported literally
  from DREAM `PXiExternalLoss.cpp`, including an apparent index inconsistency: both terms
  share the last interior face's `𝒱'` and are
  multiplied by that face's own `Aᵖ`, while the second term's interpolation (upstream
  direction and weights) and diffusion coefficient are the second-to-last face's own; or
  `:extrapolated_flux_consistent`, `𝒱'Γᵖ` linearly extrapolated with every face's own `𝒱'`
  and `Γᵖ` throughout, matching DREAM's stated definition of `BC_DPHI_CONST` ("dΦ/dp
  constant", `Φ = 𝒱'Γᵖ`) rather than its code, which besides the `Aᵖ` index above also
  shares one `𝒱'` between the two terms (needs at least 3 momentum
  cells). Runes2P departure: DREAM comparisons keep `:extrapolated_flux` so they stay exact to
  round-off; new Runes2P setups that want `BC_DPHI_CONST`'s extrapolation without reproducing
  DREAM's apparent bugs should use this instead.
  Or `:interface`, the counterpart of the lower boundary: the face is shared with the
  grid of a kinetic unknown above, and the [`MomentumInterfaceTerm`](@ref) carries the flux.
  The ghost cells of the four-point schemes beyond an interface hold zero, as beyond any
  other boundary in p, so the last interior face on each side sees the other grid as empty.
- `r = a` (`r_boundary`): `:zero_flux`, or `:absorbing` (`f = 0` on the edge face: the
  diffusive flux is `Dʳʳ f_N` over the distance from the last cell centre to the face), or
  `:absorbing_dream`, DREAM's `BC_F_0` of its transport terms, with the distance between
  the last two cell centres, which places `f = 0` half a cell outside the edge (1 with one
  radius, as in DREAM). On a uniform grid the edge flux of `:absorbing` is twice that of
  `:absorbing_dream`. The lowest decay rate of a uniform `Dʳʳ` in a cylinder converges at
  second order with `:absorbing` and at first order with DREAM's form: 0.997 and 0.911 of
  the exact rate at 11 radii. The advective part is the upwind
  outflow in all of them.

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
  the boundary faces keep the rules listed above (the scheme's face value at p = 0,
  upwind at p_max as DREAM's `BC_F_0`).
  With fewer than three cells in a direction the face value is central, as in DREAM.

The scheme can differ per direction (`p_scheme`, `xi_scheme`, `r_scheme`). `scheme` sets
`p_scheme` and `xi_scheme`, and `r_scheme` when it is a two-point scheme. With a four-point
`scheme` the radial scheme is `:central` unless `r_scheme` names one.

A four-point scheme in r gives the face value of the radial advection.
The radial diffusion is central. Ghost cells as in DREAM: mirrored at the axis, zero beyond
the edge of the plasma. The flux through the edge is the upwind outflow. It needs one pitch
grid at all radii, and with fewer than three radii the face value is central.
"""
struct AdvectionDiffusion{T<:AbstractFloat}
    grid::KineticGrid{T}
    advection_p::Array{T,3}       # on p faces (p cells + 1, ξ cells, radii)
    diffusion_p_p::Array{T,3}
    diffusion_p_xi::Array{T,3}
    advection_xi::Array{T,3}      # on ξ faces (p cells, ξ cells + 1, radii)
    diffusion_xi_xi::Array{T,3}
    diffusion_xi_p::Array{T,3}
    advection_r::Array{T,3}       # on r faces (p cells, face pitch cells, radii + 1)
    diffusion_r_r::Array{T,3}
    unit_field_advection_p::Array{T,3}   # Aᵖ of a field of 1 V/m, on p faces
    unit_field_advection_xi::Array{T,3}  # Aᵠ of a field of 1 V/m, on ξ faces
    p_flux_at_lower_boundary::Matrix{T}      # Γᵖ at p_min (ξ cells, radii)
    advection_p_times_p2_at_zero::Matrix{T}  # lim p²Aᵖ at p = 0 (ξ cells, radii)
    p_scheme::Symbol
    xi_scheme::Symbol
    r_scheme::Symbol
    p_lower_boundary::Symbol
    p_upper_boundary::Symbol
    r_boundary::Symbol
    # grid data read in the flux loops (the same cell counts and p faces at all radii)
    momentum_grids::Vector{MomentumGrid{T}}  # per radius
    radial_overlaps::Vector{PitchOverlap{T}} # per radial face
    jacobian_centers::Array{T,3}
    jacobian_p_faces::Array{T,3}
    jacobian_xi_faces::Array{T,3}
    jacobian_r_faces::Array{T,3}
    jacobian_over_p2_at_zero::Matrix{T}
    # what the coefficients were last built from by set_fokker_planck_coefficients!
    # (`nothing` for coefficients set by hand); read by `describe`, shared by with_scheme
    coefficient_source::Base.RefValue{Any}
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
    r_scheme in (:upwind, :central, :exponential_fitting) || is_four_point_scheme(r_scheme) ||
        throw(ArgumentError("unknown scheme $r_scheme"))
    p_lower_boundary in (:zero_flux, :prescribed, :interface) ||
        throw(ArgumentError("unknown lower p boundary condition $p_lower_boundary"))
    p_upper_boundary in (:outflow, :f_zero, :constant_flux, :extrapolated_flux,
                        :extrapolated_flux_consistent, :interface) ||
        throw(ArgumentError("unknown upper p boundary condition $p_upper_boundary"))
    (has_same_cell_counts(grid) && has_same_p_faces(grid)) ||
        throw(ArgumentError("AdvectionDiffusion needs the same cell counts and the same p faces at all " *
                            "radii (the pitch faces may differ between radii)"))
    grid.radial_overlaps === nothing &&
        throw(ArgumentError("AdvectionDiffusion needs pitch grids that cover the same interval at all radii"))
    is_four_point_scheme(r_scheme) && !all(has_identical_grids, grid.radial_overlaps) &&
        throw(ArgumentError("the four-point scheme $r_scheme in r needs one pitch grid at all radii " *
                            "(issue #72); the pitch faces of this grid differ between radii"))
    np, nxi, nr = size(grid)
    face_cells = radial_face_pitch_cell_count(grid)
    if p_upper_boundary === :constant_flux && np < 2
        throw(ArgumentError("constant_flux p_max boundary needs at least 2 momentum cells"))
    end
    if p_upper_boundary === :extrapolated_flux
        np < 3 && throw(ArgumentError("extrapolated_flux p_max boundary needs at least 3 momentum cells"))
        p_scheme === :exponential_fitting &&
            throw(ArgumentError("extrapolated_flux has no exponential_fitting counterpart in DREAM " *
                                "(BC_DPHI_CONST); its flux does not split into Aᵖ times a face value"))
    end
    if p_upper_boundary === :extrapolated_flux_consistent && np < 3
        throw(ArgumentError("extrapolated_flux_consistent p_max boundary needs at least 3 momentum cells"))
    end
    if p_lower_boundary === :prescribed && momentum_grid(grid, 1).p_faces[1] == 0
        throw(ArgumentError("a prescribed p flux needs p_min > 0"))
    end
    if p_lower_boundary === :interface && momentum_grid(grid, 1).p_faces[1] == 0
        throw(ArgumentError("an interface at the lower p boundary needs p_min > 0"))
    end
    r_boundary in (:zero_flux, :absorbing, :absorbing_dream) ||
        throw(ArgumentError("unknown radial boundary condition $r_boundary"))
    on_p_faces() = zeros(T, np + 1, nxi, nr)
    on_xi_faces() = zeros(T, np, nxi + 1, nr)
    on_r_faces() = zeros(T, np, face_cells, nr + 1)
    radii = 1:nr
    return AdvectionDiffusion{T}(grid, on_p_faces(), on_p_faces(), on_p_faces(),
                                 on_xi_faces(), on_xi_faces(), on_xi_faces(),
                                 on_r_faces(), on_r_faces(), on_p_faces(), on_xi_faces(), zeros(T, nxi, nr),
                                 zeros(T, nxi, nr),
                                 p_scheme, xi_scheme, r_scheme, p_lower_boundary, p_upper_boundary, r_boundary,
                                 [momentum_grid(grid, ir) for ir in radii],
                                 [radial_face_overlap(grid, k) for k in 1:nr+1],
                                 stack(phase_space_jacobian(grid, ir) for ir in radii),
                                 stack(phase_space_jacobian_p_faces(grid, ir) for ir in radii),
                                 stack(phase_space_jacobian_xi_faces(grid, ir) for ir in radii),
                                 copy(phase_space_jacobian_r_faces(grid)),
                                 stack(phase_space_jacobian_over_p2_at_zero(grid, ir) for ir in radii),
                                 Ref{Any}(nothing))
end

"""
    with_scheme(op::AdvectionDiffusion, scheme; p_scheme=scheme, xi_scheme=scheme)

The operator `op` with other interpolation schemes in p and ξ, sharing every coefficient
array, boundary value and boundary condition with `op` (a rebuild of `op`'s coefficients,
[`set_fokker_planck_coefficients!`](@ref), is seen by both). The radial scheme is
unchanged. Used by the Jacobian modes of [`FokkerPlanckTerm`](@ref): the
residual of one scheme with the stored Jacobian of another.
"""
function with_scheme(op::AdvectionDiffusion{T}, scheme::Symbol; p_scheme::Symbol=scheme,
                     xi_scheme::Symbol=scheme) where {T}
    for s in (p_scheme, xi_scheme)
        s in (:upwind, :central, :exponential_fitting) || is_four_point_scheme(s) ||
            throw(ArgumentError("unknown scheme $s"))
    end
    p_scheme === :exponential_fitting && op.p_upper_boundary === :extrapolated_flux &&
        throw(ArgumentError("extrapolated_flux has no exponential_fitting counterpart in DREAM " *
                            "(BC_DPHI_CONST); its flux does not split into Aᵖ times a face value"))
    return AdvectionDiffusion{T}(op.grid, op.advection_p, op.diffusion_p_p, op.diffusion_p_xi, op.advection_xi,
                                 op.diffusion_xi_xi, op.diffusion_xi_p, op.advection_r, op.diffusion_r_r,
                                 op.unit_field_advection_p, op.unit_field_advection_xi,
                                 op.p_flux_at_lower_boundary, op.advection_p_times_p2_at_zero, p_scheme, xi_scheme,
                                 op.r_scheme, op.p_lower_boundary, op.p_upper_boundary, op.r_boundary,
                                 op.momentum_grids, op.radial_overlaps,
                                 op.jacobian_centers, op.jacobian_p_faces, op.jacobian_xi_faces, op.jacobian_r_faces,
                                 op.jacobian_over_p2_at_zero, op.coefficient_source)
end

# the scheme of a direction with too few cells for a four-point scheme: central, as in DREAM
@inline _two_point_scheme(scheme) = is_four_point_scheme(scheme) ? :central : scheme

# advection of a face: that of the operator, plus the electric field of the radius times the
# coefficient of a unit field when the field is given (an unknown of the system)
Base.@propagate_inbounds _advection_p(op::AdvectionDiffusion, ::Nothing, k, j, ir) = op.advection_p[k, j, ir]
Base.@propagate_inbounds _advection_p(op::AdvectionDiffusion, field, k, j, ir) =
    op.advection_p[k, j, ir] + field[ir] * op.unit_field_advection_p[k, j, ir]
Base.@propagate_inbounds _advection_xi(op::AdvectionDiffusion, ::Nothing, i, k, ir) = op.advection_xi[i, k, ir]
Base.@propagate_inbounds _advection_xi(op::AdvectionDiffusion, field, i, k, ir) =
    op.advection_xi[i, k, ir] + field[ir] * op.unit_field_advection_xi[i, k, ir]

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
    m = op.momentum_grids[ir]
    np = momentum_cell_count(m)
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
    m = op.momentum_grids[ir]
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

# Four-point face value on r face `k` for pitch grids that are the same on both sides,
# mirrored at the axis and zero beyond the edge (DREAM AD_BC_MIRRORED and AD_BC_DIRICHLET)
Base.@propagate_inbounds function _r_four_point_face(op::AdvectionDiffusion, f, state, A, i, j, k)
    rg = op.grid.radial_grid
    nr = radial_cell_count(rg)
    r_u, r_uu, r_d = A > 0 ? (k - 1, k - 2, k) : (k, k + 1, k - 1)
    centers, faces = rg.r_centers, rg.r_faces
    value(g, ir) = ir < 1 ? g[i, j, 1-ir] : (ir > nr ? zero(eltype(g)) : g[i, j, ir])
    x_d, x_u, x_uu = _ghost_center(centers, faces, r_d), _ghost_center(centers, faces, r_u), _ghost_center(centers, faces, r_uu)
    if state === nothing || !is_nonlinear_scheme(op.r_scheme)
        return _four_point_face_value(op.r_scheme, faces[k], x_d, x_u, x_uu, value(f, r_d), value(f, r_u), value(f, r_uu))
    end
    return _frozen_face_value(op.r_scheme, faces[k], x_d, x_u, x_uu, value(state, r_d), value(state, r_u),
                              value(state, r_uu), value(f, r_d), value(f, r_u), value(f, r_uu))
end

# Γᵖ (not multiplied by 𝒱') through interior p face `k` (2 <= k <= momentum cell count),
# between cells k-1 and k.
Base.@propagate_inbounds function _p_interior_gamma(op::AdvectionDiffusion, f, k, j, ir, state, field=nothing)
    m = op.momentum_grids[ir]
    np = momentum_cell_count(m)
    A = _advection_p(op, field, k, j, ir)
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
Base.@propagate_inbounds function _p_flux(op::AdvectionDiffusion, f, k, j, ir, state=nothing, field=nothing)
    m = op.momentum_grids[ir]
    np = momentum_cell_count(m)
    V = op.jacobian_p_faces[k, j, ir]
    T = eltype(f)
    if k == 1
        if m.p_faces[1] == 0
            # p²Aᵖ times the face value, which is f₁ for the two-point schemes and the
            # limiters (mirrored ghost cell) and the scheme's own value for QUICK and
            # second-order upwind (DREAM AdvectionTerm, F1PSqAtZero with the p = 0 deltas)
            face = is_four_point_scheme(op.p_scheme) && np >= 3 ?
                   _p_four_point_face(op, f, state, _advection_p(op, field, 1, j, ir), 1, j, ir) : f[1, j, ir]
            return op.jacobian_over_p2_at_zero[j, ir] * op.advection_p_times_p2_at_zero[j, ir] * face
        end
        return op.p_lower_boundary === :prescribed ? V * op.p_flux_at_lower_boundary[j, ir] : zero(T)
    end
    V == 0 && return zero(T)
    A = _advection_p(op, field, k, j, ir)
    if k == np + 1
        # the flux through an interface belongs to the MomentumInterfaceTerm
        op.p_upper_boundary === :interface && return zero(T)
        f_last = f[np, j, ir]
        if op.p_upper_boundary === :constant_flux
            # 𝒱'Γᵖ continued unchanged from the last interior face (DREAM BC_PHI_CONST)
            return _p_flux(op, f, np, j, ir, state, field)
        elseif op.p_upper_boundary === :extrapolated_flux
            # linear extrapolation of 𝒱'Γᵖ from the last two interior faces (DREAM
            # BC_DPHI_CONST, PXiExternalLoss.cpp), ported literally including an apparent
            # index inconsistency: both advective terms are multiplied by the last
            # interior face's Aᵖ (not each term's own), while the
            # second term's interpolation (upstream direction and weights) and diffusion
            # are the second-to-last face's own. Both terms share the last interior face's
            # 𝒱'.
            A_last = _advection_p(op, field, np, j, ir)
            A_prev = _advection_p(op, field, np-1, j, ir)
            gamma_last = _p_interior_gamma(op, f, np, j, ir, state, field)
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
        elseif op.p_upper_boundary === :extrapolated_flux_consistent
            # linear extrapolation of 𝒱'Γᵖ (DREAM's own "dΦ/dp constant", Φ = 𝒱'Γᵖ) using
            # each face's own 𝒱' and Γᵖ throughout, unlike DREAM's code (which
            # also shares one 𝒱' between the two terms, a second deviation from its
            # stated definition beside the Aᵖ index)
            phi_last = _p_flux(op, f, np, j, ir, state, field)
            phi_prev = _p_flux(op, f, np - 1, j, ir, state, field)
            dd = m.p_widths[np] / m.p_widths[np-1]
            return (1 + dd) * phi_last - dd * phi_prev
        elseif op.p_upper_boundary === :f_zero && np > 1  # one cell: outflow below
            distance = m.p_centers[np] - m.p_centers[np-1]
            if op.p_scheme === :exponential_fitting
                # the ghost cell with f = 0 at one spacing beyond the last centre
                return V * _face_flux(op.p_scheme, A, op.diffusion_p_p[k, j, ir], f_last, zero(T), distance, 0.5)
            end
            return V * ((A > 0 ? A * f_last : zero(T)) + op.diffusion_p_p[k, j, ir] * f_last / distance)
        end
        return V * (A > 0 ? A * f_last : zero(T))
    end
    return V * _p_interior_gamma(op, f, k, j, ir, state, field)
end

# Flux Γᵠ through ξ face `k` (between cells k-1 and k), multiplied by 𝒱'.
Base.@propagate_inbounds function _xi_flux(op::AdvectionDiffusion, f, i, k, ir, state=nothing, field=nothing)
    m = op.momentum_grids[ir]
    T = eltype(f)
    (k == 1 || k == pitch_cell_count(m) + 1) && return zero(T)
    V = op.jacobian_xi_faces[i, k, ir]
    V == 0 && return zero(T)
    f_left = f[i, k-1, ir]
    f_right = f[i, k, ir]
    A = _advection_xi(op, field, i, k, ir)
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

# How the radial fluxes of an apply! are computed, decided once per call so that the loop over
# the cells holds no test of the scheme or of the pitch overlap:
# - `_RadialTwoPoint`: one pitch grid at all radii, a two-point scheme (the scheme of too few
#   radii for a four-point scheme, as in DREAM),
# - `_RadialFourPoint`: one pitch grid at all radii, a four-point scheme and at least 3 radii,
# - `_RadialOverlap`: pitch grids that differ between radii, each face with its overlap.
struct _RadialTwoPoint
    scheme::Symbol
end
struct _RadialFourPoint end
struct _RadialOverlap end

function _radial_flux_mode(op::AdvectionDiffusion)
    all(overlap -> overlap.identical, op.radial_overlaps) || return _RadialOverlap()
    is_four_point_scheme(op.r_scheme) && radial_cell_count(op.grid) >= 3 && return _RadialFourPoint()
    return _RadialTwoPoint(_two_point_scheme(op.r_scheme))
end

# Flux Γʳ through the edge face `k = nr + 1` on the side of pitch cell `j`, multiplied by 𝒱'
Base.@propagate_inbounds function _r_edge_flux(op::AdvectionDiffusion, f, i, j, k)
    rg = op.grid.radial_grid
    nr = radial_cell_count(rg)
    T = eltype(f)
    V = op.jacobian_r_faces[i, j, k]
    (V == 0 || op.r_boundary === :zero_flux) && return zero(T)
    # absorbing edge: f = 0 on the face, or half a cell outside it (DREAM)
    A = op.advection_r[i, j, k]
    D = op.diffusion_r_r[i, j, k]
    f_left = f[i, j, nr]
    distance = op.r_boundary === :absorbing ? rg.r_faces[k] - rg.r_centers[nr] :
               nr > 1 ? rg.r_centers[nr] - rg.r_centers[nr-1] : one(eltype(rg.r_centers))
    F = (A > 0 ? A * f_left : zero(T)) + D * f_left / distance
    return V * F
end

# Flux Γʳ through r face `k` (between the radii k-1 and k) on the side of pitch cell `j` of
# radius `ir` (k - 1 or k), multiplied by 𝒱': the sum over the overlap cells of the pitch
# cell, each with its share of the cell. Pitch cell `j` itself with one pitch grid on both
# sides.
Base.@propagate_inbounds _r_flux(op::AdvectionDiffusion, f, i, j, ir, k, state=nothing) =
    _r_flux(op, f, i, j, ir, k, state, _RadialOverlap())

Base.@propagate_inbounds function _r_flux(op::AdvectionDiffusion, f, i, j, ir, k, state, mode::_RadialTwoPoint)
    T = eltype(f)
    k == 1 && return zero(T)
    rg = op.grid.radial_grid
    k == radial_cell_count(rg) + 1 && return _r_edge_flux(op, f, i, j, k)
    V = op.jacobian_r_faces[i, j, k]
    V == 0 && return zero(T)
    distance = rg.r_centers[k] - rg.r_centers[k-1]
    weight_left = (rg.r_centers[k] - rg.r_faces[k]) / distance
    F = _face_flux(mode.scheme, op.advection_r[i, j, k], op.diffusion_r_r[i, j, k],
                   f[i, j, k-1], f[i, j, k], distance, weight_left)
    return V * F
end

Base.@propagate_inbounds function _r_flux(op::AdvectionDiffusion, f, i, j, ir, k, state, ::_RadialFourPoint)
    T = eltype(f)
    k == 1 && return zero(T)
    rg = op.grid.radial_grid
    k == radial_cell_count(rg) + 1 && return _r_edge_flux(op, f, i, j, k)
    V = op.jacobian_r_faces[i, j, k]
    V == 0 && return zero(T)
    distance = rg.r_centers[k] - rg.r_centers[k-1]
    A = op.advection_r[i, j, k]
    face_value = A == 0 ? zero(T) : _r_four_point_face(op, f, state, A, i, j, k)
    return V * (A * face_value - op.diffusion_r_r[i, j, k] * (f[i, j, k] - f[i, j, k-1]) / distance)
end

Base.@propagate_inbounds function _r_flux(op::AdvectionDiffusion, f, i, j, ir, k, state, ::_RadialOverlap)
    rg = op.grid.radial_grid
    nr = radial_cell_count(rg)
    T = eltype(f)
    k == 1 && return zero(T)
    k == nr + 1 && return _r_edge_flux(op, f, i, j, k)
    overlap = op.radial_overlaps[k]
    if overlap.identical
        mode = is_four_point_scheme(op.r_scheme) && nr >= 3 ? _RadialFourPoint() :
               _RadialTwoPoint(_two_point_scheme(op.r_scheme))
        return _r_flux(op, f, i, j, ir, k, state, mode)
    end
    distance = rg.r_centers[k] - rg.r_centers[k-1]
    weight_left = (rg.r_centers[k] - rg.r_faces[k]) / distance
    total = zero(T)
    if ir == k - 1
        for s in overlap.left_ranges[j]
            V = op.jacobian_r_faces[i, s, k]
            V == 0 && continue
            F = _face_flux(op.r_scheme, op.advection_r[i, s, k], op.diffusion_r_r[i, s, k], f[i, j, k-1],
                           f[i, overlap.right_cells[s], k], distance, weight_left)
            total += overlap.left_fractions[s] * (V * F)
        end
    else
        for s in overlap.right_ranges[j]
            V = op.jacobian_r_faces[i, s, k]
            V == 0 && continue
            F = _face_flux(op.r_scheme, op.advection_r[i, s, k], op.diffusion_r_r[i, s, k],
                           f[i, overlap.left_cells[s], k-1], f[i, j, k], distance, weight_left)
            total += overlap.right_fractions[s] * (V * F)
        end
    end
    return total
end

"""
    apply!(df, op, f; scale=1, limiter_state=nothing, electric_field=nothing)

Add `scale` times the operator applied to `f` to `df` (in place, allocation free). With
`electric_field` (one value per radius, in V/m) the advection of every face is that of the
operator plus the field times the coefficients of a unit field
([`set_unit_field_coefficients!`](@ref)): the field as an argument, for an electric field
that is an unknown of the system. The
operator is linear in `f` except for a prescribed lower-p flux, which adds a constant
source, and for the flux limiters, whose face values depend on `f`. With
`limiter_state`, the limiter weights are evaluated at that state and applied linearly to
`f`: the limiter frozen at `limiter_state` (DREAM's linearly implicit solver takes them
from the previous time level; the frozen-limiter Jacobian).
"""
function apply!(df::AbstractArray{<:Any,3}, op::AdvectionDiffusion, f::AbstractArray{<:Any,3};
                scale=true, limiter_state=nothing, electric_field=nothing)
    electric_field === nothing || length(electric_field) == radial_cell_count(op.grid) ||
        throw(DimensionMismatch("the electric field needs one value per radius"))
    size(f) == size(df) == size(op.grid) || throw(DimensionMismatch("f and df must match grid"))
    limiter_state === nothing || size(limiter_state) == size(f) ||
        throw(DimensionMismatch("limiter_state must match the grid"))
    radial_mode = _radial_flux_mode(op)
    @inbounds for ir in 1:radial_cell_count(op.grid)
        apply_radius!(df, op, f, ir, scale, limiter_state, electric_field, radial_mode)
    end
    return df
end

"""
    apply_radius!(df, op, f, ir, scale)

Operator contribution to all momentum cells at radius index `ir`. Writes only to
`df[:, :, ir]`, so different radii can be processed concurrently. `radial_mode` is the
computation of the radial fluxes (`_radial_flux_mode`), a function barrier: the
loop over the cells is compiled for it.
"""
@inline function apply_radius!(df, op::AdvectionDiffusion, f, ir, scale, state=nothing, field=nothing,
                               radial_mode=_radial_flux_mode(op))
    m = op.momentum_grids[ir]
    rg = op.grid.radial_grid
    np, nxi = momentum_cell_count(m), pitch_cell_count(m)
    @inbounds for j in 1:nxi, i in 1:np
        V = op.jacobian_centers[i, j, ir]
        V == 0 && continue
        divergence = (_p_flux(op, f, i + 1, j, ir, state, field) - _p_flux(op, f, i, j, ir, state, field)) /
                     m.p_widths[i] +
                     (_xi_flux(op, f, i, j + 1, ir, state, field) - _xi_flux(op, f, i, j, ir, state, field)) /
                     m.xi_widths[j] +
                     (_r_flux(op, f, i, j, ir, ir + 1, state, radial_mode) -
                      _r_flux(op, f, i, j, ir, ir, state, radial_mode)) / rg.r_widths[ir]
        df[i, j, ir] -= scale * divergence / V
    end
    return df
end

"""
    momentum_flux(op::AdvectionDiffusion, f, face, ir=1; electric_field=nothing)

Particle flux per unit volume [m⁻³ s⁻¹] through the momentum face `face` (between cells
`face - 1` and `face`) at radius index `ir`, `(1/V') Σ_j Δξ_j 𝒱'ᵖ Γᵖ`, positive towards
larger momentum. This is the rate at which particles cross the surface `p = p_faces[face]`,
counted in the same normalization as [`density`](@ref), so a runaway rate measured through
a surface above the critical momentum is directly comparable with
[`connor_hastie_rate`](@ref). Face 1 is the flux through `p = p_min` (the `p = 0` limit or
the prescribed boundary flux) and face `momentum_cell_count + 1` the flux out of the grid.
`electric_field` (one value per radius) is the field of an operator that takes it as an
argument ([`apply!`](@ref)).
"""
function momentum_flux(op::AdvectionDiffusion, f::AbstractArray{<:Any,3}, face::Integer,
                       ir::Integer=1; electric_field=nothing)
    m = momentum_grid(op.grid, ir)
    1 <= face <= momentum_cell_count(m) + 1 ||
        throw(ArgumentError("momentum face $face outside the grid (1 to $(momentum_cell_count(m) + 1))"))
    total = zero(electric_field === nothing ? eltype(f) : promote_type(eltype(f), eltype(electric_field)))
    @inbounds for j in 1:pitch_cell_count(m)
        total += m.xi_widths[j] * _p_flux(op, f, face, j, ir, nothing, electric_field)
    end
    return total / op.grid.radial_grid.spatial_jacobian[ir]
end
