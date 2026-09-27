# Equation terms. A term adds its contribution to the right-hand side of the equation of
# one unknown (the row unknown, given by its id `row`) in the system M du/dt = rhs(u, t).

"""
    EquationTerm

Supertype of the terms of an [`Equation`](@ref). A term implements

- `dependencies(term)`: names of the unknowns it reads;
- `coupling(term)`: `:sparse` (its Jacobian block is stored; `sparsity!` lists the
  structural nonzeros) or `:matrix_free` (applied through `jacobian_vector_product!`,
  e.g. the knock-on term, whose block is dense per radius);
- `rhs!(out, term, u, layout, row, t)`: add the term to `out` in the rows of the unknown
  `row`, as a pure function of the state `u` and the time `t`. `:sparse` terms must accept
  state vectors of dual numbers, since their Jacobian comes from AD;
- `sparsity!(entries, term, layout, row)` for `:sparse` terms: push `(row index, column
  index)` pairs of the structural nonzeros;
- `jacobian_vector_product!(out, term, u, v, layout, row, t)` for `:matrix_free` terms:
  add `(∂ rhs_term / ∂u) v` to `out`;

and optionally `treatment(term)` (`:implicit` by default; `:explicit` terms are evaluated
at the previous time level and contribute nothing to the Jacobian),
`validate(term, layout, row)` (checks against the layout, run when the system is built),
[`is_linear`](@ref)`(term)` (`false` by default: whether the term is linear in the state,
up to a constant, which lets a linear system accept a Newton step on the update test alone)
and, for `:sparse` terms whose stored block is an approximation of their Jacobian
(a quasi-Newton matrix), [`jacobian_kind`](@ref)`(term) === :approximate`
with [`jacobian_correction_product!`](@ref) supplying the difference to the exact product.

A term defined outside the package adds its methods under the qualified names
(`Runes2P.dependencies(term::MyTerm) = ...`, `function Runes2P.rhs!(out, term::MyTerm, ...)`).
`validate` is not exported, since only the construction of an [`EquationSystem`](@ref)
calls it and the name is common to many packages; the other functions are exported for
calls.
"""
abstract type EquationTerm end

function dependencies end
coupling(::EquationTerm) = :sparse
treatment(::EquationTerm) = :implicit
validate(::EquationTerm, layout, row) = nothing

"""
    jacobian_kind(term)

`:exact` (default) when the stored Jacobian block of a `:sparse` term is the derivative
of its residual, `:approximate` when it is a cheaper matrix (a frozen limiter, another
scheme's Jacobian, [`FokkerPlanckTerm`](@ref)). With an approximate block the Newton
iteration on the stored matrix converges linearly (a quasi-Newton method), which
[`NewtonSolver`](@ref) accounts for; [`GMRES`](@ref)`(operator=:exact)` restores the
exact Newton operator through [`jacobian_correction_product!`](@ref).
"""
jacobian_kind(::EquationTerm) = :exact

"""
    jacobian_correction_product!(out, term, u, v, layout, row, t)

For a `:sparse` term with `jacobian_kind(term) === :approximate`: add `(J_exact - J_stored) v`
to `out`, the difference between the exact Jacobian-vector product of the term's
residual and the product of its stored block, both at the state `u`. Nothing for exact
terms.
"""
jacobian_correction_product!(out, ::EquationTerm, u, v, layout, row, t) = out

is_linear(::EquationTerm) = false

"""
    LayoutKineticArray

3D array view `f[i, j, ir]` of a kinetic unknown with a shared momentum grid in a state
vector, so that operators written for 3D arrays apply to the state directly.
"""
struct LayoutKineticArray{T,V<:AbstractVector{T},O<:AbstractVector{Int}} <: AbstractArray{T,3}
    state::V
    offsets::O   # per radius: index before the first entry (a range when equally spaced)
    p_cells::Int
    xi_cells::Int
end

Base.size(a::LayoutKineticArray) = (a.p_cells, a.xi_cells, length(a.offsets))
Base.@propagate_inbounds Base.getindex(a::LayoutKineticArray, i::Int, j::Int, ir::Int) =
    a.state[a.offsets[ir] + i + (j - 1) * a.p_cells]
Base.@propagate_inbounds Base.setindex!(a::LayoutKineticArray, x, i::Int, j::Int, ir::Int) =
    (a.state[a.offsets[ir] + i + (j - 1) * a.p_cells] = x)

"""
    kinetic_array(u, layout, id)

[`LayoutKineticArray`](@ref) of kinetic unknown `id` in state `u`; the unknown's grid must
have a shared momentum grid.
"""
function kinetic_array(u::AbstractVector, layout::UnknownLayout, id::Integer)
    grid = kinetic_grid(layout, id)
    has_shared_momentum_grid(grid) ||
        throw(ArgumentError("$(layout.unknowns[id].name) has different momentum grids at different radii"))
    np, nxi = layout.shapes[id, 1]
    return LayoutKineticArray(u, layout.radius_offsets[id], np, nxi)
end

function _equally_spaced(offsets)
    for k in 3:length(offsets)
        offsets[k] - offsets[k-1] == offsets[2] - offsets[1] || return false
    end
    return true
end

# the offsets of kinetic unknown `id` as a range; valid when they are equally spaced
function _offset_range(layout::UnknownLayout, id)
    offsets = layout.radius_offsets[id]
    # with one radius the step is never used, but `range` rejects a zero step
    step = length(offsets) > 1 ? offsets[2] - offsets[1] : 1
    return range(first(offsets); step, length=length(offsets))
end

# whether two kinetic grids have the same radii and the same momentum grid at each of them
function _same_momentum_grids(a::KineticGrid, b::KineticGrid)
    a === b && return true
    a.radial_grid.r_faces == b.radial_grid.r_faces || return false
    return all(ir -> momentum_grid(a, ir).p_faces == momentum_grid(b, ir).p_faces &&
                     momentum_grid(a, ir).xi_faces == momentum_grid(b, ir).xi_faces,
               1:radial_cell_count(a))
end

# index of cell (i, j) of kinetic unknown `id` at radius `ir`
@inline _kinetic_index(layout::UnknownLayout, id, ir, i, j) =
    layout.offsets[id, ir] + i + (j - 1) * layout.shapes[id, ir][1]

# tag of the dual numbers of jacobian_correction_product!
struct JacobianCorrectionTag end

"""
    FokkerPlanckTerm(unknown, operator::AdvectionDiffusion; jacobian=:exact)

Advection-diffusion (Fokker-Planck) operator applied to the kinetic unknown `unknown`,
added to the equation of the row unknown (normally the same). Stored sparse, with the
3×3 `(p, ξ)` stencil, the second neighbours of the four-point schemes and the two radial
neighbours. Needs a shared momentum grid.

`jacobian` selects what the stored Jacobian block holds for a flux-limiter scheme (DREAM's
`ad_jac` setting of `AdvectionInterpolation`). The residual is always the operator's own, so
every mode converges to the same solution:

- `:exact`: the Jacobian of the residual by colored forward-mode AD, including the
  limiter's derivative ψ′ (DREAM `AD_INTERP_JACOBIAN_FULL`, which carries ψ′ only for the
  limiters DREAM counts as smooth);
- `:frozen_limiter`: the limiter weights frozen at the iterate and differentiated as
  constants (`apply!(...; limiter_state)`, DREAM `AD_INTERP_JACOBIAN_LINEAR`);
- a linear scheme name (`:upwind`, `:central`, `:exponential_fitting`, `:quick`,
  `:upwind_second_order`): the Jacobian of the operator with that scheme in p and ξ
  ([`with_scheme`](@ref); `:upwind` is DREAM `AD_INTERP_JACOBIAN_UPWIND`), with that
  scheme's stencil (3-point for the two-point schemes).

For a scheme linear in `f` the frozen limiter is the exact Jacobian, and another linear
scheme's is still a different matrix (upwind's for a central residual, DREAM #126).
With an approximate block the term declares [`jacobian_kind`](@ref)`(term) === :approximate`: the Newton iteration on the
stored matrix is a quasi-Newton one that converges linearly, or, with
[`GMRES`](@ref)`(operator=:exact)`, the stored block only preconditions the exact Newton
operator ([`jacobian_correction_product!`](@ref), one dual-number pass of the operator
per product).
"""
struct FokkerPlanckTerm{T} <: EquationTerm
    unknown::Symbol
    operator::AdvectionDiffusion{T}
    jacobian::Symbol
    stored_operator::AdvectionDiffusion{T}   # the operator whose (frozen) derivative is the stored block
    values::Array{T,3}                       # the iterate's values (frozen limiter, corrections)
    correction::Array{T,3}
    dual_input::Array{ForwardDiff.Dual{JacobianCorrectionTag,T,1},3}
    dual_output::Array{ForwardDiff.Dual{JacobianCorrectionTag,T,1},3}
end

const LINEAR_JACOBIAN_SCHEMES = (:upwind, :central, :exponential_fitting, :quick, :upwind_second_order)

function FokkerPlanckTerm(unknown::Symbol, operator::AdvectionDiffusion{T}; jacobian::Symbol=:exact) where {T}
    stored = if jacobian in (:exact, :frozen_limiter)
        operator
    elseif jacobian in LINEAR_JACOBIAN_SCHEMES
        with_scheme(operator, jacobian)
    else
        throw(ArgumentError("jacobian must be :exact, :frozen_limiter or a linear scheme " *
                            "($(join(LINEAR_JACOBIAN_SCHEMES, ", "))), not $jacobian"))
    end
    dims = _jacobian_kind(jacobian, operator, stored) === :approximate ? size(operator.grid) : (0, 0, 0)
    D = ForwardDiff.Dual{JacobianCorrectionTag,T,1}
    return FokkerPlanckTerm{T}(unknown, operator, jacobian, stored, zeros(T, dims), zeros(T, dims),
                               zeros(D, dims), zeros(D, dims))
end

# a frozen limiter of a linear scheme is the exact derivative; another scheme's Jacobian
# is exact only when it is the operator's own scheme
function _jacobian_kind(jacobian, operator, stored)
    jacobian === :exact && return :exact
    jacobian === :frozen_limiter && return is_linear(operator) ? :exact : :approximate
    return (stored.p_scheme, stored.xi_scheme) == (operator.p_scheme, operator.xi_scheme) ? :exact : :approximate
end

dependencies(term::FokkerPlanckTerm) = [term.unknown]
jacobian_kind(term::FokkerPlanckTerm) = _jacobian_kind(term.jacobian, term.operator, term.stored_operator)
is_linear(term::FokkerPlanckTerm) = is_linear(term.operator)

function validate(term::FokkerPlanckTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("FokkerPlanckTerm acts on and adds to kinetic unknowns"))
    grid = term.operator.grid
    for k in (id, row), ir in 1:radial_cell_count(layout)
        layout.shapes[k, ir] == (momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) ||
            throw(DimensionMismatch("the operator grid does not match $(layout.unknowns[k].name)"))
    end
    return nothing
end

function rhs!(out, term::FokkerPlanckTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    if _equally_spaced(layout.radius_offsets[row]) && _equally_spaced(layout.radius_offsets[id])
        # equally spaced radius blocks (the usual case) index by arithmetic instead of a load
        np, nxi = layout.shapes[id, 1]
        _fokker_planck!(LayoutKineticArray(out, _offset_range(layout, row), np, nxi), term,
                        LayoutKineticArray(u, _offset_range(layout, id), np, nxi))
    else
        _fokker_planck!(kinetic_array(out, layout, row), term, kinetic_array(u, layout, id))
    end
    return out
end

# the operator on `f` into `df`, with the stored block's derivative for a dual state
function _fokker_planck!(df, term::FokkerPlanckTerm, f)
    if jacobian_kind(term) === :exact || !(eltype(f) <: ForwardDiff.Dual)
        apply!(df, term.operator, f)
    elseif term.jacobian === :frozen_limiter
        # a dual state (Jacobian assembly): the limiter weights at the values, applied as
        # constants; the value part is the full residual, since the weights at f are f's own
        term.values .= _value.(f)
        apply!(df, term.operator, f; limiter_state=term.values)
    else
        # the derivative of the other scheme, with the value part corrected to the
        # operator's own residual (constant sources cancel in the difference)
        apply!(df, term.stored_operator, f)
        term.values .= _value.(f)
        fill!(term.correction, zero(eltype(term.correction)))
        apply!(term.correction, term.operator, term.values)
        apply!(term.correction, term.stored_operator, term.values; scale=-1)
        df .+= term.correction
    end
    return df
end

function sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    op = term.stored_operator
    np, nxi, nr = size(op.grid)
    p_width = stencil_half_width(op.p_scheme)
    xi_width = stencil_half_width(op.xi_scheme)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        r = _kinetic_index(layout, row, ir, i, j)
        for dj in -1:1, di in -1:1
            (1 <= i + di <= np && 1 <= j + dj <= nxi) || continue
            push!(entries, (r, _kinetic_index(layout, id, ir, i + di, j + dj)))
        end
        # the second neighbours of the four-point schemes
        for d in 2:p_width, s in (-d, d)
            1 <= i + s <= np && push!(entries, (r, _kinetic_index(layout, id, ir, i + s, j)))
        end
        for d in 2:xi_width, s in (-d, d)
            1 <= j + s <= nxi && push!(entries, (r, _kinetic_index(layout, id, ir, i, j + s)))
        end
        for dr in (-1, 1)
            1 <= ir + dr <= nr || continue
            push!(entries, (r, _kinetic_index(layout, id, ir + dr, i, j)))
        end
        # the extrapolated p_max boundary flux couples the last row up to 3 cells in p
        # (:extrapolated_flux's second term has no p-ξ cross contribution,
        # :extrapolated_flux_consistent's does)
        pub = op.p_upper_boundary
        if pub in (:extrapolated_flux, :extrapolated_flux_consistent) && i == np
            for d in 1:3
                np - d >= 1 && push!(entries, (r, _kinetic_index(layout, id, ir, np - d, j)))
            end
            if pub === :extrapolated_flux_consistent
                for s in (-1, 1)
                    np >= 2 && 1 <= j + s <= nxi && push!(entries, (r, _kinetic_index(layout, id, ir, np - 2, j + s)))
                end
            end
        end
    end
    return entries
end

# (J_exact - J_stored) v in one dual pass: the operator's directional derivative along v
# minus the stored operator's (the frozen limiter's, or the other scheme's); allocation
# free with the term's buffers
function jacobian_correction_product!(out, term::FokkerPlanckTerm, u, v, layout, row, t)
    jacobian_kind(term) === :approximate || return out
    id = unknown_id(layout, term.unknown)
    f, direction = kinetic_array(u, layout, id), kinetic_array(v, layout, id)
    input, output = term.dual_input, term.dual_output
    D = eltype(input)
    @inbounds for k in CartesianIndices(input)
        input[k] = D(f[k], ForwardDiff.Partials((direction[k],)))
    end
    fill!(output, zero(D))
    apply!(output, term.operator, input)
    if term.jacobian === :frozen_limiter
        term.values .= f
        apply!(output, term.operator, input; scale=-1, limiter_state=term.values)
    else
        apply!(output, term.stored_operator, input; scale=-1)
    end
    out_view = kinetic_array(out, layout, row)
    @inbounds for k in CartesianIndices(output)
        out_view[k] += ForwardDiff.partials(output[k], 1)
    end
    return out
end

"""
    PrescribedFokkerPlanckTerm(unknown, operator::AdvectionDiffusion, plasma_at;
                               settings=CollisionSettings(), synchrotron=false, jacobian=:exact)

Fokker-Planck term whose collision coefficients follow a prescribed, time-dependent
plasma state (a cooling plasma, `T(t)` prescribed since Runes2P has no
self-consistent temperature evolution yet). `plasma_at(t)` returns the
[`PlasmaState`](@ref) at time `t`; `operator`'s coefficients are rebuilt from it with
[`set_fokker_planck_coefficients!`](@ref) (`operator`'s scheme and boundary conditions are
unchanged). The rebuild is cached on `t`: colored AD calls `rhs!` once per colour at the
same `t`, and the coefficients do not depend on the state `u`, so rebuilding once per
distinct `t` is exact, not an approximation. Otherwise identical to
[`FokkerPlanckTerm`](@ref)`(unknown, operator; jacobian)`, which it wraps (`term.inner`).
"""
struct PrescribedFokkerPlanckTerm{T,F} <: EquationTerm
    inner::FokkerPlanckTerm{T}
    plasma_at::F
    settings::CollisionSettings
    synchrotron::Bool
    cached_t::Base.RefValue{Float64}
end

function PrescribedFokkerPlanckTerm(unknown::Symbol, operator::AdvectionDiffusion, plasma_at;
                                    settings::CollisionSettings=CollisionSettings(), synchrotron::Bool=false,
                                    jacobian::Symbol=:exact)
    return PrescribedFokkerPlanckTerm(FokkerPlanckTerm(unknown, operator; jacobian), plasma_at, settings,
                                      synchrotron, Ref(NaN))
end

dependencies(term::PrescribedFokkerPlanckTerm) = dependencies(term.inner)
validate(term::PrescribedFokkerPlanckTerm, layout, row) = validate(term.inner, layout, row)
jacobian_kind(term::PrescribedFokkerPlanckTerm) = jacobian_kind(term.inner)
is_linear(term::PrescribedFokkerPlanckTerm) = is_linear(term.inner)
sparsity!(entries, term::PrescribedFokkerPlanckTerm, layout, row) = sparsity!(entries, term.inner, layout, row)

function _rebuild_at!(term::PrescribedFokkerPlanckTerm, t)
    t == term.cached_t[] && return term.inner.operator
    set_fokker_planck_coefficients!(term.inner.operator, term.plasma_at(t), term.settings;
                                    synchrotron=term.synchrotron)
    term.cached_t[] = t
    return term.inner.operator
end

function rhs!(out, term::PrescribedFokkerPlanckTerm, u, layout, row, t)
    _rebuild_at!(term, t)
    return rhs!(out, term.inner, u, layout, row, t)
end

function jacobian_correction_product!(out, term::PrescribedFokkerPlanckTerm, u, v, layout, row, t)
    _rebuild_at!(term, t)
    return jacobian_correction_product!(out, term.inner, u, v, layout, row, t)
end

"""
    KnockOnTerm(primary, operator::KnockOnOperator; target_density, treatment=:implicit)

Knock-on source produced by the kinetic unknown `primary` on the kinetic row unknown
(`operator` maps the primary grid to the row unknown's grid, per radius, see
[`apply_at_radius!`](@ref)). `target_density` is the target (total electron) density:
the name of a radial unknown, or a vector with one value per radius. Matrix-free, since
it couples all cells of a radius; `:explicit` treatment evaluates it at the previous time
level, as DREAM does. Works with different momentum grids at different radii.
"""
struct KnockOnTerm{T,D} <: EquationTerm
    primary::Symbol
    operator::KnockOnOperator{T}
    target_density::D
    treatment::Symbol
end

function KnockOnTerm(primary::Symbol, operator::KnockOnOperator; target_density,
                     treatment::Symbol=:implicit)
    treatment in (:implicit, :explicit) || throw(ArgumentError("unknown treatment $treatment"))
    target_density isa Union{Symbol,AbstractVector} ||
        throw(ArgumentError("target_density must be an unknown name or a vector"))
    return KnockOnTerm(primary, operator, target_density, treatment)
end

# linear in the primary with a prescribed target density; bilinear when it is an unknown
is_linear(term::KnockOnTerm) = !(term.target_density isa Symbol)
dependencies(term::KnockOnTerm) =
    term.target_density isa Symbol ? [term.primary, term.target_density] : [term.primary]
coupling(::KnockOnTerm) = :matrix_free
treatment(term::KnockOnTerm) = term.treatment

function validate(term::KnockOnTerm, layout, row)
    primary = unknown_id(layout, term.primary)
    layout.kinds[primary] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("KnockOnTerm acts on and adds to kinetic unknowns"))
    op = term.operator
    for ir in 1:radial_cell_count(layout)
        knockon_shape = (momentum_cell_count(op.knockon, ir), pitch_cell_count(op.knockon, ir))
        primary_shape = (momentum_cell_count(op.primary, ir), pitch_cell_count(op.primary, ir))
        layout.shapes[row, ir] == knockon_shape ||
            throw(DimensionMismatch("the knock-on grid does not match $(layout.unknowns[row].name)"))
        layout.shapes[primary, ir] == primary_shape ||
            throw(DimensionMismatch("the primary grid does not match $(term.primary)"))
    end
    if term.target_density isa Symbol
        density = unknown_id(layout, term.target_density)
        layout.kinds[density] === :radial && layout.sizes[density, 1] == 1 ||
            throw(ArgumentError("the target density must be a radial unknown with one component"))
    else
        length(term.target_density) == radial_cell_count(layout) ||
            throw(DimensionMismatch("target_density needs one value per radius"))
    end
    return nothing
end

_target_density(term::KnockOnTerm{T,Symbol}, u, layout, ir) where {T} =
    radial_view(u, layout, unknown_id(layout, term.target_density), ir)[1]
_target_density(term::KnockOnTerm, u, layout, ir) = term.target_density[ir]

function rhs!(out, term::KnockOnTerm, u, layout, row, t)
    primary = unknown_id(layout, term.primary)
    for ir in 1:radial_cell_count(layout)
        apply_at_radius!(kinetic_view(out, layout, row, ir), term.operator,
                         kinetic_view(u, layout, primary, ir), _target_density(term, u, layout, ir), ir)
    end
    return out
end

# C is linear in f and in the target density: J v = C(v_f; n) + C(f; v_n)
function jacobian_vector_product!(out, term::KnockOnTerm, u, v, layout, row, t)
    primary = unknown_id(layout, term.primary)
    for ir in 1:radial_cell_count(layout)
        df = kinetic_view(out, layout, row, ir)
        apply_at_radius!(df, term.operator, kinetic_view(v, layout, primary, ir),
                         _target_density(term, u, layout, ir), ir)
        if term.target_density isa Symbol
            apply_at_radius!(df, term.operator, kinetic_view(u, layout, primary, ir),
                             _target_density(term, v, layout, ir), ir)
        end
    end
    return out
end

"""
    LinearTerm(unknown; coefficient=1)

`coefficient` times the unknown `unknown`, added entry by entry to the row unknown, which
must have the same kind and sizes (e.g. `-1` times an unknown in its own algebraic
equation, or a decay rate).
"""
struct LinearTerm <: EquationTerm
    unknown::Symbol
    coefficient::Float64
end
LinearTerm(unknown::Symbol; coefficient::Real=1.0) = LinearTerm(unknown, Float64(coefficient))

dependencies(term::LinearTerm) = [term.unknown]
is_linear(::LinearTerm) = true

function validate(term::LinearTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === layout.kinds[row] && layout.sizes[id, :] == layout.sizes[row, :] ||
        throw(DimensionMismatch("LinearTerm needs an unknown of the same kind and sizes as its row"))
    return nothing
end

# the (row range, column range) pairs of a LinearTerm: one per radius, or one for scalars
_range_count(layout, row) = layout.kinds[row] === :scalar ? 1 : radial_cell_count(layout)
_paired_ranges(layout, row, id, k) =
    layout.kinds[row] === :scalar ? (unknown_range(layout, row), unknown_range(layout, id)) :
    (unknown_range(layout, row, k), unknown_range(layout, id, k))

function rhs!(out, term::LinearTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    c = term.coefficient
    for k in 1:_range_count(layout, row)
        rows, columns = _paired_ranges(layout, row, id, k)
        @inbounds for (r, s) in zip(rows, columns)
            out[r] += c * u[s]
        end
    end
    return out
end

function sparsity!(entries, term::LinearTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for k in 1:_range_count(layout, row)
        append!(entries, zip(_paired_ranges(layout, row, id, k)...))
    end
    return entries
end

"""
    DensityMomentTerm(kinetic; coefficient=1)

`coefficient` times the density `n(r) = (1/V') Σ 𝒱' Δp Δξ f` of the kinetic unknown
`kinetic`, added to a radial row unknown with one component (e.g. a hot-electron density
defined by an algebraic equation). Works with different momentum grids at different radii.
"""
struct DensityMomentTerm <: EquationTerm
    kinetic::Symbol
    coefficient::Float64
end
DensityMomentTerm(kinetic::Symbol; coefficient::Real=1.0) = DensityMomentTerm(kinetic, Float64(coefficient))

dependencies(term::DensityMomentTerm) = [term.kinetic]
is_linear(::DensityMomentTerm) = true

function validate(term::DensityMomentTerm, layout, row)
    layout.kinds[unknown_id(layout, term.kinetic)] === :kinetic ||
        throw(ArgumentError("DensityMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("DensityMomentTerm adds to a radial unknown with one component"))
    return nothing
end

rhs!(out, term::DensityMomentTerm, u, layout, row, t) =
    _add_density_moment!(out, term, kinetic_grid(layout, unknown_id(layout, term.kinetic)), u, layout, row)

function _add_density_moment!(out, term::DensityMomentTerm, grid::KineticGrid, u, layout, row)
    id = unknown_id(layout, term.kinetic)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            n += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        out[first(unknown_range(layout, row, ir))] +=
            term.coefficient * n / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

function sparsity!(entries, term::DensityMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for s in unknown_range(layout, id, ir)
            push!(entries, (r, s))
        end
    end
    return entries
end

"""
    CurrentMomentTerm(kinetic, grid; coefficient=1, pitch_rule=:linear)

`coefficient` times the current density of the kinetic unknown `kinetic` on `grid`
([`current_density`](@ref); `pitch_rule = :cell_center` is DREAM's `j_hot` moment), added
to a radial row unknown with one component (the ohmic or hot-electron current defined by
an algebraic equation). The pitch weights are built per radius when the term is created,
so different momentum grids at different radii work; `grid` must be the unknown's grid
(the same radii and momentum faces, checked when the system is built).
"""
struct CurrentMomentTerm{G<:KineticGrid} <: EquationTerm
    kinetic::Symbol
    grid::G
    coefficient::Float64
    pitch_rule::Symbol
    pitch_weights::Vector{Vector{Float64}}   # per radius
end
function CurrentMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0, pitch_rule::Symbol=:linear)
    weights = [pitch_moment_weights(momentum_grid(grid, ir); pitch_rule) for ir in 1:radial_cell_count(grid)]
    return CurrentMomentTerm(kinetic, grid, Float64(coefficient), pitch_rule, weights)
end

dependencies(term::CurrentMomentTerm) = [term.kinetic]
is_linear(::CurrentMomentTerm) = true

function validate(term::CurrentMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("CurrentMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("CurrentMomentTerm adds to a radial unknown with one component"))
    # the pitch weights belong to the term's grid: equal cell counts are not enough
    _same_momentum_grids(term.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the term's grid does not match $(term.kinetic)"))
    return nothing
end

function rhs!(out, term::CurrentMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        j = _current_moment(momentum_grid(grid, ir), term.pitch_weights[ir], phase_space_jacobian(grid, ir),
                            kinetic_view(u, layout, id, ir))
        out[first(unknown_range(layout, row, ir))] += term.coefficient * j / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

sparsity!(entries, term::CurrentMomentTerm, layout, row) =
    sparsity!(entries, DensityMomentTerm(term.kinetic, term.coefficient), layout, row)

"""
    MaxwellianSourceTerm(kinetic, grid, temperature; target_density, rate,
                         bulk_momentum=Inf)

Source with the shape of a non-relativistic Maxwellian `exp(-p²/2Θ)`, `Θ = T/mₑc²`, at
`temperature` [eV] (DREAM's `ParticleSourceTerm` uses the relativistic one), whose
magnitude drives the density of the kinetic unknown `kinetic` back to `target_density` (a
number, or one value per radius) at the rate `rate` [1/s]:

    S(p, ξ) = rate (n_target - n(f)) f_M(p) / n_M,

with `f_M` the Maxwellian on `grid` and `n_M` its density, so the source adds
`rate (n_target - n(f))` particles per second and per unit volume and nothing else.
`kinetic` and the row unknown both live on `grid`. It keeps a thermal bulk from draining
into the runaway region, which is what the published Dreicer rates assume; with
`rate = 1/Δt` a step restores the particles the previous step lost.

`n(f)` counts the cells below `bulk_momentum`, and `n_target` is the density of the bulk
there. The bound matters: with the whole grid counted, particles moving from the bulk into
the runaway tail leave the total unchanged, the source sees no deficit, and the bulk still
drains. Set it below the critical momentum and above the thermal momentum.

The term is `:explicit` (evaluated at the previous time level), so the density moment it
reads adds nothing to the Jacobian and the sparsity stays local. This is a benchmark
device, not the self-consistent particle source: DREAM's `ParticleSourceTerm` fixes the
magnitude from the cold-electron density through quasi-neutrality, which needs the fluid
unknowns.
"""
struct MaxwellianSourceTerm{T,D} <: EquationTerm
    kinetic::Symbol
    grid::KineticGrid{T}
    shape::Vector{Matrix{T}}   # normalized Maxwellian per radius, density 1
    bulk_cells::Vector{Int}    # per radius: momentum cells counted, 1:bulk_cells
    target_density::D
    rate::Float64
    temperature::Float64       # [eV], of the shape
    bulk_momentum::Float64
end

function MaxwellianSourceTerm(kinetic::Symbol, grid::KineticGrid{T}, temperature::Real;
                              target_density, rate::Real, bulk_momentum::Real=Inf) where {T}
    temperature > 0 || throw(ArgumentError("the temperature must be positive"))
    rate > 0 || throw(ArgumentError("the rate must be positive"))
    target_density isa Union{Real,AbstractVector} ||
        throw(ArgumentError("target_density must be a number or one value per radius"))
    bulk_momentum > 0 || throw(ArgumentError("bulk_momentum must be positive"))
    theta = T(temperature) / T(Constants.ELECTRON_REST_ENERGY)
    bulk_cells = map(1:radial_cell_count(grid)) do ir
        m = momentum_grid(grid, ir)
        count(<=(bulk_momentum), m.p_centers)
    end
    all(>(0), bulk_cells) || throw(ArgumentError("bulk_momentum is below the first momentum cell"))
    shape = map(1:radial_cell_count(grid)) do ir
        m = momentum_grid(grid, ir)
        values = [exp(-p^2 / (2theta)) for p in m.p_centers, _ in 1:pitch_cell_count(m)]
        jacobian = phase_space_jacobian(grid, ir)
        norm = sum(jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * values[i, j]
                   for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)) /
               grid.radial_grid.spatial_jacobian[ir]
        values ./ norm
    end
    return MaxwellianSourceTerm{T,typeof(target_density)}(kinetic, grid, shape, bulk_cells, target_density,
                                                         Float64(rate), Float64(temperature),
                                                         Float64(bulk_momentum))
end

dependencies(term::MaxwellianSourceTerm) = [term.kinetic]
treatment(::MaxwellianSourceTerm) = :explicit

function validate(term::MaxwellianSourceTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("MaxwellianSourceTerm reads and adds to kinetic unknowns"))
    # the density is counted on the grid of `kinetic` and the source added on the row's
    for k in (id, row)
        _same_momentum_grids(term.grid, kinetic_grid(layout, k)) ||
            throw(DimensionMismatch("the term's grid does not match $(layout.unknowns[k].name)"))
    end
    term.target_density isa AbstractVector && length(term.target_density) != radial_cell_count(layout) &&
        throw(DimensionMismatch("target_density needs one value per radius"))
    return nothing
end

_target(term::MaxwellianSourceTerm, ir) =
    term.target_density isa AbstractVector ? term.target_density[ir] : term.target_density

function rhs!(out, term::MaxwellianSourceTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:term.bulk_cells[ir]
            n += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        n /= grid.radial_grid.spatial_jacobian[ir]
        magnitude = term.rate * (_target(term, ir) - n)
        shape = term.shape[ir]
        added = kinetic_view(out, layout, row, ir)
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            added[i, j] += magnitude * shape[i, j]
        end
    end
    return out
end

# explicit terms carry no Jacobian block
sparsity!(entries, ::MaxwellianSourceTerm, layout, row) = entries
