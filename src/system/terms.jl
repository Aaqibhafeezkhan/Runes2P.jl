# Equation terms. A term adds its contribution to the right-hand side of the equation of
# one unknown (the row unknown, given by its id `row`) in the system M du/dt = rhs(u, t).

using SparseArrays: nzrange, rowvals, nonzeros

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

A `:sparse` term whose row holds many unknowns (a moment of a distribution, the loss
through the edge) declares [`jacobian_assembly`](@ref)`(term) === :direct` and adds its
Jacobian entries itself with [`add_jacobian!`](@ref), so that the coloring of the AD
Jacobian does not see its rows.

A term may also define `treatment(term)` (`:implicit` by default; `:explicit` terms are evaluated
at the previous time level and contribute nothing to the Jacobian), [`prepare_step!`](@ref)
(a part of an implicit term lagged to the previous time level),
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

"""
    dependencies(term)

Names of the unknowns that the [`EquationTerm`](@ref) `term` reads, as a collection of
`Symbol`s. Every term defines it.
"""
function dependencies end

"""
    coupling(term)

`:sparse` (default) when the Jacobian block of the [`EquationTerm`](@ref) `term` is
stored, with its structural nonzeros listed by [`sparsity!`](@ref). `:matrix_free` when it
is applied through [`jacobian_vector_product!`](@ref), as for the knock-on term, whose
block is dense per radius.
"""
coupling(::EquationTerm) = :sparse

"""
    treatment(term)

`:implicit` (default) or `:explicit`. An explicit [`EquationTerm`](@ref) is evaluated at
the previous time level and contributes nothing to the Jacobian.
"""
treatment(::EquationTerm) = :implicit

"""
    rhs!(out, term, u, layout, row, t)

Add the [`EquationTerm`](@ref) `term` to `out` in the rows of the unknown with id `row`,
as a pure function of the state `u` and the time `t`. A `:sparse` term (see
[`coupling`](@ref)) must accept state vectors of dual numbers, since its Jacobian comes
from automatic differentiation.
"""
rhs!

"""
    sparsity!(entries, term, layout, row)

Push the `(row index, column index)` pairs of the structural nonzeros of the Jacobian
block of a `:sparse` [`EquationTerm`](@ref) (see [`coupling`](@ref)) to `entries`.
"""
sparsity!

"""
    jacobian_vector_product!(out, term, u, v, layout, row, t)

Add the product `(∂ rhs_term / ∂u) v` of a `:matrix_free` [`EquationTerm`](@ref) (see
[`coupling`](@ref)) to `out`.
"""
jacobian_vector_product!
validate(::EquationTerm, layout, row) = nothing
# with the initial time of the system, for terms that evaluate a function of time
validate(term::EquationTerm, layout, row, t) = validate(term, layout, row)

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
    jacobian_assembly(term)

How the stored Jacobian block of a `:sparse` term is filled: `:automatic` (default) by
colored forward-mode AD of `rhs!`, or `:direct` by the term's own [`add_jacobian!`](@ref).

The colored AD needs one pass of the sparse terms per color, and two columns that share a
row need different colors. A row that holds every cell of a radius (the density or current
moment of a distribution, the loss through the edge) would therefore take one color per
cell. Such terms assemble their block directly, and the coloring sees the other terms
alone. A direct term is tested against the AD of its `rhs!`.
"""
jacobian_assembly(::EquationTerm) = :automatic

"""
    prepare_step!(term, u_old, layout, row, t)

Called with the state at the start of the step before every evaluation of the residual
([`residual!`](@ref)), and with the state itself in the ODE form ([`rhs!`](@ref)). A term
with a part lagged to the previous time level inside an implicit term (the knock-on term
with `treatment = :explicit_primary`) evaluates that part here. Nothing by default.
"""
prepare_step!(::EquationTerm, u_old, layout, row, t) = nothing

"""
    add_jacobian!(J, term, u, layout, row, t)

Add the entries `∂ rhs_term / ∂u` of a term with [`jacobian_assembly`](@ref)`(term) ===
:direct` to the stored matrix `J` at the state `u` and the time `t`. The entries lie in
the pattern of `sparsity!`. Allocation free.
"""
function add_jacobian! end

# add `value` to the stored entry (row, column) of J
@inline function _add_entry!(J::SparseMatrixCSC, row::Integer, column::Integer, value)
    range = nzrange(J, column)
    rows = rowvals(J)
    k = searchsortedfirst(rows, row, first(range), last(range), Base.Order.Forward)
    (k <= last(range) && rows[k] == row) ||
        throw(ArgumentError("the entry ($row, $column) is not in the pattern of the stored Jacobian"))
    nonzeros(J)[k] += value
    return J
end

"""
    LayoutKineticArray

3D array view `f[i, j, ir]` of a kinetic unknown with the same cell counts at all radii in
a state vector, so that operators written for 3D arrays apply to the state directly.
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
have the same cell counts at all radii.
"""
function kinetic_array(u::AbstractVector, layout::UnknownLayout, id::Integer)
    grid = kinetic_grid(layout, id)
    has_same_cell_counts(grid) ||
        throw(ArgumentError("$(layout.unknowns[id].name) has different cell counts at different radii"))
    np, nxi = layout.shapes[id, 1]
    return LayoutKineticArray(u, layout.radius_offsets[id], np, nxi)
end

"""
    LayoutRadialValues

The values of a radial unknown with one component over the radii, as a vector view of a
state vector: `values[ir]` is the value at radius index `ir`. Operators take the electric
field of the state in this form.
"""
struct LayoutRadialValues{T,V<:AbstractVector{T},O<:AbstractVector{Int}} <: AbstractVector{T}
    state::V
    offsets::O   # per radius: index before the entry
end

Base.size(a::LayoutRadialValues) = (length(a.offsets),)
Base.@propagate_inbounds Base.getindex(a::LayoutRadialValues, ir::Int) = a.state[a.offsets[ir] + 1]
Base.@propagate_inbounds Base.setindex!(a::LayoutRadialValues, x, ir::Int) = (a.state[a.offsets[ir] + 1] = x)

"""
    radial_values(u, layout, id)

[`LayoutRadialValues`](@ref) of the radial unknown `id`, which has one component, in the
state `u`.
"""
function radial_values(u::AbstractVector, layout::UnknownLayout, id::Integer)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a radial unknown with one component"))
    return LayoutRadialValues(u, layout.radius_offsets[id])
end

# the electric field of the state for a term: nothing when the term has none as an unknown
_field_values(::Nothing, u, layout) = nothing
_field_values(name::Symbol, u, layout) = radial_values(u, layout, unknown_id(layout, name))

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
    FokkerPlanckTerm(unknown, operator::AdvectionDiffusion; jacobian=:exact, electric_field=nothing,
                     plasma=nothing, settings=nothing, synchrotron=nothing,
                     coefficients=:implicit, lagged=())

Advection-diffusion (Fokker-Planck) operator applied to the kinetic unknown `unknown`,
added to the equation of the row unknown (normally the same). Stored sparse, with the
3×3 `(p, ξ)` stencil, the second neighbours of the four-point schemes and the radial
neighbours: the cell of the same pitch at each neighbouring radius, or with different
pitch grids the cells that overlap it. Needs the same cell counts and p faces at all radii.

`electric_field` names the radial unknown of the electric field [V/m] when the field is an
unknown of the system. The advection of every face is then that of the
operator, which holds the collisions, plus the field of the radius times the coefficients
of a unit field ([`set_unit_field_coefficients!`](@ref)), inside the scheme. The operator
must not hold the field of a prescribed plasma as well
([`fokker_planck_operator`](@ref)`(...; electric_field = false)`). The term is bilinear in
the distribution and the field, and its stored block holds the column of the field of each
radius, which is DREAM's Jacobian of its `ElectricFieldTerm`.

`plasma` is the source of the plasma that the collision coefficients follow: a
[`PlasmaSource`](@ref), a [`PlasmaState`](@ref) or a function of time that returns
one ([`PrescribedPlasma`](@ref)). The coefficients of `operator` are rebuilt from it with
`settings` and `synchrotron` ([`set_fokker_planck_coefficients!`](@ref)), and the scheme and
the boundary conditions of `operator` are kept. Without them the term takes those that
built `operator` ([`fokker_planck_operator`](@ref)), and `CollisionSettings()` with the
synchrotron loss for an operator built otherwise, which is on by default (decision of the
author, 2026-10-04). The rebuild happens when the plasma values of any radius differ from
those of the last rebuild ([`update_key!`](@ref)), not at every new time, and rebuilds all
radii. Without `plasma` the coefficients of `operator` are fixed.
[`PrescribedFokkerPlanckTerm`](@ref) builds such a term from a function of time.

With a plasma that reads unknowns ([`PlasmaFromUnknowns`](@ref)) the term depends on them
and is not linear. `coefficients` selects the state its coefficients are built from:

- `:implicit`: the state of the Newton iterate, as DREAM does. The stored Jacobian holds
  the columns of the plasma unknowns ([`add_plasma_columns!`](@ref)) next to the block in
  `f`.
- `:lagged`: the state at the start of the step, set in [`prepare_step!`](@ref), with the
  prescribed part of the plasma at the time of the evaluation. The term is linear in `f`
  within a step and its block is exact. In the ODE form ([`rhs!`](@ref) of the system, the
  SciML interface) `prepare_step!` sees the state itself, so a lagged term is implicit
  there.

`lagged = (:ion_densities,)` lags one group of plasma scalars alone: with
`coefficients = :implicit` the ion densities of a plasma that
reads them come from the start of the step, and the temperature and the density from the
iterate. The stored Jacobian then has the columns of the temperature and the density and
none of the charge states, and the residual is consistent within the step. DREAM's
reference case drops these columns from its Jacobian alone (`enableIonJacobian(False)`).
Without the lag the columns of the charge states are part of the plasma columns.

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
struct FokkerPlanckTerm{T,P} <: EquationTerm
    unknown::Symbol
    operator::AdvectionDiffusion{T,T}
    jacobian::Symbol
    electric_field::Union{Nothing,Symbol}    # the unknown of the field, when it is one
    stored_operator::AdvectionDiffusion{T,T}   # the operator whose (frozen) derivative is the stored block
    values::Array{T,3}                       # the iterate's values (frozen limiter, corrections)
    correction::Array{T,3}
    dual_input::Array{ForwardDiff.Dual{JacobianCorrectionTag,T,1},3}
    dual_output::Array{ForwardDiff.Dual{JacobianCorrectionTag,T,1},3}
    field_values::Vector{T}                  # the iterate's field (corrections)
    dual_field::Vector{ForwardDiff.Dual{JacobianCorrectionTag,T,1}}
    plasma::P                                # nothing, or the PlasmaSource of the coefficients
    settings::CollisionSettings
    synchrotron::Bool
    plasma_key::PlasmaKey                    # the plasma values of the last rebuild
    rebuilds::Base.RefValue{Int}             # the number of rebuilds, for the caches that read the operator
    coefficients::Symbol                     # :implicit or :lagged, for a plasma that reads unknowns
    dual_operator::Base.RefValue{Any}        # the coefficients in dual numbers, for the plasma columns
    dual_columns::Base.RefValue{Any}         # the term in dual numbers, for the plasma columns
    lagged::Tuple{Vararg{Symbol}}            # the groups of plasma scalars taken at the start of the step
    ion_state::Vector{Float64}               # the state at the start of the step, for lagged ions
end

const LINEAR_JACOBIAN_SCHEMES = (:upwind, :central, :exponential_fitting, :quick, :upwind_second_order)

function FokkerPlanckTerm(unknown::Symbol, operator::AdvectionDiffusion{T}; jacobian::Symbol=:exact,
                          electric_field::Union{Nothing,Symbol}=nothing, plasma=nothing,
                          settings::Union{Nothing,CollisionSettings}=nothing, synchrotron::Union{Nothing,Bool}=nothing,
                          coefficients::Symbol=:implicit, lagged=()) where {T}
    coefficients in (:implicit, :lagged) ||
        throw(ArgumentError("coefficients must be :implicit or :lagged, not $coefficients"))
    lagged = Tuple(Symbol(name) for name in lagged)
    # the settings that built the operator, before the field branch below forgets them: a
    # rebuild with other settings would change the coefficients at the first refresh
    built = operator.coefficient_source[]
    settings === nothing && (settings = built === nothing ? CollisionSettings() : built.settings)
    synchrotron === nothing && (synchrotron = built === nothing ? true : built.synchrotron)
    plasma_from = plasma === nothing ? nothing : plasma_source(plasma)
    # the rebuild of the coefficients leaves the field out when it is an unknown
    plasma_from !== nothing && electric_field !== nothing && (operator.coefficient_source[] = nothing)
    if electric_field !== nothing
        source = operator.coefficient_source[]
        source !== nothing && get(source, :electric_field, true) && any(!iszero, source.plasma.electric_field) &&
            throw(ArgumentError("the operator holds the electric field of its plasma, and the term would add the " *
                                "unknown $electric_field to it: build the operator with electric_field = false"))
        set_unit_field_coefficients!(operator)
    end
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
    radii = electric_field === nothing ? 0 : radial_cell_count(operator.grid)
    return FokkerPlanckTerm{T,typeof(plasma_from)}(unknown, operator, jacobian, electric_field, stored, zeros(T, dims),
                                              zeros(T, dims), zeros(D, dims), zeros(D, dims), zeros(T, radii),
                                              zeros(D, radii), plasma_from, settings, synchrotron, PlasmaKey(), Ref(0),
                                              coefficients, Ref{Any}(nothing), Ref{Any}(nothing), lagged, Float64[])
end

lags_in_ode_form(term::FokkerPlanckTerm) = _is_lagged(term) || _lags_ions(term)

# whether the term takes the ion densities of its plasma from the start of the step
_lags_ions(term::FokkerPlanckTerm) = ion_dependency(term.plasma) !== nothing && ion_dependency(term.plasma) in term.lagged
# the state of the ion densities of the plasma of a term
_ion_state(term::FokkerPlanckTerm, u) = _lags_ions(term) && !isempty(term.ion_state) ? term.ion_state : u
# the plasma of the term at the state u, with lagged ions from the start of the step
function _fokker_planck_plasma(term::FokkerPlanckTerm, u, layout, t)
    state = _ion_state(term, u)
    return state === u ? plasma_values(term.plasma, u, layout, t) : plasma_values(term.plasma, u, layout, t; ion_state=state)
end

"""
    refresh_coefficients!(term, u, layout, t)

Bring the coefficients of a term up to date with its plasma at the state `u` and the time
`t`, and return the operator that holds them. A [`FokkerPlanckTerm`](@ref) with a plasma
source rebuilds its operator when the plasma values of any radius changed since the last
rebuild. A term without a plasma source returns its operator unchanged, and so does a term
with `coefficients = :lagged`, which [`prepare_step!`](@ref) refreshes. `u` and `layout`
may be `nothing` for a prescribed plasma.
"""
function refresh_coefficients!(term::FokkerPlanckTerm, u, layout, t)
    _is_lagged(term) && return term.operator
    return _rebuild_coefficients!(term, u, layout, t)
end

_is_lagged(term::FokkerPlanckTerm) = term.coefficients === :lagged && is_state_dependent(term.plasma)

# the coefficients of the plasma at the state u and the time t
function _rebuild_coefficients!(term::FokkerPlanckTerm, u, layout, t)
    term.plasma === nothing && return term.operator
    field = term.electric_field === nothing
    plasma = changed_plasma!(term.plasma_key, term.plasma, u, layout, t; electric_field=field,
                             ion_state=_ion_state(term, u))
    plasma === nothing && return term.operator
    try
        set_fokker_planck_coefficients!(term.operator, plasma, term.settings; synchrotron=term.synchrotron,
                                        electric_field=field)
    catch
        reset_key!(term.plasma_key)
        rethrow()
    end
    term.rebuilds[] += 1
    return term.operator
end

"""
    has_plasma_source(term)

Whether the coefficients of a term follow a plasma source ([`refresh_coefficients!`](@ref)).
"""
has_plasma_source(term::FokkerPlanckTerm) = term.plasma !== nothing
has_plasma_source(::Any) = false

# a lagged term takes the plasma of the state at the start of the step
function prepare_step!(term::FokkerPlanckTerm, u_old, layout, row, t)
    if _lags_ions(term)
        resize!(term.ion_state, length(u_old))
        term.ion_state .= _value.(u_old)
    end
    _is_lagged(term) && _rebuild_coefficients!(term, u_old, layout, t)
    return nothing
end

"""
    lags_in_ode_form(term)

Whether a term takes a part of its residual at the state of [`prepare_step!`](@ref), the
start of the step: a knock-on term with a lagged primary, a Fokker-Planck term with lagged
coefficients or lagged ions, the energy loss of such a term, a field-particle term whose
plasma reads unknowns. In the ODE form ([`rhs!`](@ref) of the system) that state is the
state itself, so the stored Jacobian, which treats the lagged part as fixed, is not the
Jacobian of the ODE, and `ode_function` passes none.
"""
lags_in_ode_form(::Any) = false

"""
    has_plasma_columns(term)

Whether the stored Jacobian holds columns of the plasma unknowns of a term: the derivative
of the term with respect to the temperature and the density of each radius, which the
colored AD of the term does not see, since its coefficients hold numbers. Such a term lists
them with [`plasma_column_sparsity!`](@ref) and adds them with
[`add_plasma_columns!`](@ref). True for a [`FokkerPlanckTerm`](@ref) whose plasma reads
unknowns and whose coefficients are implicit.
"""
has_plasma_columns(::Any) = false
has_plasma_columns(term::FokkerPlanckTerm) = is_state_dependent(term.plasma) && !_is_lagged(term)

"""
    plasma_column_sparsity!(entries, term, layout, row)

Push the entries of the plasma columns of a term with [`has_plasma_columns`](@ref): every
cell of the row unknown at a radius against each plasma unknown of that radius.
"""
function plasma_column_sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    for ir in 1:radial_cell_count(layout), column in plasma_columns(term.plasma, layout, ir; ions=!_lags_ions(term))
        append!(entries, ((r, column) for r in unknown_range(layout, row, ir)))
    end
    return entries
end

"""
    add_plasma_columns!(J, term, u, layout, row, t)

Add the plasma columns of a term with [`has_plasma_columns`](@ref) to `J` at the state `u`
and the time `t`. A [`FokkerPlanckTerm`](@ref) builds its coefficients from the plasma with
dual numbers in the temperature and the density of every radius ([`dual_plasma`](@ref)) and
applies them to `f`: the partials are the derivatives of the residual through the scheme's
own face weights, the Péclet number of exponential fitting included. The scheme is the
operator's own, as is the residual, whatever `jacobian` selects for the block in `f`.
"""
function add_plasma_columns!(J, term::FokkerPlanckTerm, u, layout, row, t)
    plasma = _fokker_planck_plasma(term, u, layout, t)
    with_ions = ion_dependency(term.plasma) !== nothing && !_lags_ions(term)
    dual = dual_plasma(plasma, term.plasma.density !== nothing, with_ions)
    D = eltype(dual.cold_electron_temperature)
    op = _dual_operator!(term, D)
    with_field = term.electric_field === nothing
    set_fokker_planck_coefficients!(op, dual, term.settings; synchrotron=term.synchrotron, electric_field=with_field)
    id = unknown_id(layout, term.unknown)
    f = kinetic_array(u, layout, id)
    df = term.dual_columns[]::Array{D,3}
    fill!(df, zero(D))
    apply!(df, op, f; electric_field=_field_values(term.electric_field, u, layout))
    np, nxi, nr = size(df)
    for ir in 1:nr
        columns = plasma_columns(term.plasma, layout, ir; ions=with_ions)
        for j in 1:nxi, i in 1:np
            r = _kinetic_index(layout, row, ir, i, j)
            partials = ForwardDiff.partials(df[i, j, ir])
            for k in eachindex(columns)
                _add_entry!(J, r, columns[k], partials[k])
            end
        end
    end
    return J
end

# the operator of dual coefficients of a term, made at the first use
function _dual_operator!(term::FokkerPlanckTerm, ::Type{D}) where {D}
    op = term.dual_operator[]
    if !(op isa AdvectionDiffusion{<:Any,D})
        op = similar_coefficients(term.operator, D)
        # the coefficients of a unit field do not depend on the plasma
        term.electric_field === nothing || set_unit_field_coefficients!(op)
        term.dual_operator[] = op
        term.dual_columns[] = zeros(D, size(term.operator.grid))
    end
    return op::AdvectionDiffusion{<:Any,D}
end

# a frozen limiter of a linear scheme is the exact derivative; another scheme's Jacobian
# is exact only when it is the operator's own scheme
function _jacobian_kind(jacobian, operator, stored)
    jacobian === :exact && return :exact
    jacobian === :frozen_limiter && return is_linear(operator) ? :exact : :approximate
    return (stored.p_scheme, stored.xi_scheme) == (operator.p_scheme, operator.xi_scheme) ? :exact : :approximate
end

dependencies(term::FokkerPlanckTerm) =
    unique!([term.unknown; term.electric_field === nothing ? Symbol[] : [term.electric_field];
             collect(Symbol, source_dependencies(term.plasma))])
# the kind of the stored block in f: another scheme's Jacobian or a frozen limiter
_scheme_jacobian_kind(term::FokkerPlanckTerm) = _jacobian_kind(term.jacobian, term.operator, term.stored_operator)
# the columns of the plasma unknowns are exact for the operator's own scheme, so the kind
# is that of the block in f
jacobian_kind(term::FokkerPlanckTerm) = _scheme_jacobian_kind(term)
# with the field as an unknown the term is bilinear in the distribution and the field; with a
# plasma from the unknowns the coefficients depend on the state
is_linear(term::FokkerPlanckTerm) =
    term.electric_field === nothing && !is_state_dependent(term.plasma) && is_linear(term.operator)

validate(term::FokkerPlanckTerm, layout, row) = _check_fokker_planck(term, layout, row, 0.0)

# the checks of the grid, the field and the plasma unknowns; the composition of the ions is
# read at the time `t`, the start of the run, where the prescribed part is defined
function _check_fokker_planck(term::FokkerPlanckTerm, layout, row, t)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("FokkerPlanckTerm acts on and adds to kinetic unknowns"))
    grid = term.operator.grid
    for k in (id, row), ir in 1:radial_cell_count(layout)
        layout.shapes[k, ir] == (momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) ||
            throw(DimensionMismatch("the operator grid does not match $(layout.unknowns[k].name)"))
    end
    if term.electric_field !== nothing
        field = unknown_id(layout, term.electric_field)
        layout.kinds[field] === :radial && layout.sizes[field, 1] == 1 ||
            throw(ArgumentError("the electric field of a FokkerPlanckTerm is a radial unknown with one component"))
    end
    for name in term.lagged
        name === ion_dependency(term.plasma) ||
            throw(ArgumentError("the FokkerPlanckTerm of $(term.unknown) lags $name, which is not the ion densities " *
                                "of its plasma"))
    end
    for name in source_dependencies(term.plasma)
        id = unknown_id(layout, name)
        if name === ion_dependency(term.plasma)
            _check_ion_unknown(layout, id, IonComposition(plasma_values(term.plasma.prescribed, nothing, nothing, t)))
            continue
        end
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("the plasma unknown $name of a FokkerPlanckTerm is a radial unknown with one component"))
    end
    return nothing
end

# The target of the e-e collisions in a system that holds the cold density as an unknown,
# the closure n_cold = n_free − n_hot − n_re: a plasma that keeps the density
# of its prescribed part, or a prescribed plasma in the fully kinetic mode whose cold
# density is the free-electron default, would count the electrons of the grids twice, once
# in the target and once as f.
function validate(term::FokkerPlanckTerm, layout, row, t)
    _check_fokker_planck(term, layout, row, t)
    haskey(layout.ids, :cold_electron_density) || return nothing
    source = term.plasma
    if source isa PlasmaFromUnknowns
        source.density === nothing &&
            throw(ArgumentError("the plasma of the FokkerPlanckTerm of $(term.unknown) keeps the density of its " *
                                "prescribed part (density = nothing), while the system has the unknown " *
                                "cold_electron_density: read it with density = :cold_electron_density (issue #154)"))
    elseif source isa PrescribedPlasma && term.settings.mode === :full
        plasma_values(source, nothing, nothing, t).cold_density_given ||
            throw(ArgumentError("the prescribed plasma of the FokkerPlanckTerm of $(term.unknown) takes the " *
                                "free-electron density as the cold density, which counts the electrons of the " *
                                "grids twice in a system with the unknown cold_electron_density: give the cold " *
                                "density, or read it from the unknown (issue #154)"))
    end
    return nothing
end

function rhs!(out, term::FokkerPlanckTerm, u, layout, row, t)
    refresh_coefficients!(term, u, layout, t)
    id = unknown_id(layout, term.unknown)
    field = _field_values(term.electric_field, u, layout)
    if _equally_spaced(layout.radius_offsets[row]) && _equally_spaced(layout.radius_offsets[id])
        # equally spaced radius blocks (the usual case) index by arithmetic instead of a load
        np, nxi = layout.shapes[id, 1]
        _fokker_planck!(LayoutKineticArray(out, _offset_range(layout, row), np, nxi), term,
                        LayoutKineticArray(u, _offset_range(layout, id), np, nxi), field)
    else
        _fokker_planck!(kinetic_array(out, layout, row), term, kinetic_array(u, layout, id), field)
    end
    return out
end

# the field without its derivative parts, in the buffer of the term
_plain_field!(term::FokkerPlanckTerm, ::Nothing) = nothing
function _plain_field!(term::FokkerPlanckTerm, field)
    for ir in eachindex(term.field_values)
        term.field_values[ir] = _value(field[ir])
    end
    return term.field_values
end

# the operator on `f` into `df`, with the stored block's derivative for a dual state
function _fokker_planck!(df, term::FokkerPlanckTerm, f, field=nothing)
    if _scheme_jacobian_kind(term) === :exact || !(eltype(f) <: ForwardDiff.Dual)
        apply!(df, term.operator, f; electric_field=field)
    elseif term.jacobian === :frozen_limiter
        # a dual state (Jacobian assembly): the limiter weights at the values, applied as
        # constants; the value part is the full residual, since the weights at f are f's own
        term.values .= _value.(f)
        apply!(df, term.operator, f; limiter_state=term.values, electric_field=field)
    else
        # the derivative of the other scheme, with the value part corrected to the
        # operator's own residual (constant sources cancel in the difference)
        apply!(df, term.stored_operator, f; electric_field=field)
        term.values .= _value.(f)
        plain = _plain_field!(term, field)
        fill!(term.correction, zero(eltype(term.correction)))
        apply!(term.correction, term.operator, term.values; electric_field=plain)
        apply!(term.correction, term.stored_operator, term.values; scale=-1, electric_field=plain)
        df .+= term.correction
    end
    return df
end

function sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    op = term.stored_operator
    np, nxi, nr = size(op.grid)
    widths = (p_width=stencil_half_width(op.p_scheme), xi_width=stencil_half_width(op.xi_scheme),
              r_width=stencil_half_width(op.r_scheme), p_upper_boundary=op.p_upper_boundary)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        r = _kinetic_index(layout, row, ir, i, j)
        _visit_stencil(op.grid, i, j, ir; widths...) do i2, j2, ir2
            push!(entries, (r, _kinetic_index(layout, id, ir2, i2, j2)))
        end
        # the field of the radius acts on every cell of the radius
        term.electric_field === nothing ||
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, term.electric_field), ir))))
    end
    return entries
end

# (J_exact - J_stored) v in one dual pass: the operator's directional derivative along v
# minus the stored operator's (the frozen limiter's, or the other scheme's); allocation
# free with the term's buffers
function jacobian_correction_product!(out, term::FokkerPlanckTerm, u, v, layout, row, t)
    _scheme_jacobian_kind(term) === :approximate || return out
    refresh_coefficients!(term, u, layout, t)
    id = unknown_id(layout, term.unknown)
    f, direction = kinetic_array(u, layout, id), kinetic_array(v, layout, id)
    input, output = term.dual_input, term.dual_output
    D = eltype(input)
    @inbounds for k in CartesianIndices(input)
        input[k] = D(f[k], ForwardDiff.Partials((direction[k],)))
    end
    # the field and its direction, when the field is an unknown
    field = nothing
    if term.electric_field !== nothing
        values, change = _field_values(term.electric_field, u, layout), _field_values(term.electric_field, v, layout)
        for ir in eachindex(term.dual_field)
            term.dual_field[ir] = D(values[ir], ForwardDiff.Partials((change[ir],)))
        end
        field = term.dual_field
    end
    fill!(output, zero(D))
    apply!(output, term.operator, input; electric_field=field)
    if term.jacobian === :frozen_limiter
        term.values .= f
        apply!(output, term.operator, input; scale=-1, limiter_state=term.values, electric_field=field)
    else
        apply!(output, term.stored_operator, input; scale=-1, electric_field=field)
    end
    out_view = kinetic_array(out, layout, row)
    @inbounds for k in CartesianIndices(output)
        out_view[k] += ForwardDiff.partials(output[k], 1)
    end
    return out
end

"""
    PrescribedFokkerPlanckTerm(unknown, operator::AdvectionDiffusion, plasma_at;
                               settings=nothing, synchrotron=nothing, jacobian=:exact,
                               electric_field=nothing)

A [`FokkerPlanckTerm`](@ref) whose collision coefficients follow a plasma prescribed in time
(a cooling plasma with `T(t)` prescribed): `plasma_at(t)` returns the [`PlasmaState`](@ref)
at the time `t`. The same as `FokkerPlanckTerm(unknown, operator; plasma = plasma_at,
settings, synchrotron, jacobian, electric_field)`, which this function returns. The
coefficients are rebuilt when the plasma values change ([`refresh_coefficients!`](@ref)).

With `electric_field`, the name of the unknown of the field, the field of the prescribed
plasma is left out of the coefficients, and the term takes the field of the state.
"""
PrescribedFokkerPlanckTerm(unknown::Symbol, operator::AdvectionDiffusion, plasma_at;
                           settings::Union{Nothing,CollisionSettings}=nothing, synchrotron::Union{Nothing,Bool}=nothing,
                           jacobian::Symbol=:exact, electric_field::Union{Nothing,Symbol}=nothing) =
    FokkerPlanckTerm(unknown, operator; jacobian, electric_field, plasma=PrescribedPlasma(plasma_at), settings,
                     synchrotron)

"""
    KnockOnTerm(primary, operator::KnockOnOperator; target_density, treatment=:explicit_primary)

Knock-on source produced by the kinetic unknown `primary` on the kinetic row unknown
(`operator` maps the primary grid to the row unknown's grid, per radius, see
[`apply_at_radius!`](@ref)). `target_density` is the target (total electron) density:
the name of a radial unknown, or a vector with one value per radius. Works with different
momentum grids at different radii.

`treatment`:

- `:explicit_primary` (default, DREAM's Boltzmann operator `MollerBoltzmannOperator`): the
  primary distribution at the previous time level, the target density at the new one. The
  term is then the target density times a source fixed for the step, evaluated once per
  residual from the state at the start of the step ([`prepare_step!`](@ref)), and its
  Jacobian is the column of the target density at each radius. With a prescribed target
  density it is the same as `:explicit`.
- `:explicit`: the whole term at the previous time level, target density included.
- `:implicit`: the whole term at the new time level, matrix-free, since it couples all cells
  of a radius. Every Krylov iteration of [`GMRES`](@ref) applies the operator once, and the
  direct solver does not take it.

The explicit forms are the cheaper step. The implicit form is the more accurate one per
step: on DREAM's avalanche example the implicit term reaches the accuracy of
the explicit one at about four times the step (+8.6 % at 8 ms against −8.7 % at 2 ms in the
density at 0.2 s). Then its step cost 3.6 times the explicit one, which cancelled the gain.
With [`FluidSchurLU`](@ref) it costs 1.86 times (production grid, 20 radii), so
where the avalanche growth limits the time step, the implicit term with a longer step can
reach the same accuracy at a lower cost. The time-step control
is the place to choose between them. The explicit default follows DREAM.
"""
struct KnockOnTerm{T,D} <: EquationTerm
    primary::Symbol
    operator::KnockOnOperator{T}
    target_density::D
    treatment::Symbol
    lagged::Vector{Matrix{T}}              # :explicit_primary: the source per unit target density, per radius
    prepared::Base.RefValue{Bool}
    prepared_from::Vector{Matrix{T}}       # the primary of the last preparation, per radius
    preparations::Base.RefValue{Int}       # the applications of the kernel by prepare_step!
end

function KnockOnTerm(primary::Symbol, operator::KnockOnOperator{T}; target_density,
                     treatment::Symbol=:explicit_primary) where {T}
    treatment in (:implicit, :explicit, :explicit_primary) || throw(ArgumentError("unknown treatment $treatment"))
    target_density isa Union{Symbol,AbstractVector} ||
        throw(ArgumentError("target_density must be an unknown name or a vector"))
    grid = operator.knockon
    lagged = treatment === :explicit_primary && target_density isa Symbol ?
             [zeros(T, momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) for ir in 1:radial_cell_count(grid)] :
             Matrix{T}[]
    prepared_from = [similar(source) for source in lagged]
    return KnockOnTerm(primary, operator, target_density, treatment, lagged, Ref(false), prepared_from, Ref(0))
end

# the primary lagged and the target density an unknown: the term is the target density
# times a source fixed for the step
_lagged_primary(term::KnockOnTerm) = term.treatment === :explicit_primary && term.target_density isa Symbol
lags_in_ode_form(term::KnockOnTerm) = _lagged_primary(term)

# linear in the primary with a prescribed target density; bilinear when it is an unknown
is_linear(term::KnockOnTerm) = _lagged_primary(term) || !(term.target_density isa Symbol)
dependencies(term::KnockOnTerm) =
    _lagged_primary(term) ? [term.target_density] :
    term.target_density isa Symbol ? [term.primary, term.target_density] : [term.primary]
coupling(term::KnockOnTerm) = _lagged_primary(term) ? :sparse : :matrix_free
jacobian_assembly(term::KnockOnTerm) = _lagged_primary(term) ? :direct : :automatic
treatment(term::KnockOnTerm) =
    term.treatment === :explicit_primary ? (term.target_density isa Symbol ? :implicit : :explicit) : term.treatment

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

# The kernel is O(N_p²) per radius: it is applied once per start state, not at every
# residual of the Newton iteration and its line search. The primary of the last
# preparation is kept, and a residual at the same start state reuses the source.
function prepare_step!(term::KnockOnTerm, u_old, layout, row, t)
    _lagged_primary(term) || return nothing
    primary = unknown_id(layout, term.primary)
    if term.prepared[]
        same = true
        for ir in 1:radial_cell_count(layout)
            _same_primary(term.prepared_from[ir], kinetic_view(u_old, layout, primary, ir)) || (same = false; break)
        end
        same && return nothing
    end
    for ir in 1:radial_cell_count(layout)
        source = term.lagged[ir]
        f = kinetic_view(u_old, layout, primary, ir)
        fill!(source, zero(eltype(source)))
        apply_at_radius!(source, term.operator, f, one(eltype(source)), ir)
        term.prepared_from[ir] .= _value.(f)
    end
    term.preparations[] += 1
    term.prepared[] = true
    return nothing
end

function _same_primary(stored, f)
    @inbounds for k in eachindex(stored, f)
        isequal(stored[k], _value(f[k])) || return false
    end
    return true
end

function rhs!(out, term::KnockOnTerm, u, layout, row, t)
    if _lagged_primary(term)
        term.prepared[] ||
            throw(ArgumentError("the knock-on source of the previous time level is not set: it is set by residual! " *
                                "and by the right-hand side of the system (prepare_step!)"))
        density = unknown_id(layout, term.target_density)
        for ir in 1:radial_cell_count(layout)
            added = kinetic_view(out, layout, row, ir)
            source = term.lagged[ir]
            n = radial_view(u, layout, density, ir)[1]
            @inbounds for k in eachindex(source)
                added[k] += n * source[k]
            end
        end
        return out
    end
    primary = unknown_id(layout, term.primary)
    for ir in 1:radial_cell_count(layout)
        apply_at_radius!(kinetic_view(out, layout, row, ir), term.operator,
                         kinetic_view(u, layout, primary, ir), _target_density(term, u, layout, ir), ir)
    end
    return out
end

# :explicit_primary: the column of the target density at each radius
# (zero before the first residual!: the constructor of the system fills its prototype here)
function add_jacobian!(J, term::KnockOnTerm, u, layout, row, t)
    density = unknown_id(layout, term.target_density)
    for ir in 1:radial_cell_count(layout)
        column = first(unknown_range(layout, density, ir))
        source = term.lagged[ir]
        for (k, r) in enumerate(unknown_range(layout, row, ir))
            _add_entry!(J, r, column, source[k])
        end
    end
    return J
end

function sparsity!(entries, term::KnockOnTerm, layout, row)
    density = unknown_id(layout, term.target_density)
    for ir in 1:radial_cell_count(layout)
        column = first(unknown_range(layout, density, ir))
        append!(entries, ((r, column) for r in unknown_range(layout, row, ir)))
    end
    return entries
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

# tag of the dual numbers of the local columns of a term
struct LocalColumnTag end

"""
    add_local_jacobian!(J, term, u, layout, row, t)

Add the Jacobian of a term whose rows at a radius read only the radial unknowns of that
radius (its [`dependencies`](@ref)) to `J`, by one evaluation of its `rhs!` with dual numbers:
partial `k` is seeded on the `k`-th component of the dependencies at every radius at once,
so that the partials of a row at radius `ir` are its derivatives with respect to the
unknowns at `ir`. A term with a row that reads all charge states of a radius (the radiated
power, the conductivity, the energy exchange with the ions) assembles its
block this way, so that the coloring of the automatic Jacobian does not need a color per
charge state. The tag is fixed, so the code compiles once per number of local columns.
"""
function add_local_jacobian!(J, term::EquationTerm, u, layout, row, t)
    ids = [unknown_id(layout, name) for name in dependencies(term)]
    all(id -> layout.kinds[id] === :radial, ids) ||
        throw(ArgumentError("the local Jacobian of a $(nameof(typeof(term))) needs radial unknowns alone"))
    count = sum(id -> layout.sizes[id, 1], ids)
    return _add_local_jacobian!(J, term, u, layout, row, t, ids, Val(count))
end

function _add_local_jacobian!(J, term, u, layout, row, t, ids, ::Val{K}) where {K}
    D = ForwardDiff.Dual{LocalColumnTag,Float64,K}
    seeded = D.(u)
    for ir in 1:radial_cell_count(layout)
        k = 0
        for id in ids, i in unknown_range(layout, id, ir)
            k += 1
            seeded[i] = D(u[i], ForwardDiff.Partials(ntuple(m -> Float64(m == k), Val(K))))
        end
    end
    out = zeros(D, length(u))
    rhs!(out, term, seeded, layout, row, t)
    for ir in 1:radial_cell_count(layout)
        columns = [i for id in ids for i in unknown_range(layout, id, ir)]
        for r in unknown_range(layout, row, ir)
            partials = ForwardDiff.partials(out[r])
            for k in 1:K
                _add_entry!(J, r, columns[k], partials[k])
            end
        end
    end
    return J
end

"""
    LinearTerm(unknown; coefficient=1)

`coefficient` times the unknown `unknown`, added entry by entry to the row unknown, which
must have the same kind and sizes (e.g. `-1` times an unknown in its own algebraic
equation, or a decay rate). `coefficient` is a number, or one number per radius for a
radial or kinetic unknown: a conductivity in Ohm's law `0 = σ E − j`. The vector is
referenced, so an update of the run may change its values between the intervals.
"""
struct LinearTerm{C<:Union{Float64,Vector{Float64}}} <: EquationTerm
    unknown::Symbol
    coefficient::C
end
LinearTerm(unknown::Symbol; coefficient::Union{Real,AbstractVector{<:Real}}=1.0) =
    LinearTerm(unknown, coefficient isa Real ? Float64(coefficient) :
                        coefficient isa Vector{Float64} ? coefficient : collect(Float64, coefficient))

dependencies(term::LinearTerm) = [term.unknown]
is_linear(::LinearTerm) = true

function validate(term::LinearTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === layout.kinds[row] && layout.sizes[id, :] == layout.sizes[row, :] ||
        throw(DimensionMismatch("LinearTerm needs an unknown of the same kind and sizes as its row"))
    if term.coefficient isa Vector
        layout.kinds[row] !== :scalar && length(term.coefficient) == radial_cell_count(layout) ||
            throw(DimensionMismatch("a coefficient per radius needs a radial or kinetic unknown and one value " *
                                    "per radius"))
    end
    return nothing
end

Base.@propagate_inbounds _coefficient_at(coefficient::Float64, k) = coefficient
Base.@propagate_inbounds _coefficient_at(coefficient::Vector{Float64}, k) = coefficient[k]

# the (row range, column range) pairs of a LinearTerm: one per radius, or one for scalars
_range_count(layout, row) = layout.kinds[row] === :scalar ? 1 : radial_cell_count(layout)
_paired_ranges(layout, row, id, k) =
    layout.kinds[row] === :scalar ? (unknown_range(layout, row), unknown_range(layout, id)) :
    (unknown_range(layout, row, k), unknown_range(layout, id, k))

function rhs!(out, term::LinearTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for k in 1:_range_count(layout, row)
        rows, columns = _paired_ranges(layout, row, id, k)
        c = _coefficient_at(term.coefficient, k)
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
    PrescribedValueTerm(values)

A prescribed value added to the row of a radial unknown with one component or of a scalar
unknown: a number, one number per radius, or a function of time that returns either. For a
radial unknown the function may also return a callable of r, such as an interpolator:
it is evaluated on the cell centres once per time, and a profile on points of
its own that is constant in time is `t -> profile`. With
`LinearTerm(name; coefficient = -1)` in an algebraic equation it holds the unknown at the
prescribed value, `0 = value(t) - unknown`, which makes a prescribed profile an unknown of
the system: a prescribed electric field `E(r, t)` that the kinetic terms read from the
state.
"""
struct PrescribedValueTerm{V} <: EquationTerm
    values::V
    points::Base.RefValue{Vector{Float64}}                  # cell centres, set by validate
    evaluated::Base.RefValue{Tuple{Float64,Vector{Float64}}} # (time, values) of a profile
end

# The points and the cache belong to the term object: a term shared between systems on
# different radial grids takes the points of the last `validate`, and the cache is not safe
# for concurrent calls of `rhs!` at different times.
PrescribedValueTerm(values) = PrescribedValueTerm(values, Ref(Float64[]), Ref((NaN, Float64[])))

dependencies(::PrescribedValueTerm) = Symbol[]
is_linear(::PrescribedValueTerm) = true
sparsity!(entries, ::PrescribedValueTerm, layout, row) = entries

_prescribed_values(values, t) = values
_prescribed_values(values::Function, t) = values(t)

function _prescribed_values(term::PrescribedValueTerm, t)
    values = _prescribed_values(term.values, t)
    _is_radial_profile(values) || return values
    cached_time, cached = term.evaluated[]
    cached_time == t && return cached
    evaluated = collect(Float64, radial_values(values, term.points[]))
    term.evaluated[] = (Float64(t), evaluated)
    return evaluated
end

function validate(term::PrescribedValueTerm, layout, row)
    kind = layout.kinds[row]
    kind === :scalar || (kind === :radial && layout.sizes[row, 1] == 1) ||
        throw(ArgumentError("PrescribedValueTerm adds to a scalar unknown or to a radial unknown with one component"))
    kind === :radial && (term.points[] = collect(Float64, layout.radial_grid.r_centers))
    term.values isa Function && return nothing
    term.values isa Real || (term.values isa AbstractVector{<:Real} && kind === :radial &&
                             length(term.values) == radial_cell_count(layout)) ||
        throw(ArgumentError("the values of a PrescribedValueTerm are a number, one number per radius of a radial " *
                            "unknown, or a function of time"))
    return nothing
end

function rhs!(out, term::PrescribedValueTerm, u, layout, row, t)
    values = _prescribed_values(term, t)
    if layout.kinds[row] === :scalar
        out[first(unknown_range(layout, row))] += values
    else
        for ir in 1:radial_cell_count(layout)
            out[first(unknown_range(layout, row, ir))] += values isa Real ? values : values[ir]
        end
    end
    return out
end

"""
    DensityMomentTerm(kinetic; coefficient=1)
    DensityMomentTerm(kinetic, grid; coefficient=1, region=nothing)

`coefficient` times the density `n(r) = (1/V') Σ 𝒱' Δp Δξ f` of the kinetic unknown
`kinetic`, added to a radial row unknown with one component (e.g. a hot-electron density
defined by an algebraic equation). Works with different momentum grids at different radii.

With `region`, a [`MomentumRegion`](@ref), the moment covers that part of the momentum
grid: the hot electrons above the threshold of the hot region. The limit
of the region at the time of the evaluation counts, and `grid` is the grid of the unknown.
The limit is prescribed data: the term is linear in the distribution.
"""
struct DensityMomentTerm{E} <: EquationTerm
    kinetic::Symbol
    coefficient::Float64
    envelopes::E    # nothing, or the envelopes of the region per radius
end
DensityMomentTerm(kinetic::Symbol; coefficient::Real=1.0) = DensityMomentTerm(kinetic, Float64(coefficient), nothing)
DensityMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0,
                  region::Union{Nothing,MomentumRegion}=nothing) =
    DensityMomentTerm(kinetic, Float64(coefficient), RegionEnvelopes(region, grid))

_region_unknowns(envelopes) = envelopes === nothing ? () : region_temperature(envelopes.region)
dependencies(term::DensityMomentTerm) = [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
is_linear(term::DensityMomentTerm) = isempty(_region_unknowns(term.envelopes))

# the envelopes of a term fit the grid of its unknown
_check_envelopes(::Nothing, grid) = nothing
function _check_envelopes(envelopes::RegionEnvelopes, grid::KineticGrid)
    length(envelopes.values) == radial_cell_count(grid) &&
        all(ir -> length(envelopes.values[ir]) == momentum_cell_count(grid, ir), 1:radial_cell_count(grid)) ||
        throw(DimensionMismatch("the momentum region of the term was built for another grid"))
    return nothing
end

function validate(term::DensityMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("DensityMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("DensityMomentTerm adds to a radial unknown with one component"))
    _check_envelopes(term.envelopes, kinetic_grid(layout, id))
    return nothing
end

rhs!(out, term::DensityMomentTerm, u, layout, row, t) =
    _add_density_moment!(out, term, kinetic_grid(layout, unknown_id(layout, term.kinetic)), u, layout, row, t)

function _add_density_moment!(out, term::DensityMomentTerm, grid::KineticGrid, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = _moment_envelope(term.envelopes, grid, ir, u, layout, t)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            n += _envelope_value(envelope, i) * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        out[first(unknown_range(layout, row, ir))] +=
            term.coefficient * n / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::DensityMomentTerm) = :direct

function add_jacobian!(J, term::DensityMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        r = first(unknown_range(layout, row, ir))
        factor = term.coefficient / grid.radial_grid.spatial_jacobian[ir]
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                        factor * _envelope_value(envelope, i) * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j])
        end
    end
    _add_threshold_column!(J, term.envelopes, grid, u, layout, row, id) do ir, i, j
        m = momentum_grid(grid, ir)
        term.coefficient / grid.radial_grid.spatial_jacobian[ir] * phase_space_jacobian(grid, ir)[i, j] *
            m.p_widths[i] * m.xi_widths[j]
    end
    return J
end

sparsity!(entries, term::DensityMomentTerm, layout, row) =
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)

# the row of a moment holds every cell of its radius, and the temperature of the radius when
# the limit of its region follows that unknown
function _moment_sparsity!(entries, kinetic::Symbol, layout, row, envelopes=nothing)
    id = unknown_id(layout, kinetic)
    temperature = envelopes === nothing ? () : region_temperature(envelopes.region)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for s in unknown_range(layout, id, ir)
            push!(entries, (r, s))
        end
        for name in temperature
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, name), ir))))
        end
    end
    return entries
end

# the envelope for the residual of a moment: in the number type of a state with derivative
# parts when the limit follows the temperature, so that the AD of rhs! holds its column
function _moment_envelope(envelopes, grid, ir, u, layout, t)
    if envelopes !== nothing && !isempty(region_temperature(envelopes.region)) && eltype(u) <: ForwardDiff.Dual
        limit = _state_limits(envelopes.region, u, layout)[ir]
        return momentum_envelope!(zeros(typeof(limit), momentum_cell_count(grid, ir)), envelopes.region,
                                  momentum_grid(grid, ir), limit)
    end
    return _envelope_at(envelopes, grid, ir, u, layout, t)
end

# the column of the temperature of a moment over a region that follows it: the derivative of
# the envelope with respect to the temperature times the moment weight `weight(ir, i, j)` of
# each cell and f
function _add_threshold_column!(weight, J, envelopes, grid, u, layout, row, id)
    envelopes === nothing && return J
    temperature = region_temperature(envelopes.region)
    isempty(temperature) && return J
    T_id = unknown_id(layout, only(temperature))
    D = ForwardDiff.Dual{PlasmaColumnTag,Float64,1}
    T = radial_values(u, layout, T_id)
    limit = envelopes.region.limit
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        p0 = limit.factor * sqrt(2 * D(T[ir], ForwardDiff.Partials((1.0,))) / Constants.ELECTRON_REST_ENERGY)
        envelope = momentum_envelope!(zeros(D, momentum_cell_count(m)), envelopes.region, m, p0)
        f = kinetic_view(u, layout, id, ir)
        value = 0.0
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            value += ForwardDiff.partials(envelope[i], 1) * weight(ir, i, j) * f[i, j]
        end
        _add_entry!(J, first(unknown_range(layout, row, ir)), first(unknown_range(layout, T_id, ir)), value)
    end
    return J
end

"""
    CurrentMomentTerm(kinetic, grid; coefficient=1, pitch_rule=:linear, region=nothing)

`coefficient` times the current density of the kinetic unknown `kinetic` on `grid`
([`current_density`](@ref); `pitch_rule = :cell_center` is DREAM's `j_hot` moment), added
to a radial row unknown with one component (the ohmic or hot-electron current defined by
an algebraic equation). The pitch weights are built per radius when the term is created,
so different momentum grids at different radii work; `grid` must be the unknown's grid
(the same radii and momentum faces, checked when the system is built).

With `region`, a [`MomentumRegion`](@ref), the moment covers that part of the momentum
grid, with the limit of the region at the time of the evaluation: the
current of the hot electrons above the threshold of the hot region.
"""
struct CurrentMomentTerm{G<:KineticGrid,E} <: EquationTerm
    kinetic::Symbol
    grid::G
    coefficient::Float64
    pitch_rule::Symbol
    pitch_weights::Vector{Vector{Float64}}   # per radius
    envelopes::E                             # nothing, or the envelopes of the region per radius
end
function CurrentMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0, pitch_rule::Symbol=:linear,
                           region::Union{Nothing,MomentumRegion}=nothing)
    weights = [pitch_moment_weights(momentum_grid(grid, ir); pitch_rule) for ir in 1:radial_cell_count(grid)]
    return CurrentMomentTerm(kinetic, grid, Float64(coefficient), pitch_rule, weights, RegionEnvelopes(region, grid))
end

dependencies(term::CurrentMomentTerm) = [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
is_linear(term::CurrentMomentTerm) = isempty(_region_unknowns(term.envelopes))

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
                            kinetic_view(u, layout, id, ir), _moment_envelope(term.envelopes, grid, ir, u, layout, t))
        out[first(unknown_range(layout, row, ir))] += term.coefficient * j / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::CurrentMomentTerm) = :direct

function add_jacobian!(J, term::CurrentMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    charge = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        weights = term.pitch_weights[ir]
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        r = first(unknown_range(layout, row, ir))
        factor = term.coefficient * charge / grid.radial_grid.spatial_jacobian[ir]
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            p = m.p_centers[i]
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                        factor * _envelope_value(envelope, i) * weights[j] * jacobian[i, j] * m.p_widths[i] * p /
                        sqrt(1 + p^2))
        end
    end
    _add_threshold_column!(J, term.envelopes, grid, u, layout, row, id) do ir, i, j
        m = momentum_grid(grid, ir)
        p = m.p_centers[i]
        term.coefficient * charge / grid.radial_grid.spatial_jacobian[ir] * term.pitch_weights[ir][j] *
            phase_space_jacobian(grid, ir)[i, j] * m.p_widths[i] * p / sqrt(1 + p^2)
    end
    return J
end

sparsity!(entries, term::CurrentMomentTerm, layout, row) =
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)

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
