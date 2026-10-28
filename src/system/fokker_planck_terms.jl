# The Fokker-Planck term of a kinetic unknown: the advection-diffusion operator with its
# collision coefficients, the electric field and the plasma it follows.

"""
    FokkerPlanckTerm(unknown, operator::AdvectionDiffusion; jacobian=:exact, electric_field=nothing,
                     plasma=nothing, settings=nothing, synchrotron=nothing, lagged=:ions)

Advection-diffusion (Fokker-Planck) operator applied to the kinetic unknown `unknown`,
added to the equation of the row unknown (normally the same). Stored sparse, with the
3×3 `(p, ξ)` stencil, the second neighbours of the four-point schemes and the radial
neighbours: the cell of the same pitch at each neighbouring radius, or with different
pitch grids the cells that overlap it. Needs the same cell counts and p faces at all radii.

`electric_field` names the radial unknown of the electric field [V/m] when the field is an
unknown of the system. The advection of every face is then that of the operator, which holds
the collisions, plus the field of the radius times the coefficients of a unit field
([`set_unit_field_coefficients!`](@ref)), inside the scheme. The operator must not hold the
field of a prescribed plasma as well ([`fokker_planck_operator`](@ref)`(...;
field_in_coefficients = false)`). The term is bilinear in the distribution and the field,
and its stored block holds the column of the field of each radius, which is DREAM's Jacobian
of its `ElectricFieldTerm`.

`plasma` is the source of the plasma that the collision coefficients follow: a
[`PlasmaSource`](@ref), a [`PlasmaState`](@ref) or a function of time that returns
one ([`PrescribedPlasma`](@ref)). The coefficients of `operator` are rebuilt from it with
`settings` and `synchrotron` ([`set_fokker_planck_coefficients!`](@ref)), and the scheme and
the boundary conditions of `operator` are kept. Without them the term takes those that
built `operator` ([`fokker_planck_operator`](@ref)), and `CollisionSettings()` with the
synchrotron loss for an operator built otherwise (on by default, a choice of this package;
see the docs page *Choices of this package*). The rebuild happens when the plasma values
of any radius differ from those of the last rebuild ([`update_key!`](@ref)), not at every
new time, and rebuilds all radii. Without `plasma` the coefficients of `operator` are
fixed. [`prescribed_fokker_planck_term`](@ref) builds such a term from a function of time.

With a plasma that reads unknowns ([`PlasmaFromUnknowns`](@ref)) the term depends on them
and is not linear. `lagged` names the part of that plasma taken at the start of the step;
the rest is taken at the state of the Newton iterate, as DREAM does, and the stored
Jacobian holds its columns ([`add_plasma_columns!`](@ref)) next to the block in `f`. With
the ion densities lagged and the temperature and the density implicit, the stored Jacobian
has the columns of the temperature and the density and none of the charge states, and the
residual is consistent within the step. DREAM's disruption example drops these columns
from its Jacobian alone (`enableIonJacobian(False)`, `examples/Disruptions/generate.py:243`).

- `lagged = :all`: the whole plasma at the start of the step, set in
  [`prepare_step!`](@ref), with the prescribed part of the plasma at the time of the
  evaluation. The term is linear in `f` within a step and its block is exact. In the ODE
  form ([`rhs!`](@ref) of the system, the SciML interface) `prepare_step!` sees the state
  itself, so the term is implicit there.
- `lagged = :ions` (the default, a choice of this package; see the docs page *Choices of
  this package*): the ion densities of the plasma when it reads them, no group otherwise.
  A term that lags them is of first order under BDF2, and `ode_function` passes no Jacobian
  for a system with it ([`lags_in_ode_form`](@ref)). The kinetic ionization and a corrected
  conductivity read the charge states of the iterate, so within a step the ions enter the
  collisions of `f` and those terms at two states, a difference of first order in the step.
- `lagged = ()`: nothing lagged; the columns of the charge states are part of the plasma
  columns.
- `lagged = (:ion_densities,)`: the ion densities by name, as `:ions` for that unknown.

`jacobian` selects what the stored Jacobian block holds for a flux-limiter scheme (DREAM's
`ad_jac` setting of `AdvectionInterpolation`). The residual is always
the operator's own, so every mode converges to the same solution:

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
scheme's is still a different matrix (upwind's for a central residual, the approximation
that rescues some DREAM runs at high momentum resolution with central interpolation; see
the docs page *Comparison with DREAM*).
With an approximate block the term declares [`jacobian_kind`](@ref)`(term) === :approximate`: the Newton iteration on the
stored matrix is a quasi-Newton one that converges linearly, or, with
[`GMRES`](@ref)`(operator=:exact)`, the stored block only preconditions the exact Newton
operator ([`jacobian_correction_product!`](@ref), one dual-number pass of the operator
per product).
"""
struct FokkerPlanckTerm{T, P} <: EquationTerm
    unknown::Symbol
    operator::AdvectionDiffusion{T, T}
    jacobian::Symbol
    electric_field::Union{Nothing, Symbol}    # the unknown of the field, when it is one
    stored_operator::AdvectionDiffusion{T, T}   # the operator whose (frozen) derivative is the stored block
    values::Array{T, 3}                       # the iterate's values (frozen limiter, corrections)
    correction::Array{T, 3}
    dual_input::Array{ForwardDiff.Dual{JacobianCorrectionTag, T, 1}, 3}
    dual_output::Array{ForwardDiff.Dual{JacobianCorrectionTag, T, 1}, 3}
    field_values::Vector{T}                  # the iterate's field (corrections)
    dual_field::Vector{ForwardDiff.Dual{JacobianCorrectionTag, T, 1}}
    plasma::P                                # nothing, or the PlasmaSource of the coefficients
    settings::CollisionSettings
    synchrotron::Bool
    plasma_key::PlasmaKey                    # the plasma values of the last rebuild
    rebuilds::Base.RefValue{Int}             # the number of rebuilds, for the caches that read the operator
    lag_all::Bool                            # the whole plasma at the start of the step (lagged = :all)
    dual_operator::Base.RefValue{Any}        # the coefficients in dual numbers, for the plasma columns
    dual_columns::Base.RefValue{Any}         # the term in dual numbers, for the plasma columns
    lagged::Tuple{Vararg{Symbol}}            # the groups of plasma scalars taken at the start of the step
    ion_state::Vector{Float64}               # the state at the start of the step, for lagged ions
end

const LINEAR_JACOBIAN_SCHEMES = (
    :upwind, :central, :exponential_fitting, :quick, :upwind_second_order)

function FokkerPlanckTerm(
        unknown::Symbol, operator::AdvectionDiffusion{T}; jacobian::Symbol = :exact,
        electric_field::Union{Nothing, Symbol} = nothing, plasma = nothing,
        settings::Union{Nothing, CollisionSettings} = nothing, synchrotron::Union{
            Nothing, Bool} = nothing,
        lagged = :ions) where {T}
    lag_all = lagged === :all
    # the settings that built the operator, before the field branch below forgets them: a
    # rebuild with other settings would change the coefficients at the first refresh
    built = operator.coefficient_source[]
    settings === nothing &&
        (settings = built === nothing ? CollisionSettings() : built.settings)
    synchrotron === nothing && (synchrotron = built === nothing ? true : built.synchrotron)
    plasma_from = plasma === nothing ? nothing : plasma_source(plasma)
    lagged = if lag_all
        ()
    elseif lagged === :ions
        ions = ion_dependency(plasma_from)
        ions === nothing ? () : (ions,)
    else
        lagged isa Symbol &&
            throw(ArgumentError("lagged is :all, :ions or a collection of names, not $lagged"))
        Tuple(Symbol(name) for name in lagged)
    end
    # the rebuild of the coefficients leaves the field out when it is an unknown
    plasma_from !== nothing && electric_field !== nothing &&
        (operator.coefficient_source[] = nothing)
    if electric_field !== nothing
        source = operator.coefficient_source[]
        source !== nothing && get(source, :electric_field, true) &&
            any(!iszero, source.plasma.electric_field) &&
            throw(ArgumentError("the operator holds the electric field of its plasma, and the term would add the " *
                                "unknown $electric_field to it: build the operator with field_in_coefficients = false"))
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
    dims = _jacobian_kind(jacobian, operator, stored) === :approximate ?
           size(operator.grid) : (0, 0, 0)
    D = ForwardDiff.Dual{JacobianCorrectionTag, T, 1}
    radii = electric_field === nothing ? 0 : radial_cell_count(operator.grid)
    return FokkerPlanckTerm{T, typeof(plasma_from)}(
        unknown, operator, jacobian, electric_field, stored, zeros(T, dims),
        zeros(T, dims), zeros(D, dims), zeros(D, dims), zeros(T, radii),
        zeros(D, radii), plasma_from, settings, synchrotron, PlasmaKey(), Ref(0),
        lag_all, Ref{Any}(nothing), Ref{Any}(nothing), lagged, Float64[])
end

lags_in_ode_form(term::FokkerPlanckTerm) = _is_lagged(term) || _lags_ions(term)

# whether the term takes the ion densities of its plasma from the start of the step
function _lags_ions(term::FokkerPlanckTerm)
    ion_dependency(term.plasma) !== nothing && ion_dependency(term.plasma) in term.lagged
end
# the state of the ion densities of the plasma of a term
function _ion_state(term::FokkerPlanckTerm, u)
    _lags_ions(term) && !isempty(term.ion_state) ? term.ion_state : u
end
# the plasma of the term at the state u, with lagged ions from the start of the step
function _fokker_planck_plasma(term::FokkerPlanckTerm, u, layout, t)
    state = _ion_state(term, u)
    return state === u ? plasma_values(term.plasma, u, layout, t) :
           plasma_values(term.plasma, u, layout, t; ion_state = state)
end

"""
    refresh_coefficients!(term, u, layout, t)

Bring the coefficients of a term up to date with its plasma at the state `u` and the time
`t`, and return the operator that holds them. A [`FokkerPlanckTerm`](@ref) with a plasma
source rebuilds its operator when the plasma values of any radius changed since the last
rebuild. A term without a plasma source returns its operator unchanged, and so does a term
with `lagged = :all`, which [`prepare_step!`](@ref) refreshes. `u` and `layout`
may be `nothing` for a prescribed plasma.
"""
function refresh_coefficients!(term::FokkerPlanckTerm, u, layout, t)
    _is_lagged(term) && return term.operator
    return _rebuild_coefficients!(term, u, layout, t)
end

_is_lagged(term::FokkerPlanckTerm) = term.lag_all && is_state_dependent(term.plasma)

# the coefficients of the plasma at the state u and the time t
function _rebuild_coefficients!(term::FokkerPlanckTerm, u, layout, t)
    term.plasma === nothing && return term.operator
    field = term.electric_field === nothing
    plasma = changed_plasma!(
        term.plasma_key, term.plasma, u, layout, t; electric_field = field,
        ion_state = _ion_state(term, u))
    plasma === nothing && return term.operator
    try
        set_fokker_planck_coefficients!(
            term.operator, plasma, term.settings; synchrotron = term.synchrotron,
            field_in_coefficients = field)
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
function has_plasma_columns(term::FokkerPlanckTerm)
    is_state_dependent(term.plasma) && !_is_lagged(term)
end

"""
    plasma_column_sparsity!(entries, term, layout, row)

Push the entries of the plasma columns of a term with [`has_plasma_columns`](@ref): every
cell of the row unknown at a radius against each plasma unknown of that radius.
"""
function plasma_column_sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    for ir in 1:radial_cell_count(layout),
        column in plasma_columns(term.plasma, layout, ir; with_ions = !_lags_ions(term))

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
    op, with_ions = _dual_coefficients!(term, u, layout, t)
    D = eltype(op.advection_p)
    id = unknown_id(layout, term.unknown)
    f = kinetic_array(u, layout, id)
    df = term.dual_columns[]::Array{D, 3}
    fill!(df, zero(D))
    apply!(df, op, f; electric_field = _field_values(term.electric_field, u, layout))
    np, nxi, nr = size(df)
    for ir in 1:nr
        columns = plasma_columns(term.plasma, layout, ir; with_ions)
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

# The coefficients of a term in dual numbers of its plasma columns at the state u and the
# time t: its operator of dual coefficients, and whether the partials hold the charge states.
# The boundary flux of the term reads them as well.
function _dual_coefficients!(term::FokkerPlanckTerm, u, layout, t)
    plasma = _fokker_planck_plasma(term, u, layout, t)
    with_ions = ion_dependency(term.plasma) !== nothing && !_lags_ions(term)
    dual = dual_plasma(plasma, term.plasma.density !== nothing, with_ions)
    op = _dual_operator!(term, eltype(dual.cold_electron_temperature))
    set_fokker_planck_coefficients!(
        op, dual, term.settings; synchrotron = term.synchrotron,
        field_in_coefficients = term.electric_field === nothing)
    return op, with_ions
end

# the operator of dual coefficients of a term, made at the first use
function _dual_operator!(term::FokkerPlanckTerm, ::Type{D}) where {D}
    op = term.dual_operator[]
    if !(op isa AdvectionDiffusion{<:Any, D})
        op = similar_coefficients(term.operator, D)
        # the coefficients of a unit field do not depend on the plasma
        term.electric_field === nothing || set_unit_field_coefficients!(op)
        term.dual_operator[] = op
        term.dual_columns[] = zeros(D, size(term.operator.grid))
    end
    return op::AdvectionDiffusion{<:Any, D}
end

# a frozen limiter of a linear scheme is the exact derivative; another scheme's Jacobian
# is exact only when it is the operator's own scheme
function _jacobian_kind(jacobian, operator, stored)
    jacobian === :exact && return :exact
    jacobian === :frozen_limiter && return is_linear(operator) ? :exact : :approximate
    return (stored.p_scheme, stored.xi_scheme) == (operator.p_scheme, operator.xi_scheme) ?
           :exact : :approximate
end

function dependencies(term::FokkerPlanckTerm)
    unique!([term.unknown;
             term.electric_field === nothing ? Symbol[] : [term.electric_field];
             collect(Symbol, source_dependencies(term.plasma))])
end
# the kind of the stored block in f: another scheme's Jacobian or a frozen limiter
function _scheme_jacobian_kind(term::FokkerPlanckTerm)
    _jacobian_kind(term.jacobian, term.operator, term.stored_operator)
end
# the columns of the plasma unknowns are exact for the operator's own scheme, so the kind
# is that of the block in f
jacobian_kind(term::FokkerPlanckTerm) = _scheme_jacobian_kind(term)
# with the field as an unknown the term is bilinear in the distribution and the field; with a
# plasma from the unknowns the coefficients depend on the state
function is_linear(term::FokkerPlanckTerm)
    term.electric_field === nothing && !is_state_dependent(term.plasma) &&
        is_linear(term.operator)
end

validate(term::FokkerPlanckTerm, layout, row) = _check_fokker_planck(term, layout, row, 0.0)

# the checks of the grid, the field and the plasma unknowns; the composition of the ions is
# read at the time `t`, the start of the run, where the prescribed part is defined
function _check_fokker_planck(term::FokkerPlanckTerm, layout, row, t)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("FokkerPlanckTerm acts on and adds to kinetic unknowns"))
    grid = term.operator.grid
    for k in (id, row), ir in 1:radial_cell_count(layout)

        layout.shapes[k, ir] ==
        (momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) ||
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
            _check_ion_unknown(layout, id,
                IonComposition(plasma_values(term.plasma.prescribed, nothing, nothing, t)))
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
                                "cold_electron_density: read it with density = :cold_electron_density"))
    elseif source isa PrescribedPlasma && term.settings.mode === :full
        plasma_values(source, nothing, nothing, t).cold_density_given ||
            throw(ArgumentError("the prescribed plasma of the FokkerPlanckTerm of $(term.unknown) takes the " *
                                "free-electron density as the cold density, which counts the electrons of the " *
                                "grids twice in a system with the unknown cold_electron_density: give the cold " *
                                "density, or read it from the unknown"))
    end
    return nothing
end

function rhs!(out, term::FokkerPlanckTerm, u, layout, row, t)
    refresh_coefficients!(term, u, layout, t)
    id = unknown_id(layout, term.unknown)
    field = _field_values(term.electric_field, u, layout)
    if _equally_spaced(layout.radius_offsets[row]) &&
       _equally_spaced(layout.radius_offsets[id])
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
function _fokker_planck!(df, term::FokkerPlanckTerm, f, field = nothing)
    if _scheme_jacobian_kind(term) === :exact || !(eltype(f) <: ForwardDiff.Dual)
        apply!(df, term.operator, f; electric_field = field)
    elseif term.jacobian === :frozen_limiter
        # a dual state (Jacobian assembly): the limiter weights at the values, applied as
        # constants; the value part is the full residual, since the weights at f are f's own
        term.values .= _value.(f)
        apply!(df, term.operator, f; limiter_state = term.values, electric_field = field)
    else
        # the derivative of the other scheme, with the value part corrected to the
        # operator's own residual (constant sources cancel in the difference)
        apply!(df, term.stored_operator, f; electric_field = field)
        term.values .= _value.(f)
        plain = _plain_field!(term, field)
        fill!(term.correction, zero(eltype(term.correction)))
        apply!(term.correction, term.operator, term.values; electric_field = plain)
        apply!(term.correction, term.stored_operator,
            term.values; scale = -1, electric_field = plain)
        df .+= term.correction
    end
    return df
end

function sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    op = term.stored_operator
    np, nxi, nr = size(op.grid)
    widths = (p_width = stencil_half_width(op.p_scheme),
        xi_width = stencil_half_width(op.xi_scheme),
        r_width = stencil_half_width(op.r_scheme), p_upper_boundary = op.p_upper_boundary)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        r = _kinetic_index(layout, row, ir, i, j)
        _visit_stencil(op.grid, i, j, ir; widths...) do i2, j2, ir2
            push!(entries, (r, _kinetic_index(layout, id, ir2, i2, j2)))
        end
        # the field of the radius acts on every cell of the radius
        term.electric_field === nothing ||
            push!(entries, (r,
                first(unknown_range(layout, unknown_id(layout, term.electric_field), ir))))
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
        values, change = _field_values(term.electric_field, u, layout),
        _field_values(term.electric_field, v, layout)
        for ir in eachindex(term.dual_field)
            term.dual_field[ir] = D(values[ir], ForwardDiff.Partials((change[ir],)))
        end
        field = term.dual_field
    end
    fill!(output, zero(D))
    apply!(output, term.operator, input; electric_field = field)
    if term.jacobian === :frozen_limiter
        term.values .= f
        apply!(output, term.operator, input; scale = -1,
            limiter_state = term.values, electric_field = field)
    else
        apply!(output, term.stored_operator, input; scale = -1, electric_field = field)
    end
    out_view = kinetic_array(out, layout, row)
    @inbounds for k in CartesianIndices(output)
        out_view[k] += ForwardDiff.partials(output[k], 1)
    end
    return out
end

"""
    prescribed_fokker_planck_term(unknown, operator::AdvectionDiffusion, plasma_at;
                                  settings=nothing, synchrotron=nothing, jacobian=:exact,
                                  electric_field=nothing)

A [`FokkerPlanckTerm`](@ref) whose collision coefficients follow a plasma prescribed in
time (such as a cooling plasma with `T(t)` prescribed): `plasma_at(t)` returns the
[`PlasmaState`](@ref) at the time `t`. The same as
`FokkerPlanckTerm(unknown, operator; plasma = plasma_at, settings, synchrotron, jacobian,
electric_field)`, which this function returns. The coefficients are rebuilt
when the plasma values change ([`refresh_coefficients!`](@ref)).

With `electric_field`, the name of the unknown of the field, the field of the prescribed
plasma is left out of the coefficients, and the term takes the field of the state.
"""
function prescribed_fokker_planck_term(
        unknown::Symbol, operator::AdvectionDiffusion, plasma_at;
        settings::Union{Nothing, CollisionSettings} = nothing, synchrotron::Union{
            Nothing, Bool} = nothing,
        jacobian::Symbol = :exact, electric_field::Union{Nothing, Symbol} = nothing)
    FokkerPlanckTerm(unknown, operator; jacobian, electric_field,
        plasma = PrescribedPlasma(plasma_at), settings,
        synchrotron)
end
