# The knock-on term of a kinetic unknown: the matrix-free source of
# secondary electrons by large-angle collisions. Moved out of terms.jl.

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
