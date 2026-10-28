# Equation system M du/dt = rhs(u, t): one equation per unknown, built from terms; the
# stored sparse Jacobian from colored AD over the `:sparse` implicit terms only, and
# matrix-free products for the `:matrix_free` terms.

using LinearAlgebra: Diagonal
using SparseArrays: nnz, findnz, nonzeros, nzrange, rowvals
using SparseMatrixColorings: column_groups

"""
    Tolerance(absolute, relative; group=0)

Convergence tolerance of one unknown: an entry passes when `|δ| ≤ absolute + relative |x|`,
with `absolute` in the unknown's physical units. `group` adds `group Σ |x_k|` over the
components `k` of the entry's group at its radius ([`component_groups`](@ref)): the
tolerance of a charge state relative to the density of its species, so that a trace
impurity is converged on its own scale (DREAM tests the 2-norm of the whole ion block with
an absolute tolerance of 0, `src/ConvergenceChecker.cpp:110-119, 193`). A `group`
part needs a radial unknown: [`EquationSystem`](@ref) refuses it on a kinetic or scalar
unknown.
"""
struct Tolerance
    absolute::Float64
    relative::Float64
    group::Float64
    function Tolerance(absolute::Real, relative::Real; group::Real = 0.0)
        group >= 0 ||
            throw(ArgumentError("the group part of a tolerance must be nonnegative"))
        return new(Float64(absolute), Float64(relative), Float64(group))
    end
end

# a term bound to the id of its row unknown
struct BoundTerm{E <: EquationTerm}
    term::E
    row::Int
end

"""
    EquationSystem(layout, equations; time_scale=1, newton_tolerances, time_step_tolerances,
                   initial_time=0, initial_state=nothing)

The system `M du/dt = rhs(u, t)` for the unknowns of `layout`, with one
[`Equation`](@ref) per unknown. On construction it checks that every unknown has exactly
one equation, that every term's dependencies are unknowns of the layout (and passes the
term's own checks), and that every algebraic unknown appears in its own equation (some
implicit term depends on it; the numerical regularity of the algebraic block is not
checked). The terms are split into three groups:

- implicit `:sparse` terms, whose Jacobian is stored ([`jacobian!`](@ref)): by colored AD
  over the terms with automatic assembly, and by their own entries for the terms with
  [`jacobian_assembly`](@ref)`(term) === :direct` (`direct_terms`), whose rows the coloring
  does not see;
- implicit `:matrix_free` terms, which enter the Jacobian only through
  [`jacobian_operator`](@ref);
- explicit terms, evaluated at the previous time level in [`residual!`](@ref) and absent
  from the Jacobian.

Sparse terms whose stored block is an approximation ([`jacobian_kind`](@ref)`(term) ===
:approximate`) are also listed in `correction_terms`;
`stored_jacobian_exact` is false when there are any, and `linear` is true when every
implicit term is linear in the state ([`is_linear`](@ref)), both read by the Newton
iteration.

`time_scale` [s] sets the scaling of differential rows ([`residual_scales`](@ref));
`newton_tolerances` and `time_step_tolerances` map unknown names to [`Tolerance`](@ref)s
(defaults from [`tolerance`](@ref)). The time-step tolerances are those of the error
control of the adaptive steppers ([`StepDoubling`](@ref), [`StepController`](@ref)).

The construction evaluates the implicit `:sparse` terms once, at the state `initial_state`
(the zero state by default) and the time `initial_time`, for the structure of the stored
Jacobian. Set `initial_time` to the start of the run when a term is defined only on the
run's time interval (the `plasma_at` of a [`prescribed_fokker_planck_term`](@ref)), and
`initial_state` to the initial state when a term needs a valid state, such as a
Fokker-Planck term whose plasma reads the temperature from an unknown
([`PlasmaFromUnknowns`](@ref)).
"""
struct EquationSystem{
    T, S <: Tuple, M <: Tuple, X <: Tuple, C <: Tuple, D <: Tuple, K <: Tuple, F, P, B}
    layout::UnknownLayout{T}
    equations::Vector{Equation}
    sparse_terms::S
    matrix_free_terms::M
    explicit_terms::X
    correction_terms::C
    direct_terms::D                       # the sparse terms that assemble their own block
    column_terms::K                       # the sparse terms that add the columns of their plasma unknowns
    automatic_jacobian::SparseMatrixCSC{T, Int}  # the AD part, with the pattern of the coloring
    automatic_positions::Vector{Int}      # its stored entries in the stored Jacobian
    stored_jacobian_exact::Bool
    linear::Bool
    mass::Vector{T}                       # per row: 1 with a time derivative, 0 without
    mass_columns::Vector{Int}             # per row: the entry whose time derivative the row holds
    row_scales::Vector{T}                 # per row: magnitude of the quantity of the row
    pattern::SparseMatrixCSC{Bool, Int}
    sparse_rhs!::F
    preparation::P
    backend::B
    prototype::SparseMatrixCSC{T, Int}
    time::Base.RefValue{T}
    buffer::Vector{T}
    product::Vector{T}
    time_scale::T
    newton_tolerances::Dict{Symbol, Tolerance}
    time_step_tolerances::Dict{Symbol, Tolerance}
end

# Add the right-hand sides of a tuple of bound terms, unrolled at compile time.
@inline _add_terms!(out, ::Tuple{}, u, layout, t) = out
@inline function _add_terms!(out, terms::Tuple, u, layout, t)
    bound = first(terms)
    rhs!(out, bound.term, u, layout, bound.row, t)
    return _add_terms!(out, Base.tail(terms), u, layout, t)
end

@inline _prepare_terms!(::Tuple{}, u_old, layout, t) = nothing
@inline function _prepare_terms!(terms::Tuple, u_old, layout, t)
    bound = first(terms)
    prepare_step!(bound.term, u_old, layout, bound.row, t)
    return _prepare_terms!(Base.tail(terms), u_old, layout, t)
end

@inline _add_products!(out, ::Tuple{}, u, v, layout, t) = out
@inline function _add_products!(out, terms::Tuple, u, v, layout, t)
    bound = first(terms)
    jacobian_vector_product!(out, bound.term, u, v, layout, bound.row, t)
    return _add_products!(out, Base.tail(terms), u, v, layout, t)
end

@inline _add_jacobians!(J, ::Tuple{}, u, layout, t) = J
@inline function _add_jacobians!(J, terms::Tuple, u, layout, t)
    bound = first(terms)
    add_jacobian!(J, bound.term, u, layout, bound.row, t)
    return _add_jacobians!(J, Base.tail(terms), u, layout, t)
end

@inline _add_plasma_columns!(J, ::Tuple{}, u, layout, t) = J
@inline function _add_plasma_columns!(J, terms::Tuple, u, layout, t)
    bound = first(terms)
    add_plasma_columns!(J, bound.term, u, layout, bound.row, t)
    return _add_plasma_columns!(J, Base.tail(terms), u, layout, t)
end

@inline _add_corrections!(out, ::Tuple{}, u, v, layout, t) = out
@inline function _add_corrections!(out, terms::Tuple, u, v, layout, t)
    bound = first(terms)
    jacobian_correction_product!(out, bound.term, u, v, layout, bound.row, t)
    return _add_corrections!(out, Base.tail(terms), u, v, layout, t)
end

function EquationSystem(layout::UnknownLayout{T}, equations::AbstractVector;
        time_scale::Real = 1.0,
        newton_tolerances::AbstractDict = Dict{Symbol, Tolerance}(),
        time_step_tolerances::AbstractDict = Dict{Symbol, Tolerance}(),
        initial_time::Real = 0.0, initial_state::Union{Nothing, AbstractVector} = nothing) where {T}
    count = length(layout.unknowns)
    rows = zeros(Int, count)
    for eq in equations
        id = unknown_id(layout, eq.unknown)
        rows[id] == 0 ||
            throw(ArgumentError("unknown $(eq.unknown) has more than one equation"))
        rows[id] = 1
    end
    missing_equations = [layout.unknowns[id].name for id in 1:count if rows[id] == 0]
    isempty(missing_equations) ||
        throw(ArgumentError("no equation for $(join(missing_equations, ", "))"))
    # a group part needs the component groups of a radial unknown: Newton would skip it on
    # another unknown, and the step controller would fail at its first estimate
    for tolerances in (newton_tolerances, time_step_tolerances), (name, tol) in tolerances

        iszero(tol.group) || unknown(layout, name) isa RadialUnknown ||
            throw(ArgumentError("the tolerance of $name has a group part, which only a radial unknown has"))
    end

    _check_conductivity_correction(equations)
    _check_prescribed_operators(equations)
    _check_plasma_pairings(equations)
    sparse_terms, matrix_free_terms, explicit_terms = BoundTerm[], BoundTerm[], BoundTerm[]
    mass = zeros(T, length(layout))
    mass_columns = collect(1:length(layout))
    row_scales = T.(unknown_scales(layout))
    derivative_rows = zeros(Int, count)   # per unknown: the unknown whose row holds its time derivative
    for eq in equations
        row = unknown_id(layout, eq.unknown)
        eq.kind isa Union{TimeDerivative, Algebraic, Constraint} ||
            throw(ArgumentError("unknown equation kind $(eq.kind)"))
        # explicit terms are evaluated at the previous time level and determine nothing
        holds_unknown = any(
            term -> treatment(term) === :implicit &&
                    eq.unknown in dependencies(term), eq.terms)
        if eq.kind isa Algebraic
            holds_unknown ||
                throw(ArgumentError("the algebraic equation of $(eq.unknown) has no implicit term " *
                                    "depending on $(eq.unknown), so it cannot determine it"))
        elseif eq.kind isa Constraint
            eq.scale === nothing &&
                throw(ArgumentError("the equation of $(eq.unknown) is a constraint and needs the scale of the " *
                                    "quantity it balances (Equation(...; scale)): the scale of its unknown, a " *
                                    "rate, does not fit the row"))
            #! format: off
            any(other -> other.unknown !== eq.unknown &&
                         any(term -> treatment(term) === :implicit && eq.unknown in dependencies(term), other.terms),
                equations) ||
                throw(ArgumentError("the equation of $(eq.unknown) is a constraint, and no other equation has an " *
                                    "implicit term depending on $(eq.unknown), so nothing determines it"))
            #! format: on
        else
            of = eq.kind.of === nothing ? row : unknown_id(layout, eq.kind.of)
            if of != row
                layout.kinds[of] === layout.kinds[row] &&
                layout.sizes[of, :] == layout.sizes[row, :] ||
                    throw(ArgumentError("the equation of $(eq.unknown) holds the time derivative of " *
                                        "$(eq.kind.of), which needs the same kind and sizes"))
                # the unknown of the row is determined by its own row or, as the temperature by
                # the energy in DREAM's pairing, by the equation of the unknown it differentiates,
                # directly or through algebraic rows (the total energy reads the energy of the
                # cold electrons, whose row reads the temperature)
                determined_by_of = _reaches(equations, eq.kind.of, eq.unknown)
                holds_unknown || determined_by_of ||
                    throw(ArgumentError("the equation of $(eq.unknown) holds the time derivative of " *
                                        "$(eq.kind.of), and neither it nor the equation of $(eq.kind.of) has an " *
                                        "implicit term depending on $(eq.unknown), so nothing determines it"))
            end
            derivative_rows[of] == 0 ||
                throw(ArgumentError("the time derivative of $(layout.unknowns[of].name) is in the equations of " *
                                    "$(layout.unknowns[derivative_rows[of]].name) and of $(eq.unknown)"))
            derivative_rows[of] = row
            for (rows_here, columns_here) in zip(_unknown_ranges(layout, row), _unknown_ranges(layout, of))
                mass[rows_here] .= one(T)
                mass_columns[rows_here] .= columns_here
                row_scales[rows_here] .= layout.unknowns[of].scale
            end
        end
        if eq.scale !== nothing
            for range in _unknown_ranges(layout, row)
                row_scales[range] .= eq.scale
            end
        end
        for term in eq.terms
            for name in dependencies(term)
                unknown_id(layout, name)  # throws for unknown names
            end
            validate(term, layout, row, T(initial_time))
            coupling(term) in (:sparse, :matrix_free) ||
                throw(ArgumentError("unknown coupling $(coupling(term)) of $(typeof(term))"))
            bound = BoundTerm(term, row)
            if treatment(term) === :explicit
                push!(explicit_terms, bound)
            elseif treatment(term) === :implicit
                push!(coupling(term) === :sparse ? sparse_terms : matrix_free_terms, bound)
            else
                throw(ArgumentError("unknown treatment $(treatment(term)) of $(typeof(term))"))
            end
        end
    end
    #! format: off
    sparse_tuple, matrix_free_tuple, explicit_tuple = Tuple(sparse_terms), Tuple(matrix_free_terms),
                                                      Tuple(explicit_terms)
    correction_tuple = Tuple(bound for bound in sparse_terms if jacobian_kind(bound.term) === :approximate)
    #! format: on
    linear = all(bound -> is_linear(bound.term), sparse_terms) &&
             all(bound -> is_linear(bound.term), matrix_free_terms)

    for bound in sparse_terms
        jacobian_assembly(bound.term) in (:automatic, :direct) ||
            throw(ArgumentError("unknown Jacobian assembly $(jacobian_assembly(bound.term)) of $(typeof(bound.term))"))
        jacobian_assembly(bound.term) === :direct &&
            jacobian_kind(bound.term) === :approximate &&
            throw(ArgumentError("$(typeof(bound.term)) assembles its block itself, so the block is exact"))
    end
    #! format: off
    automatic_tuple = Tuple(bound for bound in sparse_terms if jacobian_assembly(bound.term) === :automatic)
    direct_tuple = Tuple(bound for bound in sparse_terms if jacobian_assembly(bound.term) === :direct)
    #! format: on
    column_tuple = Tuple(bound for bound in sparse_terms if has_plasma_columns(bound.term))

    # structural pattern of the stored Jacobian: the diagonal (for M/Δt) and the sparse
    # terms. The coloring takes the part of the terms with automatic assembly
    n = length(layout)
    entries = Tuple{Int, Int}[(i, i) for i in 1:n]
    # the entries of the mass matrix off the diagonal
    append!(entries, (i, mass_columns[i]) for i in 1:n if mass_columns[i] != i)
    for bound in automatic_tuple
        sparsity!(entries, bound.term, layout, bound.row)
    end
    as_pattern(list) = sparse(
        first.(list), last.(list), trues(length(list)), n, n, (a, b) -> a | b)
    automatic_pattern = as_pattern(entries)
    for bound in direct_tuple
        sparsity!(entries, bound.term, layout, bound.row)
    end
    # the columns of the plasma unknowns of terms whose coefficients follow them: assembled
    # by the terms, outside the coloring like the direct terms
    for bound in column_tuple
        plasma_column_sparsity!(entries, bound.term, layout, bound.row)
    end
    pattern = as_pattern(entries)

    time = Ref(T(initial_time))
    # residual of the part that AD differentiates: the coloring must not see matrix-free
    # terms, whose entries outside the pattern would be folded onto entries inside it
    sparse_rhs! = let terms = automatic_tuple, layout = layout, time = time
        (du, u) -> (
            fill!(du, zero(eltype(du))); _add_terms!(du, terms, u, layout, time[]); nothing)
    end
    # one type of dual numbers for every system (JacobianTag)
    backend = sparse_ad_backend(pattern = automatic_pattern, tag = ForwardDiff.Tag{
        JacobianTag, T}())
    initial_state === nothing || length(initial_state) == n ||
        throw(DimensionMismatch("the initial state has $(length(initial_state)) entries, the layout $n"))
    x0 = initial_state === nothing ? zeros(T, n) : collect(T, initial_state)
    buffer = zeros(T, n)
    preparation, automatic_jacobian = try
        preparation = DI.prepare_jacobian(sparse_rhs!, buffer, backend, x0)
        preparation, DI.jacobian(sparse_rhs!, buffer, preparation, backend, x0)
    catch failure
        failure isa DomainError && initial_state === nothing &&
            throw(ArgumentError("a term is not defined at the zero state, where the structure of the Jacobian is " *
                                "evaluated ($(sprint(showerror, failure))): give the initial state, " *
                                "EquationSystem(...; initial_state)"))
        rethrow()
    end
    # the prototype is evaluated at u = x0 and t = initial_time, where some derivatives
    # vanish; it must still carry the whole pattern, or jacobian! at other states would
    # miss entries
    nnz(automatic_jacobian) == nnz(automatic_pattern) ||
        error("the Jacobian prototype lost structural entries ($(nnz(automatic_jacobian)) of " *
              "$(nnz(automatic_pattern)))")
    if isempty(direct_tuple) && isempty(column_tuple)
        prototype = automatic_jacobian
        automatic_positions = Int[]
    else
        # the stored matrix holds both parts; the entries of the AD part are copied to their
        # positions in it
        rows, columns, _ = findnz(pattern)
        prototype = sparse(rows, columns, zeros(T, length(rows)), n, n)
        automatic_positions = zeros(Int, nnz(automatic_jacobian))
        automatic_rows = rowvals(automatic_jacobian)
        stored_rows = rowvals(prototype)
        for column in 1:n, k in nzrange(automatic_jacobian, column)

            range = nzrange(prototype, column)
            automatic_positions[k] = searchsortedfirst(
                stored_rows, automatic_rows[k], first(range), last(range),
                Base.Order.Forward)
        end
        _merge_jacobian!(prototype, automatic_jacobian, automatic_positions)
        _add_jacobians!(prototype, direct_tuple, x0, layout, T(initial_time))
        _add_plasma_columns!(prototype, column_tuple, x0, layout, T(initial_time))
    end
    return EquationSystem(
        layout, collect(Equation, equations), sparse_tuple, matrix_free_tuple,
        explicit_tuple, correction_tuple, direct_tuple, column_tuple, automatic_jacobian,
        automatic_positions,
        isempty(correction_tuple), linear, mass, mass_columns, row_scales, pattern,
        sparse_rhs!, preparation, backend, prototype,
        time, buffer, zeros(T, n), T(time_scale), Dict{Symbol, Tolerance}(newton_tolerances),
        Dict{Symbol, Tolerance}(time_step_tolerances))
end

# the entries of the AD part at their positions in the stored matrix, zero elsewhere
function _merge_jacobian!(J::SparseMatrixCSC, automatic::SparseMatrixCSC, positions::Vector{Int})
    stored, values = nonzeros(J), nonzeros(automatic)
    fill!(stored, zero(eltype(stored)))
    @inbounds for k in eachindex(positions)
        stored[positions[k]] = values[k]
    end
    return J
end

function _unknown_ranges(layout, id)
    layout.kinds[id] === :scalar ? (unknown_range(layout, id),) :
    (unknown_range(layout, id, ir) for ir in 1:radial_cell_count(layout))
end

"""
    rhs!(du, u, system::EquationSystem, t)

Right-hand side of `M du/dt = rhs(u, t)` with all terms at `u` (the ODE form, as passed to
a general integrator).
"""
function rhs!(du::AbstractVector, u::AbstractVector, system::EquationSystem, t)
    fill!(du, zero(eltype(du)))
    layout = system.layout
    # the ODE form has no previous time level: a lagged part takes the state itself
    _prepare_terms!(system.sparse_terms, u, layout, t)
    _prepare_terms!(system.matrix_free_terms, u, layout, t)
    _add_terms!(du, system.sparse_terms, u, layout, t)
    _add_terms!(du, system.matrix_free_terms, u, layout, t)
    _add_terms!(du, system.explicit_terms, u, layout, t)
    return du
end

"""
    residual!(F, u, u_old, system::EquationSystem, t, dt; reference=u_old)

Backward-Euler residual `F = M (u - reference)/dt - rhs(u, t)`, with explicit terms at
`u_old` and the terms prepared from `u_old` ([`prepare_step!`](@ref)). The reference is
the start state for backward Euler; the two-step BDF2 of [`StepController`](@ref)`(order = 2)`
passes its combination of the two last states and its effective step. `M` has one entry per row with a time derivative, in the column of the unknown
whose derivative the row holds. Allocation free.
"""
function residual!(F::AbstractVector, u::AbstractVector, u_old::AbstractVector,
        system::EquationSystem, t, dt; reference::AbstractVector = u_old)
    rhs = system.buffer
    fill!(rhs, zero(eltype(rhs)))
    layout = system.layout
    _prepare_terms!(system.sparse_terms, u_old, layout, t)
    _prepare_terms!(system.matrix_free_terms, u_old, layout, t)
    _add_terms!(rhs, system.sparse_terms, u, layout, t)
    _add_terms!(rhs, system.matrix_free_terms, u, layout, t)
    _add_terms!(rhs, system.explicit_terms, u_old, layout, t)
    mass, columns = system.mass, system.mass_columns
    @inbounds for i in eachindex(F)
        column = columns[i]
        F[i] = mass[i] * (u[column] - reference[column]) / dt - rhs[i]
    end
    return F
end

# whether the equation of `from` depends implicitly on `target`, directly or through the
# unknowns of algebraic equations that it depends on
function _reaches(equations, from::Symbol, target::Symbol)
    seen, queue = Set{Symbol}([from]), [from]
    while !isempty(queue)
        name = popfirst!(queue)
        for equation in equations
            equation.unknown === name || continue
            name === from || equation.kind isa Algebraic || continue
            for term in equation.terms, dependency in dependencies(term)

                treatment(term) === :implicit || continue
                dependency === target && return true
                dependency in seen || (push!(seen, dependency); push!(queue, dependency))
            end
        end
    end
    return false
end

"""
    fill_initial_values!(u, system, t; unknowns=nothing)

Set the unknowns whose [`Equation`](@ref) has `initial_value = :from_equation` from their
algebraic rows at the time `t`, entry by entry (DREAM's `INITRULE_EVAL_EQUATION`; the
energy of the cold electrons from their temperature and density). A row
`0 = c x + r(u)` that is linear in its own unknown `x` gives `x = −r/c`. The coefficient
`c` comes from the row at `x = 0, s, 2s`, with `s` the scale of the unknown, so a row
needs no particular form. A row that reads another filled unknown is solved again until
every filled row holds, whatever the order of the equations. `unknowns` limits the fill
to those names, the unknowns that a resumed run adds to the stored layout. Returns the
names of the filled unknowns.

Throws an `ArgumentError` for a row that does not depend on its unknown or is not linear in
it, and for filled rows that read each other in a cycle.
"""
function fill_initial_values!(u::AbstractVector, system::EquationSystem, t::Real; unknowns = nothing)
    layout = system.layout
    rows = [eq
            for eq in system.equations
            if eq.initial_value === :from_equation &&
        (unknowns === nothing || eq.unknown in unknowns)]
    isempty(rows) && return Symbol[]
    out = similar(u)
    #! format: off
    positions = Dict(eq.unknown => reduce(vcat, collect.(_unknown_ranges(layout, unknown_id(layout, eq.unknown))))
                     for eq in rows)
    #! format: on
    coefficients = Dict{Symbol, Vector{eltype(u)}}()
    function row_values(eq)
        fill!(out, zero(eltype(out)))
        row = unknown_id(layout, eq.unknown)
        for term in eq.terms
            rhs!(out, term, u, layout, row, t)
        end
        return out[positions[eq.unknown]]
    end
    for eq in rows, term in eq.terms

        prepare_step!(term, u, layout, unknown_id(layout, eq.unknown), t)
    end
    for _ in 1:length(rows)
        for eq in rows
            x = positions[eq.unknown]
            s = layout.unknowns[unknown_id(layout, eq.unknown)].scale
            u[x] .= 0
            r0 = row_values(eq)
            u[x] .= s
            r1 = row_values(eq)
            u[x] .= 2s
            r2 = row_values(eq)
            c = (r1 .- r0) ./ s
            for i in eachindex(c)
                magnitude = max(abs(r0[i]), abs(r1[i]), abs(r2[i]))
                # a row far larger than the unknown's scale times its coefficient loses the
                # coefficient in round-off: the initial state is in other units than the scale
                abs(c[i]) * s > 1e-6 * magnitude ||
                    throw(ArgumentError("the row of $(eq.unknown), which gives its initial value, is $magnitude in " *
                                        "magnitude against its unknown's scale $s: check the units of the initial " *
                                        "state, or that the row depends on $(eq.unknown)"))
                abs(r2[i] - 2r1[i] + r0[i]) <= 1e-10 * magnitude ||
                    throw(ArgumentError("the initial value of $(eq.unknown) comes from its equation, whose row " *
                                        "must be linear in $(eq.unknown)"))
            end
            u[x] .= .-r0 ./ c
            coefficients[eq.unknown] = c
        end
        #! format: off
        holds(eq) = all(abs(r) <= 1e-10 * abs(c) * max(abs(v), layout.unknowns[unknown_id(layout, eq.unknown)].scale)
                        for (r, c, v) in zip(row_values(eq), coefficients[eq.unknown], u[positions[eq.unknown]]))
        #! format: on
        all(holds, rows) && return [eq.unknown for eq in rows]
    end
    throw(ArgumentError("the rows of $(join((eq.unknown for eq in rows), ", ")), which give their initial values, " *
                        "read each other in a cycle"))
end

"""
    mass_matrix(system)

Mass matrix `M`, constant: 1 in the rows with a time derivative, in the column of the
unknown whose derivative the row holds, and nothing in the algebraic rows. A `Diagonal`
when every row holds the derivative of its own unknown, a sparse matrix otherwise
(`TimeDerivative(of)`).
"""
function mass_matrix(system::EquationSystem)
    n = length(system.mass)
    all(i -> system.mass_columns[i] == i, 1:n) && return Diagonal(system.mass)
    rows = findall(!iszero, system.mass)
    return sparse(rows, system.mass_columns[rows], system.mass[rows], n, n)
end

"""
    jacobian_sparsity(system)

Structural pattern of the stored Jacobian: the diagonal and the implicit `:sparse` terms.
"""
jacobian_sparsity(system::EquationSystem) = system.pattern

"""
    jacobian_prototype(system)

A new sparse matrix with the structure of [`jacobian_sparsity`](@ref), to be filled by
[`jacobian!`](@ref).
"""
jacobian_prototype(system::EquationSystem) = copy(system.prototype)

"""
    jacobian_colors(system)

Number of colors of the stored Jacobian's column coloring: the number of directional
derivatives (dual-number passes over the sparse terms) one [`jacobian!`](@ref) costs. The
terms that assemble their own block do not count.
"""
jacobian_colors(system::EquationSystem) = length(column_groups(system.preparation))

"""
    jacobian!(J, system::EquationSystem, u, t)

Values of the stored Jacobian `∂rhs/∂u` of the implicit `:sparse` terms at `(u, t)`, into
`J` (from [`jacobian_prototype`](@ref)), by colored forward-mode AD and by the entries of
the terms that assemble their own block ([`jacobian_assembly`](@ref)); for terms with an
approximate Jacobian ([`jacobian_kind`](@ref)) their stored block, not the derivative.
Matrix-free and explicit terms are not included.
"""
function jacobian!(J::SparseMatrixCSC, system::EquationSystem, u::AbstractVector, t)
    system.time[] = t
    if isempty(system.direct_terms) && isempty(system.column_terms)
        DI.jacobian!(
            system.sparse_rhs!, system.buffer, J, system.preparation, system.backend, u)
        return J
    end
    automatic = system.automatic_jacobian
    DI.jacobian!(
        system.sparse_rhs!, system.buffer, automatic, system.preparation, system.backend, u)
    _merge_jacobian!(J, automatic, system.automatic_positions)
    _add_jacobians!(J, system.direct_terms, u, system.layout, t)
    _add_plasma_columns!(J, system.column_terms, u, system.layout, t)
    return J
end

"""
    jacobian_operator(J, system::EquationSystem, u, t)

`∂rhs/∂u` at `(u, t)` as a linear operator: the stored matrix `J` (from
[`jacobian!`](@ref) at the same point) plus the products of the implicit matrix-free
terms and the corrections of the terms with an approximate stored block
([`jacobian_correction_product!`](@ref)), so the operator is the exact Jacobian whatever
the stored blocks hold. `u` is referenced, not copied. The products go through a buffer owned by the
system, shared by all operators of the same system, so they must not be applied
concurrently. The Newton-Krylov solver [`GMRES`](@ref) builds its own operator.
"""
function jacobian_operator(J::SparseMatrixCSC{T}, system::EquationSystem, u::AbstractVector, t) where {T}
    stored = LinearOperator(J)
    isempty(system.matrix_free_terms) && isempty(system.correction_terms) && return stored
    n = length(system.layout)
    product = system.product
    function matrix_free!(result, v, a, b)
        fill!(product, zero(T))
        _add_products!(product, system.matrix_free_terms, u, v, system.layout, t)
        _add_corrections!(product, system.correction_terms, u, v, system.layout, t)
        if iszero(b)
            result .= a .* product
        else
            result .= a .* product .+ b .* result
        end
        return result
    end
    return stored +
           LinearOperator{T, Vector{T}}(n, n, false, false, matrix_free!, nothing, nothing)
end

"""
    residual_scales(system)

Scale of every row of the residual: `scale/time_scale` for rows with a time derivative
and `scale` for algebraic rows. `scale` is the scale of the equation when it names one,
else the reference scale of the unknown whose time derivative the row holds, else that of
the row's unknown. Dividing the residual by these makes all rows of order one. The Newton iteration of
[`NewtonSolver`](@ref) does not use them: it weights differential rows by `dt/scale`,
since round-off in `(u - u_old)/dt` at a short step would exceed a tolerance scaled by a
fixed `time_scale`.
"""
function residual_scales(system::EquationSystem)
    scales = system.row_scales
    return [system.mass[i] == 0 ? scales[i] : scales[i] / system.time_scale
            for i in eachindex(scales)]
end

"""
    tolerance(system, name; kind=:newton)

[`Tolerance`](@ref) of the unknown `name` for Newton convergence (`kind = :newton`) or
time-step error control (`kind = :time_step`); by default `relative = 1e-10` and
`absolute = 1e-10` times the unknown's reference scale. A radial unknown with more than one
group of components ([`component_groups`](@ref), the ion densities) takes `group = 1e-10`
and `absolute = 1e-20` times its scale for the Newton iteration: each species is converged
relative to its own density, and the absolute part only covers a species absent at a
radius.
"""
function tolerance(system::EquationSystem, name::Symbol; kind::Symbol = :newton)
    settings = kind === :newton ? system.newton_tolerances :
               kind === :time_step ? system.time_step_tolerances :
               throw(ArgumentError("kind must be :newton or :time_step"))
    haskey(settings, name) && return settings[name]
    u = unknown(system.layout, name)
    kind === :newton && u isa RadialUnknown && length(component_groups(u)) > 1 &&
        return Tolerance(1e-20 * u.scale, 1e-10; group = 1e-10)
    return Tolerance(1e-10 * u.scale, 1e-10)
end

"""
    ode_function(system::EquationSystem)

SciML `ODEFunction` of the system, for OrdinaryDiffEq and NonlinearSolve. Defined by the
package extension `Runes2PSciMLBaseExt`, which loads with SciMLBase (`using SciMLBase`).
"""
function ode_function end

# A bare operator that a FokkerPlanckTerm with a plasma source owns holds the coefficients
# of the last rebuild of that term, which belong to another plasma when an interface, a
# boundary flux or a conductivity model reads them before the term is evaluated at the new
# time. Such a term takes the Fokker-Planck term, which rebuilds at the evaluation. The
# rebuild also fixes what the term that owns the operator may share:
# - one term rebuilds an operator: two would each keep their own key on shared arrays, and
#   the coefficients of the last rebuild would pass for those of the other;
# - the rebuild clears the radial coefficients, so radial transport needs an operator and
#   a term of its own;
# - a field-particle term on the same unknown restores the momentum that the e-e part of
#   the Fokker-Planck term takes, which needs the same collision settings.
function _check_prescribed_operators(equations)
    prescribed = [term for equation in equations
                  for term in equation.terms
                  if term isa FokkerPlanckTerm && has_plasma_source(term)]
    isempty(prescribed) && return nothing
    for equation in equations, term in equation.terms, source in _coefficient_sources(term)
        source isa AdvectionDiffusion && any(p -> p.operator === source, prescribed) &&
            throw(ArgumentError("a $(nameof(typeof(term))) of $(equation.unknown) reads the bare operator of a " *
                                "FokkerPlanckTerm with a plasma source, whose coefficients belong to the plasma of " *
                                "its last rebuild: give it the Fokker-Planck term in place of its operator"))
    end
    for (k, term) in enumerate(prescribed)
        any(other -> other.operator === term.operator, @view prescribed[(k + 1):end]) &&
            throw(ArgumentError("two Fokker-Planck terms with a plasma source rebuild the same operator of " *
                                "$(term.unknown): give each term an operator of its own"))
        op = term.operator
        (any(!iszero, op.advection_r) || any(!iszero, op.diffusion_r_r)) &&
            throw(ArgumentError("the operator of the Fokker-Planck term of $(term.unknown) with a plasma source " *
                                "holds radial transport, which its rebuild clears: give the radial transport an " *
                                "operator and a FokkerPlanckTerm of its own"))
    end
    for equation in equations, term in equation.terms

        term isa FieldParticleTerm && has_plasma_source(term) || continue
        for owner in prescribed
            owner.unknown === term.unknown && owner.settings != term.settings &&
                throw(ArgumentError("the field-particle term of $(term.unknown) has other collision settings than " *
                                    "the Fokker-Planck term of that unknown, so the field-particle part does not " *
                                    "restore the momentum of its electron-electron part: give both the same settings"))
        end
    end
    return nothing
end

# Where a term of the cold bulk takes its temperature from: `(:unknown, name)` for an
# unknown of the system, `(:fixed,)` for a prescribed or frozen value, `nothing` for a
# term that reads no temperature of the bulk. Other terms, such as a Maxwellian source at a
# temperature of its own, are not part of the bulk.
_temperature_source(term) = nothing
_temperature_source(term::FieldParticleTerm) = _source_temperature(term.plasma)
# a Fokker-Planck term without a plasma whose operator has no collision coefficients (radial
# transport alone, the layout the rebuilt terms ask for) reads no temperature
function _temperature_source(term::FokkerPlanckTerm)
    term.plasma === nothing && term.operator.coefficient_source[] === nothing ? nothing :
    _source_temperature(term.plasma)
end
_temperature_source(term::ConductivityTerm) = _source_temperature(term.model.plasma)
_temperature_source(term::ParticleSourceTerm) = _limit_temperature(term.temperature)
_temperature_source(term::HeatTransportTerm) = (:unknown, term.temperature)
_temperature_source(term::IonEnergyExchangeTerm) = (:unknown, term.temperature)
_temperature_source(term::RunawayFluidEnergyTransferTerm) = (:unknown, term.temperature)
_temperature_source(term::KineticEnergyLossTerm) = _temperature_source(term.source)
_temperature_source(term::RadiatedPowerTerm) = (:unknown, term.temperature)
_temperature_source(term::IonRateTerm) = (:unknown, term.temperature)
_temperature_source(term::ExternalAvalancheTerm) = (:unknown, term.temperature)
_temperature_source(term::SecondaryHeatingTerm) = (:unknown, term.temperature)
# the threshold of a kinetic ionization reads the temperature of its region
function _temperature_source(term::KineticIonizationTerm)
    term.envelopes === nothing ? nothing : _region_temperature(term.envelopes.region.limit)
end
function _temperature_source(term::Union{
        DensityMomentTerm, CurrentMomentTerm, EnergyMomentTerm})
    term.envelopes === nothing ? nothing : _region_temperature(term.envelopes.region.limit)
end
_source_temperature(::Union{Nothing, PlasmaSource}) = (:fixed,)
_source_temperature(source::PlasmaFromUnknowns) = (:unknown, source.temperature)
function _limit_temperature(temperature)
    temperature isa Symbol ? (:unknown, temperature) : (:fixed,)
end
_region_temperature(limit::ThermalMomentum) = _limit_temperature(limit.temperature)
_region_temperature(limit) = nothing      # a limit in momentum reads no temperature

# the temperature unknown that the terms of the cold bulk read, or nothing
function _bulk_temperature_unknown(equations)
    names = Set{Symbol}()
    for equation in equations, term in equation.terms

        source = _temperature_source(term)
        source !== nothing && first(source) === :unknown && push!(names, last(source))
    end
    length(names) > 1 &&
        throw(ArgumentError("the terms of the cold bulk read the temperature from the unknowns " *
                            "$(join(sort!(collect(names)), " and ")): give them one temperature"))
    return isempty(names) ? nothing : only(names)
end

# the kind of the plasma of a term, for the pairs whose coefficients must agree
_plasma_kind(::Nothing) = (:frozen,)
_plasma_kind(::PrescribedPlasma) = (:prescribed,)
function _plasma_kind(source::PlasmaFromUnknowns)
    (:unknowns, source.temperature, source.density, source.ions)
end

# the plasma source of a term whose coefficients can follow the unknowns, or nothing
_plasma_source(term) = nothing
_plasma_source(term::Union{FokkerPlanckTerm, FieldParticleTerm}) = term.plasma
_plasma_source(term::ConductivityTerm) = term.model.plasma

"""
    temperature_floors(system)

The floor [eV] of every temperature unknown that a [`PlasmaFromUnknowns`](@ref) of the
system reads, as pairs `name => floor`, with the largest floor where several sources read
one unknown. The Newton iteration takes each as the lower limit of that unknown.
"""
function temperature_floors(equations::AbstractVector)
    floors = Dict{Symbol, Float64}()
    for equation in equations, term in equation.terms

        source = _plasma_source(term)
        source isa PlasmaFromUnknowns || continue
        name = source.temperature
        floors[name] = max(get(floors, name, -Inf), source.temperature_floor)
    end
    return sort!(collect(floors); by = first)
end
temperature_floors(system::EquationSystem) = temperature_floors(system.equations)

"""
    fluid_temperature_floors(system)

The temperature unknowns that only the fluid terms read (no [`PlasmaFromUnknowns`](@ref)
gives them a floor), as pairs `name => floor` with the default floor of
`PlasmaFromUnknowns`, 0.5 eV ([`DEFAULT_TEMPERATURE_FLOOR`](@ref)). The Newton iteration
takes each as the lower limit of an unknown that declares none of its own: at and below
zero the ADAS coefficients vanish, so the radiation would switch off, and
the Rechester-Rosenbluth transport has no thermal speed. An unknown with a `LowerLimit` of
its own keeps it, so a comparison that follows DREAM below 0.5 eV can set a lower one.
`SecondaryHeatingTerm` is in the list as well: its temperature is that of the cold
electrons it heats, though it enters no coefficient of the term.
"""
function fluid_temperature_floors(equations::AbstractVector)
    floored = Set(first.(temperature_floors(equations)))
    names = Set{Symbol}()
    for equation in equations, term in equation.terms
        #! format: off
        term isa Union{IonRateTerm,RadiatedPowerTerm,HeatTransportTerm,IonEnergyExchangeTerm,
                       RunawayFluidEnergyTransferTerm,SecondaryHeatingTerm,ExternalAvalancheTerm} || continue
        #! format: on
        term.temperature in floored || push!(names, term.temperature)
    end
    return sort!([name => DEFAULT_TEMPERATURE_FLOOR for name in names]; by = first)
end
function fluid_temperature_floors(system::EquationSystem)
    fluid_temperature_floors(system.equations)
end

# Terms that must agree take their plasma from the same source and in the same treatment,
# explicit with explicit and implicit with implicit where both exist:
# - one temperature for the terms of the cold bulk: all read the unknown when one does;
# - the Fokker-Planck and the field-particle term of one unknown: the same kind of plasma,
#   and lagged with lagged when it follows the unknowns, since the field-particle term is
#   lagged only;
# - a corrected conductivity that follows the temperature (implicit) with an implicit
#   Fokker-Planck term only.
function _check_plasma_pairings(equations)
    temperature = _bulk_temperature_unknown(equations)
    if temperature !== nothing
        for equation in equations, term in equation.terms

            _temperature_source(term) == (:fixed,) &&
                throw(ArgumentError("a $(nameof(typeof(term))) in the equation of $(equation.unknown) takes a " *
                                    "prescribed or fixed temperature, while other terms of the cold bulk read the " *
                                    "unknown $temperature: give it the temperature of that unknown " *
                                    "(PlasmaFromUnknowns, or the name of the unknown)"))
        end
    end
    fokker_planck = [term for equation in equations
                     for term in equation.terms if term isa FokkerPlanckTerm]
    for equation in equations, term in equation.terms

        term isa FieldParticleTerm || continue
        for owner in fokker_planck
            owner.unknown === term.unknown || continue
            _plasma_kind(owner.plasma) == _plasma_kind(term.plasma) ||
                throw(ArgumentError("the field-particle term of $(term.unknown) and the Fokker-Planck term of that " *
                                    "unknown take their plasma from different sources, so the field-particle part " *
                                    "does not restore the momentum of the electron-electron part: give both the " *
                                    "same plasma"))
            is_state_dependent(owner.plasma) && !_is_lagged(owner) &&
                throw(ArgumentError("the field-particle term of $(term.unknown) follows the plasma of the unknowns " *
                                    "lagged and the Fokker-Planck term of that unknown implicitly, so the " *
                                    "field-particle part misses the change of the electron-electron part within a " *
                                    "step: use lagged = :all for the Fokker-Planck term"))
        end
    end
    # the moments of the hot region of one temperature unknown count the same electrons: the
    # density, the current and the energy of the hot electrons. The side may differ, since
    # the regions above and below a threshold complement each other.
    # The kinetic ionization of the split counts the hot electrons of its region as well
    # (n_cold = n_M (1 − f_>)), and the sinks charged to the total energy those of W_hot.
    regions = [term.envelopes.region for equation in equations
               for term in equation.terms
               if term isa Union{DensityMomentTerm, CurrentMomentTerm,
                          EnergyMomentTerm, KineticIonizationTerm,
                          KineticEnergyLossTerm} && term.envelopes !== nothing &&
                      term.envelopes.region.limit isa ThermalMomentum{Symbol}]
    for (k, a) in enumerate(regions), b in regions[(k + 1):end]

        a.limit.temperature === b.limit.temperature || continue
        (a.limit.factor, a.envelope, a.smooth_width) ==
        (b.limit.factor, b.envelope, b.smooth_width) ||
            throw(ArgumentError("two moments over the hot region of $(a.limit.temperature) take different thresholds " *
                                "or envelopes (factor $(a.limit.factor) and $(b.limit.factor)), so they count different " *
                                "electrons as hot: give them one region"))
    end
    # the friction of bound_friction = :ionization_split charges f for the ionization that the
    # kinetic ionization of the system counts: the same threshold, and the heating by the
    # secondaries gives the cold electrons the moment that f pays
    ionizations = [term for equation in equations
                   for term in equation.terms if term isa KineticIonizationTerm]
    heatings = [term for equation in equations
                for term in equation.terms if term isa SecondaryHeatingTerm]
    for owner in fokker_planck
        settings = owner.settings
        settings.bound_friction === :ionization_split || continue
        paired = [k for k in ionizations if k.kinetic === owner.unknown]
        isempty(paired) &&
            throw(ArgumentError("the Fokker-Planck term of $(owner.unknown) takes bound_friction = :ionization_split, " *
                                "which charges f for the ionization that a KineticIonizationTerm of $(owner.unknown) " *
                                "counts, and the system has none: add the kinetic ionization or take " *
                                "bound_friction = :bethe"))
        for k in paired
            region = k.envelopes === nothing ? nothing : k.envelopes.region
            factor = region === nothing ? 0.0 :
                     region.limit isa ThermalMomentum{Symbol} ? region.limit.factor :
                     throw(ArgumentError("bound_friction = :ionization_split takes the threshold of the kinetic " *
                                         "ionization of $(owner.unknown) as a factor of the thermal momentum of the " *
                                         "temperature unknown (ThermalMomentum with the name of the unknown)"))
            factor == settings.ionization_threshold ||
                throw(ArgumentError("the kinetic ionization of $(owner.unknown) has its threshold at $factor thermal " *
                                    "momenta and the friction of bound_friction = :ionization_split at " *
                                    "$(settings.ionization_threshold) (ionization_threshold of CollisionSettings), so f " *
                                    "pays for other ionizations than it causes: give both the same factor"))
            k.ionization_cross_section_form === settings.ionization_cross_section_form ||
                throw(ArgumentError("the kinetic ionization of $(owner.unknown) takes the cross sections of the form " *
                                    ":$(k.ionization_cross_section_form) and the friction of bound_friction = :ionization_split " *
                                    "the form :$(settings.ionization_cross_section_form), so f pays for other " *
                                    "ionizations than it causes: give both the same form"))
            # where the system books the sinks of f, the energy of the secondaries that f pays
            # goes to the cold electrons only through the heating by secondaries
            books_sinks = any(
                term -> term isa KineticEnergyLossTerm && term.source === owner &&
                        term.part === :sinks,
                (term for equation in equations for term in equation.terms))
            books_sinks && !any(heating -> heating.kinetic_ionization === k, heatings) &&
                throw(ArgumentError("the Fokker-Planck term of $(owner.unknown) takes bound_friction = " *
                                    ":ionization_split and its sinks leave the electrons (KineticEnergyLossTerm with " *
                                    "part = :sinks), and the system has no SecondaryHeatingTerm of its kinetic " *
                                    "ionization: the energy of the secondaries, which join the cold electrons through " *
                                    "the closure, would be lost. Add SecondaryHeatingTerm(kinetic_ionization) to the " *
                                    "row of the temperature"))
            widths = merge(SECONDARY_SPECTRUM_WIDTHS, settings.secondary_widths)
            for heating in heatings
                heating.kinetic_ionization === k || continue
                heating.cut === :none ||
                    throw(ArgumentError("the heating by secondaries of the kinetic ionization of $(owner.unknown) takes " *
                                        "cut = :$(heating.cut), while the friction of bound_friction = " *
                                        ":ionization_split takes the mean over every secondary from f: take cut = :none"))
                heating.spectrum === settings.secondary_spectrum ||
                    throw(ArgumentError("the heating by secondaries takes the spectrum :$(heating.spectrum) and the " *
                                        "friction of bound_friction = :ionization_split the spectrum " *
                                        ":$(settings.secondary_spectrum): give both the same"))
                heating.spectrum === :beb ||
                    all(
                        ((name, width),) -> get(widths, atomic_element(name), NaN) == width,
                        zip(k.composition.names, heating.widths)) ||
                    throw(ArgumentError("the heating by secondaries and the friction of bound_friction = " *
                                        ":ionization_split take different widths of the secondary spectrum: give " *
                                        "SecondaryHeatingTerm the secondary_widths of the CollisionSettings"))
            end
        end
    end
    # a term that reads the coefficients of a Fokker-Planck term or an interface of the system
    # takes that term itself, not a copy with other coefficients. The energy terms take
    # the term of the system always. A boundary flux or an interface takes it when its plasma
    # reads unknowns: a copy is never prepared for the step, so with lagged ions it reads the
    # charge states of the iterate without their columns, and with lagged = :all it keeps the
    # coefficients of its construction, and the flux through it is wrong without a refusal.
    # A copy with a prescribed plasma rebuilds at its evaluation and stays allowed (the
    # operator of a conductivity model is one), and a bare operator is prescribed data
    # (_check_prescribed_operators).
    terms_of_system = [term for equation in equations for term in equation.terms]
    for equation in equations, term in equation.terms
        #! format: off
        sources = term isa KineticEnergyLossTerm ? (term.source,) :
                  term isa InterfaceEnergyFluxTerm ? (term.interface,) :
                  filter(s -> s isa FokkerPlanckTerm && is_state_dependent(s.plasma), _coefficient_sources(term))
        #! format: on
        for source in sources
            any(other -> other === source, terms_of_system) ||
                throw(ArgumentError("a $(nameof(typeof(term))) in the equation of $(equation.unknown) reads a " *
                                    "$(nameof(typeof(source))) that is not a term of the system: give it the term of the system"))
        end
    end
    # the energy exchange with the ions conserves energy only with both of its sides
    exchanges = [term for equation in equations
                 for term in equation.terms if term isa IonEnergyExchangeTerm]
    for term in exchanges
        other = term.side === :ions ? :electrons : :ions
        any(
            t -> t.side === other && t.ion_energy === term.ion_energy &&
                 t.temperature === term.temperature &&
                 t.density === term.density && t.ions === term.ions &&
                 _plasma_kind(t.plasma) == _plasma_kind(term.plasma),
            exchanges) ||
            throw(ArgumentError("the energy exchange of the ions has its side :$(term.side) without the side :$other on " *
                                "the same unknowns, so it does not conserve energy: give the system both sides"))
    end
    _check_ion_pairings(equations)
    _check_sink_regions(equations)
    _check_total_energy_rows(equations)
    lagged = [term for term in fokker_planck if _is_lagged(term)]
    isempty(lagged) && return nothing
    for equation in equations, term in equation.terms

        term isa ConductivityTerm && term.model.kinetic !== :none &&
            is_state_dependent(term.model.plasma) &&
            throw(ArgumentError("the corrected conductivity in the equation of $(equation.unknown) follows the " *
                                "temperature implicitly and the Fokker-Planck term of $(first(lagged).unknown) " *
                                "lagged, so the correction and the current of the distribution belong to different " *
                                "temperatures: lag no more than the ions of the Fokker-Planck term"))
    end
    return nothing
end

# the radial unknown of the ion densities that a term reads, `nothing` for prescribed ions,
# and :none for a term that reads no ions
_ions_read(term) = :none
function _ions_read(term::Union{FokkerPlanckTerm, FieldParticleTerm})
    term.plasma isa PlasmaFromUnknowns ? term.plasma.ions :
    term.plasma === nothing ? _ions_read_operator(term) : nothing
end
# a term without a plasma source reads the ions of the plasma its operator was built from
function _ions_read_operator(term::FokkerPlanckTerm)
    term.operator.coefficient_source[] === nothing ? :none : nothing
end
_ions_read_operator(term) = :none
function _ions_read(term::ConductivityTerm)
    term.model.plasma isa PlasmaFromUnknowns ? term.model.plasma.ions :
    term.model.plasma === nothing ? :none : nothing
end
function _ions_read(term::Union{RadiatedPowerTerm, IonEnergyExchangeTerm,
        RunawayFluidEnergyTransferTerm, ExternalAvalancheTerm})
    term.ions
end

# The terms of the ion charge states and the energies that read them: the split
# of the ionization with its kinetic part in the system and the temperature of its threshold,
# the same tables and extrapolation for the rates and the binding energy, the closure of the
# cold density next to a kinetic ionization, one source of ions where the system evolves them
# and one composition per ion unknown.
function _check_ion_pairings(equations)
    terms = [term for equation in equations for term in equation.terms]
    rates = [term for term in terms if term isa IonRateTerm]
    radiations = [term for term in terms if term isa RadiatedPowerTerm]
    kinetics = [term for term in terms if term isa KineticIonizationTerm]
    for owner in [rates; radiations]
        k = owner.kinetic_ionization
        k === nothing && continue
        any(t -> t === k, kinetics) ||
            throw(ArgumentError("the $(nameof(typeof(owner))) splits the ionization with a KineticIonizationTerm that " *
                                "is not a term of the system, so the share of the electrons above its threshold is " *
                                "left out: give the system that term"))
        owner.ions === nothing || k.ions === owner.ions ||
            throw(ArgumentError("the $(nameof(typeof(owner))) of the ion densities $(owner.ions) splits the ionization " *
                                "with a kinetic ionization of $(k.ions): give both the same ion unknown"))
        _temperature_source(k) == (:unknown, owner.temperature) ||
            throw(ArgumentError("the $(nameof(typeof(owner))) reads the temperature $(owner.temperature) and the " *
                                "threshold of its kinetic ionization another one, whose Jacobian columns it does not " *
                                "list: give both the same temperature unknown"))
    end
    for rate in rates, radiation in radiations

        radiation.ions === rate.ions || continue
        radiation.kinetic_ionization === rate.kinetic_ionization ||
            throw(ArgumentError("the IonRateTerm and the RadiatedPowerTerm of the ion densities $(rate.ions) split the " *
                                "ionization differently (one with a kinetic part, the other with another or none), so " *
                                "the binding energy is charged at another rate than the rate equations take: give " *
                                "both the same kinetic_ionization keyword"))
        radiation.extrapolation === rate.extrapolation ||
            throw(ArgumentError("the IonRateTerm and the RadiatedPowerTerm of the ion densities $(rate.ions) take the " *
                                "extrapolations :$(rate.extrapolation) and :$(radiation.extrapolation) of the ADAS " *
                                "tables: give both the same"))
        for (s, name) in enumerate(rate.composition.names)
            haskey(radiation.tables, name) &&
            radiation.tables[name].scd.source == rate.tables[s].scd.source ||
                throw(ArgumentError("the IonRateTerm and the RadiatedPowerTerm of the ion densities $(rate.ions) read " *
                                    "the ionization of $name from different ADF11 files: read both from one directory"))
        end
    end
    closures = Set(term.unknown for term in terms if term isa WeightedSumTerm)
    for k in kinetics
        k.ions in closures ||
            throw(ArgumentError("the kinetic ionization of $(k.ions) raises the charge of the ions, and no closure of " *
                                "the cold density reads those ions, so no electron appears: add the " *
                                "population_equations of the composition"))
    end
    # where the system evolves the charge states, every term that reads ions takes them from it
    #! format: off
    evolved = Set(term.ions for term in terms if term isa Union{IonRateTerm,KineticIonizationTerm,IonDiffusionTerm})
    #! format: on
    if !isempty(evolved)
        for term in terms
            read = _ions_read(term)
            read === :none && continue
            read in evolved ||
                throw(ArgumentError("a $(nameof(typeof(term))) reads " *
                                    (read === nothing ? "prescribed ions" :
                                     "the ions $read") *
                                    " while the system evolves the charge states of " *
                                    "$(join(sort!(collect(evolved)), " and ")): give it the ion unknown (the ions " *
                                    "keyword, or PlasmaFromUnknowns(...; ions))"))
        end
    end
    compositions = Dict{Symbol, IonComposition}()
    for term in terms
        term isa Union{IonRateTerm, KineticIonizationTerm, IonDiffusionTerm} || continue
        known = get!(compositions, term.ions, term.composition)
        known.names == term.composition.names &&
        known.nuclear_charges == term.composition.nuclear_charges ||
            throw(ArgumentError("two terms of the ion unknown $(term.ions) take the species " *
                                "$(join(known.names, ", ")) and $(join(term.composition.names, ", ")): give them one " *
                                "composition in one order"))
    end
    for term in terms
        term isa SecondaryHeatingTerm || continue
        _temperature_source(term.kinetic_ionization) in
        (nothing, (:unknown, term.temperature)) ||
            throw(ArgumentError("the heating by secondaries reads the temperature $(term.temperature) and the threshold " *
                                "of its kinetic ionization another one: give both the same temperature unknown"))
    end
    return nothing
end

# A sinks term charged to the total energy holds the hot region of W_hot: without a region it
# charges the bulk as well, whose radiation the cold electrons pay through RadiatedPowerTerm.
# The regions themselves are compared with the other moments.
function _check_sink_regions(equations)
    terms = [term for equation in equations for term in equation.terms]
    for sinks in terms
        sinks isa KineticEnergyLossTerm && sinks.part === :sinks &&
        sinks.envelopes === nothing || continue
        any(
            term -> term isa EnergyMomentTerm && term.kinetic === sinks.kinetic &&
                    term.envelopes !== nothing &&
                    term.envelopes.region.limit isa ThermalMomentum,
            terms) &&
            throw(ArgumentError("the sinks of $(sinks.kinetic) are charged without a region while the energy of its " *
                                "hot electrons is a moment over a thermal threshold, so the sinks of the bulk are " *
                                "charged as well: give KineticEnergyLossTerm the region of the energy moment"))
    end
    return nothing
end

# whether a Fokker-Planck term holds collisions, whose flux through p_max and sinks move energy
function _has_collisions(term::FokkerPlanckTerm)
    term.plasma !== nothing || term.operator.coefficient_source[] !== nothing
end
# whether its collisions have sinks: radiation or the friction on bound electrons, which the
# partially screened frequencies hold and the non-screened ones as well, since their target
# density counts the bound electrons
function _has_sinks(term::FokkerPlanckTerm)
    term.synchrotron || term.settings.bremsstrahlung === :stopping_power ||
        term.settings.screening in (:partial, :none)
end
# whether two regions count the same electrons: the same limit, side, envelope and width
function _same_limit(a::ThermalMomentum, b::ThermalMomentum)
    a.temperature == b.temperature && a.factor == b.factor
end
_same_limit(a, b) = a == b
function _same_region(a::MomentumRegion, b::MomentumRegion)
    _same_limit(a.limit, b.limit) && a.side === b.side && a.envelope === b.envelope &&
        a.smooth_width == b.smooth_width
end
_same_region(a, b) = a === nothing && b === nothing
_region_of(envelopes) = envelopes === nothing ? nothing : envelopes.region
# whether a term moves f in radius
function _moves_in_radius(term::FokkerPlanckTerm)
    any(!iszero, term.operator.diffusion_r_r) || any(!iszero, term.operator.advection_r)
end

# The row of a total energy holds the time derivative of W_tot = W_cold + W_hot, with W_hot
# the energy moments of the distributions in the closure of W_tot, so every change of W_hot
# that no term of the row lists is charged to the cold electrons. For each distribution of
# W_hot the row lists the sinks of its collisions, the energy that leaves through p_max (the
# flux of the boundary, or the interface to the runaway grid), and refuses the channels
# that no row term carries: radial transport of f and knock-on secondaries on its grid.
function _check_total_energy_rows(equations)
    for row in equations
        row.kind isa TimeDerivative && row.kind.of !== nothing || continue
        total = row.kind.of
        closures = [e for e in equations if e.unknown === total && e.kind isa Algebraic]
        isempty(closures) && continue
        distributions = unique([t.kinetic for e in closures
                                for t in e.terms if t isa EnergyMomentTerm])
        listed = row.terms
        for d in distributions, equation in equations

            equation.unknown === d || continue
            #! format: off
            moment = first(t for e in closures for t in e.terms if t isa EnergyMomentTerm && t.kinetic === d)
            #! format: on
            for term in equation.terms
                if term isa FokkerPlanckTerm && _has_collisions(term)
                    # the sinks term of this Fokker-Planck term, charged as a loss, over the
                    # region of the energy moment
                    sinks = [t
                             for t in listed
                             if t isa KineticEnergyLossTerm && t.source === term &&
                                    t.part === :sinks]
                    _has_sinks(term) && isempty(sinks) &&
                        throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy " *
                                            "$total, and the collisions of $d have sinks (radiation or bound electrons) " *
                                            "that no term of the row lists, so the cold electrons would pay them: add " *
                                            "KineticEnergyLossTerm(term; part = :sinks, region) to the row"))
                    for t in sinks
                        t.coefficient < 0 ||
                            throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy " *
                                                "$total, and its sinks term of $d has the coefficient $(t.coefficient), " *
                                                "which adds the sinks to the total in place of removing them: give it " *
                                                "a negative coefficient"))
                        _same_region(_region_of(t.envelopes), _region_of(moment.envelopes)) ||
                            throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy " *
                                                "$total, and its sinks term of $d takes another region than the energy " *
                                                "moment of $d, so it charges electrons outside W_hot: give it the region " *
                                                "of the moment"))
                    end
                    boundary = term.operator.p_upper_boundary
                    if boundary === :interface
                        crossings = [t
                                     for t in listed
                                     if t isa InterfaceEnergyFluxTerm &&
                            t.interface.lower === d]
                        isempty(crossings) &&
                            throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy " *
                                                "$total, and the electrons of $d that cross the interface to the runaway " *
                                                "grid leave W_hot without a term of the row: add InterfaceEnergyFluxTerm"))
                        for t in crossings
                            t.interface.lower_term === term ||
                                throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total " *
                                                    "energy $total, and its interface term of $d reads the interface of " *
                                                    "another term of $d: give it the interface of this term"))
                            t.coefficient < 0 ||
                                throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total " *
                                                    "energy $total, and its interface term of $d has the coefficient " *
                                                    "$(t.coefficient), which adds the energy that crosses the interface " *
                                                    "to the total in place of removing it: give it a negative coefficient"))
                        end
                    else
                        any(
                            t -> t isa MomentumBoundaryFluxTerm && t.kinetic === d &&
                                 t.source === term &&
                                 t.weight === :energy && t.coefficient < 0,
                            listed) ||
                            throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy " *
                                                "$total, and the electrons of $d that leave the grid through p_max " *
                                                "(p_upper_boundary = :$boundary) take their energy without a term of the " *
                                                "row: add MomentumBoundaryFluxTerm($d, term; weight = :energy, " *
                                                "coefficient = -1) of this term"))
                    end
                end
                (term isa FokkerPlanckTerm && _moves_in_radius(term) ||
                 term isa RadialEdgeLossTerm) &&
                    throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy $total, " *
                                        "and $d moves in radius, whose energy flux no term of the row carries: the " *
                                        "electrons that move out would heat the cold electrons where they leave. Radial " *
                                        "transport of f is refused under the total energy"))
                term isa KnockOnTerm &&
                    throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy $total, " *
                                        "and the knock-on secondaries on the grid of $d take their energy from the " *
                                        "cold electrons, since no term carries the energy of the secondaries: refused"))
                term isa RosenbluthPutvinskiSourceTerm &&
                    throw(ArgumentError("the row of $(row.unknown) holds the time derivative of the total energy $total, " *
                                        "and the knock-ons of the fluid runaways above the grid of $d take their energy " *
                                        "from the cold electrons, since the primaries have no energy unknown: refused"))
            end
        end
    end
    return nothing
end
