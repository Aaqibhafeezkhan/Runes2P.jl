# Equation system M du/dt = rhs(u, t): one equation per unknown, built from terms; the
# stored sparse Jacobian from colored AD over the `:sparse` implicit terms only, and
# matrix-free products for the `:matrix_free` terms.

using LinearAlgebra: Diagonal
using SparseArrays: nnz, findnz, nonzeros, nzrange, rowvals
using SparseMatrixColorings: column_groups

"""
    TimeDerivative()
    TimeDerivative(of)
    Algebraic()
    Constraint()

Kinds of [`Equation`](@ref): `du/dt = Σ terms` or `0 = Σ terms`. `TimeDerivative(of)`
puts the time derivative of another unknown into the row, `d(of)/dt = Σ terms`, for an
unknown `of` of the same kind and sizes as the unknown of the equation.
Faraday's law is of this form: the equation of the electric field is
`dψ/dt = 2π E`, with the poloidal flux as `of`, and the equation of the flux is Ampère's
law, which is algebraic. Every algebraic row then holds its own unknown. The unknown of
an equation with the derivative of another unknown must appear in the terms of that
equation.

`Algebraic()` is an equation `0 = Σ terms` that holds its unknown. `Constraint()` is one
that does not: the unknown is fixed through the equations of other unknowns, which hold
it. The magnitude of the particle source is of this kind: its equation is the density that
the distribution has to keep, `0 = n(f) − n_cold − n_hot`, and the magnitude stands in the
kinetic equation (DREAM's `S_particle`). The system then has the index 2
in that unknown, which backward Euler solves, and the diagonal entry of the row in the
Jacobian is zero. A constraint names the scale of the quantity it balances
(`Equation(...; scale)`): its unknown is a rate, whose scale does not fit the row.
"""
struct TimeDerivative
    of::Union{Nothing,Symbol}
end
TimeDerivative() = TimeDerivative(nothing)
struct Algebraic end
struct Constraint end

"""
    Equation(unknown, kind, terms...; scale=nothing)

Equation of the unknown `unknown` (a name in the layout): `TimeDerivative()`,
`TimeDerivative(of)` or `Algebraic()`, with the right-hand side the sum of `terms`
([`EquationTerm`](@ref)s).

`scale` is the magnitude of the quantity whose rate or balance the row holds, for the
weights of the residual ([`residual_scales`](@ref)). By default it is the scale of the
unknown whose time derivative the row holds, and the scale of the unknown of the equation
for an algebraic row. An algebraic row that balances another quantity than its unknown
names the scale of that quantity: the row of the poloidal flux holds Ampère's law.
"""
struct Equation{K,TT<:Tuple}
    unknown::Symbol
    kind::K
    terms::TT
    scale::Union{Nothing,Float64}
end
function Equation(unknown::Symbol, kind, terms::Tuple; scale::Union{Nothing,Real}=nothing)
    scale === nothing || scale > 0 || throw(ArgumentError("the scale of an equation must be positive"))
    return Equation(unknown, kind, terms, scale === nothing ? nothing : Float64(scale))
end
Equation(unknown::Symbol, kind, terms::EquationTerm...; scale::Union{Nothing,Real}=nothing) =
    Equation(unknown, kind, terms; scale)

"""
    Tolerance(absolute, relative)

Convergence tolerance of one unknown: an entry passes when `|δ| ≤ absolute + relative |x|`,
with `absolute` in the unknown's physical units.
"""
struct Tolerance
    absolute::Float64
    relative::Float64
end

# a term bound to the id of its row unknown
struct BoundTerm{E<:EquationTerm}
    term::E
    row::Int
end

"""
    EquationSystem(layout, equations; time_scale=1, newton_tolerances, time_step_tolerances,
                   initial_time=0)

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
(defaults from [`tolerance`](@ref)). The time-step tolerances are stored for time-step
error control, which [`integrate`](@ref) does not have yet (fixed steps with halving on
failure).

The construction evaluates the implicit `:sparse` terms once, at the zero state and the
time `initial_time`, for the structure of the stored Jacobian. Set `initial_time` to the
start of the run when a term is defined only on the run's time interval (the `plasma_at`
of a [`PrescribedFokkerPlanckTerm`](@ref)).
"""
struct EquationSystem{T,S<:Tuple,M<:Tuple,X<:Tuple,C<:Tuple,D<:Tuple,F,P,B}
    layout::UnknownLayout{T}
    equations::Vector{Equation}
    sparse_terms::S
    matrix_free_terms::M
    explicit_terms::X
    correction_terms::C
    direct_terms::D                       # the sparse terms that assemble their own block
    automatic_jacobian::SparseMatrixCSC{T,Int}  # the AD part, with the pattern of the coloring
    automatic_positions::Vector{Int}      # its stored entries in the stored Jacobian
    stored_jacobian_exact::Bool
    linear::Bool
    mass::Vector{T}                       # per row: 1 with a time derivative, 0 without
    mass_columns::Vector{Int}             # per row: the entry whose time derivative the row holds
    row_scales::Vector{T}                 # per row: magnitude of the quantity of the row
    pattern::SparseMatrixCSC{Bool,Int}
    sparse_rhs!::F
    preparation::P
    backend::B
    prototype::SparseMatrixCSC{T,Int}
    time::Base.RefValue{T}
    buffer::Vector{T}
    product::Vector{T}
    time_scale::T
    newton_tolerances::Dict{Symbol,Tolerance}
    time_step_tolerances::Dict{Symbol,Tolerance}
end

# Add the right-hand sides of a tuple of bound terms, unrolled at compile time.
@inline _add_terms!(out, ::Tuple{}, u, layout, t) = out
@inline function _add_terms!(out, terms::Tuple, u, layout, t)
    bound = first(terms)
    rhs!(out, bound.term, u, layout, bound.row, t)
    return _add_terms!(out, Base.tail(terms), u, layout, t)
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

@inline _add_corrections!(out, ::Tuple{}, u, v, layout, t) = out
@inline function _add_corrections!(out, terms::Tuple, u, v, layout, t)
    bound = first(terms)
    jacobian_correction_product!(out, bound.term, u, v, layout, bound.row, t)
    return _add_corrections!(out, Base.tail(terms), u, v, layout, t)
end

function EquationSystem(layout::UnknownLayout{T}, equations::AbstractVector;
                        time_scale::Real=1.0,
                        newton_tolerances::AbstractDict=Dict{Symbol,Tolerance}(),
                        time_step_tolerances::AbstractDict=Dict{Symbol,Tolerance}(),
                        initial_time::Real=0.0) where {T}
    count = length(layout.unknowns)
    rows = zeros(Int, count)
    for eq in equations
        id = unknown_id(layout, eq.unknown)
        rows[id] == 0 || throw(ArgumentError("unknown $(eq.unknown) has more than one equation"))
        rows[id] = 1
    end
    missing_equations = [layout.unknowns[id].name for id in 1:count if rows[id] == 0]
    isempty(missing_equations) || throw(ArgumentError("no equation for $(join(missing_equations, ", "))"))

    _check_conductivity_correction(equations)
    sparse_terms, matrix_free_terms, explicit_terms = BoundTerm[], BoundTerm[], BoundTerm[]
    mass = zeros(T, length(layout))
    mass_columns = collect(1:length(layout))
    row_scales = T.(unknown_scales(layout))
    derivative_rows = zeros(Int, count)   # per unknown: the unknown whose row holds its time derivative
    for eq in equations
        row = unknown_id(layout, eq.unknown)
        eq.kind isa Union{TimeDerivative,Algebraic,Constraint} ||
            throw(ArgumentError("unknown equation kind $(eq.kind)"))
        # explicit terms are evaluated at the previous time level and determine nothing
        holds_unknown = any(term -> treatment(term) === :implicit && eq.unknown in dependencies(term), eq.terms)
        if eq.kind isa Algebraic
            holds_unknown ||
                throw(ArgumentError("the algebraic equation of $(eq.unknown) has no implicit term " *
                                    "depending on $(eq.unknown), so it cannot determine it"))
        elseif eq.kind isa Constraint
            eq.scale === nothing &&
                throw(ArgumentError("the equation of $(eq.unknown) is a constraint and needs the scale of the " *
                                    "quantity it balances (Equation(...; scale)): the scale of its unknown, a " *
                                    "rate, does not fit the row"))
            any(other -> other.unknown !== eq.unknown &&
                         any(term -> treatment(term) === :implicit && eq.unknown in dependencies(term), other.terms),
                equations) ||
                throw(ArgumentError("the equation of $(eq.unknown) is a constraint, and no other equation has an " *
                                    "implicit term depending on $(eq.unknown), so nothing determines it"))
        else
            of = eq.kind.of === nothing ? row : unknown_id(layout, eq.kind.of)
            if of != row
                layout.kinds[of] === layout.kinds[row] && layout.sizes[of, :] == layout.sizes[row, :] ||
                    throw(ArgumentError("the equation of $(eq.unknown) holds the time derivative of " *
                                        "$(eq.kind.of), which needs the same kind and sizes"))
                holds_unknown ||
                    throw(ArgumentError("the equation of $(eq.unknown) holds the time derivative of " *
                                        "$(eq.kind.of) and has no implicit term depending on $(eq.unknown), " *
                                        "so it cannot determine it"))
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
            validate(term, layout, row)
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
    sparse_tuple, matrix_free_tuple, explicit_tuple = Tuple(sparse_terms), Tuple(matrix_free_terms),
                                                      Tuple(explicit_terms)
    correction_tuple = Tuple(bound for bound in sparse_terms if jacobian_kind(bound.term) === :approximate)
    linear = all(bound -> is_linear(bound.term), sparse_terms) && all(bound -> is_linear(bound.term), matrix_free_terms)

    for bound in sparse_terms
        jacobian_assembly(bound.term) in (:automatic, :direct) ||
            throw(ArgumentError("unknown Jacobian assembly $(jacobian_assembly(bound.term)) of $(typeof(bound.term))"))
        jacobian_assembly(bound.term) === :direct && jacobian_kind(bound.term) === :approximate &&
            throw(ArgumentError("$(typeof(bound.term)) assembles its block itself, so the block is exact"))
    end
    automatic_tuple = Tuple(bound for bound in sparse_terms if jacobian_assembly(bound.term) === :automatic)
    direct_tuple = Tuple(bound for bound in sparse_terms if jacobian_assembly(bound.term) === :direct)

    # structural pattern of the stored Jacobian: the diagonal (for M/Δt) and the sparse
    # terms. The coloring takes the part of the terms with automatic assembly
    n = length(layout)
    entries = Tuple{Int,Int}[(i, i) for i in 1:n]
    # the entries of the mass matrix off the diagonal
    append!(entries, (i, mass_columns[i]) for i in 1:n if mass_columns[i] != i)
    for bound in automatic_tuple
        sparsity!(entries, bound.term, layout, bound.row)
    end
    as_pattern(list) = sparse(first.(list), last.(list), trues(length(list)), n, n, (a, b) -> a | b)
    automatic_pattern = as_pattern(entries)
    for bound in direct_tuple
        sparsity!(entries, bound.term, layout, bound.row)
    end
    pattern = as_pattern(entries)

    time = Ref(T(initial_time))
    # residual of the part that AD differentiates: the coloring must not see matrix-free
    # terms, whose entries outside the pattern would be folded onto entries inside it
    sparse_rhs! = let terms = automatic_tuple, layout = layout, time = time
        (du, u) -> (fill!(du, zero(eltype(du))); _add_terms!(du, terms, u, layout, time[]); nothing)
    end
    backend = sparse_ad_backend(pattern=automatic_pattern)
    x0 = zeros(T, n)
    buffer = zeros(T, n)
    preparation = DI.prepare_jacobian(sparse_rhs!, buffer, backend, x0)
    automatic_jacobian = DI.jacobian(sparse_rhs!, buffer, preparation, backend, x0)
    # the prototype is evaluated at u = 0 and t = initial_time, where some derivatives
    # vanish; it must still carry the whole pattern, or jacobian! at other states would
    # miss entries
    nnz(automatic_jacobian) == nnz(automatic_pattern) ||
        error("the Jacobian prototype lost structural entries ($(nnz(automatic_jacobian)) of " *
              "$(nnz(automatic_pattern)))")
    if isempty(direct_tuple)
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
            automatic_positions[k] = searchsortedfirst(stored_rows, automatic_rows[k], first(range), last(range),
                                                       Base.Order.Forward)
        end
        _merge_jacobian!(prototype, automatic_jacobian, automatic_positions)
        _add_jacobians!(prototype, direct_tuple, x0, layout, T(initial_time))
    end
    return EquationSystem(layout, collect(Equation, equations), sparse_tuple, matrix_free_tuple,
                          explicit_tuple, correction_tuple, direct_tuple, automatic_jacobian, automatic_positions,
                          isempty(correction_tuple), linear, mass, mass_columns, row_scales, pattern,
                          sparse_rhs!, preparation, backend, prototype,
                          time, buffer, zeros(T, n), T(time_scale), Dict{Symbol,Tolerance}(newton_tolerances),
                          Dict{Symbol,Tolerance}(time_step_tolerances))
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

_unknown_ranges(layout, id) = layout.kinds[id] === :scalar ? (unknown_range(layout, id),) :
                              (unknown_range(layout, id, ir) for ir in 1:radial_cell_count(layout))

"""
    rhs!(du, u, system::EquationSystem, t)

Right-hand side of `M du/dt = rhs(u, t)` with all terms at `u` (the ODE form, as passed to
a general integrator).
"""
function rhs!(du::AbstractVector, u::AbstractVector, system::EquationSystem, t)
    fill!(du, zero(eltype(du)))
    layout = system.layout
    _add_terms!(du, system.sparse_terms, u, layout, t)
    _add_terms!(du, system.matrix_free_terms, u, layout, t)
    _add_terms!(du, system.explicit_terms, u, layout, t)
    return du
end

"""
    residual!(F, u, u_old, system::EquationSystem, t, dt)

Backward-Euler residual `F = M (u - u_old)/dt - rhs(u, t)`, with explicit terms at
`u_old`. `M` has one entry per row with a time derivative, in the column of the unknown
whose derivative the row holds. Allocation free.
"""
function residual!(F::AbstractVector, u::AbstractVector, u_old::AbstractVector,
                   system::EquationSystem, t, dt)
    rhs = system.buffer
    fill!(rhs, zero(eltype(rhs)))
    layout = system.layout
    _add_terms!(rhs, system.sparse_terms, u, layout, t)
    _add_terms!(rhs, system.matrix_free_terms, u, layout, t)
    _add_terms!(rhs, system.explicit_terms, u_old, layout, t)
    mass, columns = system.mass, system.mass_columns
    @inbounds for i in eachindex(F)
        column = columns[i]
        F[i] = mass[i] * (u[column] - u_old[column]) / dt - rhs[i]
    end
    return F
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
    jacobian_prototype(system)

Structural pattern of the stored Jacobian (the diagonal and the implicit `:sparse` terms),
and a sparse matrix with that structure, to be filled by [`jacobian!`](@ref).
"""
jacobian_sparsity(system::EquationSystem) = system.pattern
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
    if isempty(system.direct_terms)
        DI.jacobian!(system.sparse_rhs!, system.buffer, J, system.preparation, system.backend, u)
        return J
    end
    automatic = system.automatic_jacobian
    DI.jacobian!(system.sparse_rhs!, system.buffer, automatic, system.preparation, system.backend, u)
    _merge_jacobian!(J, automatic, system.automatic_positions)
    _add_jacobians!(J, system.direct_terms, u, system.layout, t)
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
    return stored + LinearOperator{T,Vector{T}}(n, n, false, false, matrix_free!, nothing, nothing)
end

"""
    residual_scales(system)

Scale of every row of the residual: `scale/time_scale` for rows with a time derivative
and `scale` for algebraic rows. `scale` is the scale of the equation when it names one,
else the reference scale of the unknown whose time derivative the row holds, else that of
the row's unknown. Dividing the residual by these makes all rows of order one. The Newton
iteration of [`NewtonSolver`](@ref) does not use them: it weights differential rows by
`dt/scale`, since round-off in `(u - u_old)/dt` at a short step would exceed a tolerance
scaled by a fixed `time_scale`.
"""
function residual_scales(system::EquationSystem)
    scales = system.row_scales
    return [system.mass[i] == 0 ? scales[i] : scales[i] / system.time_scale for i in eachindex(scales)]
end

"""
    tolerance(system, name; kind=:newton)

[`Tolerance`](@ref) of the unknown `name` for Newton convergence (`kind = :newton`) or
time-step error control (`kind = :time_step`); by default `relative = 1e-10` and
`absolute = 1e-10` times the unknown's reference scale.
"""
function tolerance(system::EquationSystem, name::Symbol; kind::Symbol=:newton)
    settings = kind === :newton ? system.newton_tolerances :
               kind === :time_step ? system.time_step_tolerances :
               throw(ArgumentError("kind must be :newton or :time_step"))
    haskey(settings, name) && return settings[name]
    return Tolerance(1e-10 * unknown(system.layout, name).scale, 1e-10)
end

"""
    ode_function(system::EquationSystem)

SciML `ODEFunction` of the system, for OrdinaryDiffEq and NonlinearSolve. Defined by the
package extension `Runes2PSciMLBaseExt`, which loads with SciMLBase (`using SciMLBase`).
"""
function ode_function end
