# Equation system M du/dt = rhs(u, t): one equation per unknown, built from terms; the
# stored sparse Jacobian from colored AD over the `:sparse` implicit terms only, and
# matrix-free products for the `:matrix_free` terms.

using LinearAlgebra: Diagonal
using SparseArrays: nnz
using SparseMatrixColorings: ncolors

"""
    TimeDerivative()
    Algebraic()

Kinds of [`Equation`](@ref): `du/dt = Σ terms` (mass 1) or `0 = Σ terms` (mass 0).
"""
struct TimeDerivative end
struct Algebraic end

"""
    Equation(unknown, kind, terms...)

Equation of the unknown `unknown` (a name in the layout): `TimeDerivative()` or
`Algebraic()`, with the right-hand side the sum of `terms` ([`EquationTerm`](@ref)s).
"""
struct Equation{K,TT<:Tuple}
    unknown::Symbol
    kind::K
    terms::TT
end
Equation(unknown::Symbol, kind, terms::EquationTerm...) = Equation(unknown, kind, terms)

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
    EquationSystem(layout, equations; time_scale=1, newton_tolerances, time_step_tolerances)

The system `M du/dt = rhs(u, t)` for the unknowns of `layout`, with one
[`Equation`](@ref) per unknown. On construction it checks that every unknown has exactly
one equation, that every term's dependencies are unknowns of the layout (and passes the
term's own checks), and that every algebraic unknown appears in its own equation (some
term depends on it; the numerical regularity of the algebraic block is not checked). The
terms are split into three groups:

- implicit `:sparse` terms, whose Jacobian is stored ([`jacobian!`](@ref), colored AD over
  these terms alone);
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
(defaults from [`tolerance`](@ref)).
"""
struct EquationSystem{T,S<:Tuple,M<:Tuple,X<:Tuple,C<:Tuple,F,P,B}
    layout::UnknownLayout{T}
    equations::Vector{Equation}
    sparse_terms::S
    matrix_free_terms::M
    explicit_terms::X
    correction_terms::C
    stored_jacobian_exact::Bool
    linear::Bool
    mass::Vector{T}
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

@inline _add_corrections!(out, ::Tuple{}, u, v, layout, t) = out
@inline function _add_corrections!(out, terms::Tuple, u, v, layout, t)
    bound = first(terms)
    jacobian_correction_product!(out, bound.term, u, v, layout, bound.row, t)
    return _add_corrections!(out, Base.tail(terms), u, v, layout, t)
end

function EquationSystem(layout::UnknownLayout{T}, equations::AbstractVector;
                        time_scale::Real=1.0,
                        newton_tolerances::AbstractDict=Dict{Symbol,Tolerance}(),
                        time_step_tolerances::AbstractDict=Dict{Symbol,Tolerance}()) where {T}
    count = length(layout.unknowns)
    rows = zeros(Int, count)
    for eq in equations
        id = unknown_id(layout, eq.unknown)
        rows[id] == 0 || throw(ArgumentError("unknown $(eq.unknown) has more than one equation"))
        rows[id] = 1
    end
    missing_equations = [layout.unknowns[id].name for id in 1:count if rows[id] == 0]
    isempty(missing_equations) || throw(ArgumentError("no equation for $(join(missing_equations, ", "))"))

    sparse_terms, matrix_free_terms, explicit_terms = BoundTerm[], BoundTerm[], BoundTerm[]
    mass = zeros(T, length(layout))
    for eq in equations
        row = unknown_id(layout, eq.unknown)
        eq.kind isa Union{TimeDerivative,Algebraic} ||
            throw(ArgumentError("unknown equation kind $(eq.kind)"))
        if eq.kind isa Algebraic
            any(term -> eq.unknown in dependencies(term), eq.terms) ||
                throw(ArgumentError("the algebraic equation of $(eq.unknown) has no term depending on " *
                                    "$(eq.unknown), so it cannot determine it"))
        else
            for range in _unknown_ranges(layout, row)
                mass[range] .= one(T)
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

    # structural pattern of the stored Jacobian: the diagonal (for M/Δt) and the sparse terms
    n = length(layout)
    entries = Tuple{Int,Int}[(i, i) for i in 1:n]
    for bound in sparse_terms
        sparsity!(entries, bound.term, layout, bound.row)
    end
    pattern = sparse(first.(entries), last.(entries), trues(length(entries)), n, n, (a, b) -> a | b)

    time = Ref(zero(T))
    # residual of the stored part alone: the coloring must not see matrix-free terms, whose
    # entries outside the pattern would be folded onto entries inside it
    sparse_rhs! = let terms = sparse_tuple, layout = layout, time = time
        (du, u) -> (fill!(du, zero(eltype(du))); _add_terms!(du, terms, u, layout, time[]); nothing)
    end
    backend = sparse_ad_backend(pattern=pattern)
    x0 = zeros(T, n)
    buffer = zeros(T, n)
    preparation = DI.prepare_jacobian(sparse_rhs!, buffer, backend, x0)
    prototype = DI.jacobian(sparse_rhs!, buffer, preparation, backend, x0)
    # the prototype is evaluated at u = 0, where some derivatives vanish; it must still carry
    # the whole pattern, or jacobian! at other states would miss entries
    nnz(prototype) == nnz(pattern) ||
        error("the Jacobian prototype lost structural entries ($(nnz(prototype)) of $(nnz(pattern)))")
    return EquationSystem(layout, collect(Equation, equations), sparse_tuple, matrix_free_tuple,
                          explicit_tuple, correction_tuple, isempty(correction_tuple), linear, mass, pattern,
                          sparse_rhs!, preparation, backend, prototype,
                          time, buffer, zeros(T, n), T(time_scale), Dict{Symbol,Tolerance}(newton_tolerances),
                          Dict{Symbol,Tolerance}(time_step_tolerances))
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
`u_old`. Allocation free.
"""
function residual!(F::AbstractVector, u::AbstractVector, u_old::AbstractVector,
                   system::EquationSystem, t, dt)
    rhs = system.buffer
    fill!(rhs, zero(eltype(rhs)))
    layout = system.layout
    _add_terms!(rhs, system.sparse_terms, u, layout, t)
    _add_terms!(rhs, system.matrix_free_terms, u, layout, t)
    _add_terms!(rhs, system.explicit_terms, u_old, layout, t)
    @inbounds for i in eachindex(F)
        F[i] = system.mass[i] * (u[i] - u_old[i]) / dt - rhs[i]
    end
    return F
end

"""
    mass_matrix(system)

Diagonal mass matrix `M`: 1 on the rows of differential unknowns, 0 on algebraic ones.
"""
mass_matrix(system::EquationSystem) = Diagonal(system.mass)

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
derivatives (dual-number passes over the sparse terms) one [`jacobian!`](@ref) costs.
"""
jacobian_colors(system::EquationSystem) = ncolors(system.preparation.coloring_result)

"""
    jacobian!(J, system::EquationSystem, u, t)

Values of the stored Jacobian `∂rhs/∂u` of the implicit `:sparse` terms at `(u, t)`, into
`J` (from [`jacobian_prototype`](@ref)), by colored forward-mode AD; for terms with an
approximate Jacobian ([`jacobian_kind`](@ref)) their stored block, not the derivative.
Matrix-free and explicit terms are not included.
"""
function jacobian!(J::SparseMatrixCSC, system::EquationSystem, u::AbstractVector, t)
    system.time[] = t
    DI.jacobian!(system.sparse_rhs!, system.buffer, J, system.preparation, system.backend, u)
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

Scale of every row of the residual: `scale/time_scale` for differential rows and `scale`
for algebraic rows, with `scale` the reference scale of the row's unknown. Dividing the
residual by these makes all rows of order one.
"""
function residual_scales(system::EquationSystem)
    scales = unknown_scales(system.layout)
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
