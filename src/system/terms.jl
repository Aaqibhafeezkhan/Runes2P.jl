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
The functions that callers use are exported. [`validate`](@ref), whose name is common to
many packages, [`add_local_jacobian!`](@ref), [`lags_in_ode_form`](@ref) and the plasma
columns of a term ([`has_plasma_columns`](@ref), [`plasma_column_sparsity!`](@ref),
[`add_plasma_columns!`](@ref)), which a term implements and the system calls, are declared
`public` and not exported.
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
function rhs! end

"""
    sparsity!(entries, term, layout, row)

Push the `(row index, column index)` pairs of the structural nonzeros of the Jacobian
block of a `:sparse` [`EquationTerm`](@ref) (see [`coupling`](@ref)) to `entries`.
"""
function sparsity! end

"""
    jacobian_vector_product!(out, term, u, v, layout, row, t)

Add the product `(∂ rhs_term / ∂u) v` of a `:matrix_free` [`EquationTerm`](@ref) (see
[`coupling`](@ref)) to `out`.
"""
function jacobian_vector_product! end

"""
    validate(term, layout, row[, t])

Check `term` against the layout `layout` of its system, in the equation of the unknown `row`,
and throw an `ArgumentError` when it does not fit; run when an [`EquationSystem`](@ref) is
built, with its initial time `t` for terms that evaluate a function of time. The default
accepts every term.
"""
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
the pattern of `sparsity!`. A method may allocate work arrays per call, such as the dual
numbers of [`ParticleSourceTerm`](@ref), [`InterfaceParticleFluxTerm`](@ref) and
[`InterfaceEnergyFluxTerm`](@ref) or the row of [`KineticEnergyLossTerm`](@ref).
"""
function add_jacobian! end

# add `value` to the stored entry (row, column) of J
@inline function _add_entry!(J::SparseMatrixCSC, row::Integer, column::Integer, value)
    nonzeros(J)[_pattern_position(J, row, column)] += value
    return J
end

# the position in the stored values of J of the entry (row, column), an `ArgumentError` when
# the entry is not in the pattern
@inline function _pattern_position(J::SparseMatrixCSC, row::Integer, column::Integer)
    range = nzrange(J, column)
    rows = rowvals(J)
    k = searchsortedfirst(rows, row, first(range), last(range), Base.Order.Forward)
    (k <= last(range) && rows[k] == row) ||
        throw(ArgumentError("the entry ($row, $column) is not in the pattern of the stored Jacobian"))
    return k
end

# the position in the stored values of J of the entry (first(rows), column), where the rows
# `rows` of the column are stored one after the other: the entry of row r is at the position
# plus r - first(rows); the stored rows of a column increase strictly, so the run holds every
# row of `rows` when its first and last positions hold the first and last rows, and otherwise
# an entry is not in the pattern, an `ArgumentError` as in `_add_entry!`
function _run_start(J::SparseMatrixCSC, rows::UnitRange{Int}, column::Integer)
    isempty(rows) && throw(ArgumentError("an empty run of rows in column $column"))
    range = nzrange(J, column)
    stored = rowvals(J)
    start = searchsortedfirst(
        stored, first(rows), first(range), last(range), Base.Order.Forward)
    stop = start + length(rows) - 1
    if !(stop <= last(range) && stored[start] == first(rows) && stored[stop] == last(rows))
        throw(ArgumentError("the entries ($rows, $column) are not all in the pattern of the stored Jacobian"))
    end
    return start
end

# add partial k of the dual number `value` to the entry (row, columns[k]) of J, for every k
# in the order of `columns`
@inline function _add_partials!(J::SparseMatrixCSC, row::Integer, columns::Vector{Int}, value)
    length(columns) == ForwardDiff.npartials(value) ||
        throw(DimensionMismatch("$(length(columns)) plasma columns for $(ForwardDiff.npartials(value)) partials"))
    partials = ForwardDiff.partials(value)
    for k in eachindex(columns)
        _add_entry!(J, row, columns[k], partials[k])
    end
    return J
end

"""
    LayoutKineticArray

3D array view `f[i, j, ir]` of a kinetic unknown with the same cell counts at all radii in
a state vector, so that operators written for 3D arrays apply to the state directly.
"""
struct LayoutKineticArray{T, V <: AbstractVector{T}, O <: AbstractVector{Int}} <:
       AbstractArray{T, 3}
    state::V
    offsets::O   # per radius: index before the first entry (a range when equally spaced)
    p_cells::Int
    xi_cells::Int
end

Base.size(a::LayoutKineticArray) = (a.p_cells, a.xi_cells, length(a.offsets))
#! format: off
Base.@propagate_inbounds Base.getindex(a::LayoutKineticArray, i::Int, j::Int, ir::Int) =
    a.state[a.offsets[ir] + i + (j - 1) * a.p_cells]
Base.@propagate_inbounds Base.setindex!(a::LayoutKineticArray, x, i::Int, j::Int, ir::Int) =
    (a.state[a.offsets[ir] + i + (j - 1) * a.p_cells] = x)
#! format: on

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
field of the state in this form. [`unknown_values`](@ref) builds it.
"""
struct LayoutRadialValues{T, V <: AbstractVector{T}, O <: AbstractVector{Int}} <:
       AbstractVector{T}
    state::V
    offsets::O   # per radius: index before the entry
end

Base.size(a::LayoutRadialValues) = (length(a.offsets),)
Base.@propagate_inbounds Base.getindex(a::LayoutRadialValues, ir::Int) = a.state[a.offsets[ir] + 1]
Base.@propagate_inbounds Base.setindex!(a::LayoutRadialValues, x, ir::Int) = (a.state[a.offsets[ir] + 1] = x)

function unknown_values(u::AbstractVector, layout::UnknownLayout, id::Integer)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a radial unknown with one component"))
    return LayoutRadialValues(u, layout.radius_offsets[id])
end

# the electric field of the state for a term: nothing when the term has none as an unknown
_field_values(::Nothing, u, layout) = nothing
_field_values(name::Symbol, u, layout) = unknown_values(u, layout, unknown_id(layout, name))

function _equally_spaced(offsets)
    for k in 3:length(offsets)
        offsets[k] - offsets[k - 1] == offsets[2] - offsets[1] || return false
    end
    return true
end

# the offsets of kinetic unknown `id` as a range; valid when they are equally spaced
function _offset_range(layout::UnknownLayout, id)
    offsets = layout.radius_offsets[id]
    # with one radius the step is never used, but `range` rejects a zero step
    step = length(offsets) > 1 ? offsets[2] - offsets[1] : 1
    return range(first(offsets); step, length = length(offsets))
end

# whether two kinetic grids have the same radii and the same momentum grid at each of them
function _same_momentum_grids(a::KineticGrid, b::KineticGrid)
    a === b && return true
    a.radial_grid.r_faces == b.radial_grid.r_faces || return false
    for ir in 1:radial_cell_count(a)
        if !(momentum_grid(a, ir).p_faces == momentum_grid(b, ir).p_faces &&
             momentum_grid(a, ir).xi_faces == momentum_grid(b, ir).xi_faces)
            return false
        end
    end
    return true
end

# index of cell (i, j) of kinetic unknown `id` at radius `ir`
#! format: off
@inline _kinetic_index(layout::UnknownLayout, id, ir, i, j) =
    layout.offsets[id, ir] + i + (j - 1) * layout.shapes[id, ir][1]
#! format: on

# tag of the dual numbers of jacobian_correction_product!
struct JacobianCorrectionTag end

# tag of the dual numbers of the local columns of a term
struct LocalColumnTag end

# the dual number of type D with value x and partial `partial` equal to one; none when
# `partial` lies outside the partials of D
function _seeded_dual(::Type{ForwardDiff.Dual{T, V, N}}, x, partial) where {T, V, N}
    seed = ntuple(q -> q == partial ? one(V) : zero(V), Val(N))
    return ForwardDiff.Dual{T, V, N}(x, ForwardDiff.Partials(seed))
end

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

The evaluation runs on a state of the dependencies and the row alone, in a layout of these
radial unknowns on the same radial grid ([`local_layout`](@ref)), so that its cost does not
grow with the kinetic unknowns of the system.
"""
function add_local_jacobian!(J, term::EquationTerm, u, layout, row, t)
    buffers = _local_jacobian_buffers(term, layout, row)
    return _add_local_jacobian!(J, term, u, layout, t, buffers)
end

# the buffers that a term keeps between its evaluations on the state of the system: those of
# add_local_jacobian! and the plasma states of `_with_state_ions!`. The Jacobian buffers are
# those of one layout and row, so a term evaluated in turn on two rows or two systems makes
# them again at each call; the results are the same. The public add_local_jacobian! makes a
# local layout per call, so on a term that reads the ion densities of the state each of its
# calls adds a plasma state, and the list, emptied when full, drops the states of the residual
# and of the kept Jacobian buffers with it.
struct LocalTermBuffers
    jacobian::Base.RefValue{Any}   # the LocalJacobianBuffers of the last layout and row
    plasmas::Vector{Any}           # the StateIonsBuffer per layout and number type
end
LocalTermBuffers() = LocalTermBuffers(Ref{Any}(nothing), Any[])

# add_local_jacobian! with the buffers of `buffers`, made again when the layout or the row
# differ from those of the last call
function _add_buffered_local_jacobian!(J, term::EquationTerm, u, layout, row, t,
        buffers::LocalTermBuffers)
    kept = buffers.jacobian[]
    if !(kept isa LocalJacobianBuffers && kept.layout === layout && kept.row == row)
        kept = _local_jacobian_buffers(term, layout, row)
        buffers.jacobian[] = kept
    end
    return _add_local_jacobian!(J, term, u, layout, t, kept)
end

"""
    local_layout(layout, ids)

The layout of the radial unknowns `ids` of `layout` alone, on its radial grid and in the order
of `ids`, without their lower limits, whose references may be missing. With
[`add_local_jacobian!`](@ref) a term evaluates on the state of this layout.
"""
function local_layout(layout::UnknownLayout, ids)
    all(id -> layout.kinds[id] === :radial, ids) ||
        throw(ArgumentError("a local layout holds radial unknowns alone"))
    unknowns = [_local_unknown(layout.unknowns[id]) for id in ids]
    return UnknownLayout(layout.radial_grid, unknowns)
end

# a copy of the radial unknown `u` for a local layout
function _local_unknown(u::RadialUnknown)
    groups = u.grouped ? u.groups : nothing
    return RadialUnknown(u.name; components = u.components, scale = u.scale, groups)
end

# the state of the dependencies and the row of a term in the local layout, with the partials
# seeded, for the layout `layout` of the system and the row `row`
struct LocalJacobianBuffers{D <: ForwardDiff.Dual, L <: UnknownLayout}
    layout::UnknownLayout    # the layout of the system
    row::Int
    ids::Vector{Int}         # the dependencies, whose components the partials take in order
    reduced::L               # the layout of the dependencies and the row (local_layout)
    local_row::Int           # the row in `reduced`
    sources::Vector{Int}     # per entry of the local state: its index in the state of the system
    seeded::Vector{D}        # the local state, its partials seeded
    out::Vector{D}           # the output of the evaluation
end

# the buffers of the local Jacobian of `term` for the layout and the row
function _local_jacobian_buffers(term::EquationTerm, layout, row)
    ids = [unknown_id(layout, name) for name in dependencies(term)]
    all(id -> layout.kinds[id] === :radial, ids) ||
        throw(ArgumentError("the local Jacobian of a $(nameof(typeof(term))) needs radial unknowns alone"))
    layout.kinds[row] === :radial ||
        throw(ArgumentError("the local Jacobian of a $(nameof(typeof(term))) adds to a radial unknown"))
    count = sum(id -> layout.sizes[id, 1], ids)
    return _local_jacobian_buffers(layout, row, ids, Val(count))
end

function _local_jacobian_buffers(layout, row, ids, ::Val{K}) where {K}
    D = ForwardDiff.Dual{LocalColumnTag, Float64, K}
    local_ids = row in ids ? ids : [ids; row]
    reduced = local_layout(layout, local_ids)
    local_row = findfirst(==(row), local_ids)::Int
    sources = zeros(Int, length(reduced))
    seeded = zeros(D, length(reduced))
    for ir in 1:radial_cell_count(layout)
        k = 0
        for (m, id) in enumerate(local_ids)
            target = unknown_range(reduced, m, ir)
            for (a, i) in enumerate(unknown_range(layout, id, ir))
                sources[target[a]] = i
                if m <= length(ids)
                    k += 1
                    seeded[target[a]] = _seeded_dual(D, 0.0, k)
                end
            end
        end
    end
    return LocalJacobianBuffers(
        layout, row, ids, reduced, local_row, sources, seeded, zeros(D, length(reduced)))
end

function _add_local_jacobian!(
        J, term, u, layout, t, buffers::LocalJacobianBuffers{D}) where {D}
    seeded = buffers.seeded
    # the values of the state, the seeded partials kept
    for (j, i) in enumerate(buffers.sources)
        seeded[j] = D(u[i], ForwardDiff.partials(seeded[j]))
    end
    out = buffers.out
    fill!(out, zero(D))
    rhs!(out, term, seeded, buffers.reduced, buffers.local_row, t)
    for ir in 1:radial_cell_count(layout)
        rows = unknown_range(layout, buffers.row, ir)
        local_rows = unknown_range(buffers.reduced, buffers.local_row, ir)
        for (r, r_local) in zip(rows, local_rows)
            partials = ForwardDiff.partials(out[r_local])
            # partial k is the k-th component of the dependencies at the radius
            k = 0
            for id in buffers.ids
                for c in unknown_range(layout, id, ir)
                    k += 1
                    _add_entry!(J, r, c, partials[k])
                end
            end
        end
    end
    return J
end

# the interface of a term whose Jacobian is a sum of low-rank blocks (the field-particle
# term), which the linear solvers add by a Woodbury update
"""
    LowRankBlock(rows, columns, left, right)

Part `left * right'` of a term's Jacobian, with rows `rows` and columns `columns` of the
state vector (contiguous ranges). `left` is `length(rows) × k`, `right` is
`length(columns) × k`, `k` the rank.
"""
struct LowRankBlock{T}
    rows::UnitRange{Int}
    columns::UnitRange{Int}
    left::Matrix{T}
    right::Matrix{T}
end

"""
    low_rank_jacobian(term, u, layout, row, t)

For a `:matrix_free` term whose Jacobian at the state `u` is a sum of low-rank blocks:
the vector of [`LowRankBlock`](@ref)s. `nothing` (the default) for any other term. The
block preconditioners of [`GMRES`](@ref) and the direct solvers [`SparseLU`](@ref) and
[`MKLPardiso`](@ref) add the blocks by a Woodbury update, so the term needs no stored
Jacobian.
"""
low_rank_jacobian(::EquationTerm, u, layout, row, t) = nothing

"""
    has_low_rank_jacobian(term)

Whether [`low_rank_jacobian`](@ref) returns the whole Jacobian of `term` (false by
default). A direct solver takes such matrix-free terms by a Woodbury update.
"""
has_low_rank_jacobian(::EquationTerm) = false
