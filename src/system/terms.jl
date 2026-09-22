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
at the previous time level and contribute nothing to the Jacobian) and
`validate(term, layout, row)` (checks against the layout, run when the system is built).
"""
abstract type EquationTerm end

function dependencies end
coupling(::EquationTerm) = :sparse
treatment(::EquationTerm) = :implicit
validate(::EquationTerm, layout, row) = nothing

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
    step = length(offsets) > 1 ? offsets[2] - offsets[1] : 0
    return range(first(offsets); step, length=length(offsets))
end

# index of cell (i, j) of kinetic unknown `id` at radius `ir`
@inline _kinetic_index(layout::UnknownLayout, id, ir, i, j) =
    layout.offsets[id, ir] + i + (j - 1) * layout.shapes[id, ir][1]

"""
    FokkerPlanckTerm(unknown, operator::AdvectionDiffusion)

Advection-diffusion (Fokker-Planck) operator applied to the kinetic unknown `unknown`,
added to the equation of the row unknown (normally the same). Stored sparse, with the
3×3 `(p, ξ)` stencil and the two radial neighbours. Needs a shared momentum grid.
"""
struct FokkerPlanckTerm{T} <: EquationTerm
    unknown::Symbol
    operator::AdvectionDiffusion{T}
end

dependencies(term::FokkerPlanckTerm) = [term.unknown]

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
        apply!(LayoutKineticArray(out, _offset_range(layout, row), np, nxi), term.operator,
               LayoutKineticArray(u, _offset_range(layout, id), np, nxi))
    else
        apply!(kinetic_array(out, layout, row), term.operator, kinetic_array(u, layout, id))
    end
    return out
end

function sparsity!(entries, term::FokkerPlanckTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    np, nxi, nr = size(term.operator.grid)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        r = _kinetic_index(layout, row, ir, i, j)
        for dj in -1:1, di in -1:1
            (1 <= i + di <= np && 1 <= j + dj <= nxi) || continue
            push!(entries, (r, _kinetic_index(layout, id, ir, i + di, j + dj)))
        end
        for dr in (-1, 1)
            1 <= ir + dr <= nr || continue
            push!(entries, (r, _kinetic_index(layout, id, ir + dr, i, j)))
        end
    end
    return entries
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
