# Layout of the unknowns in the state vector. Unknowns are stored radius by radius: at each
# radius the radial unknowns (components contiguous) and then the kinetic unknowns (p
# fastest, then ξ), followed by the scalar unknowns. Code outside this file reaches the
# unknowns through the views and ranges below and never computes offsets itself, so the
# order can change.

"""
    LowerLimit(; value=0, tolerance=0, reference=nothing)

Lower limit of an unknown, enforced on Newton steps (DREAM damps the Newton update to keep
selected unknowns above a tenth of their value, `MaximalPhysicalStepLength` in
`src/Solver/SolverNonLinear.cpp:441-524`):
every entry stays at or above `value - tolerance * reference value`, where the reference
value is the radial unknown `reference` (one component) at the same radius, or the unknown's
own scale when `reference` is `nothing`, or the sum of the magnitudes of the components of
the entry's group at the same radius when `reference` is `:group` (the charge states of one
ion species, [`RadialUnknown`](@ref) with `groups`). Declared on the unknown with
`nonnegative = true` (value 0, tolerance 0) or `nonnegative = LowerLimit(...)`, e.g.
`RadialUnknown(:ion_densities; components=3, nonnegative=LowerLimit(tolerance=1e-12,
reference=:total_electron_density))`. Scalar unknowns take no reference. An unknown that
can sit at exactly 0 (an empty charge state) needs a positive tolerance: with tolerance 0 a
round-off negative update of an entry at its limit fails the Newton step
(`:below_lower_limit`), and shorter time steps do not help.

`value` is an absolute limit in the units of the unknown, such as the floor of a
temperature: `RadialUnknown(:cold_electron_temperature; nonnegative=LowerLimit(value=0.5))`.
A Newton update loses at most the fraction `max_reduction` of the distance
to the limit, so an iterate approaches the floor and does not cross it, and a step whose
solution lies below it does not converge and is refused. The temperature unknown of a
[`PlasmaFromUnknowns`](@ref) takes the floor of that source as its limit without a
declaration, and the larger of the two values where both exist. A temperature that only
fluid terms read takes the default floor [`DEFAULT_TEMPERATURE_FLOOR`](@ref) without a
declaration ([`fluid_temperature_floors`](@ref)); a declared limit is kept as it is.
"""
struct LowerLimit
    value::Float64
    tolerance::Float64
    reference::Union{Nothing, Symbol}
end
function LowerLimit(; value::Real = 0.0, tolerance::Real = 0.0, reference::Union{
        Nothing, Symbol} = nothing)
    tolerance >= 0 ||
        throw(ArgumentError("the tolerance of a lower limit must be nonnegative"))
    isfinite(value) || throw(ArgumentError("the value of a lower limit must be finite"))
    return LowerLimit(Float64(value), Float64(tolerance), reference)
end

_lower_limit(nonnegative::Bool) = nonnegative ? LowerLimit() : nothing
_lower_limit(limit::LowerLimit) = limit

"""
    ScalarUnknown(name; scale=1, nonnegative=false)

Unknown with one value for the whole system (e.g. `:plasma_current`). `scale` is its
reference magnitude; `nonnegative` declares a [`LowerLimit`](@ref).
"""
struct ScalarUnknown
    name::Symbol
    scale::Float64
    lower_limit::Union{Nothing, LowerLimit}
end
function ScalarUnknown(name::Symbol; scale::Real = 1.0, nonnegative::Union{
        Bool, LowerLimit} = false)
    ScalarUnknown(name, Float64(scale), _lower_limit(nonnegative))
end

"""
    RadialUnknown(name; components=1, scale=1, nonnegative=false, groups=nothing)

Unknown with `components` values at each radius (e.g. `:electric_field`, or
`:ion_densities` with one component per charge state). `groups` splits the components into
consecutive ranges, such as the charge states of each ion species
([`component_groups`](@ref)).
"""
struct RadialUnknown
    name::Symbol
    components::Int
    scale::Float64
    lower_limit::Union{Nothing, LowerLimit}
    groups::Vector{UnitRange{Int}}
end
function RadialUnknown(name::Symbol; components::Integer = 1, scale::Real = 1.0,
        nonnegative::Union{Bool, LowerLimit} = false, groups = nothing)
    components >= 1 || throw(ArgumentError("a radial unknown needs at least one component"))
    groups = groups === nothing ? [1:Int(components)] : UnitRange{Int}[g for g in groups]
    _splits(groups, components) ||
        throw(ArgumentError("the groups of $name must split its components 1:$components into consecutive ranges"))
    return RadialUnknown(
        name, Int(components), Float64(scale), _lower_limit(nonnegative), groups)
end

# whether the ranges split 1:n into consecutive nonempty ranges
function _splits(groups, n)
    next = 1
    for g in groups
        (isempty(g) || first(g) != next) && return false
        next = last(g) + 1
    end
    return next == n + 1
end

"""
    component_groups(unknown::RadialUnknown)

The groups of the components of a radial unknown, consecutive ranges that split
`1:components`: the charge states of each ion species for the ion densities
([`ion_densities_unknown`](@ref)), one group of all components otherwise. A [`LowerLimit`](@ref)
with `reference = :group` and a [`Tolerance`](@ref) with a `group` part are relative to the
sum over the group of an entry.
"""
component_groups(u::RadialUnknown) = u.groups

"""
    KineticUnknown(name, grid; scale=1, nonnegative=false)

Distribution function on the kinetic grid `grid`, one value per momentum cell at each
radius (e.g. `:hot_tail_distribution`). The momentum grid may differ between radii.
A distribution is normally declared without a lower limit: negative values produced by
the discretization belong to the scheme, not to the Newton steps.
"""
struct KineticUnknown{T}
    name::Symbol
    grid::KineticGrid{T}
    scale::Float64
    lower_limit::Union{Nothing, LowerLimit}
end
function KineticUnknown(name::Symbol, grid::KineticGrid; scale::Real = 1.0,
        nonnegative::Union{Bool, LowerLimit} = false)
    KineticUnknown(name, grid, Float64(scale), _lower_limit(nonnegative))
end

const Unknown = Union{ScalarUnknown, RadialUnknown, KineticUnknown}

"""
    UnknownLayout(radial_grid, unknowns; scalars=:last)

Positions of the `unknowns` (a list of [`ScalarUnknown`](@ref), [`RadialUnknown`](@ref) and
[`KineticUnknown`](@ref)) in the state vector. Each radius owns one contiguous block, with
the radial and then the kinetic unknowns in declaration order; the scalar unknowns form
one block after the radius blocks (`scalars = :first` places it before them, which only
serves to check that no code depends on the order). Kinetic unknowns must live on
`radial_grid`. The layout is immutable; adding an unknown builds a new layout.
"""
struct UnknownLayout{T <: AbstractFloat}
    radial_grid::RadialGrid{T}
    unknowns::Vector{Unknown}
    ids::Dict{Symbol, Int}
    kinds::Vector{Symbol}         # :scalar, :radial or :kinetic, per unknown
    offsets::Matrix{Int}          # (unknown, radius): index before the first entry; 0 for scalars
    radius_offsets::Vector{Vector{Int}}  # per unknown, the offsets over the radii (a row of `offsets`)
    sizes::Matrix{Int}            # (unknown, radius): number of entries
    shapes::Matrix{Tuple{Int, Int}}  # (unknown, radius): (p cells, ξ cells) of kinetic unknowns
    kinetic_grids::Vector{Union{Nothing, KineticGrid{T}}}  # per unknown: its grid, or nothing
    scalar_offsets::Vector{Int}   # per unknown: index of the value for scalars, 0 otherwise
    radius_blocks::Vector{UnitRange{Int}}
    scalar_block::UnitRange{Int}
    length::Int
    scalars::Symbol
end

function UnknownLayout(radial_grid::RadialGrid{T}, unknowns::AbstractVector;
        scalars::Symbol = :last) where {T}
    scalars in (:last, :first) || throw(ArgumentError("scalars must be :last or :first"))
    list = Unknown[u for u in unknowns]
    ids = Dict{Symbol, Int}()
    for (id, u) in enumerate(list)
        haskey(ids, u.name) && throw(ArgumentError("unknown $(u.name) declared twice"))
        ids[u.name] = id
        if u isa KineticUnknown
            u.grid.radial_grid.r_faces == radial_grid.r_faces ||
                throw(ArgumentError("kinetic unknown $(u.name) lives on another radial grid"))
        end
    end
    for u in list
        limit = u.lower_limit
        (limit === nothing || limit.reference === nothing) && continue
        u isa ScalarUnknown &&
            throw(ArgumentError("the lower limit of scalar unknown $(u.name) cannot take a reference"))
        if limit.reference === :group
            u isa RadialUnknown ||
                throw(ArgumentError("the lower limit of $(u.name) is relative to a group of components, which only " *
                                    "a radial unknown has"))
            continue
        end
        id = get(ids, limit.reference, 0)
        id != 0 && list[id] isa RadialUnknown && list[id].components == 1 ||
            throw(ArgumentError("the lower-limit reference of $(u.name) must be a radial unknown " *
                                "with one component"))
    end
    nr = radial_cell_count(radial_grid)
    count = length(list)
    kinds = [u isa ScalarUnknown ? :scalar : u isa RadialUnknown ? :radial : :kinetic
             for u in list]
    sizes = zeros(Int, count, nr)
    shapes = fill((0, 0), count, nr)
    for (id, u) in enumerate(list), ir in 1:nr

        if u isa RadialUnknown
            sizes[id, ir] = u.components
        elseif u isa KineticUnknown
            shapes[id, ir] = (momentum_cell_count(u.grid, ir), pitch_cell_count(u.grid, ir))
            sizes[id, ir] = prod(shapes[id, ir])
        end
    end
    scalar_ids = [id for (id, u) in enumerate(list) if u isa ScalarUnknown]
    radial_ids = [id for (id, u) in enumerate(list) if u isa RadialUnknown]
    kinetic_ids = [id for (id, u) in enumerate(list) if u isa KineticUnknown]
    scalar_count = length(scalar_ids)
    position = scalars === :first ? scalar_count : 0
    offsets = zeros(Int, count, nr)
    radius_blocks = Vector{UnitRange{Int}}(undef, nr)
    for ir in 1:nr
        first_index = position + 1
        for id in Iterators.flatten((radial_ids, kinetic_ids))
            offsets[id, ir] = position
            position += sizes[id, ir]
        end
        radius_blocks[ir] = first_index:position
    end
    scalar_start = scalars === :first ? 0 : position
    scalar_offsets = zeros(Int, count)
    for (n, id) in enumerate(scalar_ids)
        scalar_offsets[id] = scalar_start + n
    end
    scalar_block = (scalar_start + 1):(scalar_start + scalar_count)
    total = position + (scalars === :first ? 0 : scalar_count)
    radius_offsets = [offsets[id, :] for id in 1:count]
    kinetic_grids = Union{Nothing, KineticGrid{T}}[u isa KineticUnknown ? u.grid : nothing
                                                   for u in list]
    return UnknownLayout{T}(
        radial_grid, list, ids, kinds, offsets, radius_offsets, sizes, shapes,
        kinetic_grids, scalar_offsets, radius_blocks, scalar_block, total, scalars)
end

Base.length(layout::UnknownLayout) = layout.length
radial_cell_count(layout::UnknownLayout) = radial_cell_count(layout.radial_grid)

"""
    unknown_id(layout, name)

Integer id of the unknown `name`, for use in inner loops; setup code uses names.
"""
function unknown_id(layout::UnknownLayout, name::Symbol)
    id = get(layout.ids, name, 0)
    id == 0 && throw(ArgumentError("no unknown named $name in the layout"))
    return id
end

"""
    unknown_names(layout)

Names of the unknowns in declaration order.
"""
unknown_names(layout::UnknownLayout) = [u.name for u in layout.unknowns]

"""
    unknown(layout, id_or_name)

Declaration of an unknown ([`ScalarUnknown`](@ref), [`RadialUnknown`](@ref) or
[`KineticUnknown`](@ref)).
"""
unknown(layout::UnknownLayout, id::Integer) = layout.unknowns[id]
unknown(layout::UnknownLayout, name::Symbol) = layout.unknowns[unknown_id(layout, name)]

"""
    kinetic_grid(layout, id)

Kinetic grid of the kinetic unknown `id`.
"""
function kinetic_grid(layout::UnknownLayout, id::Integer)
    grid = layout.kinetic_grids[id]
    grid === nothing &&
        throw(ArgumentError("$(layout.unknowns[id].name) is not a kinetic unknown"))
    return grid
end

"""
    unknown_range(layout, id, ir)
    unknown_range(layout, id)

Index range of unknown `id` at radius index `ir` (radial and kinetic unknowns), or of a
scalar unknown.
"""
function unknown_range(layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :scalar &&
        throw(ArgumentError("$(layout.unknowns[id].name) is a scalar unknown; omit the radius"))
    offset = layout.offsets[id, ir]
    return (offset + 1):(offset + layout.sizes[id, ir])
end
function unknown_range(layout::UnknownLayout, id::Integer)
    layout.kinds[id] === :scalar ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a scalar unknown; give a radius"))
    index = layout.scalar_offsets[id]
    return index:index
end

"""
    radius_block_range(layout, ir)

Contiguous index range of all unknowns at radius index `ir`. The per-radius
preconditioner and a radial domain decomposition work on these blocks.
"""
radius_block_range(layout::UnknownLayout, ir::Integer) = layout.radius_blocks[ir]

"""
    scalar_block_range(layout)

Contiguous index range of all scalar unknowns, the block next to those of
[`radius_block_range`](@ref).
"""
scalar_block_range(layout::UnknownLayout) = layout.scalar_block

"""
    radial_view(u, layout, id, ir)

View into the state vector `u` of the components of the radial unknown `id` at radius index
`ir`, as a vector. Type-stable for a given `u`, including vectors of dual numbers, so terms
use it in inner loops with ids from [`unknown_id`](@ref). See also [`kinetic_view`](@ref)
and [`scalar_view`](@ref).
"""
function radial_view(u::AbstractVector, layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :radial ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a radial unknown"))
    offset = layout.offsets[id, ir]
    return view(u, (offset + 1):(offset + layout.sizes[id, ir]))
end

"""
    kinetic_view(u, layout, id, ir)

View into the state vector `u` of the kinetic unknown `id` at radius index `ir`, as a
`(p cells, ξ cells)` matrix of the momentum grid there. Type-stable as
[`radial_view`](@ref).
"""
function kinetic_view(u::AbstractVector, layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a kinetic unknown"))
    offset = layout.offsets[id, ir]
    return reshape(view(u, (offset + 1):(offset + layout.sizes[id, ir])), layout.shapes[id, ir])
end

"""
    scalar_view(u, layout, id)

View into the state vector `u` of the scalar unknown `id`, 0-dimensional. Type-stable as
[`radial_view`](@ref).
"""
function scalar_view(u::AbstractVector, layout::UnknownLayout, id::Integer)
    layout.kinds[id] === :scalar ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a scalar unknown"))
    return view(u, layout.scalar_offsets[id])
end

"""
    unknown_view(u, layout, name, ir)
    unknown_view(u, layout, name)

View of the unknown `name` by kind (see [`radial_view`](@ref)); convenient for setup and
tests, not type-stable.
"""
function unknown_view(u::AbstractVector, layout::UnknownLayout, name::Symbol, ir::Integer)
    id = unknown_id(layout, name)
    return layout.kinds[id] === :kinetic ? kinetic_view(u, layout, id, ir) :
           radial_view(u, layout, id, ir)
end
function unknown_view(u::AbstractVector, layout::UnknownLayout, name::Symbol)
    scalar_view(u, layout, unknown_id(layout, name))
end

# declared before physics/plasma_source.jl, which reads it (the include order of
# test_include_order.jl); its method is in system/terms.jl, with LayoutRadialValues
function unknown_values end

"""
    unknown_scales(layout)

Vector of the reference scale of every entry of the state vector.
"""
function unknown_scales(layout::UnknownLayout)
    scales = zeros(length(layout))
    for (id, u) in enumerate(layout.unknowns)
        if u isa ScalarUnknown
            scales[unknown_range(layout, id)] .= u.scale
        else
            for ir in 1:radial_cell_count(layout)
                scales[unknown_range(layout, id, ir)] .= u.scale
            end
        end
    end
    return scales
end

"""
    copy_unknowns!(u_to, layout_to, u_from, layout_from)

Copy the values of every unknown present in both layouts, matched by name. The sizes, the
radial grid, the momentum grids of a kinetic unknown and the component groups of a radial
unknown (the ion species) must agree. Two species with the same count of charge states
that trade places are not detected. Returns the names copied. This is how a state is carried to a layout with another
order or with unknowns added or removed (restarts).
"""
function copy_unknowns!(u_to::AbstractVector, layout_to::UnknownLayout,
        u_from::AbstractVector, layout_from::UnknownLayout)
    length(u_to) == length(layout_to) ||
        throw(DimensionMismatch("u_to does not match layout_to"))
    length(u_from) == length(layout_from) ||
        throw(DimensionMismatch("u_from does not match layout_from"))
    copied = Symbol[]
    for (name, id_from) in layout_from.ids
        id_to = get(layout_to.ids, name, 0)
        id_to == 0 && continue
        layout_from.kinds[id_from] === layout_to.kinds[id_to] ||
            throw(ArgumentError("$name changes kind between the layouts"))
        if layout_from.kinds[id_from] === :scalar
            u_to[unknown_range(layout_to, id_to)] .= u_from[unknown_range(layout_from, id_from)]
        else
            layout_from.sizes[id_from, :] == layout_to.sizes[id_to, :] ||
                throw(DimensionMismatch("$name has different sizes in the two layouts"))
            layout_from.radial_grid.r_faces == layout_to.radial_grid.r_faces ||
                throw(ArgumentError("$name lives on different radial grids in the two layouts"))
            if layout_from.kinds[id_from] === :radial
                component_groups(layout_from.unknowns[id_from]) ==
                component_groups(layout_to.unknowns[id_to]) ||
                    throw(ArgumentError("$name has other component groups in the two layouts"))
            end
            if layout_from.kinds[id_from] === :kinetic
                grid_from = layout_from.unknowns[id_from].grid
                grid_to = layout_to.unknowns[id_to].grid
                all(
                    ir -> momentum_grid(grid_from, ir).p_faces ==
                          momentum_grid(grid_to, ir).p_faces &&
                          momentum_grid(grid_from, ir).xi_faces ==
                          momentum_grid(grid_to, ir).xi_faces,
                    1:radial_cell_count(layout_from)) ||
                    throw(ArgumentError("$name lives on different momentum grids in the two layouts"))
            end
            for ir in 1:radial_cell_count(layout_from)
                u_to[unknown_range(layout_to, id_to, ir)] .= u_from[unknown_range(layout_from, id_from, ir)]
            end
        end
        push!(copied, name)
    end
    return sort!(copied)
end
