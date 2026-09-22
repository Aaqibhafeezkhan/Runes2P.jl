# Layout of the unknowns in the state vector. Unknowns are stored radius by radius: at
# each radius the radial unknowns
# (components contiguous) and then the kinetic unknowns (p fastest, then ξ), followed by
# the scalar unknowns. Code outside this file reaches the unknowns through the views and
# ranges below and never computes offsets itself, so the order can change.

"""
    ScalarUnknown(name; scale=1)

Unknown with one value for the whole system (e.g. `:plasma_current`). `scale` is its
reference magnitude.
"""
struct ScalarUnknown
    name::Symbol
    scale::Float64
end
ScalarUnknown(name::Symbol; scale::Real=1.0) = ScalarUnknown(name, Float64(scale))

"""
    RadialUnknown(name; components=1, scale=1)

Unknown with `components` values at each radius (e.g. `:electric_field`, or
`:ion_densities` with one component per charge state).
"""
struct RadialUnknown
    name::Symbol
    components::Int
    scale::Float64
end
function RadialUnknown(name::Symbol; components::Integer=1, scale::Real=1.0)
    components >= 1 || throw(ArgumentError("a radial unknown needs at least one component"))
    return RadialUnknown(name, Int(components), Float64(scale))
end

"""
    KineticUnknown(name, grid; scale=1)

Distribution function on the kinetic grid `grid`, one value per momentum cell at each
radius (e.g. `:hot_tail_distribution`). The momentum grid may differ between radii.
"""
struct KineticUnknown{T}
    name::Symbol
    grid::KineticGrid{T}
    scale::Float64
end
KineticUnknown(name::Symbol, grid::KineticGrid; scale::Real=1.0) =
    KineticUnknown(name, grid, Float64(scale))

const Unknown = Union{ScalarUnknown,RadialUnknown,KineticUnknown}

"""
    UnknownLayout(radial_grid, unknowns; scalars=:last)

Positions of the `unknowns` (a list of [`ScalarUnknown`](@ref), [`RadialUnknown`](@ref) and
[`KineticUnknown`](@ref)) in the state vector. Each radius owns one contiguous block, with
the radial and then the kinetic unknowns in declaration order; the scalar unknowns form
one block after the radius blocks (`scalars = :first` places it before them, which only
serves to check that no code depends on the order). Kinetic unknowns must live on
`radial_grid`. The layout is immutable; adding an unknown builds a new layout.
"""
struct UnknownLayout{T<:AbstractFloat}
    radial_grid::RadialGrid{T}
    unknowns::Vector{Unknown}
    ids::Dict{Symbol,Int}
    kinds::Vector{Symbol}         # :scalar, :radial or :kinetic, per unknown
    offsets::Matrix{Int}          # (unknown, radius): index before the first entry; 0 for scalars
    sizes::Matrix{Int}            # (unknown, radius): number of entries
    shapes::Matrix{Tuple{Int,Int}}  # (unknown, radius): (p cells, ξ cells) of kinetic unknowns
    scalar_offsets::Vector{Int}   # per unknown: index of the value for scalars, 0 otherwise
    radius_blocks::Vector{UnitRange{Int}}
    scalar_block::UnitRange{Int}
    length::Int
    scalars::Symbol
end

function UnknownLayout(radial_grid::RadialGrid{T}, unknowns::AbstractVector;
                       scalars::Symbol=:last) where {T}
    scalars in (:last, :first) || throw(ArgumentError("scalars must be :last or :first"))
    list = Unknown[u for u in unknowns]
    ids = Dict{Symbol,Int}()
    for (id, u) in enumerate(list)
        haskey(ids, u.name) && throw(ArgumentError("unknown $(u.name) declared twice"))
        ids[u.name] = id
        if u isa KineticUnknown
            u.grid.radial_grid.r_faces == radial_grid.r_faces ||
                throw(ArgumentError("kinetic unknown $(u.name) lives on another radial grid"))
        end
    end
    nr = radial_cell_count(radial_grid)
    count = length(list)
    kinds = [u isa ScalarUnknown ? :scalar : u isa RadialUnknown ? :radial : :kinetic for u in list]
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
    scalar_block = scalar_start+1:scalar_start+scalar_count
    total = position + (scalars === :first ? 0 : scalar_count)
    return UnknownLayout{T}(radial_grid, list, ids, kinds, offsets, sizes, shapes, scalar_offsets,
                            radius_blocks, scalar_block, total, scalars)
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
    unknown_range(layout, id, ir)
    unknown_range(layout, id)

Index range of unknown `id` at radius index `ir` (radial and kinetic unknowns), or of a
scalar unknown.
"""
function unknown_range(layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :scalar &&
        throw(ArgumentError("$(layout.unknowns[id].name) is a scalar unknown; omit the radius"))
    offset = layout.offsets[id, ir]
    return offset+1:offset+layout.sizes[id, ir]
end
function unknown_range(layout::UnknownLayout, id::Integer)
    layout.kinds[id] === :scalar ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a scalar unknown; give a radius"))
    index = layout.scalar_offsets[id]
    return index:index
end

"""
    radius_block_range(layout, ir)
    scalar_block_range(layout)

Contiguous index range of all unknowns at radius index `ir`, and of all scalar unknowns.
The per-radius preconditioner and a radial domain decomposition work on these blocks.
"""
radius_block_range(layout::UnknownLayout, ir::Integer) = layout.radius_blocks[ir]
scalar_block_range(layout::UnknownLayout) = layout.scalar_block

"""
    radial_view(u, layout, id, ir)
    kinetic_view(u, layout, id, ir)
    scalar_view(u, layout, id)

Views into the state vector `u`: the components of a radial unknown at radius index `ir`
(vector), a kinetic unknown at `ir` as a `(p cells, ξ cells)` matrix of the momentum grid
there, and a scalar unknown (0-dimensional). Type-stable for a given `u`, including
vectors of dual numbers, so terms use them in inner loops with ids from
[`unknown_id`](@ref).
"""
function radial_view(u::AbstractVector, layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :radial ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a radial unknown"))
    offset = layout.offsets[id, ir]
    return view(u, offset+1:offset+layout.sizes[id, ir])
end

function kinetic_view(u::AbstractVector, layout::UnknownLayout, id::Integer, ir::Integer)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("$(layout.unknowns[id].name) is not a kinetic unknown"))
    offset = layout.offsets[id, ir]
    return reshape(view(u, offset+1:offset+layout.sizes[id, ir]), layout.shapes[id, ir])
end

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
unknown_view(u::AbstractVector, layout::UnknownLayout, name::Symbol) =
    scalar_view(u, layout, unknown_id(layout, name))

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
    describe(layout)

Plain description of the layout (names, kinds, sizes per radius, grids, order), written
to output and restart files. [`copy_unknowns!`](@ref) matches unknowns by name, so files
stay readable when the internal order changes.
"""
function describe(layout::UnknownLayout)
    entries = map(layout.unknowns) do u
        id = layout.ids[u.name]
        entry = Dict{String,Any}("name" => String(u.name), "scale" => u.scale)
        if u isa ScalarUnknown
            entry["kind"] = "scalar"
        elseif u isa RadialUnknown
            entry["kind"] = "radial"
            entry["components"] = u.components
        else
            entry["kind"] = "kinetic"
            nr = radial_cell_count(layout)
            entry["p_faces"] = [momentum_grid(u.grid, ir).p_faces for ir in 1:nr]
            entry["xi_faces"] = [momentum_grid(u.grid, ir).xi_faces for ir in 1:nr]
        end
        entry["sizes"] = u isa ScalarUnknown ? [1] : layout.sizes[id, :]
        entry
    end
    return Dict{String,Any}("layout_version" => 1, "order" => "radius_blocks_scalars_$(layout.scalars)",
                            "r_faces" => layout.radial_grid.r_faces, "unknowns" => entries)
end

"""
    copy_unknowns!(u_to, layout_to, u_from, layout_from)

Copy the values of every unknown present in both layouts, matched by name; the sizes must
agree. Returns the names copied. This is how a state is carried to a layout with another
order or with unknowns added or removed (restarts).
"""
function copy_unknowns!(u_to::AbstractVector, layout_to::UnknownLayout,
                        u_from::AbstractVector, layout_from::UnknownLayout)
    length(u_to) == length(layout_to) || throw(DimensionMismatch("u_to does not match layout_to"))
    length(u_from) == length(layout_from) || throw(DimensionMismatch("u_from does not match layout_from"))
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
            if layout_from.kinds[id_from] === :kinetic
                grid_from = layout_from.unknowns[id_from].grid
                grid_to = layout_to.unknowns[id_to].grid
                all(ir -> momentum_grid(grid_from, ir).p_faces == momentum_grid(grid_to, ir).p_faces &&
                          momentum_grid(grid_from, ir).xi_faces == momentum_grid(grid_to, ir).xi_faces,
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
