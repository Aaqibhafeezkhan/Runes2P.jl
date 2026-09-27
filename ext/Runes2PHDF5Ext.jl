"""
HDF5 output of a `TimeSolution`: loaded when HDF5 is, so that
Runes2P does not depend on HDF5.
"""
module Runes2PHDF5Ext

using Runes2P: Runes2P, TimeSolution, SolverStatistics, describe, radial_cell_count, has_shared_momentum_grid,
            unknown_range
using HDF5: HDF5, h5open, create_group, attributes

# the plain record of `describe`: dictionaries become groups, lists of arrays groups with
# one dataset per entry ("1", "2", ...), the list of unknowns one group per unknown name
function _write_record(parent, name::AbstractString, value::AbstractDict)
    group = create_group(parent, name)
    for (key, entry) in value
        _write_record(group, string(key), entry)
    end
    return group
end
function _write_record(parent, name::AbstractString, value::AbstractVector{<:AbstractDict})
    group = create_group(parent, name)
    group["order"] = [String(entry["name"]) for entry in value]
    for entry in value
        _write_record(group, String(entry["name"]), entry)
    end
    return group
end
function _write_record(parent, name::AbstractString, value::AbstractVector{<:AbstractVector})
    group = create_group(parent, name)
    for (k, entry) in enumerate(value)
        group[string(k)] = entry
    end
    return group
end
# HDF5 has no boolean type: a byte, marked by the attribute `bool` so that it is read back
# as a boolean
function _write_record(parent, name::AbstractString, value::Bool)
    parent[name] = UInt8(value)
    attributes(parent[name])["bool"] = UInt8(1)
    return parent[name]
end
_write_record(parent, name::AbstractString, value) = (parent[name] = value)

function _read_record(group::Union{HDF5.File,HDF5.Group})
    record = Dict{String,Any}()
    for name in keys(group)
        object = group[name]
        record[name] = object isa HDF5.Group ? _read_record(object) : _read_value(object)
    end
    return record
end
function _read_value(dataset::HDF5.Dataset)
    value = read(dataset)
    return haskey(attributes(dataset), "bool") ? value != 0 : value
end

function _write_unknown(group, solution::TimeSolution, id)
    layout = solution.layout
    states = solution.states
    u = layout.unknowns[id]
    name = String(u.name)
    nt = size(states, 2)
    kind = layout.kinds[id]
    if kind === :scalar
        group[name] = states[only(unknown_range(layout, id)), :]
    elseif kind === :radial
        nr = radial_cell_count(layout)
        data = zeros(eltype(states), u.components, nr, nt)
        for ir in 1:nr
            data[:, ir, :] .= states[unknown_range(layout, id, ir), :]
        end
        group[name] = data
    elseif has_shared_momentum_grid(u.grid)
        nr = radial_cell_count(layout)
        np, nxi = layout.shapes[id, 1]
        data = zeros(eltype(states), np, nxi, nr, nt)
        for ir in 1:nr
            data[:, :, ir, :] .= reshape(states[unknown_range(layout, id, ir), :], np, nxi, nt)
        end
        group[name] = data
    else
        per_radius = create_group(group, name)
        for ir in 1:radial_cell_count(layout)
            np, nxi = layout.shapes[id, ir]
            per_radius["radius_$ir"] = reshape(states[unknown_range(layout, id, ir), :], np, nxi, nt)
        end
    end
    return nothing
end

function _write_statistics(group, stats::SolverStatistics)
    for field in fieldnames(SolverStatistics)
        value = getfield(stats, field)
        group[String(field)] = eltype(value) === Symbol ? String.(value) :
                               eltype(value) === Bool ? UInt8.(value) : value
    end
    return group
end

function Runes2P.write_output(filename::AbstractString, solution::TimeSolution)
    h5open(filename, "w") do file
        attributes(file)["format_version"] = Runes2P.output_format_version()
        code = get(solution.settings, "code", nothing)
        code === nothing && (code = Runes2P.code_version())
        attributes(file)["runes2p_commit"] = code["commit"]
        attributes(file)["runes2p_version"] = code["package_version"]
        attributes(file)["runes2p_tree_clean"] = UInt8(code["tree_clean"])
        isempty(solution.settings) || _write_record(file, "settings", solution.settings)
        _write_record(file, "layout", describe(solution.layout))
        file["time"] = solution.times
        unknowns = create_group(file, "unknowns")
        for id in eachindex(solution.layout.unknowns)
            _write_unknown(unknowns, solution, id)
        end
        _write_statistics(create_group(file, "solver_statistics"), solution.statistics)
    end
    return filename
end

function Runes2P.read_settings(filename::AbstractString)
    return h5open(filename, "r") do file
        haskey(file, "settings") ? _read_record(file["settings"]) : Dict{String,Any}()
    end
end

end # module Runes2PHDF5Ext
