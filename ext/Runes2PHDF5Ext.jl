"""
HDF5 output of a `TimeSolution`: loaded when
HDF5 is, so that Runes2P does not depend on HDF5.
"""
module Runes2PHDF5Ext

using Runes2P: Runes2P, TimeSolution, SolverStatistics, describe, radial_cell_count, has_shared_momentum_grid,
            unknown_range
using HDF5: HDF5, h5open, create_group, attributes

const FORMAT_VERSION = 1

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
_write_record(parent, name::AbstractString, value) = (parent[name] = value)

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
        attributes(file)["format_version"] = FORMAT_VERSION
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

end # module Runes2PHDF5Ext
