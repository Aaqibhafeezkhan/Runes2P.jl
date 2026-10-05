"""
HDF5 files of Runes2P: the output of a `TimeSolution`, the
checkpoints of a run and their readers. Loaded when HDF5 is, so that Runes2P does not depend
on HDF5.
"""
module Runes2PHDF5Ext

using Runes2P: Runes2P, TimeSolution, SolverStatistics, StoredState, CheckpointSettings, UnknownLayout, ScalarUnknown,
            RadialUnknown, KineticUnknown, LowerLimit, RadialGrid, MomentumGrid, cylindrical_grid, describe,
            radial_cell_count, has_shared_momentum_grid, unknown_range, checkpoint_file, previous_checkpoint_file,
            segment_files, write_atomically
using HDF5: HDF5, h5open, create_group, attributes

# ---------------------------------------------------------------------------------------
# records

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

# ---------------------------------------------------------------------------------------
# unknowns

# the unknown `id` of the states (one column per time) as its dataset or group
function _write_unknown(group, layout::UnknownLayout, states::AbstractMatrix, id)
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

# the states of the saved times `indices` (a range, which HDF5 reads as one block) of the
# unknowns group, in the order of `layout`
function _read_states(group, layout::UnknownLayout{T}, indices::AbstractUnitRange{<:Integer}) where {T}
    states = zeros(T, length(layout), length(indices))
    isempty(indices) && return states
    nr = radial_cell_count(layout)
    for (id, u) in enumerate(layout.unknowns)
        name = String(u.name)
        kind = layout.kinds[id]
        if kind === :scalar
            states[only(unknown_range(layout, id)), :] .= group[name][indices]
        elseif kind === :radial
            data = group[name][:, :, indices]
            for ir in 1:nr
                states[unknown_range(layout, id, ir), :] .= data[:, ir, :]
            end
        elseif has_shared_momentum_grid(u.grid)
            data = group[name][:, :, :, indices]
            for ir in 1:nr
                states[unknown_range(layout, id, ir), :] .= reshape(data[:, :, ir, :], :, length(indices))
            end
        else
            for ir in 1:nr
                data = group[name]["radius_$ir"][:, :, indices]
                states[unknown_range(layout, id, ir), :] .= reshape(data, :, length(indices))
            end
        end
    end
    return states
end

# the kinetic grid of a record of `describe(grid::KineticGrid)`. The geometry is taken
# as cylindrical: the file has the faces, and the Jacobians of the rebuilt grids are not
# used
function _read_kinetic_grid(record, radial::RadialGrid)
    nr = radial_cell_count(radial)
    momentum(entry) = MomentumGrid(read(entry["p_faces"]), read(entry["xi_faces"]))
    haskey(record, "momentum_grid") && return cylindrical_grid(momentum(record["momentum_grid"]), radial)
    grids = [momentum(record["momentum_grids"]["radius_$ir"]) for ir in 1:nr]
    return cylindrical_grid(grids, radial)
end

# the layout of a file, rebuilt from its root group `layout` for matching unknowns by
# name and comparing grids: version 2 is the settings record
# with the grids inline, version 1 (before format 0.14) held the faces per unknown and
# the order as one string
function _read_layout(group)
    radial = RadialGrid(read(group["r_faces"]))
    nr = radial_cell_count(radial)
    version = haskey(group, "layout_version") ? Int(read(group["layout_version"])) : 1
    scalars = version == 1 ? (endswith(read(group["order"]), "first") ? :first : :last) : Symbol(read(group["scalars"]))
    entries = group["unknowns"]
    order = version == 1 ? read(entries["order"]) : read(group["order"])
    unknowns = map(order) do name
        entry = entries[name]
        kind, scale = read(entry["kind"]), read(entry["scale"])
        limit = if haskey(entry, "lower_limit_tolerance")
            reference = haskey(entry, "lower_limit_reference") ? Symbol(read(entry["lower_limit_reference"])) : nothing
            value = haskey(entry, "lower_limit_value") ? read(entry["lower_limit_value"]) : 0.0
            LowerLimit(; value, tolerance=read(entry["lower_limit_tolerance"]), reference)
        else
            false
        end
        if kind == "scalar"
            ScalarUnknown(Symbol(name); scale, nonnegative=limit)
        elseif kind == "radial"
            components = read(entry["components"])
            groups = if haskey(entry, "group_starts")
                starts = read(entry["group_starts"])
                [starts[k]:(k == length(starts) ? components : starts[k+1] - 1) for k in eachindex(starts)]
            end
            RadialUnknown(Symbol(name); components, scale, nonnegative=limit, groups)
        elseif version == 1
            grids = [MomentumGrid(read(entry["p_faces"][string(ir)]), read(entry["xi_faces"][string(ir)]))
                     for ir in 1:nr]
            KineticUnknown(Symbol(name), cylindrical_grid(grids, radial); scale, nonnegative=limit)
        else
            grid = _read_kinetic_grid(group["grids"][read(entry["grid"])], radial)
            KineticUnknown(Symbol(name), grid; scale, nonnegative=limit)
        end
    end
    return UnknownLayout(radial, unknowns; scalars)
end

# ---------------------------------------------------------------------------------------
# statistics and update states

function _write_statistics(group, stats::SolverStatistics)
    for field in fieldnames(SolverStatistics)
        value = getfield(stats, field)
        isempty(value) && continue   # a segment without a step: an absent dataset is an empty one
        group[String(field)] = eltype(value) === Symbol ? String.(value) :
                               eltype(value) === Bool ? UInt8.(value) : value
    end
    return group
end

function _read_statistics(group)
    values = map(fieldnames(SolverStatistics)) do field
        type = fieldtype(SolverStatistics, field)
        haskey(group, String(field)) || return type()
        data = read(group[String(field)])
        eltype(type) === Symbol ? Symbol.(data) : eltype(type) === Bool ? data .!= 0 : convert(type, data)
    end
    return SolverStatistics(values...)
end

function _append_statistics!(all::SolverStatistics, part::SolverStatistics)
    attempts = length(all.time)
    for field in fieldnames(SolverStatistics)
        values = getfield(part, field)
        append!(getfield(all, field), field === :iteration_attempt ? values .+ attempts : values)
    end
    return all
end

# the statistics of the first `count` attempted steps
_statistics_first(stats::SolverStatistics, count::Integer) =
    _statistics_of(stats, 1:min(count, length(stats.time)))

# the statistics of the attempted steps that end at or before `time`
_statistics_until(stats::SolverStatistics, time::Real) = _statistics_of(stats, findall(<=(time), stats.time))

function _statistics_of(stats::SolverStatistics, attempts::AbstractVector{<:Integer})
    length(attempts) == length(stats.time) && return stats
    kept = Set(attempts)
    iterations = findall(in(kept), stats.iteration_attempt)
    position = Dict(attempt => k for (k, attempt) in enumerate(attempts))
    values = map(fieldnames(SolverStatistics)) do field
        data = getfield(stats, field)
        field === :iteration_attempt && return Int[position[a] for a in data[iterations]]
        startswith(String(field), "iteration_") ? data[iterations] : data[attempts]
    end
    return SolverStatistics(values...)
end

# ---------------------------------------------------------------------------------------
# files

function _write_root(file, settings::AbstractDict)
    attributes(file)["format_version"] = Runes2P.output_format_version()
    code = get(settings, "code", nothing)
    code === nothing && (code = Runes2P.code_version())
    attributes(file)["runes2p_commit"] = code["commit"]
    attributes(file)["runes2p_version"] = code["package_version"]
    attributes(file)["runes2p_tree_clean"] = UInt8(code["tree_clean"])
    isempty(settings) || _write_record(file, "settings", settings)
    return file
end

# A segment that has not saved a state yet has no `time` and no `unknowns`: HDF5 keeps
# no shape of an array without entries.
function _write_states(file, layout::UnknownLayout, times::AbstractVector, states::AbstractMatrix)
    _write_record(file, "layout", describe(layout))
    isempty(times) && return file
    file["time"] = collect(times)
    unknowns = create_group(file, "unknowns")
    for id in eachindex(layout.unknowns)
        _write_unknown(unknowns, layout, states, id)
    end
    return file
end

function _write_solution(filename::AbstractString, solution::TimeSolution)
    h5open(filename, "w") do file
        _write_root(file, solution.settings)
        _write_states(file, solution.layout, solution.times, solution.states)
        _write_statistics(create_group(file, "solver_statistics"), solution.statistics)
        updates = create_group(file, "update_states")
        for (name, states) in solution.update_states
            isempty(states) || (updates[name] = states)
        end
    end
    return filename
end

function Runes2P.write_output(filename::AbstractString, solution::TimeSolution; atomic::Bool=false)
    atomic ? write_atomically(temporary -> _write_solution(temporary, solution), filename) :
    _write_solution(filename, solution)
    return filename
end

function Runes2P.read_settings(filename::AbstractString)
    return h5open(filename, "r") do file
        haskey(file, "settings") ? _read_record(file["settings"]) : Dict{String,Any}()
    end
end

function _check_version(file, filename)
    version = read(attributes(file)["format_version"])
    version isa AbstractString ||
        throw(ArgumentError("$filename has the format of before the numbering (format_version = $version), " *
                            "which holds no state to resume from"))
    VersionNumber(version) > Runes2P.OUTPUT_FORMAT_VERSION &&
        throw(ArgumentError("$filename has the format version $version, newer than this Runes2P reads " *
                            "($(Runes2P.output_format_version()))"))
    return VersionNumber(version)
end

# ---------------------------------------------------------------------------------------
# checkpoints

function Runes2P.write_checkpoint(filename::AbstractString, stored::StoredState; previous=nothing)
    # a checkpoint that cannot be read does not replace the good one before it
    previous !== nothing && isfile(filename) && !HDF5.ishdf5(filename) && (previous = nothing)
    write_atomically(filename; previous) do temporary
        h5open(temporary, "w") do file
            _write_root(file, stored.settings)
            _write_states(file, stored.layout, [stored.time], reshape(stored.state, :, 1))
            group = create_group(file, "checkpoint")
            group["time"] = stored.time
            group["segment"] = stored.segment
            group["saved_count"] = stored.saved_count
            _write_record(group, "summary", stored.summary)
            _write_record(group, "stepper", stored.stepper)
            updates = create_group(group, "update_states")
            for (name, state) in stored.update_states
                updates[name] = state
            end
        end
    end
    return filename
end
Runes2P.write_checkpoint(settings::CheckpointSettings, stored::StoredState) =
    Runes2P.write_checkpoint(checkpoint_file(settings), stored; previous=previous_checkpoint_file(settings))

function _read_update_states(group, column)
    states = Dict{String,Vector{Float64}}()
    for name in keys(group)
        data = read(group[name])
        states[name] = column === nothing ? vec(data) : data[:, column]
    end
    return states
end

function Runes2P.read_state(filename::AbstractString; time_index::Union{Nothing,Integer}=nothing)
    return h5open(filename, "r") do file
        _check_version(file, filename)
        layout = _read_layout(file["layout"])
        settings = haskey(file, "settings") ? _read_record(file["settings"]) : Dict{String,Any}()
        times = haskey(file, "time") ? read(file["time"]) : Float64[]
        if haskey(file, "checkpoint")
            time_index === nothing || time_index == 1 ||
                throw(ArgumentError("$filename is a checkpoint, which holds one state"))
            group = file["checkpoint"]
            state = vec(_read_states(file["unknowns"], layout, 1:1))
            # a checkpoint of a format before 0.14 has no record of the stepper: its time
            # is a time point of the run, where a stepper starts fresh
            stepper = haskey(group, "stepper") ? _read_record(group["stepper"]) : Dict{String,Any}()
            return StoredState(Float64(read(group["time"])), state, layout,
                               _read_update_states(group["update_states"], nothing), settings,
                               Int(read(group["segment"])), Int(read(group["saved_count"])),
                               _read_record(group["summary"]), stepper, String(filename))
        end
        index = time_index === nothing ? length(times) : Int(time_index)
        1 <= index <= length(times) ||
            throw(ArgumentError("$filename has $(length(times)) saved states, there is no state $index"))
        segment = get(settings, "segment", Dict{String,Any}())
        updates = haskey(file, "update_states") ? _read_update_states(file["update_states"], index) :
                  Dict{String,Vector{Float64}}()
        return StoredState(Float64(times[index]), vec(_read_states(file["unknowns"], layout, index:index)), layout,
                           updates, settings, Int(get(segment, "index", 1)),
                           Int(get(segment, "saved_before", 0)) + index, Dict{String,Any}(), Dict{String,Any}(),
                           String(filename))
    end
end

Runes2P.read_checkpoint(filename::AbstractString) = Runes2P.read_state(filename)

# A run whose checkpoints all exist but cannot be read must not start again from its
# initial state: with resume = true it would replace the file of the first segment.
# Only a run without any checkpoint file starts fresh.
function Runes2P.read_checkpoint(settings::CheckpointSettings)
    unreadable = String[]
    for file in (checkpoint_file(settings), previous_checkpoint_file(settings))
        isfile(file) || continue
        try
            return Runes2P.read_state(file)
        catch error
            @warn "the checkpoint $file cannot be read" error
            push!(unreadable, file)
        end
    end
    isempty(unreadable) ||
        throw(ArgumentError("no checkpoint of the run can be read ($(join(unreadable, ", "))): repair or remove " *
                            "them to start the run again from its initial state"))
    return nothing
end

# ---------------------------------------------------------------------------------------
# reading outputs

function _read_solution(filename::AbstractString)
    return h5open(filename, "r") do file
        _check_version(file, filename)
        haskey(file, "checkpoint") && throw(ArgumentError("$filename is a checkpoint, read it with read_state"))
        layout = _read_layout(file["layout"])
        times = haskey(file, "time") ? read(file["time"]) : Float64[]
        states = isempty(times) ? zeros(length(layout), 0) : _read_states(file["unknowns"], layout, 1:length(times))
        settings = haskey(file, "settings") ? _read_record(file["settings"]) : Dict{String,Any}()
        updates = Dict{String,Matrix{Float64}}()
        if haskey(file, "update_states")
            for name in keys(file["update_states"])
                updates[name] = read(file["update_states"][name])
            end
        end
        TimeSolution(layout, times, states, _read_statistics(file["solver_statistics"]), settings, updates)
    end
end

_start_time(solution::TimeSolution) = Float64(get(get(solution.settings, "segment", Dict{String,Any}()),
                                                  "start_time", -Inf))
function _attempts_before(solution::TimeSolution)
    value = get(get(solution.settings, "segment", Dict{String,Any}()), "attempts_before", nothing)
    return value === nothing ? nothing : Int(value)
end

function Runes2P.read_output(filenames::AbstractString...)
    isempty(filenames) && throw(ArgumentError("no output file given"))
    parts = [_read_solution(filename) for filename in filenames]
    length(parts) == 1 && return only(parts)
    first_part = first(parts)
    times = empty(first_part.times)
    columns = Matrix{eltype(first_part.states)}[]
    statistics = SolverStatistics()
    # a segment without a saved state has no update states either
    names = Set{String}()
    foreach(part -> union!(names, keys(part.update_states)), parts)
    updates = Dict(name => Matrix{Float64}[] for name in names)
    for (k, part) in enumerate(parts)
        Runes2P.unknown_names(part.layout) == Runes2P.unknown_names(first_part.layout) &&
            length(part.layout) == length(first_part.layout) ||
            throw(ArgumentError("$(filenames[k]) has another layout than $(first(filenames)); " *
                                "read the segments of a run whose unknowns changed one by one"))
        # what a segment did after the start of the following one is repeated by it: the
        # statistics are cut by the count of attempts the following segment resumed
        # from (a failed attempt is recorded at the time it aimed at), by the time before
        # format 0.14
        limit = k < length(parts) ? _start_time(parts[k+1]) : Inf
        before, after = _attempts_before(part), k < length(parts) ? _attempts_before(parts[k+1]) : nothing
        if before !== nothing && after !== nothing
            _append_statistics!(statistics, _statistics_first(part.statistics, after - before))
        else
            _append_statistics!(statistics, _statistics_until(part.statistics, limit))
        end
        kept = findall(<=(limit), part.times)
        isempty(kept) && continue
        Set(keys(part.update_states)) == names ||
            throw(ArgumentError("$(filenames[k]) has other interval updates than the other files"))
        append!(times, part.times[kept])
        push!(columns, part.states[:, kept])
        for (name, states) in part.update_states
            push!(updates[name], states[:, kept])
        end
    end
    issorted(times) || throw(ArgumentError("the files are not the segments of one run in order"))
    states = isempty(columns) ? zeros(eltype(first_part.states), length(first_part.layout), 0) : reduce(hcat, columns)
    return TimeSolution(first_part.layout, times, states, statistics, last(parts).settings,
                        Dict(name => reduce(hcat, parts) for (name, parts) in updates if !isempty(parts)))
end

Runes2P.read_output(settings::CheckpointSettings) = Runes2P.read_output(segment_files(settings)...)

end # module Runes2PHDF5Ext
