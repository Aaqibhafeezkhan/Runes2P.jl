# Checkpoints and segments of a run.
# A run that does not fit one allocation is a chain of segments. Each segment writes a
# checkpoint at requested time points and its own output file, flushed with every
# checkpoint. The files are HDF5 and written by the package extension `Runes2PHDF5Ext`.

"""
    CheckpointSettings(name; directory=".", interval=1800, overwrite=false)

Checkpoints of a run called `name`, written to `directory`:

- `<name>.checkpoint.h5`, the last checkpoint, and `<name>.checkpoint.prev.h5`, the one
  before it, kept so that a failed write never loses the last good state;
- `<name>.segment<k>.h5` (`k` with four digits), the output of segment `k`, holding the
  states the segment saved up to its last checkpoint.

A checkpoint is written at a requested time point when `interval` seconds of wall time
have passed since the last one (0 writes one at every time point), and always at the end
of the segment. Files are written to a temporary file, synced and renamed.

A segment refuses to replace the segment file of an earlier run unless `overwrite` is
true. A chain of segments never meets one: a segment file exists only once its segment
has written a checkpoint, and the next segment then has the next number.
"""
struct CheckpointSettings
    name::String
    directory::String
    interval::Float64
    overwrite::Bool
end
function CheckpointSettings(name::AbstractString; directory::AbstractString=".", interval::Real=1800,
                            overwrite::Bool=false)
    isempty(name) && throw(ArgumentError("the checkpoints need a name"))
    occursin('/', name) && throw(ArgumentError("the name of the checkpoints must not contain '/'; use directory"))
    interval >= 0 || throw(ArgumentError("the checkpoint interval must not be negative"))
    return CheckpointSettings(String(name), String(directory), Float64(interval), overwrite)
end

"File of the last checkpoint."
checkpoint_file(settings::CheckpointSettings) = joinpath(settings.directory, settings.name * ".checkpoint.h5")
"File of the checkpoint before the last."
previous_checkpoint_file(settings::CheckpointSettings) =
    joinpath(settings.directory, settings.name * ".checkpoint.prev.h5")
"Output file of segment `k`."
segment_file(settings::CheckpointSettings, k::Integer) =
    joinpath(settings.directory, settings.name * ".segment" * lpad(k, 4, '0') * ".h5")

"""
    segment_files(settings::CheckpointSettings)

The segment output files of the run that exist, in the order of the segments, for
[`read_output`](@ref).
"""
function segment_files(settings::CheckpointSettings)
    pattern = Regex("^" * replace(settings.name, r"([^A-Za-z0-9_])" => s"\\\1") * raw"\.segment(\d+)\.h5$")
    found = Tuple{Int,String}[]
    isdir(settings.directory) || return String[]
    for file in readdir(settings.directory)
        m = match(pattern, file)
        m === nothing || push!(found, (parse(Int, m.captures[1]), joinpath(settings.directory, file)))
    end
    return [file for (_, file) in sort!(found)]
end

describe(settings::CheckpointSettings) = _describe_fields(settings)

"""
    StoredState

A state of a run read from a file, from which [`integrate`](@ref) resumes (`resume`):
the `time`, the `state` vector in the `layout` it was stored with (rebuilt from the
file's layout record, for matching the unknowns by name), the states of the interval
updates by name, the settings record of the run that wrote it, the number of the segment
that wrote it, the number of states the run had saved up to the time (`saved_count`), a
`summary` of the run so far (attempted and accepted steps, Newton and Krylov iterations,
wall time) and the file it came from. [`read_checkpoint`](@ref) and
[`read_state`](@ref) return one.
"""
struct StoredState{T}
    time::Float64
    state::Vector{T}
    layout::UnknownLayout{T}
    update_states::Dict{String,Vector{Float64}}
    settings::Dict{String,Any}
    segment::Int
    saved_count::Int
    summary::Dict{String,Any}
    source::String
end

const SUMMARY_COUNTS = ("attempted_steps", "accepted_steps", "newton_iterations", "krylov_iterations")

# the summary of a run after a segment with the statistics `stats`, given the summary
# `before` the segment (empty for the first)
function _run_summary(before::AbstractDict, stats, wall_time::Real)
    summary = SettingsRecord()
    summary["attempted_steps"] = get(before, "attempted_steps", 0) + length(stats.accepted)
    summary["accepted_steps"] = get(before, "accepted_steps", 0) + count(stats.accepted)
    summary["newton_iterations"] = get(before, "newton_iterations", 0) + sum(stats.newton_iterations)
    summary["krylov_iterations"] = get(before, "krylov_iterations", 0) + sum(stats.krylov_iterations)
    summary["wall_time"] = Float64(get(before, "wall_time", 0.0)) + Float64(wall_time)
    return summary
end

# flush a file's data to the storage device; a file system that refuses is not an error
function _sync(path::AbstractString)
    try
        file = Base.Filesystem.open(path, Base.Filesystem.JL_O_RDONLY)
        try
            ccall(:fsync, Cint, (Base.OS_HANDLE,), Base.fd(file))
        finally
            close(file)
        end
    catch
    end
    return nothing
end

"""
    write_atomically(write_file, path; previous=nothing)

Write a file so that `path` holds either its old content or the new one, whatever
happens during the write: `write_file(temporary)` writes to a temporary file in the
directory of `path`, the file is synced and renamed onto `path`, and the directory is
synced. With `previous`, the file at `path` is first renamed to `previous`. A failure of
`write_file` removes the temporary file and leaves `path` and `previous` as they were.
Between the two renames `path` does not exist and `previous` holds the last good file,
which is why a reader of checkpoints falls back on it.
"""
function write_atomically(write_file, path::AbstractString; previous::Union{Nothing,AbstractString}=nothing)
    directory = dirname(abspath(path))
    temporary = joinpath(directory, "." * basename(path) * "." * string(getpid()) * ".tmp")
    try
        write_file(temporary)
    catch
        rm(temporary; force=true)
        rethrow()
    end
    _sync(temporary)
    previous !== nothing && isfile(path) && Base.Filesystem.rename(path, previous)
    Base.Filesystem.rename(temporary, path)
    _sync(directory)
    return path
end

"""
    write_checkpoint(settings::CheckpointSettings, stored::StoredState)
    read_checkpoint(settings::CheckpointSettings)
    read_checkpoint(filename)

Write a checkpoint ([`write_atomically`](@ref), keeping the previous one), and read one
as a [`StoredState`](@ref). Reading with the settings falls back on the previous
checkpoint when the last one is missing or cannot be read, and returns `nothing` when
there is neither. Defined by the package extension `Runes2PHDF5Ext`, which loads with HDF5.
"""
function write_checkpoint end
function read_checkpoint end

"""
    read_state(filename; time_index=nothing)

The state saved at `time_index` (the last one by default) of an output file, or the
state of a checkpoint file, as a [`StoredState`](@ref) to resume from. Defined by the
package extension `Runes2PHDF5Ext`, which loads with HDF5.
"""
function read_state end

"""
    read_output(filenames...)
    read_output(settings::CheckpointSettings)

The [`TimeSolution`](@ref) of one output file, or of the segments of a run in order,
joined. The layout is rebuilt from the file's layout record. Of a segment that went on
after its last checkpoint and was followed by another, the states and statistics after
the start of the following segment are left out, since that segment repeats them. The
settings are those of the last file. Defined by the package extension `Runes2PHDF5Ext`,
which loads with HDF5.
"""
function read_output end

# the stored state to resume from: `resume` is false, true (the checkpoint of `checkpoint`
# when there is one), a file name or a StoredState
function _resume_state(resume, checkpoint)
    resume === false && return nothing
    resume isa StoredState && return resume
    if resume === true
        checkpoint === nothing &&
            throw(ArgumentError("resume = true needs checkpoint settings; pass a file or a StoredState otherwise"))
        _require_hdf5(read_checkpoint, "resuming from a checkpoint")
        return read_checkpoint(checkpoint)
    end
    resume isa AbstractString || throw(ArgumentError("resume must be true, false, a file name or a StoredState"))
    _require_hdf5(read_state, "resuming from a file")
    return read_state(resume)
end

_require_hdf5(f, what) =
    isempty(methods(f)) && throw(ArgumentError("$what needs the HDF5 package: add `using HDF5` to the script"))

# sections of the setup whose hash differs between two settings records
function changed_sections(before::AbstractDict, now::AbstractDict)
    old, new = get(before, "hashes", SettingsRecord()), get(now, "hashes", SettingsRecord())
    return sort!(String[section for section in union(keys(old), keys(new))
                        if get(old, section, "") != get(new, section, "")])
end

# record of the segment in the settings of a run: outside the setup hash
function _segment_record(segment::Integer, stored, settings::AbstractDict, start_time::Real, saved_before::Integer)
    record = SettingsRecord("index" => Int(segment), "start_time" => Float64(start_time),
                            "resumed" => stored !== nothing, "saved_before" => Int(saved_before))
    stored === nothing && return record
    record["resumed_from"] = stored.source
    record["resumed_from_segment"] = stored.segment
    previous_code = get(stored.settings, "code", SettingsRecord())
    record["previous_commit"] = String(get(previous_code, "commit", "unknown"))
    record["commit_changed"] = record["previous_commit"] != settings["code"]["commit"]
    record["previous_setup_hash"] = String(get(stored.settings, "setup_hash", "unknown"))
    changed = changed_sections(stored.settings, settings)
    record["settings_changed"] = !isempty(changed)
    isempty(changed) || (record["changed_sections"] = changed)
    return record
end
