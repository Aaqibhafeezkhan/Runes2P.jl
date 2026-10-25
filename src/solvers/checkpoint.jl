# Checkpoints and segments of a run. A run that does not fit one allocation is a chain of
# segments. Each segment writes a checkpoint after accepted steps and its own output file,
# rewritten with every checkpoint. The files are HDF5 and written by the package extension
# `Runes2PHDF5Ext`.

"""
    CheckpointSettings(name; directory=".", interval=1800, overwrite=false)

Checkpoints of a run called `name`, written to `directory`:

- `<name>.checkpoint.h5`, the last checkpoint, and `<name>.checkpoint.prev.h5`, the one
  before it, kept so that a failed write never loses the last good state;
- `<name>.segment<k>.h5` (`k` with four digits), the output of segment `k`, holding the
  states the segment saved up to its last checkpoint.

A checkpoint is written after an accepted step when `interval` seconds of wall time have
passed since the last one (0 writes one after every accepted step), and always at the end of
the segment. Each checkpoint rewrites the output file of the segment. Files are written to a
temporary file, synced and renamed.

A run refuses to replace the segment file of another run unless `overwrite` is true.
A run continued with `resume = true` replaces the file with the number of its segment:
the output is written before the checkpoint, so an attempt of the segment that was killed
between the two leaves its file behind, and the checkpoint still names the segment before.
A run refuses to start while segment files of its name with higher numbers than its
segment exist, with `overwrite = true` as well: [`read_output`](@ref) would join those of
an earlier run under the name, or of a continuation that the run replaces, to its output.
The run deletes no file.
"""
struct CheckpointSettings
    name::String
    directory::String
    interval::Float64
    overwrite::Bool
end
function CheckpointSettings(
        name::AbstractString; directory::AbstractString = ".", interval::Real = 1800,
        overwrite::Bool = false)
    isempty(name) && throw(ArgumentError("the checkpoints need a name"))
    occursin('/', name) &&
        throw(ArgumentError("the name of the checkpoints must not contain '/'; use directory"))
    interval >= 0 || throw(ArgumentError("the checkpoint interval must not be negative"))
    return CheckpointSettings(String(name), String(directory), Float64(interval), overwrite)
end

"File of the last checkpoint."
function checkpoint_file(settings::CheckpointSettings)
    joinpath(settings.directory, settings.name * ".checkpoint.h5")
end
"File of the checkpoint before the last."
function previous_checkpoint_file(settings::CheckpointSettings)
    joinpath(settings.directory, settings.name * ".checkpoint.prev.h5")
end
"Output file of segment `k`."
function segment_file(settings::CheckpointSettings, k::Integer)
    joinpath(settings.directory, settings.name * ".segment" * lpad(k, 4, '0') * ".h5")
end

"The number of a segment output file `<name>.segment<k>.h5`."
function segment_number(file::AbstractString)
    parse(Int, match(r"\.segment(\d+)\.h5$", file).captures[1])
end

"""
    segment_files(settings::CheckpointSettings)

The segment output files of the run that exist, in the order of the segments, for
[`read_output`](@ref).
"""
function segment_files(settings::CheckpointSettings)
    pattern = Regex("^" * replace(settings.name, r"([^A-Za-z0-9_])" => s"\\\1") *
                    raw"\.segment(\d+)\.h5$")
    found = Tuple{Int, String}[]
    isdir(settings.directory) || return String[]
    for file in readdir(settings.directory)
        m = match(pattern, file)
        m === nothing ||
            push!(found, (parse(Int, m.captures[1]), joinpath(settings.directory, file)))
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
wall time), the record of the time `stepper` at the time (its step length and what it
carries between steps, [`FixedSteps`](@ref); empty for a state read from an output file,
whose time is then a time point of the run) and the file it came from.
[`read_checkpoint`](@ref) and [`read_state`](@ref) return one.
"""
struct StoredState{T}
    time::Float64
    state::Vector{T}
    layout::UnknownLayout{T}
    update_states::Dict{String, Vector{Float64}}
    settings::Dict{String, Any}
    segment::Int
    saved_count::Int
    summary::Dict{String, Any}
    stepper::Dict{String, Any}
    source::String
end

const SUMMARY_COUNTS = (
    "attempted_steps", "accepted_steps", "newton_iterations", "krylov_iterations")

# the summary of a run after a segment with the statistics `stats`, given the summary
# `before` the segment (empty for the first)
function _run_summary(before::AbstractDict, stats, wall_time::Real)
    summary = SettingsRecord()
    summary["attempted_steps"] = get(before, "attempted_steps", 0) + length(stats.accepted)
    summary["accepted_steps"] = get(before, "accepted_steps", 0) + count(stats.accepted)
    summary["newton_iterations"] = get(before, "newton_iterations", 0) +
                                   sum(stats.newton_iterations)
    summary["krylov_iterations"] = get(before, "krylov_iterations", 0) +
                                   sum(stats.krylov_iterations)
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
function write_atomically(write_file, path::AbstractString; previous::Union{
        Nothing, AbstractString} = nothing)
    directory = dirname(abspath(path))
    temporary = joinpath(directory, "." * basename(path) * "." * string(getpid()) * ".tmp")
    try
        write_file(temporary)
    catch
        rm(temporary; force = true)
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

Write a checkpoint ([`write_atomically`](@ref), keeping the previous one). Defined by the
package extension `Runes2PHDF5Ext`, which loads with HDF5. See also [`read_checkpoint`](@ref).
"""
function write_checkpoint end

"""
    read_checkpoint(settings::CheckpointSettings)
    read_checkpoint(filename)

Read a checkpoint as a [`StoredState`](@ref). Reading with the settings falls back on the
previous checkpoint when the last one is missing or cannot be read, and returns `nothing`
when neither file exists. When checkpoint files exist and none of them can be read it
throws an `ArgumentError`, so that a run with `resume = true` does not start again from
its initial state over its own files. Defined by the package extension `Runes2PHDF5Ext`,
which loads with HDF5.
"""
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
settings are those of the last file. Segments in the wrong order, or whose joined times
repeat one (two runs from the same start), are refused with an `ArgumentError`. So is a
segment that did not resume from the one before it: its `resumed_from_segment` must be the
index of that segment and its `previous_setup_hash` the setup hash of that segment, where
both records hold them. The higher segments of an earlier run under the same name with the
same setup pass this check; a run refuses to start while segment files with higher numbers
than its own exist ([`CheckpointSettings`](@ref)). Defined by the package extension
`Runes2PHDF5Ext`, which loads with HDF5.
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
    resume isa AbstractString ||
        throw(ArgumentError("resume must be true, false, a file name or a StoredState"))
    _require_hdf5(read_state, "resuming from a file")
    return read_state(resume)
end

function _require_hdf5(f, what)
    isempty(methods(f)) &&
        throw(ArgumentError("$what needs the HDF5 package: add `using HDF5` to the script"))
end

# sections of the setup whose hash differs between two settings records
function changed_sections(before::AbstractDict, now::AbstractDict)
    old, new = get(before, "hashes", SettingsRecord()), get(now, "hashes", SettingsRecord())
    return sort!(String[section
                        for section in union(keys(old), keys(new))
                        if get(old, section, "") != get(new, section, "")])
end

# record of the segment in the settings of a run: outside the setup hash. The status, the
# time reached and the number of accepted steps follow the run (`_update_segment!`).
function _segment_record(segment::Integer, stored, settings::AbstractDict,
        start_time::Real, saved_before::Integer)
    record = SettingsRecord("index" => Int(segment), "start_time" => Float64(start_time),
        "resumed" => stored !== nothing, "saved_before" => Int(saved_before),
        "status" => "running", "end_time" => Float64(start_time), "steps" => 0,
        "checkpoint_written" => false)
    # the attempted steps of the run before the segment, by which read_output cuts the
    # statistics of the segment before at the checkpoint (a failed attempt is recorded at
    # the time it aimed at, which may lie beyond the checkpoint); a state from an output
    # file has no count, and the join then falls back on the time
    if stored === nothing
        record["attempts_before"] = 0
    elseif haskey(stored.summary, "attempted_steps")
        record["attempts_before"] = Int(stored.summary["attempted_steps"])
    end
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

# Clean stop of a segment. Julia exits on SIGTERM without running user code, so a run cannot
# react to the signal at the time limit. It stops by itself instead, after an accepted step,
# when the wall time left does not cover another step or when a stop file appears, which a
# batch script can create on the warning signal of the batch system.

"""
    StopCondition(; end_time=nothing, wall_budget=nothing, stop_file=nothing, slack=300)

When a segment stops before the last time point. [`integrate`](@ref) asks after every
accepted step. It stops when

- the wall time left until `end_time` is less than the margin: the longest accepted step
  of the segment so far, plus the longest write of a checkpoint with its output so far,
  plus `slack` seconds (the rule of SCR: step, write time and a slack; the default slack
  is the margin of GS2, 300 s);
- the file `stop_file` exists.

`end_time` is a UNIX time in seconds. By default it is `SLURM_JOB_END_TIME` of the
environment when that is set, so a run in a Slurm job stops before its time limit without
further settings. `wall_budget` gives the end as seconds from now. With both, the earlier
end counts.

A stop file left by an earlier segment is removed when the segment starts. A segment
always completes one step before it stops, so that every segment of a run makes
progress.
"""
struct StopCondition
    end_time::Float64
    stop_file::String
    slack::Float64
end

function StopCondition(; end_time::Union{Nothing, Real} = nothing,
        wall_budget::Union{Nothing, Real} = nothing,
        stop_file::Union{Nothing, AbstractString} = nothing, slack::Real = 300)
    slack >= 0 || throw(ArgumentError("the slack must not be negative"))
    wall_budget === nothing || wall_budget > 0 ||
        throw(ArgumentError("the wall budget must be positive"))
    ends = Float64[]
    if end_time !== nothing
        push!(ends, Float64(end_time))
    elseif haskey(ENV, "SLURM_JOB_END_TIME")
        from_slurm = tryparse(Float64, ENV["SLURM_JOB_END_TIME"])
        from_slurm === nothing || push!(ends, from_slurm)
    end
    wall_budget === nothing || push!(ends, time() + Float64(wall_budget))
    return StopCondition(
        isempty(ends) ? Inf : minimum(ends), stop_file === nothing ? "" : String(stop_file),
        Float64(slack))
end

function describe(stop::StopCondition)
    SettingsRecord("type" => "StopCondition", "end_time" => stop.end_time,
        "stop_file" => stop.stop_file,
        "slack" => stop.slack)
end

"""
    stop_reason(stop, longest_step, longest_write; now=time())

Why the segment stops now, `"stop file"` or `"wall time"`, or `nothing` when it goes on.
`longest_step` and `longest_write` are the wall times in seconds of the longest
accepted step and of the longest checkpoint write of the segment so far.
"""
function stop_reason(stop::StopCondition, longest_step::Real, longest_write::Real; now::Real = time())
    isempty(stop.stop_file) || !isfile(stop.stop_file) || return "stop file"
    now + longest_step + longest_write + stop.slack >= stop.end_time && return "wall time"
    return nothing
end
stop_reason(::Nothing, longest_step, longest_write; now = time()) = nothing

"Exit status of a script whose run finished."
const EXIT_FINISHED = 0
"Exit status of a script whose segment stopped at a checkpoint, to be followed by another segment."
const EXIT_STOPPED = 85
"Exit status of a script whose segment stopped without a checkpoint to continue from."
const EXIT_STOPPED_WITHOUT_CHECKPOINT = 86

"""
    run_status(solution)

`:finished` when the run reached its last time point, `:stopped` when the segment stopped
before it ([`StopCondition`](@ref)). `solution.settings["segment"]` has the reason
(`stop_reason`) and the time reached (`end_time`).
"""
function run_status(solution)
    Symbol(get(get(solution.settings, "segment", SettingsRecord()), "status", "finished"))
end

"""
    exit_status(solution)

Exit status for the script of a segment, which a batch script that chains the segments
reads: $EXIT_FINISHED when the run finished, $EXIT_STOPPED when the segment stopped and
wrote a checkpoint, so that another segment follows, and $EXIT_STOPPED_WITHOUT_CHECKPOINT
when it stopped without checkpoints, where another segment would start from the beginning.
"""
function exit_status(solution)
    run_status(solution) === :finished && return EXIT_FINISHED
    segment = solution.settings["segment"]
    return get(segment, "checkpoint_written", false) ? EXIT_STOPPED :
           EXIT_STOPPED_WITHOUT_CHECKPOINT
end
