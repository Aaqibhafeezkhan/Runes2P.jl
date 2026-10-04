# Progress log of a run. A run reports while it is going: a header that
# identifies it, an entry at milestones of the simulated time, at the ends of a segment
# and after a silence, and the events of the run when they happen. The size of the log is
# bounded whatever the number of steps: about 120 milestone entries, the entries of the
# silence guard, and a few lines per kind of event between two entries.

using Logging: Logging
using Printf: @sprintf
using Profile: Profile

"Fractions of the span of a run at which a progress entry is written: 0.01 %, 0.1 %, then every 1 %."
const DEFAULT_MILESTONES = Float64[1e-4; 1e-3; (1:100) ./ 100]

"Minutes of silence after which the guard writes an entry, in the order used."
const DEFAULT_GUARD_MINUTES = (1, 2, 5, 10, 20, 30, 60, 90, 120)

"""
    ProgressLog(; file=nothing, io=stdout, color=(io is a terminal), milestones, guard_minutes,
                events_shown=3, state=true, on_signal=true, request_file=nothing)

Progress log of a run, the keyword `log` of [`integrate`](@ref). It writes to `io`, or
appends to `file`, which the segments of a run share.

- A **header** when the segment starts: the commit of Runes2P and whether its tree was
  clean, the Julia version and the machine, the segment, then the solver settings, the
  time points, the layout and grids, the terms of every equation and what the caller
  described, from the record of the run ([`run_settings`](@ref)). Arrays are summarized,
  so the header identifies a run without the settings file and stays short.
- A **progress entry** at the `milestones`, fractions of the span of the run (0.01 %,
  0.1 %, 1 % and then every 1 % by default), and at the start and the end of the segment:
  the simulated time and its fraction, the steps since the last entry with the halved
  ones, the time step, the wall time since the start and the last entry, per step and its
  split, the Newton and Krylov counts, the memory, and an estimate of the wall time left
  from the recent intervals. Below it, set off, the state: per kinetic unknown the
  density and the current density per radius, the total current and the negative cells,
  and the values of the radial and scalar unknowns (`state = false` leaves it out).
- A **guard** entry after a silence. The silence allowed grows through `guard_minutes`
  (1, 2, 5, 10, 20, 30, 60, 90, 120 by default) and stays at the last value. The sequence
  only advances: an entry at a milestone does not set it back, so a slow phase is
  reported often at first and then at a steady low rate.
- The **events** when they happen: a step that failed and is retried with half the time
  step, a step accepted at the residual floor, negative values, a checkpoint written, a
  stop request. Of each kind `events_shown` are written between two entries; the rest is
  counted in the next entry.
- An entry **on demand**, after the step in progress: when the file `request_file`
  appears (`touch` it; the log removes it), and with `on_signal = true` when the signal
  SIGUSR1 reaches the Julia process. The entry takes the place of Julia's profile report.
  Julia still prints its notice and a stack trace of about 45 lines to the error stream
  and profiles for a second, so the file is the quiet way.

Warnings of `integrate` (a resume with changed settings) go to the log as well. With a
terminal as `io`, warnings and errors are coloured.
"""
mutable struct ProgressLog <: Logging.AbstractLogger
    io::IO
    owns_io::Bool
    color::Bool
    milestones::Vector{Float64}
    guard_seconds::Vector{Float64}
    events_shown::Int
    state::Bool
    on_signal::Bool
    request_file::String
    # state of the log during a run
    next_milestone::Int
    guard_index::Int
    started::Float64
    last_entry::Float64
    last_attempt::Int
    last_intervals::Int
    event_counts::Dict{String,Int}
    requested::Threads.Atomic{Bool}
    entries::Int
end

function ProgressLog(; file::Union{Nothing,AbstractString}=nothing, io::IO=stdout,
                     color::Bool=file === nothing && io isa Base.TTY,
                     milestones=DEFAULT_MILESTONES, guard_minutes=DEFAULT_GUARD_MINUTES, events_shown::Integer=3,
                     state::Bool=true, on_signal::Bool=true,
                     request_file::Union{Nothing,AbstractString}=nothing)
    milestones = sort!(collect(Float64, milestones))
    all(m -> 0 < m <= 1, milestones) || throw(ArgumentError("the milestones are fractions in (0, 1]"))
    isempty(guard_minutes) && throw(ArgumentError("the guard needs at least one silence limit"))
    all(>(0), guard_minutes) || throw(ArgumentError("the silence limits of the guard must be positive"))
    events_shown >= 0 || throw(ArgumentError("events_shown must not be negative"))
    if file !== nothing
        mkpath(dirname(abspath(file)))
        io = open(file, "a")
    end
    return ProgressLog(io, file !== nothing, color, milestones, 60.0 .* collect(Float64, guard_minutes),
                       Int(events_shown), state, on_signal, request_file === nothing ? "" : String(request_file),
                       1, 1, time(), time(), 0, 0, Dict{String,Int}(), Threads.Atomic{Bool}(false), 0)
end

describe(log::ProgressLog) =
    SettingsRecord("type" => "ProgressLog", "milestones" => copy(log.milestones),
                   "guard_minutes" => log.guard_seconds ./ 60, "events_shown" => log.events_shown,
                   "state" => log.state, "on_signal" => log.on_signal, "request_file" => log.request_file)

Base.close(log::ProgressLog) = (log.owns_io && isopen(log.io) && close(log.io); nothing)

# ---------------------------------------------------------------------------------------
# formats

_timestamp() = Libc.strftime("%Y-%m-%dT%H:%M:%S", time())

function _duration(seconds::Real)
    isfinite(seconds) || return "unknown"
    seconds < 1 && return @sprintf("%.1f ms", 1000 * seconds)
    seconds < 60 && return @sprintf("%.1f s", seconds)
    total = round(Int, seconds)
    total < 3600 && return @sprintf("%d min %02d s", total ÷ 60, total % 60)
    total < 86400 && return @sprintf("%d h %02d min", total ÷ 3600, (total % 3600) ÷ 60)
    return @sprintf("%d d %02d h", total ÷ 86400, (total % 86400) ÷ 3600)
end

_bytes(bytes::Real) = bytes < 1024^3 ? @sprintf("%.0f MB", bytes / 1024^2) : @sprintf("%.2f GB", bytes / 1024^3)

_number(x::Bool) = string(x)
_number(x::Integer) = string(x)
_number(x::AbstractFloat) = @sprintf("%.6g", x)
_number(x) = string(x)

const MAX_VALUES_SHOWN = 8

# the values of an array in a line: all of a short one, the range and the ends of a long one
function _values(values::AbstractArray{<:Real})
    isempty(values) && return "none"
    length(values) <= MAX_VALUES_SHOWN && ndims(values) <= 1 && return join((_number(v) for v in values), " ")
    smallest, largest = extrema(values)
    shape = join(size(values), " × ")
    return "$shape values from $(_number(smallest)) to $(_number(largest)), " *
           "first $(_number(first(values))), last $(_number(last(values)))"
end
_values(values::AbstractVector{<:AbstractString}) = join(values, ", ")

_text(value) = value isa AbstractArray ? _values(value) : _number(value)

# the values of a record in one line, the type and empty texts left out
_in_line(record::AbstractDict) =
    join(("$key = $(_text(record[key]))" for key in sort!(collect(keys(record)))
          if key != "type" && !(record[key] isa AbstractDict) && record[key] != ""), ", ")

# a record as indented lines `key = value`, the type first, records in it indented further
function _print_record(io::IO, record::AbstractDict, indent::AbstractString; skip=())
    names = sort!(collect(keys(record)); by=name -> (name != "type", name))
    for name in names
        name in skip && continue
        value = record[name]
        value isa AbstractDict && continue
        println(io, indent, name, " = ", _text(value))
    end
    for name in names
        name in skip && continue
        value = record[name]
        value isa AbstractDict || continue
        if isempty(value)
            println(io, indent, name, ": none")
        else
            println(io, indent, name, ":")
            _print_record(io, value, indent * "  ")
        end
    end
    return nothing
end

const RULE = "="^88
const SECTION = "-"^20

_section(io::IO, title) = println(io, SECTION, " ", title, " ", SECTION)

function _paint(log::ProgressLog, level::Logging.LogLevel, text::AbstractString)
    log.color || return text
    code = level >= Logging.Error ? "\e[1;31m" : level >= Logging.Warn ? "\e[1;33m" : "\e[1;36m"
    return code * text * "\e[0m"
end

_level_name(level::Logging.LogLevel) =
    level >= Logging.Error ? "ERROR" : level >= Logging.Warn ? "WARNING" : level >= Logging.Info ? "info" : "debug"

# ---------------------------------------------------------------------------------------
# the logger interface: messages of @info, @warn and @error inside a run

Logging.min_enabled_level(::ProgressLog) = Logging.Info
Logging.shouldlog(::ProgressLog, level, _module, group, id) = true
Logging.catch_exceptions(::ProgressLog) = true

function Logging.handle_message(log::ProgressLog, level::Logging.LogLevel, message, _module, group, id, file,
                                line; kwargs...)
    details = isempty(kwargs) ? "" : " (" * join(("$key = $(_text(value))" for (key, value) in kwargs), ", ") * ")"
    _write_line(log, level, string(message) * details)
    return nothing
end

function _write_line(log::ProgressLog, level::Logging.LogLevel, text::AbstractString)
    println(log.io, "[", _timestamp(), "] ", _paint(log, level, _level_name(level)), ": ", text)
    flush(log.io)
    return nothing
end

"""
    log_event!(log, kind, level, message)

Write an event of the run to the log when it happens. Of each `kind`, `events_shown` are
written between two progress entries and the rest is counted, which the next entry
reports, so that a run with many events of a kind does not fill the log.
"""
function log_event!(log::ProgressLog, kind::AbstractString, level::Logging.LogLevel, message::AbstractString)
    count = get(log.event_counts, kind, 0) + 1
    log.event_counts[kind] = count
    if count <= log.events_shown
        note = count == log.events_shown ? " (further events of this kind are counted in the next entry)" : ""
        _write_line(log, level, kind * ": " * message * note)
    end
    return nothing
end
log_event!(::Nothing, kind, level, message) = nothing

# ---------------------------------------------------------------------------------------
# header

_record(settings::AbstractDict, name) = get(settings, name, SettingsRecord())

function _describe_grid_line(name, grid::AbstractDict)
    radial = grid["r_faces"]
    text = "$name: $(get(grid, "geometry", "unknown")) geometry, $(length(radial) - 1) radii from r = " *
           "$(_number(first(radial))) to $(_number(last(radial))) m"
    momentum(m) = "$(length(m["p_faces"]) - 1) momentum cells from p = $(_number(first(m["p_faces"]))) to " *
                  "$(_number(last(m["p_faces"]))) (first cell $(_number(m["p_faces"][2] - m["p_faces"][1]))), " *
                  "$(length(m["xi_faces"]) - 1) pitch cells"
    if haskey(grid, "momentum_grid")
        return text * ", " * momentum(grid["momentum_grid"])
    end
    grids = grid["momentum_grids"]
    counts = [length(grids["radius_$ir"]["p_faces"]) - 1 for ir in 1:length(grids)]
    return text * ", a momentum grid per radius with $(minimum(counts)) to $(maximum(counts)) momentum cells"
end

"""
    write_header(log, settings)

The header of a segment, from the record of the run: what identifies the run first (code,
machine, segment), then the solver settings, then the physics and grid settings.
"""
function write_header(log::ProgressLog, settings::AbstractDict)
    io = log.io
    code, environment, segment = _record(settings, "code"), _record(settings, "environment"), _record(settings, "segment")
    run = _record(settings, "run")
    println(io, RULE)
    println(io, "Runes2P run", haskey(run, "name") ? " \"$(run["name"])\"" : "", ", segment ", get(segment, "index", 1))
    println(io, "  code        Runes2P ", get(code, "package_version", "unknown"), ", commit ", get(code, "commit", "unknown"),
            get(code, "commit_known", false) ? (get(code, "tree_clean", false) ? " (clean tree)" : " (changed tree)") : "",
            ", branch ", get(code, "branch", "unknown"))
    println(io, "  julia       ", get(code, "julia_version", "unknown"), " on ", get(environment, "hostname", "unknown"),
            " (", get(environment, "cpu_model", "unknown"), "), ", get(environment, "julia_threads", "?"),
            " Julia threads, ", get(environment, "blas_threads", "?"), " BLAS threads (",
            get(environment, "blas_libraries", "unknown"), ")")
    println(io, "  started     ", get(environment, "start_time", "unknown"),
            haskey(environment, "slurm_job_id") ? ", Slurm job $(environment["slurm_job_id"])" : "")
    if get(segment, "resumed", false)
        println(io, "  resumed     from ", segment["resumed_from"], " at t = ", _number(segment["start_time"]), " s, ",
                get(segment, "settings_changed", false) ?
                "changed sections: " * join(segment["changed_sections"], ", ") : "settings unchanged",
                get(segment, "commit_changed", false) ? ", commit changed from $(segment["previous_commit"])" : "")
    end
    println(io, "  setup hash  ", get(settings, "setup_hash", "unknown"),
            get(settings, "complete", true) ? "" : " (the record of the run is not complete)")
    _section(io, "solver")
    _print_record(io, _record(settings, "solver"), "  ")
    _section(io, "time")
    time_record = _record(settings, "time")
    if haskey(time_record, "times")
        times = time_record["times"]
        intervals = length(times) > 1 ? diff(times) : [0.0]
        println(io, "  ", length(times), " time points from t = ", _number(first(times)), " to ", _number(last(times)),
                " s, intervals from ", _number(minimum(intervals)), " to ", _number(maximum(intervals)), " s")
    end
    stepper = get(time_record, "stepper", SettingsRecord())
    println(io, "  ", get(time_record, "save_count", 0), " save times, stepper ", get(stepper, "type", "unknown"),
            isempty(_in_line(stepper)) ? "" : " with " * _in_line(stepper))
    for name in ("checkpoint", "stop", "log")
        record = _record(settings, name)
        get(record, "type", "none") == "none" && continue
        println(io, "  ", name, ": ", _in_line(record))
    end
    _section(io, "layout and grids")
    layout = _record(settings, "layout")
    for name in get(layout, "order", String[])
        unknown = layout["unknowns"][name]
        println(io, "  ", name, ": ", unknown["kind"], haskey(unknown, "grid") ? " on $(unknown["grid"])" : "",
                get(unknown, "components", 1) > 1 ? ", $(unknown["components"]) components" : "", ", scale ",
                _number(unknown["scale"]), ", ", sum(unknown["sizes"]), " entries",
                haskey(unknown, "lower_limit_tolerance") ? ", lower limit with tolerance $(_number(unknown["lower_limit_tolerance"]))" : "",
                haskey(unknown, "lower_limit_value") ? " at $(_number(unknown["lower_limit_value"]))" : "")
    end
    println(io, "  ", get(layout, "unknown_count", "?"), " entries of the state in all")
    grids = _record(settings, "grids")
    for name in sort!(collect(keys(grids)))
        println(io, "  ", _describe_grid_line(name, grids[name]))
    end
    _section(io, "system")
    _print_record(io, _record(_record(settings, "system"), "equations"), "  ")
    println(io, "  time_scale = ", _number(get(_record(settings, "system"), "time_scale", NaN)))
    isempty(_record(settings, "updates")) || (println(io, "  updates:"); _print_record(io, settings["updates"], "    "))
    if !isempty(_record(settings, "prescribed"))
        _section(io, "prescribed")
        _print_record(io, settings["prescribed"], "  ")
    end
    if !isempty(run)
        _section(io, "run")
        _print_record(io, run, "  ")
    end
    println(io, RULE)
    flush(io)
    return nothing
end
write_header(::Nothing, settings) = nothing

# ---------------------------------------------------------------------------------------
# the state of the run in an entry

# density, current density and negative cells of a kinetic unknown per radius
function _kinetic_moments(u::AbstractVector, layout::UnknownLayout, id::Integer)
    grid = layout.kinetic_grids[id]
    nr = radial_cell_count(layout)
    densities, currents = zeros(nr), zeros(nr)
    negative, smallest, largest = 0, Inf, -Inf
    for ir in 1:nr
        f = kinetic_view(u, layout, id, ir)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        volume = grid.radial_grid.spatial_jacobian[ir]
        total = 0.0
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            total += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        densities[ir] = total / volume
        currents[ir] = _current_moment(m, pitch_moment_weights(m), jacobian, f) / volume
        negative += count(<(0), f)
        low, high = extrema(f)
        smallest, largest = min(smallest, low), max(largest, high)
    end
    # the cross section of a cylinder per radial cell is 2π V' Δr with V' = r
    radial = grid.radial_grid
    total_current = 2pi * sum(radial.spatial_jacobian .* radial.r_widths .* currents)
    return (; densities, currents, total_current, negative, smallest, largest)
end

# the state per unknown, with the error estimate of the last accepted step per unknown
# when the stepper has one (`errors`, NaN for an unknown without a check)
function _write_state(io::IO, u::AbstractVector, layout::UnknownLayout, errors=nothing)
    for (id, unknown) in enumerate(layout.unknowns)
        name, kind = String(unknown.name), layout.kinds[id]
        error_text = errors === nothing || isnan(errors[id]) ? "" :
                     ", error estimate " * _number(errors[id]) * " of the tolerance"
        if kind === :kinetic
            moments = _kinetic_moments(u, layout, id)
            println(io, "    ", name, ": density [m^-3] ", _values(moments.densities))
            println(io, "    ", " "^length(name), "  current density [A/m^2] ", _values(moments.currents),
                    ", total current [A] ", _number(moments.total_current))
            println(io, "    ", " "^length(name), "  negative cells ", moments.negative,
                    moments.negative == 0 ? "" : " (smallest value $(_number(moments.smallest)), largest $(_number(moments.largest)))",
                    error_text)
        elseif kind === :radial
            values = [radial_view(u, layout, id, ir)[c] for c in 1:unknown.components, ir in 1:radial_cell_count(layout)]
            println(io, "    ", name, ": ", unknown.components == 1 ? _values(vec(values)) : _values(values), error_text)
        else
            println(io, "    ", name, ": ", _number(only(scalar_view(u, layout, id))), error_text)
        end
    end
    return nothing
end

# number of negative entries of the kinetic unknowns
function _negative_cells(u::AbstractVector, layout::UnknownLayout)
    negative = 0
    for (id, kind) in enumerate(layout.kinds), ir in 1:radial_cell_count(layout)
        kind === :kinetic || continue
        negative += count(<(0), kinetic_view(u, layout, id, ir))
    end
    return negative
end

# ---------------------------------------------------------------------------------------
# entries

"""
    write_entry(log, reason, u, layout, t, times, interval, stats; errors=nothing)

A progress entry: `reason` says why it is written (`milestone`, `silence`, `segment
start`, `segment end`, `on demand`), `u` is the state at the time `t`, `interval` the
number of intervals of `times` done, `stats` the statistics of the segment, `errors` the
error estimate per unknown of the last accepted step when the stepper has one.
"""
function write_entry(log::ProgressLog, reason::AbstractString, u::AbstractVector, layout::UnknownLayout, t::Real,
                     times::AbstractVector, interval::Integer, stats; errors=nothing)
    io = log.io
    now = time()
    span = last(times) - first(times)
    fraction = span > 0 ? (t - first(times)) / span : 1.0
    println(io, "[", _timestamp(), "] ", _paint(log, Logging.Info, "progress"), " ", @sprintf("%7.3f", 100 * fraction),
            " %  t = ", @sprintf("%.6e", t), " s of ", @sprintf("%.6e", last(times)), " s, time point ", interval + 1,
            " of ", length(times), "  (", reason, ")")
    attempts = log.last_attempt+1:length(stats.time)
    accepted = count(@view stats.accepted[attempts])
    if !isempty(attempts)
        newton = sum(@view stats.newton_iterations[attempts])
        krylov = sum(@view stats.krylov_iterations[attempts])
        wall = sum(@view stats.wall_time[attempts])
        accepted_steps = findall(@view stats.accepted[attempts])
        time_step = isempty(accepted_steps) ? NaN : stats.time_step[attempts[last(accepted_steps)]]
        # an attempt that did not converge failed; one that converged and was not accepted
        # was rejected by the stepper on its error, or was a half step of a doubling
        failed = count(s -> s !== :converged, @view stats.status[attempts])
        half = count(==(:half_step), @view stats.kind[attempts])
        rejected = length(attempts) - accepted - failed - half
        println(io, "  steps   ", accepted, " accepted of ", length(attempts), " attempted since the last entry (",
                failed, " failed, ", rejected, " rejected, ", half, " half steps, ",
                count(@view stats.accepted_at_residual_floor[attempts]), " at the residual floor, ",
                count(@view stats.accepted_as_linear[attempts]), " as a linear system), last time step ",
                _number(time_step), " s")
        share(field) = wall > 0 ? @sprintf("%.0f %%", 100 * sum(@view getfield(stats, field)[attempts]) / wall) : "-"
        println(io, "  solver  Newton iterations ", @sprintf("%.2f", newton / max(length(attempts), 1)),
                " per step, Krylov iterations ", @sprintf("%.2f", krylov / max(newton, 1)),
                " per Newton iteration, factorizations ",
                @sprintf("%.2f", sum(@view stats.preconditioner_updates[attempts]) / max(length(attempts), 1)), " per step")
        println(io, "  cost    ", _duration(wall / length(attempts)), " per step: Jacobian ", share(:jacobian_time),
                ", residual ", share(:residual_time), ", factorization ", share(:factorization_time),
                ", linear solve with it ", share(:linear_solve_time))
    else
        println(io, "  steps   none since the last entry")
    end
    intervals_done = interval - log.last_intervals
    remaining = length(times) - 1 - interval
    estimate = intervals_done > 0 ? remaining * (now - log.last_entry) / intervals_done : NaN
    println(io, "  wall    ", _duration(now - log.started), " since the start of the segment, ",
            _duration(now - log.last_entry), " since the last entry, estimate for the ", remaining,
            " time points left: ", remaining == 0 ? "none" : _duration(estimate))
    println(io, "  memory  ", _bytes(Sys.maxrss()), " at most resident, ", _bytes(Base.gc_live_bytes()), " live in Julia")
    hidden = [(kind, count) for (kind, count) in log.event_counts if count > 0]
    if !isempty(hidden)
        println(io, "  events  since the last entry: ",
                join(("$count × $kind" * (count > log.events_shown ? " ($(count - log.events_shown) not shown)" : "")
                      for (kind, count) in sort!(hidden)), ", "))
    end
    if log.state
        println(io)
        println(io, "  ", SECTION, " state at t = ", @sprintf("%.6e", t), " s ", SECTION)
        _write_state(io, u, layout, errors)
        println(io)
    end
    flush(io)
    log.last_entry = now
    log.last_attempt = length(stats.time)
    log.last_intervals = interval
    empty!(log.event_counts)
    log.entries += 1
    return nothing
end
write_entry(::Nothing, reason, u, layout, t, times, interval, stats; errors=nothing) = nothing

"""
    start_log!(log, settings, u, layout, t, times, interval, stats)

Begin the log of a segment: the header and the entry of the segment start. With
`on_signal`, SIGUSR1 asks for an entry from now on, until [`integrate`](@ref) returns and
puts Julia's own report back.
"""
function start_log!(log::ProgressLog, settings::AbstractDict, u, layout, t, times, interval, stats)
    now = time()
    log.started = log.last_entry = now
    log.last_attempt, log.last_intervals, log.guard_index, log.entries = 0, interval, 1, 0
    empty!(log.event_counts)
    log.requested[] = false
    span = last(times) - first(times)
    fraction = span > 0 ? (t - first(times)) / span : 1.0
    # the milestones behind a resumed segment are not written again
    log.next_milestone = something(findfirst(>(fraction), log.milestones), length(log.milestones) + 1)
    write_header(log, settings)
    write_entry(log, "segment start", u, layout, t, times, interval, stats)
    log.on_signal && (Profile.peek_report[] = () -> (log.requested[] = true; nothing))
    return nothing
end
start_log!(::Nothing, settings, u, layout, t, times, interval, stats) = nothing

# whether an entry was asked for, by signal or by the request file, which is removed
function _requested!(log::ProgressLog)
    requested = log.requested[]
    log.requested[] = false
    if !isempty(log.request_file) && isfile(log.request_file)
        rm(log.request_file; force=true)
        requested = true
    end
    return requested
end

"""
    report_progress!(log, u, layout, t, times, interval, stats; errors=nothing)

Called by [`integrate`](@ref) after every accepted step: writes the entry of a
milestone passed, of a request by signal or of the silence guard, whichever applies.
"""
function report_progress!(log::ProgressLog, u, layout, t, times, interval, stats; errors=nothing)
    log.on_signal && yield()      # the task that receives the signal runs here
    span = last(times) - first(times)
    fraction = span > 0 ? (t - first(times)) / span : 1.0
    reason = nothing
    if log.next_milestone <= length(log.milestones) && fraction >= log.milestones[log.next_milestone]
        log.next_milestone = something(findfirst(>(fraction), log.milestones), length(log.milestones) + 1)
        reason = "milestone"
    end
    _requested!(log) && (reason = "on demand")
    if reason === nothing && time() - log.last_entry >= log.guard_seconds[log.guard_index]
        log.guard_index = min(log.guard_index + 1, length(log.guard_seconds))
        reason = "silence"
    end
    reason === nothing || write_entry(log, reason, u, layout, t, times, interval, stats; errors)
    return nothing
end
report_progress!(::Nothing, u, layout, t, times, interval, stats; errors=nothing) = nothing

"""
    report_step!(log, u, layout, t, times, interval, stats; errors=nothing)

Called by [`integrate`](@ref) after every failed or rejected step: the guard and the
request by signal are served with the state of the last accepted step, so that a
stretch of failed steps is not silent.
"""
function report_step!(log::ProgressLog, u, layout, t, times, interval, stats; errors=nothing)
    log.on_signal && yield()      # the task that receives the signal runs here
    reason = nothing
    if _requested!(log)
        reason = "on demand"
    elseif time() - log.last_entry >= log.guard_seconds[log.guard_index]
        log.guard_index = min(log.guard_index + 1, length(log.guard_seconds))
        reason = "silence"
    end
    reason === nothing || write_entry(log, reason, u, layout, t, times, interval, stats; errors)
    return nothing
end
report_step!(::Nothing, u, layout, t, times, interval, stats; errors=nothing) = nothing
