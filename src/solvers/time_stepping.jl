# Backward-Euler time stepping: steps between the time points of any
# array, a failed Newton solve retried with shorter steps, saved states at their own
# times, and solver statistics per attempted step with their own time array (DREAM #296).

"""
    SolverStatistics

Solver statistics of a run, one entry per attempted step (accepted or not), with its own
time array (DREAM #296):

- `time`, `time_step`: end time and length of the attempted step;
- `accepted`, `status`: whether the step was accepted, and the [`NewtonResult`](@ref)
  status;
- `newton_iterations`, `krylov_iterations`, `preconditioner_updates` (factorizations of
  the Krylov preconditioner);
- `initial_residual_norm`, `residual_norm`, `update_norm`: scaled norms before the first
  and after the last Newton iteration (see [`NewtonSolver`](@ref));
- `convergence_order`: estimated order of the Newton iteration (about 2 with a correct
  Jacobian; about 1 flags an inexact Jacobian, DREAM #226);
- `smallest_step_length`: the shortest Newton step length used (1 when neither the lower
  limits nor the line search shortened a step);
- `accepted_at_residual_floor`: the step converged with its residual above
  `residual_tolerance`, at the round-off floor of a stiff step (see [`NewtonSolver`](@ref));
- `accepted_as_linear`: the step was accepted as a linear system, on the update test
  alone at the second Newton iteration (see [`NewtonSolver`](@ref));
- `wall_time`: seconds spent in the step, of which `jacobian_time` in the assembly of the
  stored Jacobian blocks, `residual_time` in residual evaluations (line search included),
  `linear_solve_time` in the linear solves and, within it, `factorization_time` in the
  LU or preconditioner factorizations (the rest of the solve time is the Krylov
  iteration, with the matrix-free terms' products).

The history of every Newton iteration is in `iteration_attempt` (index of the attempted
step it belongs to), `iteration_residual_norm`, `iteration_update_norm` and
`iteration_step_length`.
"""
struct SolverStatistics
    time::Vector{Float64}
    time_step::Vector{Float64}
    accepted::Vector{Bool}
    status::Vector{Symbol}
    newton_iterations::Vector{Int}
    krylov_iterations::Vector{Int}
    preconditioner_updates::Vector{Int}
    initial_residual_norm::Vector{Float64}
    residual_norm::Vector{Float64}
    update_norm::Vector{Float64}
    convergence_order::Vector{Float64}
    smallest_step_length::Vector{Float64}
    accepted_at_residual_floor::Vector{Bool}
    accepted_as_linear::Vector{Bool}
    wall_time::Vector{Float64}
    jacobian_time::Vector{Float64}
    residual_time::Vector{Float64}
    factorization_time::Vector{Float64}
    linear_solve_time::Vector{Float64}
    iteration_attempt::Vector{Int}
    iteration_residual_norm::Vector{Float64}
    iteration_update_norm::Vector{Float64}
    iteration_step_length::Vector{Float64}
end
SolverStatistics() = SolverStatistics(Float64[], Float64[], Bool[], Symbol[], Int[], Int[], Int[], Float64[],
                                      Float64[], Float64[], Float64[], Float64[], Bool[], Bool[], Float64[],
                                      Float64[], Float64[], Float64[], Float64[], Int[], Float64[], Float64[], Float64[])

function _record!(stats::SolverStatistics, t, dt, result::NewtonResult, ws::NewtonWorkspace, wall)
    push!(stats.time, t)
    push!(stats.time_step, dt)
    push!(stats.accepted, result.status === :converged)
    push!(stats.status, result.status)
    push!(stats.newton_iterations, result.iterations)
    push!(stats.krylov_iterations, result.krylov_iterations)
    push!(stats.preconditioner_updates, result.preconditioner_updates)
    push!(stats.initial_residual_norm, result.initial_residual_norm)
    push!(stats.residual_norm, result.residual_norm)
    push!(stats.update_norm, result.update_norm)
    push!(stats.convergence_order, result.convergence_order)
    push!(stats.smallest_step_length, result.smallest_step_length)
    push!(stats.accepted_at_residual_floor, result.status === :converged && result.residual_floor)
    push!(stats.accepted_as_linear, result.status === :converged && result.linear_step)
    push!(stats.wall_time, wall)
    push!(stats.jacobian_time, result.jacobian_time)
    push!(stats.residual_time, result.residual_time)
    push!(stats.factorization_time, result.factorization_time)
    push!(stats.linear_solve_time, result.linear_solve_time)
    attempt = length(stats.time)
    append!(stats.iteration_attempt, fill(attempt, length(ws.update_norms)))
    append!(stats.iteration_residual_norm, ws.residual_norms)
    append!(stats.iteration_update_norm, ws.update_norms)
    append!(stats.iteration_step_length, ws.step_lengths)
    return stats
end

"""
    TimeSolution

Result of [`integrate`](@ref): the `layout`, the saved `times`, the saved `states` (one
column per saved time; views of the unknowns through [`unknown_view`](@ref)), the
[`SolverStatistics`](@ref), the record of the run's `settings`
([`run_settings`](@ref)) and `update_states`, per interval update the state in force when
each saved state was computed (one column per saved time). Of a resumed run it holds the
states saved by this segment, those after the time it resumed from.
"""
struct TimeSolution{T,L<:UnknownLayout}
    layout::L
    times::Vector{T}
    states::Matrix{T}
    statistics::SolverStatistics
    settings::Dict{String,Any}
    update_states::Dict{String,Matrix{Float64}}
end
TimeSolution(layout::UnknownLayout, times::Vector, states::Matrix, statistics::SolverStatistics,
             settings::Dict{String,Any}=Dict{String,Any}()) =
    TimeSolution(layout, times, states, statistics, settings, Dict{String,Matrix{Float64}}())

"""
    TimeStepFailure

Thrown by [`integrate`](@ref) when a step still fails after the allowed number of
halvings; holds the `time` and `time_step` of the last attempt, its Newton `status` and
the statistics up to the failure.
"""
struct TimeStepFailure <: Exception
    time::Float64
    time_step::Float64
    status::Symbol
    statistics::SolverStatistics
end

Base.showerror(io::IO, e::TimeStepFailure) =
    print(io, "TimeStepFailure: the step to t = $(e.time) failed with status :$(e.status) ",
          "after halving the time step to $(e.time_step)")

# the part of a run's saved states a segment holds: columns 1:count of the arrays
function _segment_solution(layout, save_times, first_save, count, saved, stats, settings, updates, saved_updates)
    T = eltype(saved)
    update_states = Dict{String,Matrix{Float64}}(update_name(update) => saved_updates[k][:, 1:count]
                                                 for (k, update) in enumerate(updates))
    return TimeSolution(layout, collect(T, save_times[first_save:first_save+count-1]), saved[:, 1:count], stats,
                        settings, update_states)
end

"""
    integrate(system, u0, times; solver=NewtonSolver(), save_times=times, max_step_halvings=10,
              description=Dict(), updates=(), checkpoint=nothing, resume=false, stop=nothing)

Backward-Euler integration of `system` from the state `u0` at `times[1]` through the
time points `times` (any increasing array, e.g. nonuniform or matching the restarts of
another run, DREAM #502). Each interval is one step, solved by [`newton_solve!`](@ref)
with explicit terms at the previous time level. When a step fails, it is retried with
half the time step, up to `max_step_halvings` times; the shorter steps continue to the
end of the interval, whose end point is always hit exactly, and the next interval starts
again with its full length. A step that still fails throws [`TimeStepFailure`](@ref).

States are saved at `save_times` (increasing, inside `[times[1], times[end]]`),
independently of the steps (DREAM #81, #93): a save time that falls between the ends of a
step gets the linear interpolation of its end states, the dense output of backward Euler.
Returns a [`TimeSolution`](@ref).

The solution carries the record of the run's settings ([`run_settings`](@ref)), taken
before the first step and before a stored state is restored, so it describes the setup
as the script built it and is the same in every segment of a run. It holds the grids, the
layout, every term, the solver, the time points, the code version with the commit of
Runes2P, and `description`, a dictionary of plain values the caller adds (a run name, the
origin of the initial state).

`updates` are [`IntervalUpdate`](@ref)s, applied at every time point of `times` before
the interval that starts there (the knock-on cutoff that follows the plasma,
[`KnockOnCutoffUpdate`](@ref)).

`checkpoint` ([`CheckpointSettings`](@ref)) writes checkpoints at time points of `times`
and the output of the segment with each of them; it needs HDF5 loaded.

`resume` continues a run from a stored state: `true` takes the last checkpoint of
`checkpoint` when there is one and starts from `u0` otherwise, so the same script serves
every segment of a run; a file name takes the checkpoint or the last saved state of an
output file; a [`StoredState`](@ref) is what [`read_checkpoint`](@ref) and
[`read_state`](@ref) return. The stored time must be one of `times`, equal to the last
bit, and the run continues from there; a script that builds its time points the same way
in every segment gets the same numbers. The unknowns are matched by name: those in the stored state replace
the values of `u0`, on the same grids, and an unknown the stored state does not have
keeps its value of `u0`. The updates are set back to their stored states. The settings
may differ from those of the run that stored the state (tolerances, the solver, later
time points, terms); the record names the sections that changed (`segment` in the
settings) and a warning lists them.

`stop` ([`StopCondition`](@ref)) ends the segment at a requested time point before the
last one, when the wall time left does not cover another interval or a stop file
appears. The segment writes its checkpoint and returns what it has saved;
[`run_status`](@ref) of the solution is then `:stopped`, and [`exit_status`](@ref) is the
exit status for the script.
"""
function integrate(system::EquationSystem{T}, u0::AbstractVector, times::AbstractVector;
                   solver::NewtonSolver=NewtonSolver(), save_times::AbstractVector=times,
                   max_step_halvings::Integer=10, description=Dict{String,Any}(),
                   updates=(), checkpoint::Union{Nothing,CheckpointSettings}=nothing, resume=false,
                   stop::Union{Nothing,StopCondition}=nothing) where {T}
    layout = system.layout
    n = length(layout)
    length(u0) == n || throw(DimensionMismatch("u0 does not match the layout"))
    isempty(times) && throw(ArgumentError("times is empty"))
    all(k -> times[k+1] > times[k], 1:length(times)-1) ||
        throw(ArgumentError("times must be strictly increasing"))
    issorted(save_times) || throw(ArgumentError("save_times must be increasing"))
    all(s -> first(times) <= s <= last(times), save_times) ||
        throw(ArgumentError("save_times must lie inside [times[1], times[end]]"))
    updates = _checked_updates(updates)
    checkpoint === nothing || _require_hdf5(write_checkpoint, "writing checkpoints")

    settings = run_settings(system; solver, times, save_times, max_step_halvings, description, updates,
                            checkpoint, stop)
    stored = _resume_state(resume, checkpoint)
    u_old = Vector{T}(u0)
    first_interval, first_save, segment = 1, 1, 1
    summary_before = SettingsRecord()
    if stored !== nothing
        found = findfirst(==(stored.time), times)
        found === nothing &&
            throw(ArgumentError("the stored time $(stored.time) of $(stored.source) is not one of the time points"))
        first_interval = found
        copy_unknowns!(u_old, layout, stored.state, stored.layout)
        for update in updates
            state = get(stored.update_states, update_name(update), nothing)
            state === nothing || restore_update!(update, state)
        end
        # the save times up to the stored time belong to the segments before
        first_save = searchsortedlast(save_times, stored.time) + 1
        segment = stored.segment + 1
        summary_before = stored.summary
    end
    settings["segment"] = _segment_record(segment, stored, settings, times[first_interval],
                                          stored === nothing ? 0 : stored.saved_count)
    if checkpoint !== nothing
        mkpath(checkpoint.directory)
        file = segment_file(checkpoint, segment)
        # With resume = true the run owns its name: a file with the number of this segment
        # is what an attempt of it left that was killed between its output and its
        # checkpoint, and is replaced. Any other run must not replace a segment file.
        isfile(file) && !checkpoint.overwrite && resume !== true &&
            throw(ArgumentError("the segment file $file exists; continue the run with resume = true, choose " *
                                "another name or set overwrite = true in the checkpoint settings"))
    end
    if stored !== nothing && settings["segment"]["settings_changed"]
        @warn "resuming $(stored.source) with changed settings" sections = settings["segment"]["changed_sections"]
    end

    # a stop file of an earlier segment is not a request to this one
    stop === nothing || isempty(stop.stop_file) || rm(stop.stop_file; force=true)
    segment_record = settings["segment"]
    function update_segment!(status, time, intervals; reason=nothing, checkpoint_written=false)
        segment_record["status"] = status
        segment_record["end_time"] = Float64(time)
        segment_record["intervals"] = intervals
        reason === nothing || (segment_record["stop_reason"] = reason)
        checkpoint_written && (segment_record["checkpoint_written"] = true)
        return nothing
    end

    ws = NewtonWorkspace(system, solver)
    stats = SolverStatistics()
    save_count = length(save_times) - first_save + 1
    saved = Matrix{T}(undef, n, save_count)
    saved_updates = [Matrix{Float64}(undef, length(update_state(update)), save_count) for update in updates]
    u = copy(u_old)
    next_save = first_save
    function save!(values)
        column = next_save - first_save + 1
        saved[:, column] .= values
        for (k, update) in enumerate(updates)
            saved_updates[k][:, column] .= update_state(update)
        end
        next_save += 1
        return nothing
    end
    apply_updates!(t) = foreach(update -> apply_update!(update, u_old, layout, t), updates)
    segment_solution() = _segment_solution(layout, save_times, first_save, next_save - first_save, saved, stats,
                                           settings, updates, saved_updates)

    started_run = time()
    last_checkpoint = started_run
    longest_interval, longest_write = 0.0, 0.0
    apply_updates!(T(times[first_interval]))
    # the state at the first time point is saved by a run that starts there
    while stored === nothing && next_save <= length(save_times) && save_times[next_save] == first(times)
        save!(u_old)
    end

    for interval in first_interval:length(times)-1
        t, t_end = T(times[interval]), T(times[interval+1])
        interval > first_interval && apply_updates!(t)
        started_interval = time()
        dt = t_end - t
        halvings = 0
        while t < t_end
            # the last step of the interval lands exactly on its end
            t_next = t_end - t <= dt * (1 + 1e-10) ? t_end : t + dt
            h = t_next - t
            copyto!(u, u_old)
            started = time_ns()
            result = newton_solve!(u, u_old, ws, t_next, h)
            _record!(stats, t_next, h, result, ws, (time_ns() - started) / 1e9)
            if result.status !== :converged
                halvings += 1
                halvings > max_step_halvings && throw(TimeStepFailure(t_next, h, result.status, stats))
                dt = h / 2
                continue
            end
            while next_save <= length(save_times) && save_times[next_save] <= t_next
                s = save_times[next_save]
                if s == t_next
                    save!(u)
                else
                    fraction = (s - t) / h
                    save!((1 - fraction) .* u_old .+ fraction .* u)
                end
            end
            copyto!(u_old, u)
            t = t_next
        end
        longest_interval = max(longest_interval, time() - started_interval)
        last_interval = interval == length(times) - 1
        # the last time point is the end of the run, not a stop
        reason = last_interval ? nothing : stop_reason(stop, longest_interval, longest_write)
        status = last_interval ? "finished" : reason === nothing ? "running" : "stopped"
        intervals = interval - first_interval + 1
        if checkpoint !== nothing
            if last_interval || reason !== nothing || time() - last_checkpoint >= checkpoint.interval
                started_write = time()
                update_segment!(status, t_end, intervals; reason, checkpoint_written=true)
                # the output first: a checkpoint then never names states its segment file lacks
                write_output(segment_file(checkpoint, segment), segment_solution(); atomic=true)
                write_checkpoint(checkpoint,
                                 StoredState(Float64(t_end), copy(u_old), layout,
                                             Dict{String,Vector{Float64}}(update_name(update) => update_state(update)
                                                                          for update in updates),
                                             settings, segment, next_save - 1,
                                             _run_summary(summary_before, stats, time() - started_run),
                                             checkpoint_file(checkpoint)))
                last_checkpoint = time()
                longest_write = max(longest_write, last_checkpoint - started_write)
            end
        end
        update_segment!(status, t_end, intervals; reason)
        reason === nothing || return segment_solution()
    end
    update_segment!("finished", times[end], length(times) - first_interval)
    return segment_solution()
end

"""
    write_output(filename, solution::TimeSolution; atomic=false)

Write a [`TimeSolution`](@ref) to an HDF5 file: the root
attributes `format_version` ([`OUTPUT_FORMAT_VERSION`](@ref)), `runes2p_commit`,
`runes2p_version` and `runes2p_tree_clean`, the record of the run under `settings`
([`run_settings`](@ref), read back by [`read_settings`](@ref)), the
[`describe`](@ref) record of the layout under `layout`, the saved `time`, one dataset per
unknown under `unknowns` (scalar `(t)`, radial `(components, r, t)`, kinetic on a shared
grid `(p, xi, r, t)`, on per-radius grids a group with one `(p, xi, t)` dataset per radius
named `radius_<ir>`), and the [`SolverStatistics`](@ref) under `solver_statistics` with
their own time array, and the states of the interval updates per saved time under
`update_states`. `atomic = true` writes to a temporary file that is renamed onto
`filename` ([`write_atomically`](@ref)). [`read_output`](@ref) reads the file back.
Defined by the package extension `Runes2PHDF5Ext`, which loads with HDF5 (`using HDF5`).
"""
function write_output end
