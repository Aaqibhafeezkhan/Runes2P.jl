# Backward-Euler time stepping: steps whose length a stepper
# sets, every time point of the run the end of a step, a failed Newton solve retried with
# shorter steps, saved states at their own times, updates, checkpoints and the stop after
# accepted steps, and solver statistics per attempted step with their own time array
# (DREAM #296).

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

Thrown by [`integrate`](@ref) when a step still fails after the retries the stepper
allows; holds the `time` and `time_step` of the last attempt, its Newton `status` and
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
          "at the time step $(e.time_step)")

# ---------------------------------------------------------------------------------------
# time steppers: what sets the length of the next step and what a
# stepper carries from step to step, which goes into the checkpoint so that a resumed
# run repeats the uninterrupted one

"""
    TimeStepper

Supertype of the steppers of [`integrate`](@ref), which set the length of every step:
[`FixedSteps`](@ref). The state a stepper carries between steps (its step length, the
count of its failures) is recorded in every checkpoint and restored by a resume.
"""
abstract type TimeStepper end

"""
    FixedSteps(; max_halvings=10)

The stepper of DREAM's constant time stepper: each interval between two time points of
the run is one step. When a step fails, it is retried with half the time step, up to
`max_halvings` times; the shorter steps continue to the end of the interval, whose end
point is always hit exactly, and the next interval starts again with its full length. A
step that still fails throws [`TimeStepFailure`](@ref).
"""
struct FixedSteps <: TimeStepper
    max_halvings::Int
end
function FixedSteps(; max_halvings::Integer=10)
    max_halvings >= 0 || throw(ArgumentError("max_halvings must not be negative"))
    return FixedSteps(Int(max_halvings))
end

describe(stepper::TimeStepper) = _describe_fields(stepper)

# the state of FixedSteps: the step length of the interval, halved on failure
mutable struct FixedStepState
    dt::Float64
    halvings::Int
end

# the length of the interval of `times` that holds the time t (its end when t is a time
# point), zero at the last time point
function _interval_length(times, t)
    k = searchsortedlast(times, t)
    return k >= length(times) ? 0.0 : Float64(times[k+1] - times[k])
end

"""
    stepper_state(stepper, times, t, record)

The state of `stepper` at the time `t` of the run with the time points `times`: restored
from the `record` of a checkpoint written by the same kind of stepper, fresh otherwise.
"""
function stepper_state(stepper::FixedSteps, times, t, record::AbstractDict)
    # inside an interval the stored length and halvings continue it; at a time point the
    # interval that starts there sets the length, whatever the time points of the segment
    # that wrote the record were
    get(record, "type", "") == "FixedSteps" && t ∉ times &&
        return FixedStepState(Float64(record["dt"]), Int(record["halvings"]))
    return FixedStepState(_interval_length(times, t), 0)
end

"Record of the state of a stepper, stored in the checkpoint."
stepper_record(state::FixedStepState) =
    SettingsRecord("type" => "FixedSteps", "dt" => state.dt, "halvings" => state.halvings)

"The length of the next step the stepper asks for; the run caps it at the next time point."
step_length(::FixedSteps, state::FixedStepState, t) = state.dt

# after a failed step of length h: whether the stepper retries, with its new length
function step_failed!(stepper::FixedSteps, state::FixedStepState, h)
    state.halvings += 1
    state.halvings > stepper.max_halvings && return false
    state.dt = h / 2
    return true
end

# after an accepted step to t_next; at a time point the next interval starts with its
# full length
function step_accepted!(::FixedSteps, state::FixedStepState, times, t_next, at_time_point)
    if at_time_point
        state.dt = _interval_length(times, t_next)
        state.halvings = 0
    end
    return nothing
end

# the message of a failed step for the log and the exception
_failure_text(stepper::FixedSteps, state::FixedStepState) = "after $(stepper.max_halvings) halvings"
_retry_text(::FixedSteps, state::FixedStepState) = "is retried with half the time step"

# ---------------------------------------------------------------------------------------
# run options

"""
    RunOptions(; solver=NewtonSolver(), stepper=FixedSteps(), save_times=nothing,
               updates=(), checkpoint=nothing, resume=false, stop=nothing, log=nothing,
               description=Dict())

The options of a run of [`integrate`](@ref), which takes them as keywords or as one
record. `save_times = nothing` saves at the time points. The others are described with
[`integrate`](@ref).
"""
Base.@kwdef struct RunOptions{S<:NewtonSolver,P<:TimeStepper,U,C<:Union{Nothing,CheckpointSettings},R,
                              X<:Union{Nothing,StopCondition},L<:Union{Nothing,ProgressLog},D}
    solver::S = NewtonSolver()
    stepper::P = FixedSteps()
    save_times::Union{Nothing,AbstractVector} = nothing
    updates::U = ()
    checkpoint::C = nothing
    resume::R = false
    stop::X = nothing
    log::L = nothing
    description::D = Dict{String,Any}()
end

# the record of a run with these options
run_settings(system::EquationSystem, times::AbstractVector, options::RunOptions) =
    run_settings(system; options.solver, options.stepper, times,
                 save_times=options.save_times === nothing ? times : options.save_times, options.description,
                 options.updates, options.checkpoint, options.stop, options.log)

# the part of a run's saved states a segment holds: columns 1:count of the arrays
function _segment_solution(layout, save_times, first_save, count, saved, stats, settings, updates, saved_updates)
    T = eltype(saved)
    update_states = Dict{String,Matrix{Float64}}(update_name(update) => saved_updates[k][:, 1:count]
                                                 for (k, update) in enumerate(updates))
    return TimeSolution(layout, collect(T, save_times[first_save:first_save+count-1]), saved[:, 1:count], stats,
                        settings, update_states)
end

"""
    integrate(system, u0, times; solver=NewtonSolver(), stepper=FixedSteps(), save_times=times,
              description=Dict(), updates=(), checkpoint=nothing, resume=false, stop=nothing,
              log=nothing)
    integrate(system, u0, times, options::RunOptions)

Backward-Euler integration of `system` from the state `u0` at `times[1]` through the
time points `times` (any increasing array, e.g. nonuniform or matching the restarts of
another run, DREAM #502). The `stepper` ([`TimeStepper`](@ref)) sets the length of every
step, and every time point is the end of a step: with [`FixedSteps`](@ref), the default,
each interval between two time points is one step, retried with half the time step when
it fails (`max_step_halvings = n` is a shorthand for `stepper = FixedSteps(max_halvings=n)`).
Each step is solved by [`newton_solve!`](@ref) with explicit terms at the previous time
level. A step that still fails throws [`TimeStepFailure`](@ref).

States are saved at `save_times` (increasing, inside `[times[1], times[end]]`),
independently of the steps (DREAM #81, #93): a save time that falls between the ends of a
step gets the linear interpolation of its end states, the dense output of backward Euler.
Returns a [`TimeSolution`](@ref).

The solution carries the record of the run's settings ([`run_settings`](@ref)), taken
before the first step and before a stored state is restored, so it describes the setup
as the script built it and is the same in every segment of a run. It holds the grids, the
layout, every term, the solver, the stepper, the time points, the code version with the
commit of Runes2P, and `description`, a dictionary of plain values the caller adds (a run
name, the origin of the initial state).

`updates` are [`IntervalUpdate`](@ref)s, applied at the start of the run and after every
accepted step, before the step that follows (the knock-on cutoff that follows the plasma,
[`KnockOnCutoffUpdate`](@ref)). An update that reports a changed setup discards the
stored Jacobian.

`checkpoint` ([`CheckpointSettings`](@ref)) writes checkpoints after accepted steps and
the output of the segment with each of them; it needs HDF5 loaded. A checkpoint holds
the state of the stepper, so the resumed run repeats the uninterrupted one.

`resume` continues a run from a stored state: `true` takes the last checkpoint of
`checkpoint` when there is one and starts from `u0` otherwise, so the same script serves
every segment of a run; a file name takes the checkpoint or the last saved state of an
output file; a [`StoredState`](@ref) is what [`read_checkpoint`](@ref) and
[`read_state`](@ref) return. The stored time must lie inside the time points of the run;
a state from an output file, or from a checkpoint without the record of the stepper, must
be at one of them, equal to the last bit, so that the stepper can start there. The run
continues from the stored time; a script that builds its time points the same way in
every segment gets the same numbers. The unknowns are matched by name: those in the
stored state replace the values of `u0`, on the same grids, and an unknown the stored
state does not have keeps its value of `u0`. The updates are set back to their stored
states. The settings may differ from those of the run that stored the state (tolerances,
the solver, later time points, terms); the record names the sections that changed
(`segment` in the settings) and a warning lists them.

`stop` ([`StopCondition`](@ref)) ends the segment after an accepted step before the
last time point, when the wall time left does not cover another step or a stop file
appears. The segment writes its checkpoint and returns what it has saved;
[`run_status`](@ref) of the solution is then `:stopped`, and [`exit_status`](@ref) is the
exit status for the script.

`log` ([`ProgressLog`](@ref)) reports while the run is going: a header, entries at
milestones of the simulated time, after a silence and on demand, and the events of the
run when they happen. The warnings of the run are written to it.
"""
function integrate(system::EquationSystem, u0::AbstractVector, times::AbstractVector;
                   max_step_halvings::Union{Nothing,Integer}=nothing, options...)
    if max_step_halvings !== nothing
        haskey(options, :stepper) && throw(ArgumentError("give max_step_halvings or stepper, not both"))
        return integrate(system, u0, times, RunOptions(; stepper=FixedSteps(; max_halvings=max_step_halvings), options...))
    end
    return integrate(system, u0, times, RunOptions(; options...))
end

function integrate(system::EquationSystem, u0::AbstractVector, times::AbstractVector, options::RunOptions)
    log = options.log
    log === nothing && return _integrate(system, u0, times, options)
    # the log takes the warnings of the run, and the report of Julia on SIGUSR1 while it runs
    report = Profile.peek_report[]
    try
        return Logging.with_logger(() -> _integrate(system, u0, times, options), log)
    finally
        Profile.peek_report[] = report
    end
end

# checks of the arguments of a run that do not need the stored state
function _check_run(system::EquationSystem, u0, times, save_times, options::RunOptions)
    length(u0) == length(system.layout) || throw(DimensionMismatch("u0 does not match the layout"))
    isempty(times) && throw(ArgumentError("times is empty"))
    all(k -> times[k+1] > times[k], 1:length(times)-1) ||
        throw(ArgumentError("times must be strictly increasing"))
    issorted(save_times) || throw(ArgumentError("save_times must be increasing"))
    all(s -> first(times) <= s <= last(times), save_times) ||
        throw(ArgumentError("save_times must lie inside [times[1], times[end]]"))
    updates = _checked_updates(options.updates)
    foreach(update -> check_update(update, system), updates)
    for update in updates, t in forced_time_points(update)
        first(times) < t < last(times) && !any(point -> isapprox(point, t; rtol=1e-12), times) &&
            throw(ArgumentError("the update $(update_name(update)) changes at t = $t, which is not a time " *
                                "point of the run"))
    end
    options.checkpoint === nothing || _require_hdf5(write_checkpoint, "writing checkpoints")
    return updates
end

# the start of a segment: the time and the states it starts from, restored from the
# stored state when there is one, and the record of the segment
function _start_segment!(u_old, layout, times, save_times, updates, settings, stored, options::RunOptions)
    T = eltype(u_old)
    t = T(first(times))
    first_save, segment = 1, 1
    summary_before, stepper_before = SettingsRecord(), SettingsRecord()
    if stored !== nothing
        first(times) <= stored.time <= last(times) ||
            throw(ArgumentError("the stored time $(stored.time) of $(stored.source) lies outside the time points"))
        isempty(stored.stepper) && stored.time ∉ times &&
            throw(ArgumentError("the stored time $(stored.time) of $(stored.source) is not one of the time points, " *
                                "which a state without the record of the stepper must be"))
        t = T(stored.time)
        copy_unknowns!(u_old, layout, stored.state, stored.layout)
        for update in updates
            state = get(stored.update_states, update_name(update), nothing)
            state === nothing || restore_update!(update, state)
        end
        # the save times up to the stored time belong to the segments before
        first_save = searchsortedlast(save_times, stored.time) + 1
        segment = stored.segment + 1
        summary_before = stored.summary
        stepper_before = stored.stepper
    end
    settings["segment"] = _segment_record(segment, stored, settings, t, stored === nothing ? 0 : stored.saved_count)
    checkpoint = options.checkpoint
    if checkpoint !== nothing
        mkpath(checkpoint.directory)
        file = segment_file(checkpoint, segment)
        # With resume = true the run owns its name: a file with the number of this segment
        # is what an attempt of it left that was killed between its output and its
        # checkpoint, and is replaced. Any other run must not replace a segment file.
        isfile(file) && !checkpoint.overwrite && options.resume !== true &&
            throw(ArgumentError("the segment file $file exists; continue the run with resume = true, choose " *
                                "another name or set overwrite = true in the checkpoint settings"))
    end
    if stored !== nothing && settings["segment"]["settings_changed"]
        @warn "resuming $(stored.source) with changed settings" sections = settings["segment"]["changed_sections"]
    end
    return (; t, first_save, segment, summary_before, stepper_before)
end

# The state of a running segment: the states and the times, what it has saved, its
# statistics and its record, and the wall-time bookkeeping of the checkpoints and the
# stop. The loop of `_run_steps!` and the helpers below act on it.
mutable struct RunState{T,L<:UnknownLayout,U,W<:NewtonWorkspace,S,P}
    const layout::L
    const times::Vector{T}
    const save_times::Vector{T}
    const first_save::Int
    const updates::U
    const saved::Matrix{T}
    const saved_updates::Vector{Matrix{Float64}}
    const stats::SolverStatistics
    const settings::Dict{String,Any}
    const segment::Int
    const summary_before::Dict{String,Any}
    const u_old::Vector{T}
    const u::Vector{T}
    const ws::W
    const stepper::S
    const stepper_state::P
    t::T
    next_save::Int
    next_point::Int       # index of the first time point above t
    steps::Int            # accepted steps of the segment
    started_run::Float64
    last_checkpoint::Float64
    longest_step::Float64
    longest_write::Float64
end

function _save!(run::RunState, values)
    column = run.next_save - run.first_save + 1
    run.saved[:, column] .= values
    for (k, update) in enumerate(run.updates)
        run.saved_updates[k][:, column] .= update_state(update)
    end
    run.next_save += 1
    return nothing
end

# the updates at the state u_old at the time t; one that changed the setup discards the
# stored Jacobian
function _apply_updates!(run::RunState)
    changed = false
    for update in run.updates
        changed |= apply_update!(update, run.u_old, run.layout, run.t) === true
    end
    changed && invalidate_jacobian!(run.ws)
    return nothing
end

_segment_solution(run::RunState) =
    _segment_solution(run.layout, run.save_times, run.first_save, run.next_save - run.first_save, run.saved,
                      run.stats, run.settings, run.updates, run.saved_updates)

function _update_segment!(run::RunState, status; reason=nothing, checkpoint_written=false)
    record = run.settings["segment"]
    record["status"] = status
    record["end_time"] = Float64(run.t)
    record["steps"] = run.steps
    reason === nothing || (record["stop_reason"] = reason)
    checkpoint_written && (record["checkpoint_written"] = true)
    return nothing
end

function _write_checkpoint!(run::RunState, checkpoint::CheckpointSettings, log, status, reason)
    started_write = time()
    _update_segment!(run, status; reason, checkpoint_written=true)
    # the output first: a checkpoint then never names states its segment file lacks
    write_output(segment_file(checkpoint, run.segment), _segment_solution(run); atomic=true)
    write_checkpoint(checkpoint,
                     StoredState(Float64(run.t), copy(run.u_old), run.layout,
                                 Dict{String,Vector{Float64}}(update_name(update) => update_state(update)
                                                              for update in run.updates),
                                 run.settings, run.segment, run.next_save - 1,
                                 _run_summary(run.summary_before, run.stats, time() - run.started_run),
                                 stepper_record(run.stepper_state), checkpoint_file(checkpoint)))
    run.last_checkpoint = time()
    run.longest_write = max(run.longest_write, run.last_checkpoint - started_write)
    log_event!(log, "checkpoint", Logging.Info,
               "written at t = $(run.t) s in $(round(run.last_checkpoint - started_write; digits=2)) s: " *
               checkpoint_file(checkpoint))
    return nothing
end

function _integrate(system::EquationSystem{T}, u0::AbstractVector, times::AbstractVector,
                    options::RunOptions) where {T}
    layout = system.layout
    save_times = options.save_times === nothing ? times : options.save_times
    updates = _check_run(system, u0, times, save_times, options)
    settings = run_settings(system, times, options)
    stored = _resume_state(options.resume, options.checkpoint)
    u_old = Vector{T}(u0)
    start = _start_segment!(u_old, layout, times, save_times, updates, settings, stored, options)
    stop, log, checkpoint = options.stop, options.log, options.checkpoint

    # a stop file of an earlier segment is not a request to this one
    stop === nothing || isempty(stop.stop_file) || rm(stop.stop_file; force=true)
    save_count = length(save_times) - start.first_save + 1
    state = stepper_state(options.stepper, times, start.t, start.stepper_before)
    saved_updates = Matrix{Float64}[Matrix{Float64}(undef, length(update_state(update)), save_count)
                                    for update in updates]
    started_run = time()
    run = RunState(layout, collect(T, times), collect(T, save_times), start.first_save, updates,
                   Matrix{T}(undef, length(layout), save_count), saved_updates,
                   SolverStatistics(), settings, start.segment, start.summary_before, u_old, copy(u_old),
                   NewtonWorkspace(system, options.solver), options.stepper, state,
                   start.t, start.first_save, searchsortedlast(times, start.t) + 1, 0,
                   started_run, started_run, 0.0, 0.0)
    _apply_updates!(run)
    # the state at the first time point is saved by a run that starts there
    while stored === nothing && run.next_save <= length(save_times) && save_times[run.next_save] == first(times)
        _save!(run, u_old)
    end
    start_log!(log, settings, u_old, layout, run.t, times, run.next_point - 2, run.stats)
    return _run_steps!(run, options)
end

# the entry of the log at the end of the segment
_finish(run::RunState, log, reason) =
    write_entry(log, reason, run.u_old, run.layout, run.t, run.times, run.next_point - 2, run.stats)

function _run_steps!(run::RunState{T}, options::RunOptions) where {T}
    times, layout, stats, ws = run.times, run.layout, run.stats, run.ws
    u_old, u = run.u_old, run.u
    stepper, state = run.stepper, run.stepper_state
    stop, log, checkpoint = options.stop, options.log, options.checkpoint
    while run.t < last(times)
        t, ceiling = run.t, times[run.next_point]
        dt = step_length(stepper, state, t)
        # the last step before a time point lands exactly on it
        t_next = ceiling - t <= dt * (1 + 1e-10) ? ceiling : t + dt
        h = t_next - t
        copyto!(u, u_old)
        started = time_ns()
        result = newton_solve!(u, u_old, ws, t_next, h)
        _record!(stats, t_next, h, result, ws, (time_ns() - started) / 1e9)
        if result.status !== :converged
            if !step_failed!(stepper, state, h)
                log_event!(log, "run failed", Logging.Error,
                           "the step to t = $t_next s with the time step $h s failed with the status " *
                           "$(result.status) " * _failure_text(stepper, state))
                _finish(run, log, "segment end, failed")
                throw(TimeStepFailure(t_next, h, result.status, stats))
            end
            log_event!(log, "step failed", Logging.Warn,
                       "the step to t = $t_next s with the time step $h s ended with the status " *
                       "$(result.status) and " * _retry_text(stepper, state))
            report_step!(log, u_old, layout, t, times, run.next_point - 2, stats)
            continue
        end
        if log !== nothing
            result.residual_floor && !result.linear_step &&
                log_event!(log, "residual floor", Logging.Info,
                           "the step to t = $t_next s was accepted with the residual $(result.residual_norm) " *
                           "above the tolerance, where it had stopped decreasing")
            negative = _negative_cells(u, layout)
            negative > 0 && log_event!(log, "negative values", Logging.Warn,
                                       "$negative negative cells of the kinetic unknowns after the step to " *
                                       "t = $t_next s")
        end
        while run.next_save <= length(run.save_times) && run.save_times[run.next_save] <= t_next
            s = run.save_times[run.next_save]
            if s == t_next
                _save!(run, u)
            else
                fraction = (s - t) / h
                _save!(run, (1 - fraction) .* u_old .+ fraction .* u)
            end
        end
        copyto!(u_old, u)
        run.t = t_next
        run.steps += 1
        at_time_point = t_next == ceiling
        at_time_point && (run.next_point += 1)
        step_accepted!(stepper, state, times, t_next, at_time_point)
        run.longest_step = max(run.longest_step, (time_ns() - started) / 1e9)

        last_step = run.t >= last(times)
        # the last time point is the end of the run, not a stop
        reason = last_step ? nothing : stop_reason(stop, run.longest_step, run.longest_write)
        status = last_step ? "finished" : reason === nothing ? "running" : "stopped"
        if checkpoint !== nothing &&
           (last_step || reason !== nothing || time() - run.last_checkpoint >= checkpoint.interval)
            _write_checkpoint!(run, checkpoint, log, status, reason)
        end
        _update_segment!(run, status; reason)
        if reason !== nothing
            log_event!(log, "stop", Logging.Warn, "the segment stops at t = $(run.t) s: $reason")
            _finish(run, log, "segment end, stopped: $reason")
            return _segment_solution(run)
        end
        last_step && break
        # the setup for the next step follows the accepted state
        _apply_updates!(run)
        report_progress!(log, u_old, layout, run.t, times, run.next_point - 2, stats)
    end
    _update_segment!(run, "finished")
    _finish(run, log, "segment end, finished")
    return _segment_solution(run)
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
