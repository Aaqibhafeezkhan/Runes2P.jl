# Backward-Euler time stepping: steps whose length a stepper sets, every time point of the
# run the end of a step, a failed Newton solve retried with shorter steps, saved states at
# their own times, updates, checkpoints and the stop after accepted steps, and solver
# statistics per attempted step with their own time array.

"""
    SolverStatistics

Solver statistics of a run, one entry per attempted step (accepted or not), with its own
time array:

- `time`, `time_step`: end time and length of the attempted step;
- `accepted`, `status`: whether the step was accepted, and the [`NewtonResult`](@ref)
  status; a step that converged and was not accepted was rejected by the stepper on its
  error estimate, or was a half step of a step doubling;
- `kind`: `:step` for a step of the run (accepted, failed or rejected), `:half_step` for
  the two half steps of a step doubling ([`StepDoubling`](@ref));
- `error`: the error estimate of the step relative to its tolerance (accepted at most 1),
  `NaN` where the stepper made none;
- `newton_iterations`, `krylov_iterations`, `preconditioner_updates` (factorizations of
  the Krylov preconditioner);
- `initial_residual_norm`, `residual_norm`, `update_norm`: scaled norms before the first
  and after the last Newton iteration (see [`NewtonSolver`](@ref));
- `convergence_order`: estimated order of the Newton iteration (about 2 with a correct
  Jacobian; about 1 flags an inexact Jacobian, with which the iteration still converges);
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
    kind::Vector{Symbol}
    error::Vector{Float64}
end
function SolverStatistics()
    SolverStatistics(
        Float64[], Float64[], Bool[], Symbol[], Int[], Int[], Int[], Float64[],
        Float64[], Float64[], Float64[], Float64[], Bool[], Bool[], Float64[],
        Float64[], Float64[], Float64[], Float64[], Int[], Float64[], Float64[], Float64[],
        Symbol[], Float64[])
end

function _record!(
        stats::SolverStatistics, t, dt, result::NewtonResult, ws::NewtonWorkspace, wall;
        kind::Symbol = :step, accepted::Bool = result.status === :converged, error::Real = NaN)
    push!(stats.time, t)
    push!(stats.time_step, dt)
    push!(stats.accepted, accepted)
    push!(stats.status, result.status)
    push!(stats.kind, kind)
    push!(stats.error, Float64(error))
    push!(stats.newton_iterations, result.iterations)
    push!(stats.krylov_iterations, result.krylov_iterations)
    push!(stats.preconditioner_updates, result.preconditioner_updates)
    push!(stats.initial_residual_norm, result.initial_residual_norm)
    push!(stats.residual_norm, result.residual_norm)
    push!(stats.update_norm, result.update_norm)
    push!(stats.convergence_order, result.convergence_order)
    push!(stats.smallest_step_length, result.smallest_step_length)
    push!(stats.accepted_at_residual_floor, result.status === :converged &&
        result.residual_floor)
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
struct TimeSolution{T, L <: UnknownLayout}
    layout::L
    times::Vector{T}
    states::Matrix{T}
    statistics::SolverStatistics
    settings::Dict{String, Any}
    update_states::Dict{String, Matrix{Float64}}
end
function TimeSolution(
        layout::UnknownLayout, times::Vector, states::Matrix, statistics::SolverStatistics,
        settings::Dict{String, Any} = Dict{String, Any}())
    TimeSolution(
        layout, times, states, statistics, settings, Dict{String, Matrix{Float64}}())
end

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

function Base.showerror(io::IO, e::TimeStepFailure)
    print(
        io, "TimeStepFailure: the step to t = $(e.time) failed with status :$(e.status) ",
        "at the time step $(e.time_step)")
end

# ---------------------------------------------------------------------------------------
# time steppers: what sets the length of the next step and what a stepper carries from step
# to step, which goes into the checkpoint so that a resumed run repeats the uninterrupted
# one. A stepper implements
#
# - `stepper_state(stepper, system, times, t, record)`: its state at the time t, fresh or
#   restored from the record of a checkpoint;
# - `stepper_record(state)`: the record of its state for the checkpoint;
# - `ceilings(stepper, times, save_times)`: the times every step must end on;
# - `advance!(stepper, state, run, options, ceiling)`: one attempt to advance the run from
#   its time to at most `ceiling`, with its own solves and their statistics; `(t_next, h)`
#   with the accepted state in `run.u`, or `nothing` when the attempt failed or was
#   rejected and the stepper retries with another length (it throws when it gives up);
# - `reset_history!(state)`: forget the accepted states before the current one, after an
#   update changed the setup; nothing by default.

"""
    TimeStepper

Supertype of the steppers of [`integrate`](@ref), which set the length of every step:
[`FixedSteps`](@ref), [`StepDoubling`](@ref) and [`StepController`](@ref). The state a stepper carries between
steps (its step length, its counts) is recorded in every checkpoint and restored by a
resume.
"""
abstract type TimeStepper end

describe(stepper::TimeStepper) = _describe_fields(stepper)

# the time points of the run are the ceilings of every stepper
ceilings(::TimeStepper, times, save_times) = collect(Float64, times)

# the end of a step of the length the stepper asks for: the last step before a ceiling
# lands exactly on it; the slack covers the rounding of t + dt
function _step_end(t, dt, ceiling)
    ceiling - t <= dt * (1 + 1e-10) + 4eps(ceiling) ? ceiling : t + dt
end

# one solve from the state `from` to the time t_next, into `to`, recorded in the
# statistics; `to` holds the last iterate on failure
function _solve_step!(run, from, to, t_next, h; kind::Symbol = :step)
    copyto!(to, from)
    started = time_ns()
    result = newton_solve!(to, from, run.ws, t_next, h)
    # a half step is never a step of the run
    _record!(run.stats, t_next, h, result, run.ws, (time_ns() - started) / 1e9; kind,
        accepted = kind === :step && result.status === :converged)
    return result
end

# a failed solve that the stepper gives up on
function _give_up(run, options, t_next, h, result, how)
    log = options.log
    log_event!(log, "run failed", Logging.Error,
        "the step to t = $t_next s with the time step $h s failed with the status $(result.status) $how")
    _finish(run, log, "segment end, failed")
    throw(TimeStepFailure(t_next, h, result.status, run.stats))
end

# a failed solve that the stepper retries
function _retry_after_failure(run, options, t_next, h, result, how)
    log = options.log
    log_event!(log, "step failed", Logging.Warn,
        "the step to t = $t_next s with the time step $h s ended with the status $(result.status) and $how")
    report_step!(
        log, run.u_old, run.layout, run.t, run.times, run.next_point - 2, run.stats;
        errors = stepper_errors(run.stepper_state))
    return nothing
end

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
function FixedSteps(; max_halvings::Integer = 10)
    max_halvings >= 0 || throw(ArgumentError("max_halvings must not be negative"))
    return FixedSteps(Int(max_halvings))
end

# the state of FixedSteps: the step length of the interval, halved on failure
mutable struct FixedStepState
    dt::Float64
    halvings::Int
end

# the length of the interval of `times` that holds the time t (its end when t is a time
# point), zero at the last time point
function _interval_length(times, t)
    k = searchsortedlast(times, t)
    return k >= length(times) ? 0.0 : Float64(times[k + 1] - times[k])
end

"""
    stepper_state(stepper, system, times, t, record)

The state of `stepper` at the time `t` of the run of `system` with the time points
`times`: restored from the `record` of a checkpoint written by the same kind of stepper,
fresh otherwise.
"""
function stepper_state(stepper::FixedSteps, system, times, t, record::AbstractDict)
    # inside an interval the stored length and halvings continue it; at a time point the
    # interval that starts there sets the length, whatever the time points of the segment
    # that wrote the record were
    get(record, "type", "") == "FixedSteps" && t ∉ times &&
        return FixedStepState(Float64(record["dt"]), Int(record["halvings"]))
    return FixedStepState(_interval_length(times, t), 0)
end

"The error estimate per unknown of the last accepted step, for the log; `nothing` without one."
stepper_errors(state) = nothing
reset_history!(state) = nothing

"Record of the state of a stepper, stored in the checkpoint."
function stepper_record(state::FixedStepState)
    SettingsRecord("type" => "FixedSteps", "dt" => state.dt, "halvings" => state.halvings)
end

function advance!(stepper::FixedSteps, state::FixedStepState, run, options, ceiling)
    t = run.t
    t_next = _step_end(t, state.dt, ceiling)
    h = t_next - t
    result = _solve_step!(run, run.u_old, run.u, t_next, h)
    if result.status !== :converged
        state.halvings += 1
        state.halvings > stepper.max_halvings &&
            _give_up(
                run, options, t_next, h, result, "after $(stepper.max_halvings) halvings")
        state.dt = h / 2
        return _retry_after_failure(
            run, options, t_next, h, result, "is retried with half the time step")
    end
    # at a time point the next interval starts with its full length
    if t_next == run.times[run.next_point]
        state.dt = _interval_length(run.times, t_next)
        state.halvings = 0
    end
    return (t_next, h)
end

"""
    StepDoubling(; initial_step=nothing, relative_tolerance=1e-2, check_every=0, safety=0.95,
                 exponent=1.0, max_failures=5, failure_factor=0.2)

DREAM's adaptive time stepper (`TYPE_ADAPTIVE`, `TimeStepperAdaptive.cpp`), the
compatibility option of the step control. A checked step is solved three times: two steps of
half the length and one of the full length from the same state, so the error estimate costs
two solves per step of progress. The error of an unknown is the 2-norm of the difference
between the full step and the two half steps over the 2-norm of the full step,
`‖u_full - u_half‖ / (absolute + relative ‖u_full‖)`, taken over every entry of the unknown
at every radius; the error of the step is the largest over the unknowns. The step is
accepted when it is at most 1, and its solution is that of the full step, as in DREAM. The
next step is `safety · h / error^exponent` in either case, without a cap on its growth
(DREAM: safety 0.95, exponent 1.0; its stabiliser, exponent 0.7 with a memory of the last
error, is compiled out). `check_every` steps between two checked steps are taken at the
current length without a check, the documented intent of DREAM's `checkevery`. DREAM's code
takes them from a stale `initTime`, with the lengths `h₁ − h₀`, `−h₀`, `−(h₀ + h₁)`, and
`checkevery + 1` of them (`TimeStepperAdaptive.cpp:119-122`, `145-152`, `334-338`), so no
option reproduces it.

The tolerances are the [`Tolerance`](@ref)s the system holds for the time-step error
(`time_step_tolerances` of [`EquationSystem`](@ref)); an unknown without one gets
`relative_tolerance` and no absolute tolerance (DREAM's default absolute tolerance is
zero for every unknown but the runaway density and current, which get 1e-10 and
`e c 1e-10`). An unknown with both tolerances zero is left out of the check. A step whose
solve fails is retried with `failure_factor` times its length, at most `max_failures`
times in a row (DREAM's `STEP_REDUCTION_AT_EXCEPTION` and `MAX_STEPS_WITH_EXCEPTION`);
in a step between checks DREAM ends the run instead, Runes2P retries there too.
DREAM's step update reads the tolerances of the wrong unknowns when a prescribed unknown
precedes others; Runes2P uses one set of tolerances for the acceptance and the
next step.

The first step has the length `initial_step` (`nothing`: the first interval of the time
points). Every step ends at or before the next time point of the run, and a save time
must be a time point, so that every saved state is a computed one, as with
[`FixedSteps`](@ref) (DREAM saves computed steps only and has no save times). A step
shortened by a time point sets the next length from its own error, as a step to `tmax`
does in DREAM.
"""
struct StepDoubling <: TimeStepper
    initial_step::Float64      # NaN: the first interval of the time points
    relative_tolerance::Float64
    check_every::Int
    safety::Float64
    exponent::Float64
    max_failures::Int
    failure_factor::Float64
end
function StepDoubling(;
        initial_step::Union{Nothing, Real} = nothing, relative_tolerance::Real = 1e-2,
        check_every::Integer = 0, safety::Real = 0.95, exponent::Real = 1.0, max_failures::Integer = 5,
        failure_factor::Real = 0.2)
    initial_step === nothing || initial_step > 0 ||
        throw(ArgumentError("the initial step must be positive"))
    relative_tolerance > 0 ||
        throw(ArgumentError("the relative tolerance must be positive"))
    check_every >= 0 || throw(ArgumentError("check_every must not be negative"))
    0 < safety <= 1 || throw(ArgumentError("the safety factor must lie in (0, 1]"))
    exponent > 0 || throw(ArgumentError("the exponent must be positive"))
    max_failures >= 0 || throw(ArgumentError("max_failures must not be negative"))
    0 < failure_factor < 1 || throw(ArgumentError("the failure factor must lie in (0, 1)"))
    return StepDoubling(
        initial_step === nothing ? NaN : Float64(initial_step), Float64(relative_tolerance),
        Int(check_every), Float64(safety), Float64(exponent), Int(max_failures),
        Float64(failure_factor))
end

# the state of StepDoubling: the step length, the steps taken since the last check, the
# failures in a row, the tolerances per unknown, the buffers of the half steps and the
# errors per unknown of the last check
mutable struct StepDoublingState{T}
    dt::Float64
    steps_since_check::Int
    failures::Int
    const absolute::Vector{Float64}   # per unknown of the layout
    const relative::Vector{Float64}
    const half::Vector{T}             # the state after the first half step
    const doubled::Vector{T}          # the state after the two half steps
    const errors::Vector{Float64}     # per unknown, NaN before the first check
end

function stepper_state(stepper::StepDoubling, system::EquationSystem{T},
        times, t, record::AbstractDict) where {T}
    layout = system.layout
    absolute, relative = Float64[], Float64[]
    for u in layout.unknowns
        tolerance = get(system.time_step_tolerances, u.name, nothing)
        push!(absolute, tolerance === nothing ? 0.0 : tolerance.absolute)
        push!(relative, tolerance === nothing ? stepper.relative_tolerance :
                        tolerance.relative)
    end
    n = length(layout)
    if get(record, "type", "") == "StepDoubling"
        return StepDoublingState{T}(
            Float64(record["dt"]), Int(record["steps_since_check"]),
            Int(record["failures"]),
            absolute, relative, zeros(T, n), zeros(T, n), fill(NaN, length(layout.unknowns)))
    end
    dt = isnan(stepper.initial_step) ? _interval_length(times, t) : stepper.initial_step
    # the first check comes with the first step, as in DREAM
    return StepDoublingState{T}(
        dt, stepper.check_every, 0, absolute, relative, zeros(T, n), zeros(T, n),
        fill(NaN, length(layout.unknowns)))
end

stepper_errors(state::StepDoublingState) = state.errors

function stepper_record(state::StepDoublingState)
    SettingsRecord("type" => "StepDoubling", "dt" => state.dt,
        "steps_since_check" => state.steps_since_check,
        "failures" => state.failures)
end

# the error per unknown of the full step `full` against the two half steps `doubled`,
# and the largest of them; an unknown with zero tolerances is left out (NaN)
function _doubling_errors!(state::StepDoublingState, layout, full, doubled)
    largest = 0.0
    for id in eachindex(layout.unknowns)
        absolute, relative = state.absolute[id], state.relative[id]
        if absolute == 0 && relative == 0
            state.errors[id] = NaN
            continue
        end
        norm_full = norm_difference = 0.0
        for range in _unknown_ranges(layout, id), i in range

            norm_full += full[i]^2
            norm_difference += (full[i] - doubled[i])^2
        end
        scale = absolute + relative * sqrt(norm_full)
        # a scale of zero is an unknown at zero, which is no error (DREAM skips it)
        error = scale == 0 ? 0.0 : sqrt(norm_difference) / scale
        isfinite(error) || (error = Inf)
        state.errors[id] = error
        largest = max(largest, error)
    end
    return largest
end

# a failed solve of the doubling: the step length shrinks by the failure factor from the
# step tried, `step`, which a ceiling may have cut below the planned length
function _doubling_failure!(
        stepper::StepDoubling, state::StepDoublingState, run, options, t_next, h, result;
        step = h)
    state.failures += 1
    state.failures > stepper.max_failures &&
        _give_up(run, options, t_next, h, result,
            "after $(stepper.max_failures) failures in a row")
    state.dt = stepper.failure_factor * min(state.dt, step)
    return _retry_after_failure(run, options, t_next, h, result,
        "is retried with $(stepper.failure_factor) times the time step")
end

function advance!(stepper::StepDoubling, state::StepDoublingState, run, options, ceiling)
    t, u_old, u = run.t, run.u_old, run.u
    t_next = _step_end(t, state.dt, ceiling)
    h = t_next - t
    if state.steps_since_check < stepper.check_every
        # a step without a check, at the current length
        result = _solve_step!(run, u_old, u, t_next, h)
        result.status === :converged ||
            return _doubling_failure!(stepper, state, run, options, t_next, h, result)
        state.steps_since_check += 1
        state.failures = 0
        return (t_next, h)
    end
    # the two half steps, then the full step from the same state
    t_half = t + h / 2
    # DREAM counts the failures in a row of every solve (`ValidateStep` after each converged
    # one), so a converged half step resets the count
    result = _solve_step!(run, u_old, state.half, t_half, t_half - t; kind = :half_step)
    result.status === :converged ||
        return _doubling_failure!(
            stepper, state, run, options, t_half, t_half - t, result; step = h)
    state.failures = 0
    result = _solve_step!(
        run, state.half, state.doubled, t_next, t_next - t_half; kind = :half_step)
    result.status === :converged ||
        return _doubling_failure!(
            stepper, state, run, options, t_next, t_next - t_half, result; step = h)
    state.failures = 0
    copyto!(u, u_old)
    started = time_ns()
    result = newton_solve!(u, u_old, run.ws, t_next, h)
    wall = (time_ns() - started) / 1e9
    if result.status !== :converged
        _record!(run.stats, t_next, h, result, run.ws, wall)
        return _doubling_failure!(stepper, state, run, options, t_next, h, result)
    end
    error = _doubling_errors!(state, run.layout, u, state.doubled)
    accepted = error <= 1
    _record!(run.stats, t_next, h, result, run.ws, wall; accepted, error)
    # DREAM: dt = safety dt / error^exponent, from the step just measured
    state.dt = error == 0 ? Inf : stepper.safety * h / error^stepper.exponent
    # DREAM redoes a rejected step as a new doubling (`TimeStepperAdaptive.cpp:152-155`), so the
    # unchecked steps start only after an accepted check
    state.steps_since_check = accepted ? 0 : stepper.check_every
    state.failures = 0
    if !accepted
        log = options.log
        log_event!(log, "step rejected", Logging.Info,
            "the step to t = $t_next s with the time step $h s has the error $error of its tolerance " *
            "and is retried with the time step $(state.dt) s")
        report_step!(log, u_old, run.layout, t, run.times, run.next_point - 2, run.stats;
            errors = stepper_errors(state))
        return nothing
    end
    return (t_next, h)
end

"""
    StepController(; initial_step=nothing, relative_tolerance=1e-3, max_growth=2.0, min_shrink=0.2,
                   safety=0.9, gains=nothing, max_failures=5, failure_factor=0.2,
                   max_rejections=20, tail_cut=1.0, tail_floor=1e-12, order=1)

The step controller of Runes2P: an error estimate of the backward-Euler step without an
extra solve, tolerances per unknown with a floor, and a PI filter on the step length with a
cap on its growth. It is the adaptive stepper of this package; DREAM's
[`StepDoubling`](@ref) is the compatibility option.

The error estimate is the difference between the solution of the step and its predictor,
the linear extrapolation through the two last accepted states. Two backward-Euler states
satisfy `u_n - u_{n-1} = h_old f(u_n)`, so the extrapolation `u_n + (h / h_old) (u_n -
u_{n-1})` is the forward-Euler predictor `u_n + h f(u_n)` with the derivative of the last
step, whatever `h_old` is, and the implicit solution differs from it by `h (f(u_{n+1}) -
f(u_n)) = h² u''`, twice the local error `h² u'' / 2` of the step. The estimate is half
the difference and costs one vector operation. The first step of a run has no history:
it is solved as one step doubling ([`StepDoubling`](@ref), twice the cost of that step
alone), whose difference `u_full - u_half` is half the local error.

The error of an unknown is the root mean square over its entries of the local error over
`absolute + relative |u|`, with the [`Tolerance`](@ref) the system holds for the time-step
error (`time_step_tolerances` of [`EquationSystem`](@ref)); an unknown without one gets
`relative_tolerance` and the absolute tolerance `relative_tolerance` times its scale, a
floor that keeps entries near zero (the tail of a distribution, a density before its source
acts) from holding the step. A radial unknown with more than one group of components
([`component_groups`](@ref), the charge states of the ion species) gets no absolute part and
the `group` part `relative_tolerance` in its place: each entry is measured against
`relative_tolerance` times the sum over its group, the density of its species, so a trace
impurity is controlled on its own scale. An unknown with all tolerances zero is left out.
The error of the step is the largest over the unknowns; the step is accepted when it is
at most 1.

The next step follows the PI step-size control of K. Gustafsson, M. Lundh and
G. Söderlind, BIT 28, 270 (1988) and K. Gustafsson, ACM Trans. Math. Softw. 17, 533
(1991); see E. Hairer and G. Wanner, Solving Ordinary Differential Equations II, 2nd ed.
(Springer, 1996), section IV.2:
`h_new = h · safety · error^(-gains[1]) · error_old^(gains[2])`, with the factor kept
between `min_shrink` and `max_growth`. The default gains are the elementary 0.7 and 0.4
over the order of the local error, `(0.35, 0.2)` for backward Euler, whose local error is
of second order in the step (DREAM's stabiliser has them over the order 1). The two
`gains` are the fields `gain_error` and `gain_memory` of the record. The memory
`error_old` is floored at 1e-4, as in Hairer's codes, so that a step after one with an
error near zero is not shrunk by the limit. An interval update that reports a change of the
setup restarts the estimate: the next step takes it from a doubling, as the first step of
a run. A rejected step is repeated with `h · max(min_shrink, safety · error^(-1/(k+1)))`,
k the order of the step, and keeps the memory of the filter; more than `max_rejections`
rejections in a row end the run with [`TimeStepFailure`](@ref) (a settings error, such as
a `min_shrink` near 1 with a step whose error does not fall). A step whose solve fails is
retried with `failure_factor` times the length tried, at most `max_failures` times in a row.

The first step has the length `initial_step` (`nothing`: the first interval of the time
points). Every step ends at or before the next time point and the next save time, so
every saved state is a computed one. A step cut short by a time point or a save time is
estimated like any other, and the next length follows from the step taken (at most
`max_growth` times it), not from the length that was planned: after a run of close save
times the step grows back by the cap, each step under the tolerance.

The cells of the hot tail of a kinetic unknown lie far below the absolute floor of the
cell measure (`relative × scale`, 1e17 m⁻³ at a density of 1e20 m⁻³ and a tolerance of
1e-3, against 1e15 m⁻³ in the tail), so the cells alone leave the tail uncontrolled.
With `tail_cut` (a momentum in units of mₑc, `nothing` for none) the error of
every kinetic unknown also holds its moments above the cut: the density and the current
density at every radius ([`density`](@ref) and [`current_density`](@ref) with a
[`MomentumRegion`](@ref)), each estimated by the moment of the cell estimate, over
`absolute + relative |moment|` with the absolute tolerance `tail_floor` times the density
of the electrons on all kinetic grids at that radius for the density, and e c times that
for the current, so that the floor does not depend on the scale of the unknown and stays
a floor for a runaway grid whose electrons all lie above the cut. The error of the unknown
is the larger of the root mean square over its cells and that over its `2 nr` moments,
taken per radius on the momentum grid of that radius. On a two-grid layout the runaway
unknown lies above the cut as a whole, so its moments are its density and current.

With `order = 2` the steps are those of the variable-step BDF2,
`u - a u_n + b u_{n-1} = c h f(u)` with `a = (1+ω)²/(1+2ω)`, `b = ω²/(1+2ω)`, `c = (1+ω)/(1+2ω)`
and `ω` the ratio of the step to the one before. The solve is the backward-Euler residual
with the reference `a u_n - b u_{n-1}` in the time derivative and the effective step `c h`
([`residual!`](@ref)); the terms see `u_n` as with backward Euler, and the statistics
record `h`. The estimate is Milne's: the quadratic extrapolation through the last three
accepted states, `ρ (u - P)` with `ρ = h(h+h1)/((h+h1+h2)(2h+h1) + h(h+h1))`, 2/11 at
constant steps, for a local error of third order in the step. The first two steps of a run
and the first two after an update that changes the setup are backward-Euler steps, with
the gains scaled to their order. BDF2 is zero-stable for `ω < 1 + √2`, so `max_growth`
must stay below that. It keeps neither the positivity of backward Euler with an M-matrix
operator nor the second order for explicit and lagged terms, which it warns about. The
default is `order = 1`, backward Euler, which DREAM uses.

The estimate assumes that every term is implicit. An explicit or lagged term contributes
`h h_old u''/2` in place of `h² u''/2` to the difference, which the growth cap bounds to a
factor of order 2.

The state of the controller (the step length, the memory of the filter, the last accepted
state and its step) is in every checkpoint. With a direct solver a resumed run repeats the
uninterrupted one bit for bit. With [`GMRES`](@ref)`(preconditioner_refresh = :on_demand)`
the factors of the preconditioner are not in the checkpoint, so the Krylov solves after a
resume differ within the Newton tolerance, and the step lengths may differ from those of
the uninterrupted run. A run resumed onto another layout (an
unknown added or removed) keeps the step length and starts with a doubling, since the
stored state of the predictor does not fit; the count of rejections in a row starts at
zero in every segment.
"""
struct StepController <: TimeStepper
    initial_step::Float64      # NaN: the first interval of the time points
    relative_tolerance::Float64
    max_growth::Float64
    min_shrink::Float64
    safety::Float64
    gain_error::Float64      # the exponent of the error of the step in the filter
    gain_memory::Float64     # the exponent of the error of the step before
    max_failures::Int
    failure_factor::Float64
    max_rejections::Int
    tail_cut::Float64          # the momentum above which the tail moments are taken, NaN: none
    tail_floor::Float64        # the absolute tolerance of the tail density, relative to the density
    order::Int                 # 1: backward Euler, 2: BDF2
end
function StepController(;
        initial_step::Union{Nothing, Real} = nothing, relative_tolerance::Real = 1e-3,
        max_growth::Real = 2.0, min_shrink::Real = 0.2, safety::Real = 0.9, gains = nothing,
        max_failures::Integer = 5, failure_factor::Real = 0.2, max_rejections::Integer = 20,
        tail_cut::Union{Nothing, Real} = 1.0, tail_floor::Real = 1e-12, order::Integer = 1)
    order in (1, 2) ||
        throw(ArgumentError("order must be 1 (backward Euler) or 2 (BDF2), not $order"))
    # variable-step BDF2 is zero-stable for a step ratio below 1 + √2
    order == 2 && max_growth >= 1 + sqrt(2) &&
        throw(ArgumentError("BDF2 needs max_growth below 1 + √2 ≈ 2.414 for its stability, not $max_growth"))
    # the elementary gains over the order of the local error, h^(order+1)
    gains === nothing && (gains = (0.7 / (order + 1), 0.4 / (order + 1)))
    initial_step === nothing || initial_step > 0 ||
        throw(ArgumentError("the initial step must be positive"))
    tail_cut === nothing || tail_cut >= 0 ||
        throw(ArgumentError("tail_cut must not be negative"))
    tail_floor >= 0 || throw(ArgumentError("tail_floor must not be negative"))
    relative_tolerance > 0 ||
        throw(ArgumentError("the relative tolerance must be positive"))
    max_growth > 1 || throw(ArgumentError("max_growth must exceed 1"))
    0 < min_shrink < 1 || throw(ArgumentError("min_shrink must lie in (0, 1)"))
    0 < safety <= 1 || throw(ArgumentError("the safety factor must lie in (0, 1]"))
    length(gains) == 2 && gains[1] > 0 && gains[2] >= 0 ||
        throw(ArgumentError("gains are two numbers, the first positive, the second not negative"))
    max_failures >= 0 || throw(ArgumentError("max_failures must not be negative"))
    0 < failure_factor < 1 || throw(ArgumentError("the failure factor must lie in (0, 1)"))
    max_rejections >= 1 || throw(ArgumentError("max_rejections must be at least 1"))
    return StepController(
        initial_step === nothing ? NaN : Float64(initial_step), Float64(relative_tolerance),
        Float64(max_growth), Float64(min_shrink), Float64(safety),
        Float64(gains[1]), Float64(gains[2]), Int(max_failures), Float64(failure_factor),
        Int(max_rejections), tail_cut === nothing ? NaN : Float64(tail_cut), Float64(tail_floor),
        Int(order))
end

# the gains of the filter for a step of the order k: those of the stepper are for its own
# order, and the steps of backward Euler that start a BDF2 run scale them to order 1
function _gains(stepper::StepController, k)
    (stepper.gain_error, stepper.gain_memory) .* ((stepper.order + 1) / (k + 1))
end

# BDF2 is of second order for the implicit terms alone: an explicit term and a term built
# from the start of the step keep the first order of their own time level
function _warn_first_order_terms(system::EquationSystem)
    first_order = !isempty(system.explicit_terms) ||
                  any(bound -> _first_order_term(bound.term), system.sparse_terms) ||
                  any(bound -> _first_order_term(bound.term), system.matrix_free_terms)
    first_order &&
        @warn "StepController(order = 2) with explicit or lagged terms: BDF2 is of second order for the " *
              "implicit terms only, the explicit and lagged ones stay of first order"
    return first_order
end
# the terms that take a part of their residual at the start of the step: lagged coefficients
# or ions of a Fokker-Planck term, the energy loss of such a term, a lagged knock-on
# primary, a field-particle term that follows the unknowns
_first_order_term(term) = lags_in_ode_form(term)

# the state of StepController: the step length, the two last accepted states before the
# current one and their steps (NaN before there is one; the second only for BDF2), the
# error of the last accepted step for the filter, the failures in a row, the tolerances per
# unknown, the buffers and the errors per unknown of the last estimate
mutable struct StepControllerState{T}
    dt::Float64
    previous_step::Float64
    previous_step2::Float64
    error_old::Float64
    failures::Int
    rejections::Int                   # in a row
    const previous::Vector{T}         # the accepted state before u_old
    const previous2::Vector{T}        # the accepted state before that (BDF2)
    const absolute::Vector{Float64}   # per unknown of the layout
    const relative::Vector{Float64}
    const half::Vector{T}             # the buffers of the doubling of the first step and the predictor
    const doubled::Vector{T}
    const reference::Vector{T}        # the reference state of the BDF2 step
    const errors::Vector{Float64}     # per unknown, NaN before the first estimate
    const tail_region::Union{Nothing, MomentumRegion}   # the region of the tail moments
    const tail_weights::Vector{Vector{NTuple{3, Matrix{Float64}}}}   # per unknown and radius: tail density, tail current and density weights, empty without
    const tail_errors::Vector{Float64}                 # per unknown, the error of the tail moments, NaN without
    const tail_floor::Float64                          # the tail floor relative to the density of the grid electrons
    const grid_density::Vector{Float64}                # per radius, the density of the electrons on all kinetic grids
    const group::Vector{Float64}      # per unknown: the tolerance relative to the sum over a group of components
end

function stepper_state(stepper::StepController, system::EquationSystem{T},
        times, t, record::AbstractDict) where {T}
    layout = system.layout
    absolute, relative, group = Float64[], Float64[], Float64[]
    for u in layout.unknowns
        tolerance = get(system.time_step_tolerances, u.name, nothing)
        # the charge states of the ion species: each relative to the density of its species,
        # so that a trace impurity is not measured against the scale of the main one
        grouped = u isa RadialUnknown && length(component_groups(u)) > 1
        push!(absolute,
            tolerance !== nothing ? tolerance.absolute :
            grouped ? 0.0 : stepper.relative_tolerance * u.scale)
        push!(relative, tolerance === nothing ? stepper.relative_tolerance :
                        tolerance.relative)
        push!(group, tolerance !== nothing ? tolerance.group :
                     grouped ? stepper.relative_tolerance : 0.0)
    end
    stepper.order == 2 && _warn_first_order_terms(system)
    n = length(layout)
    count = length(layout.unknowns)
    # the tail moments of the kinetic unknowns: the region above the cut and the weights of
    # the moments on each kinetic unknown's grid
    tail_region = isnan(stepper.tail_cut) ? nothing : MomentumRegion(stepper.tail_cut)
    tail_weights = [tail_region !== nothing && layout.kinds[id] === :kinetic ?
                    _tail_moment_weights(u.grid, tail_region) :
                    NTuple{3, Matrix{Float64}}[] for (id, u) in enumerate(layout.unknowns)]
    #! format: off
    state(dt, step, step2, error_old, failures, previous, previous2) =
        StepControllerState{T}(dt, step, step2, error_old, failures, 0, previous, previous2, absolute, relative,
                               zeros(T, n), zeros(T, n), zeros(T, n), fill(NaN, count), tail_region, tail_weights,
                               fill(NaN, count), stepper.tail_floor, zeros(radial_cell_count(layout)), group)
    #! format: on
    if get(record, "type", "") == "StepController"
        previous = haskey(record, "previous_state") ? Vector{T}(record["previous_state"]) :
                   T[]
        if length(previous) == n
            # the second state of BDF2, when the record has one
            previous2 = haskey(record, "previous_state_2") ?
                        Vector{T}(record["previous_state_2"]) : T[]
            step2 = length(previous2) == n ? Float64(record["previous_step_2"]) : NaN
            return state(Float64(record["dt"]), Float64(record["previous_step"]),
                step2, Float64(record["error_old"]),
                Int(record["failures"]), previous, length(previous2) == n ? previous2 :
                                                   zeros(T, n))
        end
        # the run resumes onto another layout (an unknown added or removed): the stored
        # states of the predictor do not fit, so the controller starts with its stored
        # step and a doubling for the first estimate
        isempty(previous) ||
            @warn "the layout of the run differs from the stored state of the step controller ($(length(previous)) " *
                  "entries against $n): the history of the predictor is dropped, the first step is a doubling"
        return state(Float64(record["dt"]), NaN, NaN, 1.0, 0, zeros(T, n), zeros(T, n))
    end
    dt = isnan(stepper.initial_step) ? _interval_length(times, t) : stepper.initial_step
    return state(dt, NaN, NaN, 1.0, 0, zeros(T, n), zeros(T, n))
end

function stepper_record(state::StepControllerState)
    record = SettingsRecord("type" => "StepController", "dt" => state.dt,
        "previous_step" => state.previous_step,
        "error_old" => state.error_old, "failures" => state.failures)
    isnan(state.previous_step) ||
        (record["previous_state"] = collect(Float64, state.previous))
    if !isnan(state.previous_step2)
        record["previous_step_2"] = state.previous_step2
        record["previous_state_2"] = collect(Float64, state.previous2)
    end
    return record
end

function ceilings(::StepController, times, save_times)
    sort!(unique!(collect(Float64, vcat(times, save_times))))
end

stepper_errors(state::StepControllerState) = state.errors

# after an update that changed the setup the extrapolation through the states before it
# would hold the jump of the setup, a term of first order in the step: the next step takes
# its estimate from a doubling, as the first step of a run
function reset_history!(state::StepControllerState)
    (state.previous_step = NaN; state.previous_step2 = NaN; nothing)
end

# the error per unknown of the local error estimate `factor (a - b)`, the root mean
# square over the entries of the unknown of the estimate over its tolerance at the
# state `u`, and the largest of them; an unknown with zero tolerances is left out (NaN).
# A kinetic unknown with the tail moments takes the larger of that and the root mean
# square over its tail moments (`_tail_moment_error!`).
function _controller_errors!(state::StepControllerState, layout, u, a, b, factor)
    largest = 0.0
    _grid_density!(state, layout, u)
    for id in eachindex(layout.unknowns)
        absolute, relative, group = state.absolute[id], state.relative[id], state.group[id]
        state.tail_errors[id] = NaN
        if absolute == 0 && relative == 0 && group == 0
            state.errors[id] = NaN
            continue
        end
        sum_squares, count = 0.0, 0
        groups = iszero(group) ? nothing : component_groups(layout.unknowns[id])
        for range in _unknown_ranges(layout, id)
            if groups === nothing
                for i in range
                    scale = absolute + relative * abs(u[i])
                    scale == 0 && continue
                    sum_squares += (factor * (a[i] - b[i]) / scale)^2
                    count += 1
                end
            else
                for g in groups
                    entries = range[g]
                    total = group * _magnitude(u, entries)
                    for i in entries
                        scale = absolute + relative * abs(u[i]) + total
                        scale == 0 && continue
                        sum_squares += (factor * (a[i] - b[i]) / scale)^2
                        count += 1
                    end
                end
            end
        end
        error = count == 0 ? 0.0 : sqrt(sum_squares / count)
        if !isempty(state.tail_weights[id])
            tail = _tail_moment_error!(state, layout, id, u, a, b, factor, relative)
            state.tail_errors[id] = tail
            error = max(error, tail)
        end
        isfinite(error) || (error = Inf)
        state.errors[id] = error
        largest = max(largest, error)
    end
    return largest
end

# the error of the tail moments of the kinetic unknown `id`: the density and
# the current density above the cut, at every radius, of the estimate `factor (a - b)` over
# `absolute + relative |moment(u)|`, as one root mean square over the 2 nr entries. The
# absolute tolerance of the density is `tail_floor` times the density of the electrons on
# all kinetic grids at the radius (`_grid_density!`), that of the current e c times it.
# The cells of the tail lie far below the absolute floor of the cell measure, so
# the moments are what controls the tail.
function _tail_moment_error!(
        state::StepControllerState, layout, id, u, a, b, factor, relative)
    sum_squares, count = 0.0, 0
    for (ir, (density_weights, current_weights, _)) in enumerate(state.tail_weights[id])
        f, f_a, f_b = kinetic_view(u, layout, id, ir), kinetic_view(a, layout, id, ir),
        kinetic_view(b, layout, id, ir)
        density_u, current_u, density_e, current_e = 0.0, 0.0, 0.0, 0.0
        @inbounds for k in eachindex(f, f_a, f_b, density_weights, current_weights)
            estimate = factor * (f_a[k] - f_b[k])
            density_u += density_weights[k] * f[k]
            current_u += current_weights[k] * f[k]
            density_e += density_weights[k] * estimate
            current_e += current_weights[k] * estimate
        end
        floor_density = state.tail_floor * state.grid_density[ir]
        floor_current = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT *
                        floor_density
        #! format: off
        for (e, m, floor) in ((density_e, density_u, floor_density), (current_e, current_u, floor_current))
            scale = floor + relative * abs(m)
            scale == 0 && continue
            sum_squares += (e / scale)^2
            count += 1
        end
        #! format: on
    end
    return count == 0 ? 0.0 : sqrt(sum_squares / count)
end

# the density of the electrons on all kinetic grids at every radius, the reference of the
# tail floor: on a two-grid layout the runaway grid lies above the cut as a whole, and a
# floor relative to its own density would vanish with it
function _grid_density!(state::StepControllerState, layout, u)
    fill!(state.grid_density, 0.0)
    for (id, weights) in enumerate(state.tail_weights),
        (ir, (_, _, all_weights)) in enumerate(weights)

        f = kinetic_view(u, layout, id, ir)
        density = 0.0
        @inbounds for k in eachindex(f, all_weights)
            density += all_weights[k] * f[k]
        end
        state.grid_density[ir] += density
    end
    for ir in eachindex(state.grid_density)
        state.grid_density[ir] = abs(state.grid_density[ir])
    end
    return state.grid_density
end

# the weights of the density and the current density above the cut of the region, and of the
# density over all momenta, per radius on the momentum grid of that radius: the moments of
# `density` and `current_density` as sums of weight times f, for grids whose cell counts
# differ between the radii
function _tail_moment_weights(grid::KineticGrid, region::MomentumRegion)
    weights = NTuple{3, Matrix{Float64}}[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = momentum_envelope(region, grid, ir)
        pitch = pitch_moment_weights(m)
        volume = grid.radial_grid.spatial_jacobian[ir]
        density_weights = [envelope[i] * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] /
                           volume
                           for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)]
        current_weights = [Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT *
                           envelope[i] * jacobian[i, j] *
                           m.p_widths[i] * m.p_centers[i] / sqrt(1 + m.p_centers[i]^2) *
                           pitch[j] / volume
                           for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)]
        all_weights = [jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] / volume
                       for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)]
        push!(weights, (density_weights, current_weights, all_weights))
    end
    return weights
end

# a failed solve of the controller: the step length shrinks by the failure factor from the
# step tried, `step`, which a ceiling may have cut below the planned length
function _controller_failure!(stepper::StepController, state::StepControllerState,
        run, options, t_next, h, result;
        step = h)
    state.failures += 1
    state.failures > stepper.max_failures &&
        _give_up(run, options, t_next, h, result,
            "after $(stepper.max_failures) failures in a row")
    state.dt = stepper.failure_factor * min(state.dt, step)
    return _retry_after_failure(run, options, t_next, h, result,
        "is retried with $(stepper.failure_factor) times the time step")
end

# the solve of the step from u_old to u over h with its error estimate and the order of
# the step: backward Euler with the estimate from the predictor through the last accepted
# state, or from a doubling when there is none; BDF2 (order 2) once two accepted states
# precede u_old. `nothing` after a failed solve, which the controller has handled.
function _controller_estimate!(
        stepper::StepController, state::StepControllerState, run, options, t_next, h)
    u_old, u = run.u_old, run.u
    if isnan(state.previous_step)
        t = run.t
        t_half = t + h / 2
        result = _solve_step!(run, u_old, state.half, t_half, t_half - t; kind = :half_step)
        result.status === :converged ||
            return _controller_failure!(
                stepper, state, run, options, t_half, t_half - t, result; step = h)
        result = _solve_step!(
            run, state.half, state.doubled, t_next, t_next - t_half; kind = :half_step)
        result.status === :converged ||
            return _controller_failure!(
                stepper, state, run, options, t_next, t_next - t_half, result; step = h)
    end
    bdf2 = stepper.order == 2 && !isnan(state.previous_step2)
    copyto!(u, u_old)
    started = time_ns()
    if bdf2
        # variable-step BDF2, u - a u_n + b u_{n-1} = c h f(u), as a backward-Euler residual
        # with the reference a u_n - b u_{n-1} and the step c h (step_ratio = h / h_{n-1})
        step_ratio = h / state.previous_step
        a = (1 + step_ratio)^2 / (1 + 2 * step_ratio)
        b = step_ratio^2 / (1 + 2 * step_ratio)
        c = (1 + step_ratio) / (1 + 2 * step_ratio)
        @. state.reference = a * u_old - b * state.previous
        result = newton_solve!(u, u_old, run.ws, t_next, c * h; reference = state.reference)
    else
        result = newton_solve!(u, u_old, run.ws, t_next, h)
    end
    wall = (time_ns() - started) / 1e9
    if result.status !== :converged
        _record!(run.stats, t_next, h, result, run.ws, wall)
        return _controller_failure!(stepper, state, run, options, t_next, h, result)
    end
    if isnan(state.previous_step)
        # the doubling difference is half the local error of the full step
        error = _controller_errors!(state, run.layout, u, u, state.doubled, 2.0)
    elseif !bdf2
        # the extrapolation through the two last backward-Euler states is the forward-Euler
        # predictor with the derivative of the last step; the implicit solution differs
        # from it by h² u'', twice the local error
        step_ratio = h / state.previous_step
        @. state.half = u_old + step_ratio * (u_old - state.previous)
        error = _controller_errors!(state, run.layout, u, u, state.half, 0.5)
    else
        # Milne's device: the quadratic extrapolation P through the last three accepted
        # states misses by h (h + h1)(h + h1 + h2) u'''/6, BDF2 by h² (h + h1)² u'''/(6 (2h + h1)),
        # so the local error is error_factor (u - P), with error_factor = 2/11 at
        # constant steps
        h1, h2 = state.previous_step, state.previous_step2
        l0 = (h + h1) * (h + h1 + h2) / (h1 * (h1 + h2))
        l1 = -h * (h + h1 + h2) / (h1 * h2)
        l2 = h * (h + h1) / (h2 * (h1 + h2))
        @. state.half = l0 * u_old + l1 * state.previous + l2 * state.previous2
        error_factor = h * (h + h1) / ((h + h1 + h2) * (2h + h1) + h * (h + h1))
        error = _controller_errors!(state, run.layout, u, u, state.half, error_factor)
    end
    return (error, result, wall, bdf2 ? 2 : 1)
end

function advance!(
        stepper::StepController, state::StepControllerState, run, options, ceiling)
    t, u_old, u = run.t, run.u_old, run.u
    t_next = _step_end(t, state.dt, ceiling)
    h = t_next - t
    estimate = _controller_estimate!(stepper, state, run, options, t_next, h)
    estimate === nothing && return nothing
    error, result, wall, order = estimate
    accepted = error <= 1
    _record!(run.stats, t_next, h, result, run.ws, wall; accepted, error)
    state.failures = 0
    if !accepted
        state.rejections += 1
        state.rejections > stepper.max_rejections &&
            _give_up(run,
                options,
                t_next,
                h,
                NewtonResult(:rejected, result.iterations, result.initial_residual_norm,
                    result.residual_norm, result.update_norm,
                    result.convergence_order, result.smallest_step_length,
                    result.krylov_iterations, result.preconditioner_updates,
                    false, false, 0.0, 0.0, 0.0, 0.0),
                "with the error $error of its tolerance, after $(stepper.max_rejections) rejections in a row")
        # the elementary rule for a local error of the order of the step, from the
        # measured step; the filter keeps its memory
        state.dt = h * max(stepper.min_shrink, stepper.safety * error^(-1 / (order + 1)))
        log = options.log
        log_event!(log, "step rejected", Logging.Info,
            "the step to t = $t_next s with the time step $h s has the error $error of its tolerance " *
            "and is retried with the time step $(state.dt) s")
        report_step!(log, u_old, run.layout, t, run.times, run.next_point - 2, run.stats;
            errors = stepper_errors(state))
        return nothing
    end
    state.rejections = 0
    # the PI filter on the measured step, its growth capped; a zero error grows by the cap
    gain_error, gain_memory = _gains(stepper, order)
    factor = error == 0 ? stepper.max_growth :
             stepper.safety * error^(-gain_error) * state.error_old^gain_memory
    state.dt = h * clamp(factor, stepper.min_shrink, stepper.max_growth)
    # the memory of the filter floored as in Hairer's codes: a step with an error near
    # zero would otherwise shrink the next one by the limit
    state.error_old = max(error, 1e-4)
    # the history of the predictor: the second state only for BDF2
    if stepper.order == 2 && !isnan(state.previous_step)
        copyto!(state.previous2, state.previous)
        state.previous_step2 = state.previous_step
    end
    copyto!(state.previous, u_old)
    state.previous_step = h
    return (t_next, h)
end

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
Base.@kwdef struct RunOptions{
    S <: NewtonSolver, P <: TimeStepper, U, C <: Union{Nothing, CheckpointSettings}, R,
    X <: Union{Nothing, StopCondition}, L <: Union{Nothing, ProgressLog}, D}
    solver::S = NewtonSolver()
    stepper::P = FixedSteps()
    save_times::Union{Nothing, AbstractVector} = nothing
    updates::U = ()
    checkpoint::C = nothing
    resume::R = false
    stop::X = nothing
    log::L = nothing
    description::D = Dict{String, Any}()
end

# the record of a run

"Sections of the settings that define the setup of a run, covered by `setup_hash`."
const SETUP_SECTIONS = (
    "grids", "layout", "system", "updates", "prescribed", "solver", "time", "run")

function _has_undescribed(record::AbstractDict)
    get(record, "described", true) === false || any(_has_undescribed, values(record))
end
_has_undescribed(value) = false

function _describe_time(times, save_times, stepper)
    record = SettingsRecord(
        "time_count" => length(times), "save_count" => length(save_times),
        "stepper" => describe(stepper))
    isempty(times) || (record["times"] = collect(Float64, times))
    isempty(save_times) || (record["save_times"] = collect(Float64, save_times))
    return record
end

"""
    run_settings(system; solver=NewtonSolver(), stepper=FixedSteps(), times=Float64[],
                 save_times=times, description=Dict(), updates=(), checkpoint=nothing,
                 stop=nothing, log=nothing)
    run_settings(system, times, options::RunOptions)

The record of a run, written by [`write_output`](@ref) as the `settings` group. Sections:

- `grids`: the kinetic grids by name, with their faces;
- `layout`: the unknowns, their order, sizes, scales and lower limits;
- `system`: the equations with every term and its options, the tolerances;
- `updates`: the interval updates by name ([`IntervalUpdate`](@ref));
- `prescribed`: the tables of the prescribed functions of time, sampled at the time
  points ([`describe_prescribed_plasma`](@ref)), which the terms refer to by name. They
  are a section of their own so that the hash of `system` does not change with the time
  points;
- `solver`: the Newton and linear solver settings;
- `time`: the time points, the save times and the stepper ([`TimeStepper`](@ref);
  `max_step_halvings = n` stands for `stepper = FixedSteps(max_halvings=n)`);
- `run`: `description`, what the caller adds (a name, the origin of the initial state,
  the parameters a hand-built coefficient came from), as plain values;
- `code`: [`code_version`](@ref), with the commit of Runes2P;
- `environment`: [`run_environment`](@ref);
- `checkpoint`: the checkpoint settings, or the type `none`;
- `stop`: the stop condition, or the type `none`;
- `log`: the settings of the progress log, or the type `none`;
- `hashes`: [`settings_hash`](@ref) of each of the first eight sections, and
  `setup_hash` of the eight together. `code`, `environment`, `checkpoint`, `stop` and
  `log` are outside the hashes, so the same setup has the same hash on another machine
  and commit;
- `complete`: false when some object of the run has no record of its own (a term defined
  outside the package, a prescribed function that fails at a time point).

[`integrate`](@ref) builds the record at its start and returns it in the solution, with
the record of the segment added under `segment`: its number, its start time, the number
of states saved before it, its status (`running`, `stopped` or `finished`) with the time
reached, the intervals done and the reason of a stop, and for a resumed segment the file
it resumed from, the commit and setup hash of the run that wrote it and the sections whose
hash changed.
"""
function run_settings(system::EquationSystem; solver::NewtonSolver = NewtonSolver(),
        times::AbstractVector = Float64[], save_times::AbstractVector = times,
        stepper = nothing, max_step_halvings::Union{Nothing, Integer} = nothing,
        description = SettingsRecord(),
        updates = (), checkpoint = nothing, stop = nothing, log = nothing,
        max_table_bytes::Integer = MAX_PRESCRIBED_TABLE_BYTES)
    if max_step_halvings !== nothing
        stepper === nothing ||
            throw(ArgumentError("give max_step_halvings or stepper, not both"))
        stepper = FixedSteps(; max_halvings = max_step_halvings)
    end
    stepper === nothing && (stepper = FixedSteps())
    context = DescriptionContext(; times, max_table_bytes)
    record = SettingsRecord()
    record["layout"] = describe(system.layout, context)
    record["system"] = describe(system, context)
    #! format: off
    record["updates"] = SettingsRecord(update_name(update) => describe(update, context)
                                       for update in _checked_updates(updates))
    #! format: on
    # after the layout, the terms and the updates have named their grids and tables
    record["grids"] = describe(context)
    record["prescribed"] = describe_tables(context)
    record["solver"] = describe(solver)
    record["time"] = _describe_time(times, save_times, stepper)
    record["run"] = _plain(description)
    record["complete"] = !_has_undescribed(record)
    #! format: off
    record["hashes"] = SettingsRecord(section => settings_hash(record[section]) for section in SETUP_SECTIONS)
    record["setup_hash"] = settings_hash(SettingsRecord(section => record[section] for section in SETUP_SECTIONS))
    #! format: on
    record["code"] = code_version()
    record["environment"] = run_environment()
    record["checkpoint"] = checkpoint === nothing ? SettingsRecord("type" => "none") :
                           describe(checkpoint)
    record["stop"] = stop === nothing ? SettingsRecord("type" => "none") : describe(stop)
    record["log"] = log === nothing ? SettingsRecord("type" => "none") : describe(log)
    return record
end

"""
    read_settings(filename)

The `settings` group of an output file as a plain record, equal to the record written
([`run_settings`](@ref)); empty for a file without the group. Defined by the package
extension `Runes2PHDF5Ext`, which loads with HDF5 (`using HDF5`).
"""
function read_settings end

# the record of a run with these options
function run_settings(system::EquationSystem, times::AbstractVector, options::RunOptions)
    run_settings(system; options.solver, options.stepper, times,
        save_times = options.save_times === nothing ? times : options.save_times, options.description,
        options.updates, options.checkpoint, options.stop, options.log)
end

# the part of a run's saved states a segment holds: columns 1:count of the arrays
function _segment_solution(layout, save_times, first_save, count, saved,
        stats, settings, updates, saved_updates)
    T = eltype(saved)
    #! format: off
    update_states = Dict{String,Matrix{Float64}}(update_name(update) => saved_updates[k][:, 1:count]
                                                 for (k, update) in enumerate(updates))
    #! format: on
    return TimeSolution(
        layout, collect(T, save_times[first_save:(first_save + count - 1)]),
        saved[:, 1:count], stats,
        settings, update_states)
end

"""
    integrate(system, u0, times; solver=NewtonSolver(), stepper=FixedSteps(), save_times=times,
              description=Dict(), updates=(), checkpoint=nothing, resume=false, stop=nothing,
              log=nothing)
    integrate(system, u0, times, options::RunOptions)

Backward-Euler integration of `system` from the state `u0` at `times[1]` through the time
points `times` (any increasing array, e.g. nonuniform or matching the restarts of another
run). The `stepper` ([`TimeStepper`](@ref)) sets the length of every step, and every time
point is the end of a step: with [`FixedSteps`](@ref), the default, each interval between
two time points is one step, retried with half the time step when it fails
(`max_step_halvings = n` is a shorthand for `stepper = FixedSteps(max_halvings=n)`). Each
step is solved by [`newton_solve!`](@ref) with explicit terms at the previous time level. A
step that still fails throws [`TimeStepFailure`](@ref).

States are saved at `save_times` (strictly increasing, inside `[times[1], times[end]]`; a
save time within a relative 1e-12 of a time point is taken as that time point, and two save
times taken as one time point are refused), and every saved state is a computed one: with
[`FixedSteps`](@ref) and [`StepDoubling`](@ref) a save time must be a time point (an
`ArgumentError` names one that is not; DREAM saves computed steps only), while
[`StepController`](@ref) ends a step on every save time. [`time_points`](@ref) builds
`times` and `save_times` from [`Leg`](@ref)s in one arithmetic. Returns a
[`TimeSolution`](@ref).

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
state does not have keeps its value of `u0`, or takes it from its equation (below). The
updates are set back to their stored
states. The settings may differ from those of the run that stored the state (tolerances,
the solver, later time points, terms); the record names the sections that changed
(`segment` in the settings) and a warning lists them.

An unknown whose equation has `initial_value = :from_equation` (the energy of the cold
electrons, [`thermal_energy_equation`](@ref)) is set from its row at the start of the run,
whatever `u0` holds there ([`fill_initial_values!`](@ref)); a resumed run sets those that
the stored state does not have. The record of the segment names them
(`initial_values_from_equations`).

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
        max_step_halvings::Union{Nothing, Integer} = nothing, options...)
    if max_step_halvings !== nothing
        haskey(options, :stepper) &&
            throw(ArgumentError("give max_step_halvings or stepper, not both"))
        return integrate(system,
            u0,
            times,
            RunOptions(; stepper = FixedSteps(; max_halvings = max_step_halvings), options...))
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
    length(u0) == length(system.layout) ||
        throw(DimensionMismatch("u0 does not match the layout"))
    isempty(times) && throw(ArgumentError("times is empty"))
    all(k -> times[k + 1] > times[k], 1:(length(times) - 1)) ||
        throw(ArgumentError("times must be strictly increasing"))
    # strictly, after the snap to the time points: a repeated save time would be saved twice,
    # and the reader of a segmented run refuses repeated times
    all(k -> save_times[k + 1] > save_times[k], 1:(length(save_times) - 1)) ||
        throw(ArgumentError("save_times must be strictly increasing (after snapping to the time points within a relative 1e-12)"))
    all(s -> first(times) <= s <= last(times), save_times) ||
        throw(ArgumentError("save_times must lie inside [times[1], times[end]]"))
    # every saved state is a computed one: a save time is an end of a step
    step_ends = ceilings(options.stepper, times, save_times)
    for s in save_times
        s in step_ends ||
            throw(ArgumentError("the save time $s is not a time point of the run: with " *
                                "$(nameof(typeof(options.stepper))) every saved state is a computed one, " *
                                "so a save time is a time point (time_points(legs) builds matching " *
                                "times and save_times), and only StepController takes save times between them"))
    end
    updates = _checked_updates(options.updates)
    foreach(update -> check_update(update, system), updates)
    for update in updates, t in forced_time_points(update)

        first(times) < t < last(times) &&
            !any(point -> isapprox(point, t; rtol = 1e-12), times) &&
            throw(ArgumentError("the update $(update_name(update)) changes at t = $t, which is not a time " *
                                "point of the run"))
    end
    # a term whose input switches in time (the level of a heat transport)
    for equation in system.equations, term in equation.terms, t in forced_time_points(term)
        first(times) < t < last(times) &&
            !any(point -> isapprox(point, t; rtol = 1e-12), times) &&
            throw(ArgumentError("a $(nameof(typeof(term))) in the equation of $(equation.unknown) changes at " *
                                "t = $t, which is not a time point of the run"))
    end
    options.checkpoint === nothing || _require_hdf5(write_checkpoint, "writing checkpoints")
    return updates
end

# the start of a segment: the time and the states it starts from, restored from the
# stored state when there is one, and the record of the segment
function _start_segment!(
        u_old, layout, times, save_times, updates, settings, stored, options::RunOptions)
    T = eltype(u_old)
    t = T(first(times))
    first_save, segment = 1, 1
    summary_before, stepper_before = SettingsRecord(), SettingsRecord()
    copied = Symbol[]
    if stored !== nothing
        first(times) <= stored.time <= last(times) ||
            throw(ArgumentError("the stored time $(stored.time) of $(stored.source) lies outside the time points"))
        isempty(stored.stepper) && stored.time ∉ times &&
            throw(ArgumentError("the stored time $(stored.time) of $(stored.source) is not one of the time points, " *
                                "which a state without the record of the stepper must be"))
        t = T(stored.time)
        copied = copy_unknowns!(u_old, layout, stored.state, stored.layout)
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
    settings["segment"] = _segment_record(
        segment, stored, settings, t, stored === nothing ? 0 : stored.saved_count)
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
        # Segment files of the name with higher numbers than this segment belong to an
        # earlier run under the same name, or to a continuation of a segment before this
        # one that this run replaces, and read_output would join them to its output. The
        # run deletes no file: it refuses to start while they exist.
        stale = filter(f -> segment_number(f) > segment, segment_files(checkpoint))
        isempty(stale) ||
            throw(ArgumentError("the segment files $(join(stale, ", ")) of the name $(checkpoint.name) follow " *
                                "segment $segment of this run, and read_output would join them to it: remove " *
                                "them or choose another name"))
    end
    if stored !== nothing && settings["segment"]["settings_changed"]
        @warn "resuming $(stored.source) with changed settings" sections = settings["segment"]["changed_sections"]
    end
    return (; t, first_save, segment, summary_before, stepper_before, copied)
end

# The state of a running segment: the states and the times, what it has saved, its
# statistics and its record, and the wall-time bookkeeping of the checkpoints and the
# stop. The loop of `_run_steps!` and the helpers below act on it.
mutable struct RunState{T, L <: UnknownLayout, U, W <: NewtonWorkspace, S, P}
    const layout::L
    const times::Vector{T}
    const save_times::Vector{T}
    const first_save::Int
    const updates::U
    const saved::Matrix{T}
    const saved_updates::Vector{Matrix{Float64}}
    const stats::SolverStatistics
    const settings::Dict{String, Any}
    const segment::Int
    const summary_before::Dict{String, Any}
    const u_old::Vector{T}
    const u::Vector{T}
    const ws::W
    const stepper::S
    const stepper_state::P
    const ceilings::Vector{T}   # the times every step ends on: the time points, and the save times of an adaptive stepper
    t::T
    next_save::Int
    next_point::Int       # index of the first time point above t
    next_ceiling::Int     # index of the first ceiling above t
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
    if changed
        invalidate_jacobian!(run.ws)
        reset_history!(run.stepper_state)
    end
    return nothing
end

function _segment_solution(run::RunState)
    _segment_solution(run.layout, run.save_times, run.first_save,
        run.next_save - run.first_save, run.saved,
        run.stats, run.settings, run.updates, run.saved_updates)
end

function _update_segment!(run::RunState, status; reason = nothing, checkpoint_written = false)
    record = run.settings["segment"]
    record["status"] = status
    record["end_time"] = Float64(run.t)
    record["steps"] = run.steps
    reason === nothing || (record["stop_reason"] = reason)
    checkpoint_written && (record["checkpoint_written"] = true)
    return nothing
end

function _write_checkpoint!(
        run::RunState, checkpoint::CheckpointSettings, log, status, reason)
    started_write = time()
    _update_segment!(run, status; reason, checkpoint_written = true)
    # the output first: a checkpoint then never names states its segment file lacks
    write_output(segment_file(checkpoint, run.segment), _segment_solution(run); atomic = true)
    #! format: off
    write_checkpoint(checkpoint,
                     StoredState(Float64(run.t), copy(run.u_old), run.layout,
                                 Dict{String,Vector{Float64}}(update_name(update) => update_state(update)
                                                              for update in run.updates),
                                 run.settings, run.segment, run.next_save - 1,
                                 _run_summary(run.summary_before, run.stats, time() - run.started_run),
                                 stepper_record(run.stepper_state), checkpoint_file(checkpoint)))
    #! format: on
    run.last_checkpoint = time()
    run.longest_write = max(run.longest_write, run.last_checkpoint - started_write)
    log_event!(log, "checkpoint", Logging.Info,
        "written at t = $(run.t) s in $(round(run.last_checkpoint - started_write; digits=2)) s: " *
        checkpoint_file(checkpoint))
    return nothing
end

# A save time within rounding of a time point (relative 1e-12) is that time point, so that
# times and save times built by separate arithmetic give no step of the rounding length
# between them
function _snap_save_times(save_times::AbstractVector, times::AbstractVector)
    snapped = collect(promote_type(eltype(save_times), eltype(times)), save_times)
    for (k, s) in enumerate(snapped)
        i = searchsortedfirst(times, s)
        for j in (i - 1, i)
            1 <= j <= length(times) && isapprox(times[j], s; rtol = 1e-12) &&
                (snapped[k] = times[j])
        end
    end
    return snapped
end

function _integrate(system::EquationSystem{T}, u0::AbstractVector, times::AbstractVector,
        options::RunOptions) where {T}
    layout = system.layout
    save_times = options.save_times === nothing ? times :
                 _snap_save_times(options.save_times, times)
    updates = _check_run(system, u0, times, save_times, options)
    settings = run_settings(system, times, options)
    stored = _resume_state(options.resume, options.checkpoint)
    u_old = Vector{T}(u0)
    start = _start_segment!(
        u_old, layout, times, save_times, updates, settings, stored, options)
    # the unknowns that take their initial value from their equation: all of
    # them at the start of a run, those the stored layout lacks on a resume
    added = stored === nothing ? nothing :
            setdiff((unknown.name for unknown in layout.unknowns), start.copied)
    filled = fill_initial_values!(u_old, system, start.t; unknowns = added)
    isempty(filled) ||
        (settings["segment"]["initial_values_from_equations"] = String.(filled))
    stop, log, checkpoint = options.stop, options.log, options.checkpoint

    # a stop file of an earlier segment is not a request to this one
    stop === nothing || isempty(stop.stop_file) || rm(stop.stop_file; force = true)
    save_count = length(save_times) - start.first_save + 1
    state = stepper_state(options.stepper, system, times, start.t, start.stepper_before)
    step_ceilings = collect(T, ceilings(options.stepper, times, save_times))
    saved_updates = Matrix{Float64}[Matrix{Float64}(undef, length(update_state(update)), save_count)
                                    for update in updates]
    started_run = time()
    run = RunState(
        layout, collect(T, times), collect(T, save_times), start.first_save, updates,
        Matrix{T}(undef, length(layout), save_count), saved_updates,
        SolverStatistics(), settings, start.segment, start.summary_before, u_old, copy(u_old),
        NewtonWorkspace(system, options.solver), options.stepper, state, step_ceilings,
        start.t, start.first_save, searchsortedlast(times, start.t) + 1,
        searchsortedlast(step_ceilings, start.t) + 1, 0, started_run, started_run, 0.0, 0.0)
    _apply_updates!(run)
    # the state at the first time point is saved by a run that starts there
    while stored === nothing && run.next_save <= length(save_times) &&
          save_times[run.next_save] == first(times)
        _save!(run, u_old)
    end
    start_log!(log, settings, u_old, layout, run.t, times, run.next_point - 2, run.stats)
    return _run_steps!(run, options)
end

# the entry of the log at the end of the segment
function _finish(run::RunState, log, reason)
    write_entry(
        log, reason, run.u_old, run.layout, run.t, run.times, run.next_point - 2, run.stats;
        errors = stepper_errors(run.stepper_state))
end

function _run_steps!(run::RunState{T}, options::RunOptions) where {T}
    times, layout, stats = run.times, run.layout, run.stats
    u_old, u = run.u_old, run.u
    stop, log, checkpoint = options.stop, options.log, options.checkpoint
    # the wall time of a step includes its failed and rejected attempts
    started = time_ns()
    while run.t < last(times)
        t = run.t
        advanced = advance!(
            run.stepper, run.stepper_state, run, options, run.ceilings[run.next_ceiling])
        advanced === nothing && continue      # a failed or rejected attempt, retried by the stepper
        t_next, h = advanced
        last_record = length(stats.time)      # the accepted step is the last one recorded
        if log !== nothing
            stats.accepted_at_residual_floor[last_record] &&
                !stats.accepted_as_linear[last_record] &&
                log_event!(log, "residual floor", Logging.Info,
                    "the step to t = $t_next s was accepted with the residual " *
                    "$(stats.residual_norm[last_record]) above the tolerance, where it had stopped decreasing")
            negative = _negative_cells(u, layout)
            negative > 0 && log_event!(log, "negative values", Logging.Warn,
                "$negative negative cells of the kinetic unknowns after the step to " *
                "t = $t_next s")
        end
        # every save time is an end of a step (`_check_run`), so the saved state is the computed one
        while run.next_save <= length(run.save_times) &&
              run.save_times[run.next_save] <= t_next
            run.save_times[run.next_save] == t_next ||
                error("the save time $(run.save_times[run.next_save]) lies inside the step to $t_next")
            _save!(run, u)
        end
        copyto!(u_old, u)
        run.t = t_next
        run.steps += 1
        t_next == run.ceilings[run.next_ceiling] && (run.next_ceiling += 1)
        t_next == times[run.next_point] && (run.next_point += 1)
        run.longest_step = max(run.longest_step, (time_ns() - started) / 1e9)

        last_step = run.t >= last(times)
        # the last time point is the end of the run, not a stop
        reason = last_step ? nothing :
                 stop_reason(stop, run.longest_step, run.longest_write)
        status = last_step ? "finished" : reason === nothing ? "running" : "stopped"
        if checkpoint !== nothing &&
           (last_step || reason !== nothing ||
            time() - run.last_checkpoint >= checkpoint.interval)
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
        report_progress!(log, u_old, layout, run.t, times, run.next_point - 2, stats;
            errors = stepper_errors(run.stepper_state))
        started = time_ns()
    end
    _update_segment!(run, "finished")
    _finish(run, log, "segment end, finished")
    return _segment_solution(run)
end

"""
    write_output(filename, solution::TimeSolution; atomic=false)

Write a [`TimeSolution`](@ref) to an HDF5 file: the root attributes `format_version`
([`OUTPUT_FORMAT_VERSION`](@ref)), `runes2p_commit`, `runes2p_version` and
`runes2p_tree_clean`, the record of the run under `settings` ([`run_settings`](@ref), read
back by [`read_settings`](@ref)), the [`describe`](@ref) record of the layout under
`layout`, the saved `time`, one dataset per unknown under `unknowns` (scalar `(t)`, radial
`(components, r, t)`, kinetic on a shared grid `(p, xi, r, t)`, on per-radius grids a group
with one `(p, xi, t)` dataset per radius named `radius_<ir>`), and the
[`SolverStatistics`](@ref) under `solver_statistics` with their own time array, and the
states of the interval updates per saved time under `update_states`. `atomic = true` writes
to a temporary file that is renamed onto `filename` ([`write_atomically`](@ref)).
[`read_output`](@ref) reads the file back. Defined by the package extension
`Runes2PHDF5Ext`, which loads with HDF5 (`using HDF5`).
"""
function write_output end
