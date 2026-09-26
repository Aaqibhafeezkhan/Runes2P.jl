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
column per saved time; views of the unknowns through [`unknown_view`](@ref)) and the
[`SolverStatistics`](@ref).
"""
struct TimeSolution{T,L<:UnknownLayout}
    layout::L
    times::Vector{T}
    states::Matrix{T}
    statistics::SolverStatistics
end

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

"""
    integrate(system, u0, times; solver=NewtonSolver(), save_times=times, max_step_halvings=10)

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
"""
function integrate(system::EquationSystem{T}, u0::AbstractVector, times::AbstractVector;
                   solver::NewtonSolver=NewtonSolver(), save_times::AbstractVector=times,
                   max_step_halvings::Integer=10) where {T}
    layout = system.layout
    n = length(layout)
    length(u0) == n || throw(DimensionMismatch("u0 does not match the layout"))
    isempty(times) && throw(ArgumentError("times is empty"))
    all(k -> times[k+1] > times[k], 1:length(times)-1) ||
        throw(ArgumentError("times must be strictly increasing"))
    issorted(save_times) || throw(ArgumentError("save_times must be increasing"))
    all(s -> first(times) <= s <= last(times), save_times) ||
        throw(ArgumentError("save_times must lie inside [times[1], times[end]]"))

    ws = NewtonWorkspace(system, solver)
    stats = SolverStatistics()
    saved = Matrix{T}(undef, n, length(save_times))
    u_old = Vector{T}(u0)
    u = copy(u_old)
    next_save = 1
    while next_save <= length(save_times) && save_times[next_save] == first(times)
        saved[:, next_save] .= u_old
        next_save += 1
    end

    for interval in 1:length(times)-1
        t, t_end = T(times[interval]), T(times[interval+1])
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
                    saved[:, next_save] .= u
                else
                    fraction = (s - t) / h
                    @views saved[:, next_save] .= (1 - fraction) .* u_old .+ fraction .* u
                end
                next_save += 1
            end
            copyto!(u_old, u)
            t = t_next
        end
    end
    return TimeSolution(layout, collect(T, save_times), saved, stats)
end

"""
    write_output(filename, solution::TimeSolution)

Write a [`TimeSolution`](@ref) to an HDF5 file: `format_version`, the
[`describe`](@ref) record of the layout under `layout`, the saved `time`, one dataset per
unknown under `unknowns` (scalar `(t)`, radial `(components, r, t)`, kinetic on a shared
grid `(p, xi, r, t)`, on per-radius grids a group with one `(p, xi, t)` dataset per radius
named `radius_<ir>`), and the [`SolverStatistics`](@ref) under `solver_statistics` with
their own time array. Defined by the package extension `Runes2PHDF5Ext`, which loads with
HDF5 (`using HDF5`).
"""
function write_output end
