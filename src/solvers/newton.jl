# Newton iterations for one backward-Euler step: solve
# F(u) = M (u - u_old)/dt - rhs(u, t) = 0 with the Jacobian M/dt - ∂rhs/∂u, a lower-limit
# (positivity) step length, an optional line search, and convergence on both the update
# and the residual. A failure is reported as a status, which the time stepper handles.

"""
    NewtonSolver(; linear_solver=SparseLU(), max_iterations=30, max_reduction=0.9,
                 residual_tolerance=1e-8, floor_linear_tolerance=1e-8, line_search=false,
                 sufficient_decrease=1e-4, max_backtracks=20, linear_acceptance=true,
                 reuse_linear_matrix=true)

Settings of the Newton iteration of a backward-Euler step.

- Convergence needs both a small update and a small residual (DREAM #262): every entry
  of the full Newton update `δ` satisfies `|δ| ≤ absolute + relative |u|` with the
  unknown's Newton [`Tolerance`](@ref), and the residual, measured as the change it
  causes in one step relative to the unknown's scale (`dt |F|/scale` on differential
  rows, `|F|/scale` on algebraic rows), is at most `residual_tolerance` everywhere. An
  infinite absolute tolerance excludes an unknown from both tests; the
  merit function of the line search keeps all rows. The unknown of a
  [`Constraint`](@ref) is a rate: the system fixes the change `dt u` that it causes over
  the step, and round-off of the constrained quantity divided by `dt` is the accuracy it
  can reach. Its update test is `|δ| ≤ absolute + relative |u| + relative scale/dt`, with
  the scale of its equation, which bounds the error of the change over the step as the
  tolerances bound the other unknowns (this matters for the magnitude of the particle
  source at steps below 1e-6 s). A stiff step is an exception: when the transient and the
  right-hand side cancel to many digits, round-off holds the residual above
  `residual_tolerance` whatever the iterate, so a converged update is accepted once the
  residual stops decreasing (it grew, or fell by less than 10%, in the last iteration).
  Such a step is flagged (`residual_floor` in [`NewtonResult`](@ref)), since the same
  waiver would also accept a linearly converging iteration whose small update understates
  its error (an inexact Jacobian or linear solve, DREAM #262). An inexact linear solve
  with relative residual `η` contracts the error by about `η` per iteration, so the waiver
  applies only when the last linear solve was accurate to `floor_linear_tolerance`
  (always true for direct solvers; for [`GMRES`](@ref) its `relative_tolerance`), and
  never when the Newton matrix is itself approximate (a term with
  [`jacobian_kind`](@ref)`(term) === :approximate` solved on the stored matrix): such a
  quasi-Newton iteration converges linearly by design, its residual falls by a constant
  factor per iteration that the stall test would mistake for the floor, and only the
  residual criterion holds it.
- A system that is linear in the state (every implicit term [`is_linear`](@ref)) with
  an exact Newton matrix and a linear solve accurate to `floor_linear_tolerance` is
  accepted as soon as the update test passes, without the residual test
  (`linear_acceptance`, flagged as `linear_step` in [`NewtonResult`](@ref)): the first
  iteration solves the step, the second confirms it with an update at round-off, and
  the residual is then at its round-off floor whatever it is, so the iterations that
  would only establish the floor are saved (about three
  iterations per step for exponential fitting). The second iteration is kept, since a
  solve bounds the residual and not the error: accepting the first solve alone loses
  its refinement on ill-conditioned steps (1e-6 observed). A step shortened by a lower
  limit or the line search is not accepted this way.
- The stored Jacobian of such a system does not depend on the state, only on the
  coefficients, which are functions of the time. With `reuse_linear_matrix` it is
  assembled once per time level and kept for the later iterations of the step, and the
  Newton matrix and its factorization are kept while the time step is the same as well.
  An interval update of the time stepper discards them
  ([`invalidate_jacobian!`](@ref)). `reuse_linear_matrix = false` assembles and factorizes
  in every iteration, as DREAM does.
- Lower limits ([`LowerLimit`](@ref), declared on the unknowns) shorten the whole step so
  that no limited entry loses more than the fraction `max_reduction` of its distance to
  its limit in one iteration. Only steps towards the limit are shortened, never growth
  (DREAM #91). `max_reduction = 0.9` is DREAM's value (`MaximalPhysicalStepLength`).
- `line_search = true` adds backtracking on `½‖F‖²` of the scaled residual (Armijo
  condition with `sufficient_decrease`, at most `max_backtracks` reductions), starting
  from the step length the lower limits allow (DREAM PR #460). Off by default, as in
  DREAM.

The linear systems are solved in scaled form: rows weighted as the residual above,
columns multiplied by the unknowns' scales.
"""
Base.@kwdef struct NewtonSolver{L<:LinearSolver}
    linear_solver::L = SparseLU()
    max_iterations::Int = 30
    max_reduction::Float64 = 0.9
    residual_tolerance::Float64 = 1e-8
    floor_linear_tolerance::Float64 = 1e-8
    line_search::Bool = false
    sufficient_decrease::Float64 = 1e-4
    max_backtracks::Int = 20
    linear_acceptance::Bool = true
    reuse_linear_matrix::Bool = true
end

"""
    NewtonResult

Outcome of [`newton_solve!`](@ref): `status` (`:converged`, or the reason of failure:
`:max_iterations`, `:singular_matrix`, `:non_finite_residual`, `:non_finite_jacobian`
(a non-finite entry of the stored Jacobian at a finite residual), `:non_finite_update`,
`:domain_error`, `:line_search_failed`, `:linear_solve_failed`: an iterative linear
solve did not reach its tolerance, `:below_lower_limit`: an entry at its lower limit
would be pushed below it, or the converged state lies below it), the number of
`iterations`, the scaled residual norm before the first iteration and at the end (which
may exceed `residual_tolerance` on a stiff step, see [`NewtonSolver`](@ref)), the
scaled norm `max |δ|/scale` of the last update, the estimated `convergence_order` (`NaN`
when fewer than three updates are above round-off), the smallest step length used, the
Krylov iterations and preconditioner factorizations summed over the iterations (0 for
direct solvers), `residual_floor`,
true when the step converged with the residual above `residual_tolerance` because the
residual had stopped decreasing (or, for a linear step, was not checked against it), and
`linear_step`, true when the step was accepted as a linear system, on the update test
alone (at the second iteration, see [`NewtonSolver`](@ref)).
"""
struct NewtonResult
    status::Symbol
    iterations::Int
    initial_residual_norm::Float64
    residual_norm::Float64
    update_norm::Float64
    convergence_order::Float64
    smallest_step_length::Float64
    krylov_iterations::Int
    preconditioner_updates::Int
    residual_floor::Bool
    linear_step::Bool
    jacobian_time::Float64        # seconds in jacobian! (assembly of the stored blocks)
    residual_time::Float64        # seconds in residual evaluations, line search included
    factorization_time::Float64   # seconds factorizing (LU, or the preconditioner blocks)
    linear_solve_time::Float64    # seconds in the linear solves, factorization included
end

"""
    NewtonWorkspace(system, solver::NewtonSolver)

Buffers, tolerances, lower limits and the linear-solver cache for Newton iterations on
`system`, reused over all time steps. The per-iteration history of the last solve is in
the fields `residual_norms`, `update_norms` and `step_lengths`. `exact_jacobian` records
whether the Newton matrix is the exact Jacobian (no term with an approximate stored
block, or a linear solver whose operator restores it, [`GMRES`](@ref)`(operator=:exact)`),
and `linear` whether the system is linear in the state and the matrix exact, so the first
iteration solves a step and the second confirms it. For such a system `assembled_time` is
the time level of the stored Jacobian and `matrix_step` the time step of the Newton
matrix (`NaN`: none kept, see [`invalidate_jacobian!`](@ref)); `jacobian_assemblies`
counts the assemblies of the stored Jacobian since the workspace was made.
"""
struct NewtonWorkspace{T,S<:EquationSystem,L,C}
    system::S
    solver::NewtonSolver{L}
    exact_jacobian::Bool
    linear::Bool
    residual::Vector{T}
    trial_residual::Vector{T}
    update::Vector{T}
    scaled_update::Vector{T}
    scaled_rhs::Vector{T}
    trial::Vector{T}
    row_weights::Vector{T}
    scales::Vector{T}
    absolute::Vector{T}
    relative::Vector{T}
    rate_tolerance::Vector{T}   # unknowns of constraints: tolerance of the update times the step
    jacobian::SparseMatrixCSC{T,Int}
    matrix::SparseMatrixCSC{T,Int}
    diagonal_positions::Vector{Int}
    limited::Vector{Int}
    limit_tolerance::Vector{T}
    limit_reference::Vector{Int}   # index of the reference entry, 0 for the own scale
    cache::C
    residual_norms::Vector{Float64}
    update_norms::Vector{Float64}
    step_lengths::Vector{Float64}
    assembled_time::Base.RefValue{Float64}
    matrix_step::Base.RefValue{Float64}
    jacobian_assemblies::Base.RefValue{Int}
end

"""
    invalidate_jacobian!(workspace::NewtonWorkspace)

Discard the stored Jacobian and the Newton matrix that a system linear in the state keeps
between the iterations of a time level ([`NewtonSolver`](@ref), `reuse_linear_matrix`).
To be called when coefficients of the system were changed by hand at an unchanged time.
The time stepper calls it after the interval updates.
"""
function invalidate_jacobian!(ws::NewtonWorkspace)
    ws.assembled_time[] = NaN
    ws.matrix_step[] = NaN
    return ws
end

function NewtonWorkspace(system::EquationSystem{T}, solver::NewtonSolver=NewtonSolver()) where {T}
    # matrix-free terms with a low-rank Jacobian enter a direct solve by a Woodbury update
    full_rank = [bound for bound in system.matrix_free_terms if !has_low_rank_jacobian(bound.term)]
    if requires_stored_jacobian(solver.linear_solver) && !isempty(full_rank)
        names = [typeof(bound.term).name.name for bound in full_rank]
        throw(ArgumentError("$(typeof(solver.linear_solver)) needs every implicit term in the stored " *
                            "Jacobian, but the system has implicit matrix-free terms ($(join(names, ", "))); " *
                            "treat them explicitly or use the Newton-Krylov solver GMRES"))
    end
    layout = system.layout
    n = length(layout)
    scales = T.(unknown_scales(layout))
    row_weights = zeros(T, n)
    absolute, relative = zeros(T, n), zeros(T, n)
    limited, limit_tolerance, limit_reference = Int[], T[], Int[]
    for (id, u) in enumerate(layout.unknowns)
        tol = tolerance(system, u.name)
        for range in _unknown_ranges(layout, id)
            absolute[range] .= tol.absolute
            relative[range] .= tol.relative
        end
        limit = u.lower_limit
        limit === nothing && continue
        reference = limit.reference === nothing ? 0 : unknown_id(layout, limit.reference)
        for (k, range) in enumerate(_unknown_ranges(layout, id))
            append!(limited, range)
            append!(limit_tolerance, fill(limit.tolerance, length(range)))
            ref_index = reference == 0 ? 0 : first(unknown_range(layout, reference, k))
            append!(limit_reference, fill(ref_index, length(range)))
        end
    end
    rate_tolerance = zeros(T, n)
    for equation in system.equations
        equation.kind isa Constraint || continue
        id = unknown_id(layout, equation.unknown)
        for range in _unknown_ranges(layout, id)
            @. rate_tolerance[range] = relative[range] * system.row_scales[range]
        end
    end
    jacobian = jacobian_prototype(system)
    matrix = copy(jacobian)
    # positions of the entries of the mass matrix: (row, the column whose time derivative the
    # row holds), the diagonal for the rows without one
    diagonal_positions = zeros(Int, n)
    for i in 1:n
        column = system.mass_columns[i]
        for k in nzrange(matrix, column)
            matrix.rowval[k] == i && (diagonal_positions[i] = k)
        end
    end
    all(>(0), diagonal_positions) || error("the Jacobian pattern lacks the entries of the mass matrix")
    fill!(matrix.nzval, zero(T))
    matrix.nzval[diagonal_positions] .= one(T)
    cache = linear_solver_cache(solver.linear_solver, matrix, system, row_weights, scales)
    exact_jacobian = system.stored_jacobian_exact || newton_operator_exact(cache)
    return NewtonWorkspace(system, solver, exact_jacobian, system.linear && exact_jacobian, zeros(T, n),
                           zeros(T, n), zeros(T, n), zeros(T, n),
                           zeros(T, n), zeros(T, n), row_weights, scales, absolute, relative, rate_tolerance,
                           jacobian,
                           matrix, diagonal_positions, limited, limit_tolerance, limit_reference, cache,
                           Float64[], Float64[], Float64[], Ref(NaN), Ref(NaN), Ref(0))
end

# residual with the errors of evaluation outside the physical domain turned into a status
function _checked_residual!(F, u, u_old, system, t, dt)
    try
        residual!(F, u, u_old, system, t, dt)
    catch err
        err isa DomainError || rethrow()
        return :domain_error
    end
    return all(isfinite, F) ? :ok : :non_finite_residual
end

function _residual_norm(ws::NewtonWorkspace, F)
    norm = 0.0
    @inbounds for i in eachindex(F)
        isinf(ws.absolute[i]) && continue
        norm = max(norm, abs(F[i]) * ws.row_weights[i])
    end
    return norm
end

_merit(ws::NewtonWorkspace, F) = 0.5 * sum(i -> (F[i] * ws.row_weights[i])^2, eachindex(F))

# scaled Newton matrix A = W (M/dt - J) S, with W the row weights and S the unknown scales
function _newton_matrix!(ws::NewtonWorkspace, dt)
    A, J = ws.matrix, ws.jacobian
    mass = ws.system.mass
    @inbounds for k in eachindex(A.nzval)
        A.nzval[k] = -J.nzval[k]
    end
    @inbounds for i in eachindex(mass)
        A.nzval[ws.diagonal_positions[i]] += mass[i] / dt
    end
    @inbounds for j in axes(A, 2), k in nzrange(A, j)
        A.nzval[k] *= ws.row_weights[A.rowval[k]] * ws.scales[j]
    end
    return A
end

# lower limit of limited entry k at state u
@inline function _limit_value(ws::NewtonWorkspace, u, k)
    i = ws.limited[k]
    reference = ws.limit_reference[k]
    return -ws.limit_tolerance[k] * (reference == 0 ? ws.scales[i] : abs(u[reference]))
end

# longest step length ≤ 1 by which no limited entry loses more than max_reduction of its
# distance to the limit; steps away from the limit are never shortened. Zero when an
# entry at or below its limit would be pushed further down (DREAM skips such entries,
# which lets an iterate that reached its limit by round-off take the full step).
function _limited_step_length(ws::NewtonWorkspace, u, update)
    step = 1.0
    reduction = ws.solver.max_reduction
    @inbounds for k in eachindex(ws.limited)
        i = ws.limited[k]
        update[i] < 0 || continue
        room = u[i] - _limit_value(ws, u, k)
        room > 0 || return 0.0
        step = min(step, reduction * room / -update[i])
    end
    return step
end

function _below_lower_limit(ws::NewtonWorkspace, u)
    @inbounds for k in eachindex(ws.limited)
        u[ws.limited[k]] < _limit_value(ws, u, k) && return true
    end
    return false
end

# trial = u + step δ and its residual, with backtracking when the line search is on
function _take_step!(ws::NewtonWorkspace, u, u_old, t, dt, step)
    solver, system = ws.solver, ws.system
    trial, F = ws.trial, ws.trial_residual
    merit0 = _merit(ws, ws.residual)
    for backtrack in 0:(solver.line_search ? solver.max_backtracks : 0)
        @. trial = u + step * ws.update
        status = _checked_residual!(F, trial, u_old, system, t, dt)
        solver.line_search || return step, status
        if status === :ok
            merit = _merit(ws, F)
            # Armijo condition; the derivative of the merit along the Newton direction is -2 merit0
            merit <= (1 - 2 * solver.sufficient_decrease * step) * merit0 && return step, :ok
            # minimum of the quadratic through merit0, its slope and merit, safeguarded
            denominator = merit - merit0 + 2 * merit0 * step
            candidate = denominator > 0 ? merit0 * step^2 / denominator : 0.5 * step
            step = clamp(candidate, 0.1 * step, 0.5 * step)
        else
            step *= 0.5
        end
    end
    return step, :line_search_failed
end

# estimated order q from the last three update norms above round-off, e_k ≈ C e_{k-1}^q
function _convergence_order(norms)
    for k in length(norms):-1:3
        a, b, c = norms[k-2], norms[k-1], norms[k]
        min(a, b, c) > 1e3 * eps() && a > b > c || continue
        return log(c / b) / log(b / a)
    end
    return NaN
end

"""
    newton_solve!(u, u_old, workspace::NewtonWorkspace, t, dt)

Solve the backward-Euler step from `u_old` to time `t` with step `dt`, starting from the
guess in `u` and overwriting it. Returns a [`NewtonResult`](@ref); on failure `u` holds
the last iterate. Never throws for a failed iteration: a singular matrix, a non-finite
residual, Jacobian or update, a `DomainError` from a term (evaluation outside the physical
domain), a linear solve that did not converge, a failed line search, an entry below its lower limit at convergence and the
iteration limit are reported through the status, for the time stepper to handle
(DREAM #25, #81).
"""
function newton_solve!(u::AbstractVector, u_old::AbstractVector, ws::NewtonWorkspace, t, dt)
    system, solver = ws.system, ws.solver
    empty!(ws.residual_norms)
    empty!(ws.update_norms)
    empty!(ws.step_lengths)
    @inbounds for i in eachindex(ws.row_weights)
        ws.row_weights[i] = (system.mass[i] == 0 ? one(dt) : dt) / system.row_scales[i]
    end
    krylov = 0
    updates = 0
    smallest_step = 1.0
    jacobian_seconds = residual_seconds = factorization_seconds = solve_seconds = 0.0
    result(status, iterations, initial; residual_floor=false, linear_step=false) =
        NewtonResult(status, iterations, initial, isempty(ws.residual_norms) ? initial : last(ws.residual_norms),
                     isempty(ws.update_norms) ? NaN : last(ws.update_norms),
                     _convergence_order(ws.update_norms), smallest_step, krylov, updates, residual_floor,
                     linear_step, jacobian_seconds, residual_seconds, factorization_seconds, solve_seconds)

    started = time_ns()
    status = _checked_residual!(ws.residual, u, u_old, system, t, dt)
    residual_seconds += (time_ns() - started) / 1e9
    status === :ok || return result(status, 0, NaN)
    initial = _residual_norm(ws, ws.residual)
    reuse = ws.linear && solver.reuse_linear_matrix
    for iteration in 1:solver.max_iterations
        # the stored Jacobian of a system linear in the state changes with the time level
        # alone, the Newton matrix with the time level and the time step
        same_jacobian = reuse && ws.assembled_time[] == t
        if !same_jacobian
            invalidate_jacobian!(ws)
            started = time_ns()
            try
                jacobian!(ws.jacobian, system, u, t)
            catch err
                err isa DomainError || rethrow()
                return result(:domain_error, iteration - 1, initial)
            end
            jacobian_seconds += (time_ns() - started) / 1e9
            ws.jacobian_assemblies[] += 1
            all(isfinite, ws.jacobian.nzval) || return result(:non_finite_jacobian, iteration - 1, initial)
            reuse && (ws.assembled_time[] = t)
        end
        same_matrix = same_jacobian && ws.matrix_step[] == dt
        if !same_matrix
            _newton_matrix!(ws, dt)
            reuse && (ws.matrix_step[] = dt)
        end
        matrix_unchanged!(ws.cache, same_matrix)
        @. ws.scaled_rhs = -ws.row_weights * ws.residual
        started = time_ns()
        linear_status = solve!(ws.scaled_update, ws.cache, ws.matrix, ws.scaled_rhs, u, t, dt, iteration)
        solve_seconds += (time_ns() - started) / 1e9
        factorization_seconds += factorization_time(ws.cache)
        krylov += krylov_iterations(ws.cache)
        updates += preconditioner_updates(ws.cache)
        if linear_status !== :ok
            invalidate_jacobian!(ws)
            return result(linear_status, iteration - 1, initial)
        end
        @. ws.update = ws.scales * ws.scaled_update
        all(isfinite, ws.update) || return result(:non_finite_update, iteration - 1, initial)

        limited_step = _limited_step_length(ws, u, ws.update)
        limited_step > 0 || return result(:below_lower_limit, iteration - 1, initial)
        started = time_ns()
        step, status = _take_step!(ws, u, u_old, t, dt, limited_step)
        residual_seconds += (time_ns() - started) / 1e9
        smallest_step = min(smallest_step, step)
        status === :ok || return result(status, iteration - 1, initial)
        copyto!(u, ws.trial)
        copyto!(ws.residual, ws.trial_residual)

        push!(ws.update_norms, maximum(abs, ws.scaled_update))
        push!(ws.residual_norms, _residual_norm(ws, ws.residual))
        push!(ws.step_lengths, step)
        residual_small = last(ws.residual_norms) <= solver.residual_tolerance
        update_converged = all(eachindex(u)) do i
            abs(ws.update[i]) <= ws.absolute[i] + ws.relative[i] * abs(u[i]) + ws.rate_tolerance[i] / dt
        end
        # A linear system with the exact matrix and an accurate solve: once the update
        # (the error estimate of the previous iterate) passes, no further iteration can
        # improve the state, and the residual is at its round-off floor whatever it is,
        # so the residual test is not needed (it would only add the iterations that
        # establish the floor). The update test itself stays: accepting the
        # first solve alone loses the refinement the second iteration provides on
        # ill-conditioned steps (errors of the condition number times the solve's
        # accuracy, 1e-6 observed with a Krylov tolerance of 1e-10 and on a direct solve
        # of a steady-state step).
        if ws.linear && solver.linear_acceptance && step == 1 && update_converged &&
           linear_tolerance(ws.cache) <= solver.floor_linear_tolerance
            status = _below_lower_limit(ws, u) ? :below_lower_limit : :converged
            return result(status, iteration, initial; residual_floor=!residual_small, linear_step=true)
        end
        # The residual of a stiff step has a round-off floor: F = M(u - u_old)/dt - rhs
        # cancels two terms of size dt‖L‖ times the unknown, so the scaled residual cannot
        # fall below about eps dt‖L‖/scale, which exceeds `residual_tolerance` at large dt.
        # Once the update is within tolerance and the residual has stopped decreasing, the
        # iteration has converged as far as the arithmetic allows.
        # An inexact linear solve converges linearly, with a small update that understates
        # the error unless the solve was accurate (DREAM #262), so the waiver needs that;
        # so does an approximate Newton matrix (a quasi-Newton iteration).
        residual_stalled = length(ws.residual_norms) > 1 &&
                           last(ws.residual_norms) >= 0.9 * ws.residual_norms[end-1] &&
                           linear_tolerance(ws.cache) <= solver.floor_linear_tolerance && ws.exact_jacobian
        if update_converged && (residual_small || residual_stalled)
            status = _below_lower_limit(ws, u) ? :below_lower_limit : :converged
            return result(status, iteration, initial; residual_floor=!residual_small)
        end
    end
    return result(:max_iterations, solver.max_iterations, initial)
end
