# Newton iterations for one backward-Euler step: solve
# F(u) = M (u - u_old)/dt - rhs(u, t) = 0 with the Jacobian M/dt - ∂rhs/∂u, a lower-limit
# (positivity) step length, an optional line search, and convergence on both the update
# and the residual. A failure is reported as a status, which the time stepper handles.

"""
    NewtonSolver(; linear_solver=SparseLU(), max_iterations=30, max_reduction=0.9,
                 residual_tolerance=1e-8, line_search=false, sufficient_decrease=1e-4,
                 max_backtracks=20)

Settings of the Newton iteration of a backward-Euler step.

- Convergence needs both a small update and a small residual (DREAM #262): every entry
  of the full Newton update `δ` satisfies `|δ| ≤ absolute + relative |u|` with the
  unknown's Newton [`Tolerance`](@ref) (an infinite absolute tolerance excludes it), and
  the residual, measured as the change it causes in one step relative to the unknown's
  scale (`dt |F|/scale` on differential rows, `|F|/scale` on algebraic rows), is at most
  `residual_tolerance` everywhere. A stiff step is an exception: when the transient and the
  right-hand side cancel to many digits, round-off holds the residual above
  `residual_tolerance` whatever the iterate, so a converged update is accepted once the
  residual stops decreasing (it grew, or fell by less than 10%, in the last iteration).
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
    line_search::Bool = false
    sufficient_decrease::Float64 = 1e-4
    max_backtracks::Int = 20
end

"""
    NewtonResult

Outcome of [`newton_solve!`](@ref): `status` (`:converged`, or the reason of failure:
`:max_iterations`, `:singular_matrix`, `:non_finite_residual`, `:non_finite_update`,
`:domain_error`, `:line_search_failed`, `:below_lower_limit`: an entry at its lower
limit would be pushed below it, or the converged state lies below it), the number of
`iterations`, the scaled residual norm before the first iteration and at the end (which
may exceed `residual_tolerance` on a stiff step, see [`NewtonSolver`](@ref)), the
scaled norm `max |δ|/scale` of the last update, the estimated `convergence_order` (`NaN`
when fewer than three updates are above round-off), the smallest step length used and the
Krylov iterations summed over the iterations (0 for direct solvers).
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
end

"""
    NewtonWorkspace(system, solver::NewtonSolver)

Buffers, tolerances, lower limits and the linear-solver cache for Newton iterations on
`system`, reused over all time steps. The per-iteration history of the last solve is in
the fields `residual_norms`, `update_norms` and `step_lengths`.
"""
struct NewtonWorkspace{T,S<:EquationSystem,L,C}
    system::S
    solver::NewtonSolver{L}
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
end

function NewtonWorkspace(system::EquationSystem{T}, solver::NewtonSolver=NewtonSolver()) where {T}
    if requires_stored_jacobian(solver.linear_solver) && !isempty(system.matrix_free_terms)
        names = [typeof(bound.term).name.name for bound in system.matrix_free_terms]
        throw(ArgumentError("$(typeof(solver.linear_solver)) needs every implicit term in the stored " *
                            "Jacobian, but the system has implicit matrix-free terms ($(join(names, ", "))); " *
                            "treat them explicitly or use a Krylov solver"))
    end
    layout = system.layout
    n = length(layout)
    scales = T.(unknown_scales(layout))
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
    jacobian = jacobian_prototype(system)
    matrix = copy(jacobian)
    diagonal_positions = zeros(Int, n)
    for j in 1:n, k in nzrange(matrix, j)
        matrix.rowval[k] == j && (diagonal_positions[j] = k)
    end
    all(>(0), diagonal_positions) || error("the Jacobian pattern lacks diagonal entries")
    fill!(matrix.nzval, zero(T))
    matrix.nzval[diagonal_positions] .= one(T)
    cache = linear_solver_cache(solver.linear_solver, matrix)
    return NewtonWorkspace(system, solver, zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
                           zeros(T, n), zeros(T, n), zeros(T, n), scales, absolute, relative, jacobian,
                           matrix, diagonal_positions, limited, limit_tolerance, limit_reference, cache,
                           Float64[], Float64[], Float64[])
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
    @inbounds for j in axes(A, 2)
        for k in nzrange(A, j)
            A.nzval[k] = -J.nzval[k]
        end
        A.nzval[ws.diagonal_positions[j]] += mass[j] / dt
        for k in nzrange(A, j)
            A.nzval[k] *= ws.row_weights[A.rowval[k]] * ws.scales[j]
        end
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
function _limited_step_length(ws::NewtonWorkspace, u, δ)
    step = 1.0
    reduction = ws.solver.max_reduction
    @inbounds for k in eachindex(ws.limited)
        i = ws.limited[k]
        δ[i] < 0 || continue
        room = u[i] - _limit_value(ws, u, k)
        room > 0 || return 0.0
        step = min(step, reduction * room / -δ[i])
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
residual or update, a `DomainError` from a term (evaluation outside the physical
domain), a failed line search, an entry below its lower limit at convergence and the
iteration limit are reported through the status, for the time stepper to handle
(DREAM #25, #81).
"""
function newton_solve!(u::AbstractVector, u_old::AbstractVector, ws::NewtonWorkspace, t, dt)
    system, solver = ws.system, ws.solver
    empty!(ws.residual_norms)
    empty!(ws.update_norms)
    empty!(ws.step_lengths)
    @inbounds for i in eachindex(ws.row_weights)
        ws.row_weights[i] = (system.mass[i] == 0 ? one(dt) : dt) / ws.scales[i]
    end
    krylov = 0
    smallest_step = 1.0
    result(status, iterations, initial) =
        NewtonResult(status, iterations, initial, isempty(ws.residual_norms) ? initial : last(ws.residual_norms),
                     isempty(ws.update_norms) ? NaN : last(ws.update_norms),
                     _convergence_order(ws.update_norms), smallest_step, krylov)

    status = _checked_residual!(ws.residual, u, u_old, system, t, dt)
    status === :ok || return result(status, 0, NaN)
    initial = _residual_norm(ws, ws.residual)
    for iteration in 1:solver.max_iterations
        try
            jacobian!(ws.jacobian, system, u, t)
        catch err
            err isa DomainError || rethrow()
            return result(:domain_error, iteration - 1, initial)
        end
        all(isfinite, ws.jacobian.nzval) || return result(:non_finite_residual, iteration - 1, initial)
        _newton_matrix!(ws, dt)
        @. ws.scaled_rhs = -ws.row_weights * ws.residual
        solve!(ws.scaled_update, ws.cache, ws.matrix, ws.scaled_rhs) ||
            return result(:singular_matrix, iteration - 1, initial)
        krylov += krylov_iterations(ws.cache)
        @. ws.update = ws.scales * ws.scaled_update
        all(isfinite, ws.update) || return result(:non_finite_update, iteration - 1, initial)

        limited_step = _limited_step_length(ws, u, ws.update)
        limited_step > 0 || return result(:below_lower_limit, iteration - 1, initial)
        step, status = _take_step!(ws, u, u_old, t, dt, limited_step)
        smallest_step = min(smallest_step, step)
        status === :ok || return result(status, iteration - 1, initial)
        copyto!(u, ws.trial)
        copyto!(ws.residual, ws.trial_residual)

        push!(ws.update_norms, maximum(abs, ws.scaled_update))
        push!(ws.residual_norms, _residual_norm(ws, ws.residual))
        push!(ws.step_lengths, step)
        update_converged = all(eachindex(u)) do i
            abs(ws.update[i]) <= ws.absolute[i] + ws.relative[i] * abs(u[i])
        end
        # The residual of a stiff step has a round-off floor: F = M(u - u_old)/dt - rhs
        # cancels two terms of size dt‖L‖ times the unknown, so the scaled residual cannot
        # fall below about eps dt‖L‖/scale, which exceeds `residual_tolerance` at large dt.
        # Once the update is within tolerance and the residual has stopped decreasing, the
        # iteration has converged as far as the arithmetic allows.
        residual_stalled = length(ws.residual_norms) > 1 &&
                           last(ws.residual_norms) >= 0.9 * ws.residual_norms[end-1]
        if update_converged && (last(ws.residual_norms) <= solver.residual_tolerance || residual_stalled)
            status = _below_lower_limit(ws, u) ? :below_lower_limit : :converged
            return result(status, iteration, initial)
        end
    end
    return result(:max_iterations, solver.max_iterations, initial)
end
