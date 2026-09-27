# Linear solvers for the Newton iterations. A solver type holds
# the settings; `linear_solver_cache` creates the state reused between solves (the
# symbolic factorization of a fixed sparsity pattern, Krylov workspaces, preconditioners).

using LinearAlgebra: lu, lu!, issuccess, ldiv!
using SparseArrays: nzrange

"""
    LinearSolver

Supertype of the linear solvers used by [`NewtonSolver`](@ref). Each Newton iteration
solves `A x = b` with the scaled Newton matrix `A = W (M/dt - J) S` (row weights `W`,
unknown scales `S`, see [`NewtonSolver`](@ref)), where `J` is the stored Jacobian of the
implicit `:sparse` terms plus, for solvers that support them, the implicit matrix-free
terms. A solver implements

- `linear_solver_cache(solver, matrix, system, row_weights, scales)`: the state reused
  between solves; `matrix` is the scaled stored matrix (its pattern is fixed, its values
  change), `row_weights` and `scales` are the vectors `W` and `S`, which the Newton
  iteration updates in place;
- `solve!(x, cache, matrix, b, u, t, dt, iteration)`: solve with the matrix at the state
  `u`, time `t` and step `dt`, in Newton iteration `iteration` of the step; returns
  `:ok`, `:singular_matrix` or `:linear_solve_failed` (an iterative solve that did not
  reach its tolerance);
- `requires_stored_jacobian(solver)`: whether every implicit term must be in the stored
  matrix;
- `krylov_iterations(cache)`, `preconditioner_updates(cache)`: Krylov iterations and
  preconditioner factorizations of the last solve (0 for direct solvers);
- `factorization_time(cache)`: seconds spent factorizing in the last solve (the LU of a
  direct solver, the preconditioner blocks of a Krylov solver);
- `linear_tolerance(cache)`: bound on the relative residual `‖b - A x‖/‖b‖` of the last
  solve (0 for direct solvers), used by the residual-floor rule of the Newton iteration;
- `newton_operator_exact(cache)`: whether the solver's Newton operator is the exact
  Jacobian when the system has terms with an approximate stored block
  ([`jacobian_kind`](@ref); false for solvers on the stored matrix alone);
- `matrix_unchanged!(cache, unchanged)`, optional: told before every solve whether the
  matrix has the values of the last solve, so that a factorization of it can be kept
  (a system linear in the state, [`NewtonSolver`](@ref)). Without a method the solver
  treats every matrix as new.
"""
abstract type LinearSolver end

matrix_unchanged!(cache, unchanged::Bool) = nothing

"""
    SparseLU()

Monolithic sparse direct solve by LU factorization (UMFPACK, through `SparseArrays`), the
DREAM-like baseline. The symbolic analysis of the sparsity pattern is done once and
reused for every later matrix with the same pattern. Needs every implicit term in the
stored Jacobian, so a system with implicit matrix-free terms (the implicit knock-on term)
is refused; those need the Newton-Krylov solver [`GMRES`](@ref).
"""
struct SparseLU <: LinearSolver end

mutable struct SparseLUCache{F}
    factorization::F
    factorization_seconds::Float64
    factorized::Bool          # the factorization is that of the matrix of the last solve
    matrix_unchanged::Bool    # the matrix of the next solve is that of the last one
    factorizations::Int       # numerical factorizations since the cache was made
end

matrix_unchanged!(cache::SparseLUCache, unchanged::Bool) = (cache.matrix_unchanged = unchanged; nothing)

requires_stored_jacobian(::SparseLU) = true
krylov_iterations(::SparseLUCache) = 0
preconditioner_updates(::SparseLUCache) = 0
factorization_time(cache::SparseLUCache) = cache.factorization_seconds
linear_tolerance(::SparseLUCache) = 0.0
newton_operator_exact(::SparseLUCache) = false

linear_solver_cache(::SparseLU, matrix::SparseMatrixCSC, system, row_weights, scales) =
    SparseLUCache(lu(matrix; check=false), 0.0, false, false, 0)

function solve!(x::AbstractVector, cache::SparseLUCache, matrix::SparseMatrixCSC, b::AbstractVector,
                u, t, dt, iteration)
    cache.factorization_seconds = 0.0
    if !(cache.factorized && cache.matrix_unchanged)
        # reuses the symbolic analysis of the pattern
        started = time_ns()
        lu!(cache.factorization, matrix; check=false)
        cache.factorization_seconds = (time_ns() - started) / 1e9
        cache.factorizations += 1
        cache.factorized = issuccess(cache.factorization)
        cache.factorized || return :singular_matrix
    end
    ldiv!(x, cache.factorization, b)
    return :ok
end
