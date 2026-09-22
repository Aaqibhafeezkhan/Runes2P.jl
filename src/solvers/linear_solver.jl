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
- `linear_tolerance(cache)`: bound on the relative residual `‖b - A x‖/‖b‖` of the last
  solve (0 for direct solvers), used by the residual-floor rule of the Newton iteration.
"""
abstract type LinearSolver end

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
end

requires_stored_jacobian(::SparseLU) = true
krylov_iterations(::SparseLUCache) = 0
preconditioner_updates(::SparseLUCache) = 0
linear_tolerance(::SparseLUCache) = 0.0

linear_solver_cache(::SparseLU, matrix::SparseMatrixCSC, system, row_weights, scales) =
    SparseLUCache(lu(matrix; check=false))

function solve!(x::AbstractVector, cache::SparseLUCache, matrix::SparseMatrixCSC, b::AbstractVector,
                u, t, dt, iteration)
    # reuses the symbolic analysis of the pattern
    lu!(cache.factorization, matrix; check=false)
    issuccess(cache.factorization) || return :singular_matrix
    ldiv!(x, cache.factorization, b)
    return :ok
end
