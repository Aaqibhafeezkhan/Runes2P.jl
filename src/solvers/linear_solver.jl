# Linear solvers for the Newton iterations. A solver type holds the
# settings; `linear_solver_cache` creates the state reused between solves (the symbolic
# factorization of a fixed sparsity pattern).

using LinearAlgebra: lu, lu!, issuccess, ldiv!
using SparseArrays: nzrange

"""
    LinearSolver

Supertype of the linear solvers used by [`NewtonSolver`](@ref). A solver implements
`linear_solver_cache(solver, A)` and `solve!(x, cache, A, b)`, which returns `true` on
success and `false` when the matrix is singular or the factorization failed, plus
`requires_stored_jacobian(solver)` (whether all implicit terms must be in the stored
matrix) and `krylov_iterations(cache)` (Krylov iterations of the last solve, 0 for direct
solvers).
"""
abstract type LinearSolver end

"""
    SparseLU()

Monolithic sparse direct solve by LU factorization (UMFPACK, through `SparseArrays`), the
DREAM-like baseline. The symbolic analysis of the sparsity pattern is done once and
reused for every later matrix with the same pattern. Needs every implicit term in the
stored Jacobian, so a system with implicit matrix-free terms (the implicit knock-on term)
is refused; those need a Newton-Krylov solver.
"""
struct SparseLU <: LinearSolver end

mutable struct SparseLUCache{F}
    factorization::F
end

requires_stored_jacobian(::SparseLU) = true
krylov_iterations(::SparseLUCache) = 0

linear_solver_cache(::SparseLU, A::SparseMatrixCSC) = SparseLUCache(lu(A; check=false))

function solve!(x::AbstractVector, cache::SparseLUCache, A::SparseMatrixCSC, b::AbstractVector)
    # reuses the symbolic analysis of the pattern
    lu!(cache.factorization, A; check=false)
    issuccess(cache.factorization) || return false
    ldiv!(x, cache.factorization, b)
    return true
end
