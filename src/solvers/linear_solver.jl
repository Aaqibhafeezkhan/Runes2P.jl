# Linear solvers for the Newton iterations. A solver type holds the settings;
# `linear_solver_cache` creates the state reused between solves (the symbolic factorization
# of a fixed sparsity pattern, Krylov workspaces, preconditioners).

using LinearAlgebra: LinearAlgebra, lu, lu!, issuccess, ldiv!
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
  matrix, apart from matrix-free terms with a low-rank Jacobian
  ([`has_low_rank_jacobian`](@ref)), which the solver adds by a Woodbury update;
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

# Woodbury update of the factorization of a diagonal block of the scaled Newton matrix: the
# implicit matrix-free terms with a low-rank Jacobian ([`low_rank_jacobian`](@ref)) add
# `Ũ Ṽᵀ` to the block `B`, with `Ũ = -W U` and `Ṽ = S V` from the term's `U Vᵀ`, the row
# weights `W` and the unknown scales `S`. Then `(B + Ũ Ṽᵀ)⁻¹ x = y - Z (I + Ṽᵀ Z)⁻¹ Ṽᵀ y`
# with `y = B⁻¹ x` and `Z = B⁻¹ Ũ`: `k` solves with `B` when the block is factorized, one
# per application and O(n k) work.
struct LowRankCorrection{T}
    solved_left::Matrix{T}   # Z = B⁻¹ Ũ
    right::Matrix{T}         # Ṽ
    capacitance::LinearAlgebra.LU{T, Matrix{T}, Vector{Int}}   # I + Ṽᵀ Z
    coefficients::Vector{T}
end

# the low-rank blocks of the implicit matrix-free terms at the state u and time t
function _low_rank_blocks(system::EquationSystem, u, t)
    blocks = LowRankBlock[]
    for bound in system.matrix_free_terms
        found = low_rank_jacobian(bound.term, u, system.layout, bound.row, t)
        if found !== nothing
            append!(blocks, found)
        end
    end
    return blocks
end

# Ũ and Ṽ of the low-rank blocks inside `range` (in the block's own indices), or nothing;
# blocks that reach outside `range` are left out, as the other couplings between blocks
function _scaled_low_rank(
        blocks, range::UnitRange{Int}, row_weights, scales, ::Type{T}) where {T}
    lefts, rights = Matrix{T}[], Matrix{T}[]
    n = length(range)
    for b in blocks
        (issubset(b.rows, range) && issubset(b.columns, range)) || continue
        left, right = zeros(T, n, size(b.left, 2)), zeros(T, n, size(b.right, 2))
        left[b.rows .- (first(range) - 1), :] .= -view(row_weights, b.rows) .* b.left
        right[b.columns .- (first(range) - 1), :] .= view(scales, b.columns) .* b.right
        push!(lefts, left)
        push!(rights, right)
    end
    isempty(lefts) && return nothing
    return reduce(hcat, lefts), reduce(hcat, rights)
end

# the correction of a factorized block, `solve!(y, x)` solving with the block's factors;
# nothing when the capacitance matrix is singular
function _low_rank_correction(left::Matrix{T}, right::Matrix{T}, solve!) where {T}
    solved = similar(left)
    for c in axes(left, 2)
        solve!(view(solved, :, c), left[:, c])
    end
    capacitance = LinearAlgebra.lu(LinearAlgebra.I + transpose(right) * solved; check = false)
    issuccess(capacitance) || return nothing
    return LowRankCorrection(solved, right, capacitance, zeros(T, size(left, 2)))
end

# y ← y - Z (I + Ṽᵀ Z)⁻¹ Ṽᵀ y, for y = B⁻¹ x
function _apply_low_rank_correction!(y::AbstractVector, correction::LowRankCorrection)
    c = correction.coefficients
    LinearAlgebra.mul!(c, transpose(correction.right), y)
    ldiv!(correction.capacitance, c)
    LinearAlgebra.mul!(y, correction.solved_left, c, -1, 1)
    return y
end

"""
    SparseLU()

Monolithic sparse direct solve by LU factorization (UMFPACK, through `SparseArrays`), the
DREAM-like baseline. The symbolic analysis of the sparsity pattern is done once and
reused for every later matrix with the same pattern. Needs every implicit term in the
stored Jacobian, except matrix-free terms with a low-rank Jacobian
([`low_rank_jacobian`](@ref), the field-particle term), which a Woodbury update of the
factorization solves exactly. A system with other implicit matrix-free terms (the implicit
knock-on term) is refused; those need the Newton-Krylov solver [`GMRES`](@ref).
"""
struct SparseLU <: LinearSolver end

mutable struct SparseLUCache{F, S, W}
    factorization::F
    system::S
    row_weights::W
    scales::W
    low_rank::Union{Nothing, LowRankCorrection}   # Woodbury update of the factorization
    factorization_seconds::Float64
    factorized::Bool          # the factorization is that of the matrix of the last solve
    matrix_unchanged::Bool    # the matrix of the next solve is that of the last one
    factorizations::Int       # numerical factorizations since the cache was made
end

function matrix_unchanged!(cache::SparseLUCache, unchanged::Bool)
    cache.matrix_unchanged = unchanged
    return nothing
end

requires_stored_jacobian(::SparseLU) = true
krylov_iterations(::SparseLUCache) = 0
preconditioner_updates(::SparseLUCache) = 0
factorization_time(cache::SparseLUCache) = cache.factorization_seconds
linear_tolerance(::SparseLUCache) = 0.0
newton_operator_exact(::SparseLUCache) = false

function linear_solver_cache(
        ::SparseLU, matrix::SparseMatrixCSC, system, row_weights, scales)
    SparseLUCache(lu(matrix; check = false), system, row_weights,
        scales, nothing, 0.0, false, false, 0)
end

# the Woodbury update of the whole matrix for the low-rank matrix-free terms at (u, t)
function _update_low_rank!(cache::SparseLUCache, u, t)
    cache.low_rank = nothing
    system = cache.system
    isempty(system.matrix_free_terms) && return true
    scaled = _scaled_low_rank(
        _low_rank_blocks(system, u, t), 1:length(u), cache.row_weights, cache.scales,
        eltype(cache.scales))
    scaled === nothing && return true
    cache.low_rank = _low_rank_correction(scaled..., (y, x) -> ldiv!(y, cache.factorization, x))
    return cache.low_rank !== nothing
end

function solve!(
        x::AbstractVector, cache::SparseLUCache, matrix::SparseMatrixCSC, b::AbstractVector,
        u, t, dt, iteration)
    cache.factorization_seconds = 0.0
    if !(cache.factorized && cache.matrix_unchanged)
        # reuses the symbolic analysis of the pattern
        started = time_ns()
        lu!(cache.factorization, matrix; check = false)
        cache.factorization_seconds = (time_ns() - started) / 1e9
        cache.factorizations += 1
        cache.factorized = if issuccess(cache.factorization)
            _update_low_rank!(cache, u, t)
        else
            false
        end
        cache.factorized || return :singular_matrix
    end
    ldiv!(x, cache.factorization, b)
    if cache.low_rank !== nothing
        _apply_low_rank_correction!(x, cache.low_rank)
    end
    return :ok
end

"""
    MKLPardiso(; threads=1, refinement_steps=0)

Monolithic sparse direct solve by Intel MKL PARDISO, the solver of DREAM's
`LINEAR_SOLVER_MKL`, through the package Pardiso.jl. Its solves come with the extension
`Runes2PMKLPardisoExt`, loaded by `using Pardiso`; without it, a [`NewtonWorkspace`](@ref)
with this solver is refused with an `ArgumentError`. It takes the systems that
[`SparseLU`](@ref) takes, the field-particle term ([`low_rank_jacobian`](@ref)) by the same
Woodbury update. Its factorization has another pivot order than UMFPACK's, so the converged
states differ from those of [`SparseLU`](@ref) within the Newton tolerance, and the Newton
iterations of a step can differ.

PARDISO factorizes the real unsymmetric matrix with the defaults of its `pardisoinit`: a
nested-dissection ordering (METIS), a weighted matching with scaling, and pivots below
``10^{-13}`` times the infinity norm of the scaled matrix perturbed. The ordering and the
matching need the values of the matrix: they are computed at the first solve, from its
matrix, and kept for every later matrix of the same pattern; their time counts in the
factorization time of that solve ([`SolverStatistics`](@ref)). A run resumed from a
checkpoint computes them from its own first matrix, so it repeats the uninterrupted run
within round-off, not bit for bit. The numeric factorization is made in every solve, except
when the matrix is that of the last solve (a system linear in the state,
[`NewtonSolver`](@ref)). PARDISO reads the compressed columns of the matrix as the
compressed rows of its transpose and solves the transposed system, so no matrix is
converted. Its integers have 32 bits, so the factors hold at most ``2^{31} - 1`` nonzeros;
a larger factorization ends the run with an error that names the limit. The factors are in
memory that MKL allocates and Julia's garbage collector does not count: they are released
when the workspace is collected, so repeated runs in one process can hold several
factorizations until `GC.gc()`.

- `threads`: the threads of PARDISO, MKL's own; Julia's threads and the BLAS threads do not
  apply. The count is a setting of MKL for the whole process, set before every solve, so
  one workspace solves one task at a time, and two workspaces with different counts do not
  solve at the same time. 1 is DREAM's value (`fvm/Solvers/MIMKL.cpp`), and the results of
  a run are then reproducible, which MKL does not promise with more threads.
- `refinement_steps`: the largest number of steps of iterative refinement in each solve
  (PARDISO's `iparm(8)`). 0 is a choice of this package (see the docs page *Choices of
  this package*): on the disruption case ten steps take longer than the factorization, and
  the solve without them reaches a relative residual of about ``10^{-9}``. With 0, PARDISO
  still refines twice after a perturbed pivot. DREAM sets 10 (`fvm/Solvers/MIMKL.cpp`).

Every solve measures its relative residual ``\\|b - A x\\|/\\|b\\|`` with one product by
the matrix, before the Woodbury update, with refinement too, since the refinement stops at
its cap and promises no residual. That residual is the bound that the residual-floor rule
of the [`NewtonSolver`](@ref) compares with its `floor_linear_tolerance`: a perturbed pivot
that spoils the solve then turns both waivers of the residual test off, where
[`SparseLU`](@ref) reports 0.
"""
struct MKLPardiso <: LinearSolver
    threads::Int
    refinement_steps::Int
    function MKLPardiso(; threads::Integer = 1, refinement_steps::Integer = 0)
        threads >= 1 ||
            throw(ArgumentError("MKLPardiso needs at least one thread, not $threads"))
        refinement_steps >= 0 ||
            throw(ArgumentError("the refinement steps of MKLPardiso cannot be negative: $refinement_steps"))
        return new(threads, refinement_steps)
    end
end

requires_stored_jacobian(::MKLPardiso) = true

# the method without Pardiso.jl; the extension Runes2PMKLPardisoExt adds the one for a sparse
# matrix
function linear_solver_cache(::MKLPardiso, matrix, system, row_weights, scales)
    throw(ArgumentError("MKLPardiso needs the package Pardiso.jl: add `using Pardiso` to the script"))
end
