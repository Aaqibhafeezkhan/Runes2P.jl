# Newton-Krylov linear solves: GMRES (Krylov.jl) on the scaled Newton
# operator, which includes the implicit matrix-free terms (the knock-on term), right
# preconditioned by a factorization of the stored matrix in blocks, one per radius and one
# for the scalar unknowns.

using Krylov: GmresWorkspace, gmres!
using IncompleteLU: ILUFactorization, ilu
using LinearAlgebra: LinearAlgebra, mul!

"""
    RadiusBlockLU()
    RadiusBlockILU(; drop_tolerance=1e-4)

Preconditioners for [`GMRES`](@ref): the scaled stored Newton matrix restricted to the
blocks of [`radius_block_range`](@ref) (all unknowns at one radius) and
[`scalar_block_range`](@ref), each block factorized on its own; couplings between blocks
(radial transport, scalar unknowns) and the matrix-free terms are left out.
`RadiusBlockLU` factorizes each block exactly (UMFPACK, symbolic analysis reused);
`RadiusBlockILU` computes an incomplete LU factorization (Crout ILU of IncompleteLU.jl),
which drops entries of the factors below `drop_tolerance` in magnitude.
"""
abstract type BlockPreconditioner end
struct RadiusBlockLU <: BlockPreconditioner end
struct RadiusBlockILU <: BlockPreconditioner
    drop_tolerance::Float64
end
RadiusBlockILU(; drop_tolerance::Real=1e-4) = RadiusBlockILU(Float64(drop_tolerance))

"""
    GMRES(; preconditioner=RadiusBlockLU(), relative_tolerance=1e-10, max_iterations=200,
          restart_length=30, preconditioner_refresh=:every_iteration, refresh_iterations=10,
          operator=:exact)

Newton-Krylov linear solver: restarted GMRES (Krylov.jl) on the scaled Newton operator
`W (M/dt - J) S`, where `J` is the stored Jacobian plus the implicit matrix-free terms,
right preconditioned by `preconditioner` (a [`RadiusBlockLU`](@ref) or
[`RadiusBlockILU`](@ref) of the stored matrix). With right preconditioning the residual
GMRES minimizes is the true residual of the scaled system; a solve succeeds when
`‖b - A x‖ ≤ relative_tolerance ‖b‖` within `max_iterations` iterations, and otherwise
the Newton iteration fails with `:linear_solve_failed`.

`operator` sets what `J` is for terms whose stored block is an approximation of their
Jacobian ([`jacobian_kind`](@ref)): `:exact` adds their
[`jacobian_correction_product!`](@ref) to every product, so Newton is exact and the
approximate block only preconditions; `:stored` uses the stored matrix as it is, a
quasi-Newton iteration (DREAM's `AD_INTERP_JACOBIAN_LINEAR` and `UPWIND`). Without such
terms the two are the same.

`preconditioner_refresh` sets when the preconditioner is factorized again:

- `:every_iteration`: every Newton iteration, from the current matrix. For a system that
  is linear in the state the matrix of the later iterations of a step is that of the
  first, and its factors are kept, so the preconditioner is factorized once per step
  ([`NewtonSolver`](@ref), `reuse_linear_matrix`);
- `:every_step`: in the first Newton iteration of each time step;
- `:on_demand`: when the time step changes (by more than a relative 1e-8), or when the
  last solve needed more than
  `refresh_iterations` Krylov iterations. The stored matrix of a system with prescribed
  coefficients and a constant time step is the same in every step, so one
  factorization then serves the whole run.

A solve that fails with a preconditioner from an earlier iteration is repeated once with
a fresh one.

At production sizes with radially coupled radii:

- The block preconditioner leaves out the radial coupling, and the Krylov count grows with
  it: about 29 per Newton iteration at 20 radii and 59 at 40 under Rechester-Rosenbluth
  diffusion with `δB/B = 1e-3`, the largest of a step 1.4 times that. `restart_length` has
  to exceed the count. With the default of 30 the iteration stagnates and the step ends
  with `:linear_solve_failed` (600 000 and 1 200 000 unknowns, 20 radii), with 200 it
  converges.
- `relative_tolerance` stays at or below the `floor_linear_tolerance` of the
  [`NewtonSolver`](@ref) (1e-8). With 1e-6 the Newton iteration cannot accept a residual
  at its round-off floor and ends with `:max_iterations`. 1e-8 takes about 19 Krylov
  iterations per Newton iteration at 20 radii and three Newton iterations per step, 1e-10
  about 33 and two.
- The memory is that of the block factors and of the Krylov basis, `restart_length`
  vectors: 5.7 GB at 1 200 000 unknowns with a restart length of 200.
"""
Base.@kwdef struct GMRES{P<:BlockPreconditioner} <: LinearSolver
    preconditioner::P = RadiusBlockLU()
    relative_tolerance::Float64 = 1e-10
    max_iterations::Int = 200
    restart_length::Int = 30
    preconditioner_refresh::Symbol = :every_iteration
    refresh_iterations::Int = 10
    operator::Symbol = :exact
end

requires_stored_jacobian(::GMRES) = false

# The scaled Newton operator A v = matrix v - W (J_mf (S v)), with `matrix` the scaled
# stored part W (M/dt - J) S and J_mf the products of the implicit matrix-free terms at
# the current iterate, plus, with `exact`, the corrections of the terms whose stored
# block is approximate.
mutable struct NewtonOperator{T,S<:EquationSystem}
    const matrix::SparseMatrixCSC{T,Int}
    const system::S
    const row_weights::Vector{T}
    const scales::Vector{T}
    const state::Vector{T}
    const scaled_input::Vector{T}
    const product::Vector{T}
    const exact::Bool
    time::T
end

Base.size(op::NewtonOperator) = size(op.matrix)
Base.size(op::NewtonOperator, k::Integer) = size(op.matrix, k)
Base.eltype(::NewtonOperator{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, op::NewtonOperator, v::AbstractVector)
    mul!(y, op.matrix, v)
    system = op.system
    corrections = op.exact && !isempty(system.correction_terms)
    isempty(system.matrix_free_terms) && !corrections && return y
    @. op.scaled_input = op.scales * v
    fill!(op.product, zero(eltype(op.product)))
    _add_products!(op.product, system.matrix_free_terms, op.state, op.scaled_input, system.layout, op.time)
    corrections && _add_corrections!(op.product, system.correction_terms, op.state, op.scaled_input, system.layout,
                                     op.time)
    @. y -= op.row_weights * op.product
    return y
end

# the blocks of the stored matrix, their positions in its nonzeros, and the factorizations
struct BlockFactors{T,F}
    ranges::Vector{UnitRange{Int}}
    matrices::Vector{SparseMatrixCSC{T,Int}}
    positions::Vector{Vector{Int}}   # per block: index in the full matrix's nzval of each entry
    factorizations::Vector{F}
    input::Vector{Vector{T}}
    output::Vector{Vector{T}}
end

function _preconditioner_blocks(layout::UnknownLayout)
    ranges = [radius_block_range(layout, ir) for ir in 1:radial_cell_count(layout)]
    isempty(scalar_block_range(layout)) || push!(ranges, scalar_block_range(layout))
    covered = falses(length(layout))
    for range in ranges
        covered[range] .= true
    end
    all(covered) || error("the radius and scalar blocks do not cover the layout")
    return ranges
end

# block `range` of `matrix` with the same entry order as a column-wise traversal
function _extract_block(matrix::SparseMatrixCSC{T}, range::UnitRange{Int}) where {T}
    colptr = [1]
    rowval, positions = Int[], Int[]
    for j in range
        for k in nzrange(matrix, j)
            row = matrix.rowval[k]
            row in range || continue
            push!(rowval, row - first(range) + 1)
            push!(positions, k)
        end
        push!(colptr, length(rowval) + 1)
    end
    block = SparseMatrixCSC(length(range), length(range), colptr, rowval, matrix.nzval[positions])
    return block, positions
end

_factorize(::RadiusBlockLU, block) = lu(block; check=false)
_factorize(p::RadiusBlockILU, block) = ilu(block; τ=p.drop_tolerance)

function BlockFactors(preconditioner::BlockPreconditioner, matrix::SparseMatrixCSC{T}, layout) where {T}
    ranges = _preconditioner_blocks(layout)
    extracted = [_extract_block(matrix, range) for range in ranges]
    matrices, positions = first.(extracted), last.(extracted)
    factorizations = [_factorize(preconditioner, block) for block in matrices]
    return BlockFactors(ranges, matrices, positions, factorizations,
                        [zeros(T, length(r)) for r in ranges], [zeros(T, length(r)) for r in ranges])
end

_refactorize!(factors::BlockFactors, ::RadiusBlockLU, k) =
    (lu!(factors.factorizations[k], factors.matrices[k]; check=false); issuccess(factors.factorizations[k]))

function _refactorize!(factors::BlockFactors, p::RadiusBlockILU, k)
    factorization = ilu(factors.matrices[k]; τ=p.drop_tolerance)
    factors.factorizations[k] = factorization
    # a zero pivot shows up as non-finite entries of the factors
    return all(isfinite, factorization.L.nzval) && all(isfinite, factorization.U.nzval)
end

# copy the current values of `matrix` into the blocks and factorize them; false when a
# block is singular
function factorize!(factors::BlockFactors, preconditioner::BlockPreconditioner, matrix::SparseMatrixCSC)
    for k in eachindex(factors.matrices)
        block, positions = factors.matrices[k], factors.positions[k]
        @inbounds for m in eachindex(positions)
            block.nzval[m] = matrix.nzval[positions[m]]
        end
        _refactorize!(factors, preconditioner, k) || return false
    end
    return true
end

function LinearAlgebra.ldiv!(y::AbstractVector, factors::BlockFactors, x::AbstractVector)
    for k in eachindex(factors.ranges)
        range, input, output = factors.ranges[k], factors.input[k], factors.output[k]
        copyto!(input, 1, x, first(range), length(range))
        ldiv!(output, factors.factorizations[k], input)
        copyto!(y, first(range), output, 1, length(range))
    end
    return y
end

mutable struct GMRESCache{T,O<:NewtonOperator{T},F<:BlockFactors{T},W,P}
    const solver::GMRES{P}
    const operator::O
    const factors::F
    const workspace::W
    iterations::Int      # Krylov iterations of the last solve
    updates::Int         # preconditioner factorizations in the last solve
    factorized::Bool
    factorized_dt::Float64
    last_iterations::Int
    factorization_seconds::Float64   # in the last solve
    matrix_unchanged::Bool           # the stored matrix of the next solve is that of the last one
end

matrix_unchanged!(cache::GMRESCache, unchanged::Bool) = (cache.matrix_unchanged = unchanged; nothing)

krylov_iterations(cache::GMRESCache) = cache.iterations
preconditioner_updates(cache::GMRESCache) = cache.updates
factorization_time(cache::GMRESCache) = cache.factorization_seconds
linear_tolerance(cache::GMRESCache) = cache.solver.relative_tolerance
newton_operator_exact(cache::GMRESCache) = cache.solver.operator === :exact

function linear_solver_cache(solver::GMRES, matrix::SparseMatrixCSC{T}, system::EquationSystem,
                             row_weights, scales) where {T}
    solver.preconditioner_refresh in (:every_iteration, :every_step, :on_demand) ||
        throw(ArgumentError("unknown preconditioner_refresh $(solver.preconditioner_refresh)"))
    solver.operator in (:exact, :stored) || throw(ArgumentError("operator must be :exact or :stored"))
    n = size(matrix, 1)
    operator = NewtonOperator(matrix, system, row_weights, scales, zeros(T, n), zeros(T, n), zeros(T, n),
                              solver.operator === :exact, zero(T))
    factors = BlockFactors(solver.preconditioner, matrix, system.layout)
    workspace = GmresWorkspace(n, n, Vector{T}; memory=solver.restart_length)
    return GMRESCache(solver, operator, factors, workspace, 0, 0, false, NaN, 0, 0.0, false)
end

function _needs_factorization(cache::GMRESCache, dt, iteration)
    cache.factorized || return true
    refresh = cache.solver.preconditioner_refresh
    # the factors of an unchanged matrix are those of a new factorization
    refresh === :every_iteration && return !cache.matrix_unchanged
    refresh === :every_step && return iteration == 1
    # steps between the points of a range differ in the last bits, which is no reason to
    # factorize again
    return !isapprox(dt, cache.factorized_dt; rtol=1e-8) ||
           cache.last_iterations > cache.solver.refresh_iterations
end

function _factorize!(cache::GMRESCache, matrix, dt)
    cache.updates += 1
    started = time_ns()
    cache.factorized = factorize!(cache.factors, cache.solver.preconditioner, matrix)
    cache.factorization_seconds += (time_ns() - started) / 1e9
    cache.factorized_dt = dt
    return cache.factorized
end

function _gmres!(cache::GMRESCache, b)
    solver = cache.solver
    gmres!(cache.workspace, cache.operator, b; N=cache.factors, ldiv=true, restart=true, atol=0.0,
           rtol=solver.relative_tolerance, itmax=solver.max_iterations)
    stats = cache.workspace.stats
    cache.iterations += stats.niter
    cache.last_iterations = stats.niter
    return stats.solved
end

function solve!(x::AbstractVector, cache::GMRESCache, matrix::SparseMatrixCSC, b::AbstractVector,
                u, t, dt, iteration)
    operator = cache.operator
    copyto!(operator.state, u)
    operator.time = t
    cache.iterations = 0
    cache.updates = 0
    cache.factorization_seconds = 0.0
    fresh = _needs_factorization(cache, dt, iteration)
    fresh && (_factorize!(cache, matrix, dt) || return :singular_matrix)
    solved = _gmres!(cache, b)
    if !solved && !fresh
        # a preconditioner from an earlier iteration may have gone stale
        _factorize!(cache, matrix, dt) || return :singular_matrix
        solved = _gmres!(cache, b)
    end
    solved || return :linear_solve_failed
    copyto!(x, cache.workspace.x)
    return :ok
end
