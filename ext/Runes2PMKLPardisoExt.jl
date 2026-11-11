"""
Intel MKL PARDISO as the linear solver [`MKLPardiso`](@ref) of the Newton iterations,
through Pardiso.jl.
"""
module Runes2PMKLPardisoExt

using Runes2P
using Runes2P: MKLPardiso, LowRankCorrection, _low_rank_blocks, _scaled_low_rank,
               _low_rank_correction, _apply_low_rank_correction!
using Pardiso: Pardiso
using SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros
using LinearAlgebra: mul!, norm

# The state of PARDISO for one pattern. PARDISO gets the column pointers and row indices of
# the pattern, in its own integer type, as the row pointers and column indices of the
# transposed matrix, with the values of the matrix of each solve.
mutable struct MKLPardisoCache{I <: Integer, S, W}
    settings::MKLPardiso
    pardiso::Pardiso.MKLPardisoSolver
    column_pointers::Vector{I}
    row_indices::Vector{I}
    system::S
    row_weights::W
    scales::W
    low_rank::Union{Nothing, LowRankCorrection}   # Woodbury update of the factorization
    residual::Vector{Float64}       # work vector of the relative residual
    relative_residual::Float64      # of the last solve, before the Woodbury update
    analyzed::Bool                  # the ordering and the matching of the pattern exist
    factorization_seconds::Float64
    factorized::Bool          # the factorization is that of the matrix of the last solve
    matrix_unchanged::Bool    # the matrix of the next solve is that of the last one
    factorizations::Int       # numerical factorizations since the cache was made
end

function Runes2P.linear_solver_cache(
        solver::MKLPardiso, matrix::SparseMatrixCSC, system, row_weights, scales)
    eltype(matrix) === Float64 ||
        throw(ArgumentError("MKLPardiso solves a matrix of Float64, not of $(eltype(matrix))"))
    Pardiso.mkl_is_available() ||
        throw(ArgumentError("MKLPardiso needs MKL, which Pardiso.jl did not load: Pardiso.mkl_is_available() is false"))
    # PARDISO's integer type: Pardiso.jl takes 64 bits only when Julia's BLAS vendor is MKL,
    # and the vendor of current Julia is libblastrampoline, so it is 32 bits. The last column
    # pointer is the count of nonzeros plus one.
    index_type = Pardiso.MklInt
    length(nonzeros(matrix)) < typemax(index_type) ||
        throw(ArgumentError("the matrix has more nonzeros than the integers of PARDISO hold"))
    pardiso = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(pardiso, Pardiso.REAL_NONSYM)
    Pardiso.pardisoinit(pardiso)
    # solve the transposed system of the matrix PARDISO reads, the transpose of `matrix`
    Pardiso.set_iparm!(pardiso, 12, 2)
    Pardiso.set_iparm!(pardiso, 8, solver.refinement_steps)
    column_pointers = Vector{index_type}(getcolptr(matrix))
    row_indices = Vector{index_type}(rowvals(matrix))
    return MKLPardisoCache(
        solver, pardiso, column_pointers, row_indices, system, row_weights,
        scales, nothing, zeros(size(matrix, 1)), 0.0, false, 0.0, false, false, 0)
end

function Runes2P.matrix_unchanged!(cache::MKLPardisoCache, unchanged::Bool)
    cache.matrix_unchanged = unchanged
    return nothing
end

Runes2P.krylov_iterations(::MKLPardisoCache) = 0
Runes2P.preconditioner_updates(::MKLPardisoCache) = 0
Runes2P.factorization_time(cache::MKLPardisoCache) = cache.factorization_seconds
Runes2P.linear_tolerance(cache::MKLPardisoCache) = cache.relative_residual
Runes2P.newton_operator_exact(::MKLPardisoCache) = false

"""
    MAX_RELATIVE_RESIDUAL = 1e-6

The largest relative residual ``\\|b - A x\\|/\\|b\\|`` of a solve of [`MKLPardiso`](@ref)
before the Woodbury update: a solve above it returns `:singular_matrix`, as
[`SparseLU`](@ref) does on a zero pivot, so that the Newton iteration ends and the stepper
retries the step shorter. PARDISO perturbs small pivots instead of failing, so a singular or
nearly singular matrix gives a finite solution with a large residual. 1e-6 is a choice of
this package (see the docs page *Choices of this package*): three orders of magnitude above
the residuals of PARDISO on the disruption case, ``10^{-13}`` to ``10^{-9}``, and four or
more below those of singular matrices, ``10^{-2}`` to ``10``.
"""
const MAX_RELATIVE_RESIDUAL = 1e-6

const PHASE_NAMES = Dict(Pardiso.ANALYSIS => "the analysis (phase 11)",
    Pardiso.NUM_FACT => "the numeric factorization (phase 22)",
    Pardiso.SOLVE_ITERATIVE_REFINE => "the solve (phase 33)")

# One call of PARDISO in `phase`: `:ok`, or `:singular_matrix` for MKL's errors -4 and -7.
# Pardiso.jl does not pass on the number of an error but throws an exception for it: -4 (a
# zero pivot, or a failed numeric factorization or iterative refinement) comes as a
# `PardisoPosDefException`, -7 as a `PardisoException` with the text "Diagonal matrix is
# singular"; no other error of MKL has a text with "singular" in Pardiso.jl. Other errors of
# MKL throw with the meaning that Pardiso.jl gives them, -8 with the limit it stands for.
# The texts are those of `check_error` of Pardiso.jl 1.1.2, which is not public: the testset
# "the errors of MKL that the extension tells apart" of test_mkl_pardiso.jl pins them.
function _pardiso!(cache::MKLPardisoCache, phase, x, pardiso_matrix, b)
    Pardiso.set_phase!(cache.pardiso, phase)
    try
        Pardiso.pardiso(cache.pardiso, x, pardiso_matrix, b)
    catch err
        if err isa Pardiso.PardisoPosDefException
            return :singular_matrix
        end
        if err isa Pardiso.PardisoException && occursin("singular", err.info)
            return :singular_matrix
        end
        if err isa Pardiso.PardisoException && occursin("32-bit integer overflow", err.info)
            throw(ErrorException("MKL PARDISO failed in $(PHASE_NAMES[phase]): $(err.info) " *
                                 "PARDISO's 32-bit integers hold at most 2^31 - 1 nonzeros of the factors"))
        end
        if err isa Pardiso.PardisoException
            throw(ErrorException("MKL PARDISO failed in $(PHASE_NAMES[phase]): $(err.info)"))
        end
        rethrow()
    end
    return :ok
end

# a failed solve: the factorization is dropped, and so are the ordering and the matching,
# which the next factorization computes again from its own matrix, since those of another
# matrix of the pattern can spoil its solves
function _singular!(cache::MKLPardisoCache)
    cache.factorized = false
    cache.analyzed = false
    return :singular_matrix
end

# the Woodbury update of the whole matrix for the low-rank matrix-free terms at (u, t), as
# that of SparseLU, with the solves of PARDISO; false when a solve or the update fails
function _update_low_rank!(cache::MKLPardisoCache, pardiso_matrix, u, t)
    cache.low_rank = nothing
    system = cache.system
    isempty(system.matrix_free_terms) && return true
    blocks = _low_rank_blocks(system, u, t)
    scaled = _scaled_low_rank(
        blocks, 1:length(u), cache.row_weights, cache.scales, eltype(cache.scales))
    scaled === nothing && return true
    all_solved = Ref(true)
    function solve_with_factors!(y, x)
        status = _pardiso!(cache, Pardiso.SOLVE_ITERATIVE_REFINE, y, pardiso_matrix, x)
        if status !== :ok
            all_solved[] = false
        end
        return y
    end
    cache.low_rank = _low_rank_correction(scaled..., solve_with_factors!)
    return all_solved[] && cache.low_rank !== nothing
end

function Runes2P.solve!(
        x::AbstractVector, cache::MKLPardisoCache, matrix::SparseMatrixCSC, b::AbstractVector,
        u, t, dt, iteration)
    length(nonzeros(matrix)) == length(cache.row_indices) ||
        throw(DimensionMismatch("the matrix has another pattern than that of the cache of MKLPardiso"))
    cache.factorization_seconds = 0.0
    Pardiso.set_nprocs!(cache.pardiso, cache.settings.threads)
    n = size(matrix, 1)
    # the matrix in PARDISO's integer type, equal to `matrix`: PARDISO reads its compressed
    # columns as the compressed rows of the transpose
    pardiso_matrix = SparseMatrixCSC(
        n, n, cache.column_pointers, cache.row_indices, nonzeros(matrix))
    if !(cache.factorized && cache.matrix_unchanged)
        # the analysis counts in the factorization time of the solve
        started = time_ns()
        # the ordering and the matching from the values of the first matrix of the pattern,
        # and again after a failed solve
        if !cache.analyzed
            status = _pardiso!(cache, Pardiso.ANALYSIS, x, pardiso_matrix, b)
            if status !== :ok
                return _singular!(cache)
            end
            cache.analyzed = true
        end
        status = _pardiso!(cache, Pardiso.NUM_FACT, x, pardiso_matrix, b)
        cache.factorization_seconds = (time_ns() - started) / 1e9
        cache.factorizations += 1
        cache.factorized = if status === :ok
            _update_low_rank!(cache, pardiso_matrix, u, t)
        else
            false
        end
        if !cache.factorized
            return _singular!(cache)
        end
    end
    status = _pardiso!(cache, Pardiso.SOLVE_ITERATIVE_REFINE, x, pardiso_matrix, b)
    if status !== :ok
        return _singular!(cache)
    end
    # the relative residual of every solve, with refinement too: the refinement stops at its
    # cap and promises no residual
    residual = cache.residual
    mul!(residual, matrix, x)
    @. residual = b - residual
    right_hand_side_norm = norm(b)
    residual_norm = norm(residual)
    cache.relative_residual = if iszero(right_hand_side_norm)
        residual_norm
    else
        residual_norm / right_hand_side_norm
    end
    # a residual above the bound fails the solve, and linear_tolerance keeps it; so does a
    # residual that is not a number, since the matrix and the right-hand side are finite
    if !(cache.relative_residual <= MAX_RELATIVE_RESIDUAL)
        return _singular!(cache)
    end
    if cache.low_rank !== nothing
        _apply_low_rank_correction!(x, cache.low_rank)
    end
    return :ok
end

end
