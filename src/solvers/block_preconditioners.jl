# Preconditioners of GMRES: the stored Newton matrix in blocks, one per radius and one for
# the scalar unknowns, each factorized on its own. A file of its own, so that FluidSchurLU
# follows the types it extends.

using IncompleteLU: ILUFactorization, ilu

"""
    BlockPreconditioner

Preconditioners for [`GMRES`](@ref): the scaled stored Newton matrix restricted to the
blocks of [`radius_block_range`](@ref) (all unknowns at one radius) and
[`scalar_block_range`](@ref), each block factorized on its own. The subtypes are
[`RadiusBlockLU`](@ref), [`RadiusBlockILU`](@ref) and [`FluidSchurLU`](@ref).
"""
abstract type BlockPreconditioner end

"""
    RadiusBlockLU()

[`BlockPreconditioner`](@ref) that factorizes each block exactly (UMFPACK, symbolic
analysis reused). Couplings between blocks (radial transport, scalar unknowns) and the
matrix-free terms are left out.
"""
struct RadiusBlockLU <: BlockPreconditioner end

"""
    RadiusBlockILU(; drop_tolerance=1e-4)

[`BlockPreconditioner`](@ref) that computes an incomplete LU factorization of each block
(Crout ILU of IncompleteLU.jl), which drops entries of the factors below `drop_tolerance`
in magnitude. Couplings between blocks (radial transport, scalar unknowns) and the
matrix-free terms are left out.
"""
struct RadiusBlockILU <: BlockPreconditioner
    drop_tolerance::Float64
end
RadiusBlockILU(; drop_tolerance::Real = 1e-4) = RadiusBlockILU(Float64(drop_tolerance))

# the blocks of the stored matrix, their positions in its nonzeros, and the factorizations
struct BlockFactors{T, F}
    ranges::Vector{UnitRange{Int}}
    matrices::Vector{SparseMatrixCSC{T, Int}}
    positions::Vector{Vector{Int}}   # per block: index in the full matrix's nzval of each entry
    factorizations::Vector{F}
    input::Vector{Vector{T}}
    output::Vector{Vector{T}}
    low_rank::Vector{Union{Nothing, LowRankCorrection{T}}}   # per block: Woodbury update
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
    block = SparseMatrixCSC(
        length(range), length(range), colptr, rowval, matrix.nzval[positions])
    return block, positions
end

_factorize(::RadiusBlockLU, block) = lu(block; check = false)
_factorize(p::RadiusBlockILU, block) = ilu(block; τ = p.drop_tolerance)

# the factors of a preconditioner for the stored matrix
function preconditioner_factors(preconditioner::BlockPreconditioner, matrix::SparseMatrixCSC, layout)
    BlockFactors(preconditioner, matrix, layout)
end

function BlockFactors(preconditioner::BlockPreconditioner, matrix::SparseMatrixCSC{T}, layout) where {T}
    ranges = _preconditioner_blocks(layout)
    extracted = [_extract_block(matrix, range) for range in ranges]
    matrices, positions = first.(extracted), last.(extracted)
    factorizations = [_factorize(preconditioner, block) for block in matrices]
    return BlockFactors(ranges, matrices, positions, factorizations,
        [zeros(T, length(r)) for r in ranges], [zeros(T, length(r)) for r in ranges],
        Union{Nothing, LowRankCorrection{T}}[nothing for _ in ranges])
end

function _refactorize!(factors::BlockFactors, ::RadiusBlockLU, k)
    (lu!(factors.factorizations[k], factors.matrices[k]; check = false);
        issuccess(factors.factorizations[k]))
end

function _refactorize!(factors::BlockFactors, p::RadiusBlockILU, k)
    factorization = ilu(factors.matrices[k]; τ = p.drop_tolerance)
    factors.factorizations[k] = factorization
    # a zero pivot shows up as non-finite entries of the factors
    return all(isfinite, factorization.L.nzval) && all(isfinite, factorization.U.nzval)
end

# copy the current values of `matrix` into the blocks and factorize them, with the Woodbury
# update of `low_rank.blocks` (row weights and scales `low_rank.row_weights`,
# `low_rank.scales`) inside each block; false when a block
# or a capacitance matrix is singular
function factorize!(factors::BlockFactors{T},
        preconditioner::BlockPreconditioner, matrix::SparseMatrixCSC,
        low_rank = nothing) where {T}
    for k in eachindex(factors.matrices)
        block, positions = factors.matrices[k], factors.positions[k]
        @inbounds for m in eachindex(positions)
            block.nzval[m] = matrix.nzval[positions[m]]
        end
        _refactorize!(factors, preconditioner, k) || return false
        factors.low_rank[k] = nothing
        low_rank === nothing && continue
        scaled = _scaled_low_rank(
            low_rank.blocks, factors.ranges[k], low_rank.row_weights, low_rank.scales, T)
        scaled === nothing && continue
        factorization = factors.factorizations[k]
        factors.low_rank[k] = _low_rank_correction(scaled..., (y, x) -> ldiv!(y, factorization, x))
        factors.low_rank[k] === nothing && return false
    end
    return true
end

function LinearAlgebra.ldiv!(y::AbstractVector, factors::BlockFactors, x::AbstractVector)
    for k in eachindex(factors.ranges)
        range, input, output = factors.ranges[k], factors.input[k], factors.output[k]
        copyto!(input, 1, x, first(range), length(range))
        ldiv!(output, factors.factorizations[k], input)
        factors.low_rank[k] === nothing ||
            _apply_low_rank_correction!(output, factors.low_rank[k])
        copyto!(y, first(range), output, 1, length(range))
    end
    return y
end
