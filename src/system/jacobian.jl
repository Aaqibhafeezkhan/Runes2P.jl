using ADTypes: AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
import ForwardDiff
import DifferentiationInterface as DI
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using SparseArrays: SparseMatrixCSC, sparse

"""
    sparse_ad_backend(; pattern=nothing)

ForwardDiff with greedy column coloring. The sparsity pattern is taken from `pattern`
when given, otherwise detected by operator tracing (slow for large grids). The number
of Jacobian-vector products equals the number of colors, which stays bounded as the
grid is refined for local stencils.
"""
function sparse_ad_backend(; pattern=nothing)
    detector = pattern === nothing ? TracerSparsityDetector() :
               KnownJacobianSparsityDetector(pattern)
    return AutoSparse(AutoForwardDiff(); sparsity_detector=detector,
                      coloring_algorithm=GreedyColoringAlgorithm())
end

"""
    stencil_pattern(grid)

Sparsity pattern of the local transport operators: the 3×3 `(p, ξ)` neighbourhood
(including the p-ξ cross terms) and the two radial neighbours of each cell.
"""
function stencil_pattern(grid::KineticGrid)
    np, nxi, nr = size(grid)
    index = LinearIndices((np, nxi, nr))
    rows = Int[]
    columns = Int[]
    sizehint!(rows, 11 * np * nxi * nr)
    sizehint!(columns, 11 * np * nxi * nr)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        row = index[i, j, ir]
        for dj in -1:1, di in -1:1
            (1 <= i + di <= np && 1 <= j + dj <= nxi) || continue
            push!(rows, row); push!(columns, index[i+di, j+dj, ir])
        end
        for dr in (-1, 1)
            1 <= ir + dr <= nr || continue
            push!(rows, row); push!(columns, index[i, j, ir+dr])
        end
    end
    return sparse(rows, columns, trues(length(rows)), np * nxi * nr, np * nxi * nr)
end

"""
    SparseJacobian(residual!, grid; backend=sparse_ad_backend(pattern=stencil_pattern(grid)))

Prepared sparse Jacobian of an in-place residual `residual!(dF, f)` acting on arrays
shaped like `grid`. Sparsity pattern and coloring are computed once;
`jacobian!(J, sparse_jacobian, f)` then re-evaluates the values.

`residual!` must contain only stencil-local terms. Terms with global momentum coupling,
such as the knock-on operator, make the Jacobian dense per radius and are applied
matrix-free in the Krylov solver instead.
"""
struct SparseJacobian{P,B,F}
    residual!::F
    preparation::P
    backend::B
    dimensions::NTuple{3,Int}
end

function SparseJacobian(residual!, grid::KineticGrid{T};
                        backend=sparse_ad_backend(pattern=stencil_pattern(grid))) where {T}
    dimensions = size(grid)
    f0 = zeros(T, dimensions)
    y = zeros(T, dimensions)
    preparation = DI.prepare_jacobian(residual!, y, backend, f0)
    return SparseJacobian(residual!, preparation, backend, dimensions)
end

"""
    jacobian(sparse_jacobian, f)

Sparse Jacobian of `sparse_jacobian.residual!` at `f` (new matrix).
"""
function jacobian(sj::SparseJacobian, f::AbstractArray{T,3}) where {T}
    y = similar(f)
    return DI.jacobian(sj.residual!, y, sj.preparation, sj.backend, f)
end

"""
    jacobian!(J, sparse_jacobian, f)

Overwrite the values of `J` (with the pattern from `jacobian(sparse_jacobian, f)`) at `f`.
"""
function jacobian!(J::SparseMatrixCSC, sj::SparseJacobian, f::AbstractArray{T,3}) where {T}
    y = similar(f)
    DI.jacobian!(sj.residual!, y, J, sj.preparation, sj.backend, f)
    return J
end

"""
    is_local(op)

Whether `op` couples each cell only to its `stencil_pattern` neighbours. Only local
operators may be assembled with the stencil pattern: with a known pattern, column
coloring silently adds entries outside the pattern onto entries inside it. Operators
default to non-local; local operator types opt in.
"""
is_local(op) = false
is_local(::AdvectionDiffusion) = true

"""
    check_stencil_pattern(op)

Trace `op` on its grid and return `true` if every traced Jacobian entry lies inside
`stencil_pattern(op.grid)`. Slow (operator tracing); intended for tests of new local
operators.
"""
function check_stencil_pattern(op)
    M = operator_matrix(op; backend=sparse_ad_backend(), check_local=false)
    pattern = stencil_pattern(op.grid)
    return all(pattern[findall(!iszero, M)])
end

"""
    operator_matrix(op; backend=sparse_ad_backend(pattern=stencil_pattern(op.grid)))

Sparse matrix `M` with `vec(apply(op, f)) = M * vec(f) + c` for an operator `op` that is
linear in `f` up to a constant source `c`.
"""
function operator_matrix(op; backend=sparse_ad_backend(pattern=stencil_pattern(op.grid)),
                         check_local::Bool=true)
    if check_local && !is_local(op)
        throw(ArgumentError("$(typeof(op)) is not a local operator; assembling it with the " *
                            "stencil pattern would silently corrupt the matrix. Apply it " *
                            "matrix-free instead."))
    end
    residual! = (df, f) -> (fill!(df, zero(eltype(df))); apply!(df, op, f))
    sj = SparseJacobian(residual!, op.grid; backend)
    return jacobian(sj, zeros(eltype(op.advection_p), size(op.grid)))
end
