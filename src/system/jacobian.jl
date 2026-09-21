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
    n1, n2, n3 = size(grid)
    L = LinearIndices((n1, n2, n3))
    I = Int[]
    J = Int[]
    sizehint!(I, 11 * n1 * n2 * n3)
    sizehint!(J, 11 * n1 * n2 * n3)
    for ir in 1:n3, j in 1:n2, i in 1:n1
        row = L[i, j, ir]
        for dj in -1:1, di in -1:1
            (1 <= i + di <= n1 && 1 <= j + dj <= n2) || continue
            push!(I, row); push!(J, L[i+di, j+dj, ir])
        end
        for dr in (-1, 1)
            1 <= ir + dr <= n3 || continue
            push!(I, row); push!(J, L[i, j, ir+dr])
        end
    end
    return sparse(I, J, trues(length(I)), n1 * n2 * n3, n1 * n2 * n3)
end

"""
    SparseJacobian(F!, grid; backend=sparse_ad_backend(pattern=stencil_pattern(grid)))

Prepared sparse Jacobian of an in-place residual `F!(dF, f)` acting on arrays shaped
like `grid`. Sparsity pattern and coloring are computed once; `jacobian!(J, sj, f)`
then re-evaluates the values.

`F!` must contain only stencil-local terms. Terms with global momentum coupling, such
as the knock-on operator, make the Jacobian dense per radius and are applied
matrix-free in the Krylov solver instead.
"""
struct SparseJacobian{P,B,F}
    F!::F
    prep::P
    backend::B
    dims::NTuple{3,Int}
end

function SparseJacobian(F!, grid::KineticGrid{T};
                        backend=sparse_ad_backend(pattern=stencil_pattern(grid))) where {T}
    dims = size(grid)
    f0 = zeros(T, dims)
    y = zeros(T, dims)
    prep = DI.prepare_jacobian(F!, y, backend, f0)
    return SparseJacobian(F!, prep, backend, dims)
end

"""
    jacobian(sj, f)

Sparse Jacobian of `sj.F!` at `f` (new matrix).
"""
function jacobian(sj::SparseJacobian, f::AbstractArray{T,3}) where {T}
    y = similar(f)
    return DI.jacobian(sj.F!, y, sj.prep, sj.backend, f)
end

"""
    jacobian!(J, sj, f)

Overwrite the values of `J` (with the pattern from `jacobian(sj, f)`) at `f`.
"""
function jacobian!(J::SparseMatrixCSC, sj::SparseJacobian, f::AbstractArray{T,3}) where {T}
    y = similar(f)
    DI.jacobian!(sj.F!, y, J, sj.prep, sj.backend, f)
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
    P = stencil_pattern(op.grid)
    return all(P[findall(!iszero, M)])
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
    F! = (df, f) -> (fill!(df, zero(eltype(df))); apply!(df, op, f))
    sj = SparseJacobian(F!, op.grid; backend)
    return jacobian(sj, zeros(eltype(op.Ap), size(op.grid)))
end
