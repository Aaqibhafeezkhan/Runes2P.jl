using ADTypes: AutoForwardDiff, AutoSparse
import ForwardDiff
import DifferentiationInterface as DI
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using SparseArrays: SparseMatrixCSC

"""
    sparse_ad_backend()

ForwardDiff with sparsity detection and greedy column coloring. The number of
Jacobian-vector products equals the number of colors, which stays bounded as the
grid is refined for local stencils.
"""
sparse_ad_backend() = AutoSparse(AutoForwardDiff();
                                 sparsity_detector=TracerSparsityDetector(),
                                 coloring_algorithm=GreedyColoringAlgorithm())

"""
    SparseJacobian(F!, grid; backend=sparse_ad_backend())

Prepared sparse Jacobian of an in-place residual `F!(dF, f)` acting on arrays shaped
like `grid`. Sparsity pattern and coloring are computed once; `jacobian!(J, sj, f)`
then re-evaluates the values.
"""
struct SparseJacobian{P,B,F}
    F!::F
    prep::P
    backend::B
    dims::NTuple{3,Int}
end

function SparseJacobian(F!, grid::KineticGrid{T}; backend=sparse_ad_backend()) where {T}
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
    operator_matrix(op)

Sparse matrix `M` with `vec(apply(op, f)) = M * vec(f)` for a linear operator `op`.
"""
function operator_matrix(op)
    F! = (df, f) -> (fill!(df, zero(eltype(df))); apply!(df, op, f))
    sj = SparseJacobian(F!, op.grid)
    return jacobian(sj, zeros(eltype(op.Ap), size(op.grid)))
end
