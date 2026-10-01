using ADTypes: AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
import ForwardDiff
_value(x::ForwardDiff.Dual) = _value(ForwardDiff.value(x))
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
    stencil_pattern(grid; p_width=1, xi_width=1, r_width=1)
    stencil_pattern(op::AdvectionDiffusion)

Sparsity pattern of the local transport operators: the 3×3 `(p, ξ)` neighbourhood
(including the p-ξ cross terms), the cells up to `p_width` away in p and `xi_width` away
in ξ along the axes (2 for the four-point schemes), the cells up to `r_width` radii away,
and the radial neighbours of each
cell (the cell of the same pitch at the two neighbouring radii, or with different pitch
grids the cells that overlap it). For an operator, the widths follow its schemes; with `p_upper_boundary =
:extrapolated_flux` or `:extrapolated_flux_consistent` the last momentum row is
additionally coupled up to 3 cells away in p along `j` (the extrapolated flux uses the
last two interior faces), and, for `:extrapolated_flux_consistent` only,
diagonally to `(np - 2, j ± 1)` through its second term's p-ξ cross contribution
(`:extrapolated_flux`'s second term has none).
"""
stencil_pattern(op::AdvectionDiffusion) =
    stencil_pattern(op.grid; p_width=stencil_half_width(op.p_scheme), xi_width=stencil_half_width(op.xi_scheme),
                    r_width=stencil_half_width(op.r_scheme), p_upper_boundary=op.p_upper_boundary)

function stencil_pattern(grid::KineticGrid; p_width::Integer=1, xi_width::Integer=1, r_width::Integer=1,
                         p_upper_boundary::Symbol=:f_zero)
    np, nxi, nr = size(grid)
    index = LinearIndices((np, nxi, nr))
    rows = Int[]
    columns = Int[]
    sizehint!(rows, 11 * np * nxi * nr)
    sizehint!(columns, 11 * np * nxi * nr)
    for ir in 1:nr, j in 1:nxi, i in 1:np
        row = index[i, j, ir]
        _visit_stencil(grid, i, j, ir; p_width, xi_width, r_width, p_upper_boundary) do i2, j2, ir2
            push!(rows, row)
            push!(columns, index[i2, j2, ir2])
        end
    end
    # the extrapolated flux repeats cells of the 3 × 3 block
    return sparse(rows, columns, trues(length(rows)), np * nxi * nr, np * nxi * nr, |)
end

# Calls `visit(i2, j2, ir2)` for every cell that the flux divergence of cell (i, j, ir)
# reads: the 3 × 3 block in p and ξ, the second neighbours of the four-point schemes along
# the axes, the radial neighbours (the cells of the neighbouring radii that overlap pitch
# cell j), the second radial neighbours of the four-point schemes, and for the extrapolated
# flux at p_max the last row up to 3 cells in p (with the p-ξ cross term of
# `:extrapolated_flux_consistent`). A cell may be visited twice.
# Shared by `stencil_pattern` and the sparsity of `FokkerPlanckTerm`.
function _visit_stencil(visit, grid::KineticGrid, i, j, ir; p_width::Integer, xi_width::Integer, r_width::Integer,
                        p_upper_boundary::Symbol)
    np, nxi, nr = size(grid)
    for dj in -1:1, di in -1:1
        (1 <= i + di <= np && 1 <= j + dj <= nxi) || continue
        visit(i + di, j + dj, ir)
    end
    for d in 2:p_width, s in (-d, d)
        1 <= i + s <= np && visit(i + s, j, ir)
    end
    for d in 2:xi_width, s in (-d, d)
        1 <= j + s <= nxi && visit(i, j + s, ir)
    end
    for dr in (-1, 1), neighbour in radial_neighbour_pitch_cells(grid, ir, j, dr)
        visit(i, neighbour, ir + dr)
    end
    # the second radial neighbours of the four-point schemes, which need one pitch grid
    for d in 2:r_width, s in (-d, d)
        1 <= ir + s <= nr && visit(i, j, ir + s)
    end
    if i == np && p_upper_boundary in (:extrapolated_flux, :extrapolated_flux_consistent)
        for d in 1:3
            np - d >= 1 && visit(np - d, j, ir)
        end
        if p_upper_boundary === :extrapolated_flux_consistent
            for s in (-1, 1)
                np >= 2 && 1 <= j + s <= nxi && visit(np - 2, j + s, ir)
            end
        end
    end
    return nothing
end

"""
    SparseJacobian(residual!, grid; backend=sparse_ad_backend(pattern=stencil_pattern(grid)))

Prepared sparse Jacobian of an in-place residual `residual!(dF, f)` acting on arrays
shaped like `grid`. Sparsity pattern and coloring are computed once;
`jacobian!(J, sparse_jacobian, f)` then re-evaluates the values.

`residual!` must contain only stencil-local terms. Terms with global momentum coupling,
such as the knock-on operator, make the Jacobian dense per radius and are applied
matrix-free in the Krylov solver instead. The default pattern has width 1; for an operator
with four-point schemes pass `backend=sparse_ad_backend(pattern=stencil_pattern(op))`, or
the coloring folds the second neighbours silently onto the pattern.
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

Return `true` if every entry of the Jacobian of `op` lies inside `stencil_pattern(op)`.
A linear operator is traced on its grid; one with a scheme that is nonlinear in f gets its
dense Jacobian by ForwardDiff at a random positive state (tracing cannot follow the
branches of a limiter). Slow; intended for tests of local operators on small grids.
"""
function check_stencil_pattern(op)
    pattern = stencil_pattern(op)
    if is_linear(op)
        M = operator_matrix(op; backend=sparse_ad_backend(), check_local=false)
        return all(pattern[findall(!iszero, M)])
    end
    f0 = 0.5 .+ rand(size(op.grid)...)
    residual = x -> (y = zeros(eltype(x), size(op.grid)); apply!(y, op, reshape(x, size(op.grid))); vec(y))
    M = ForwardDiff.jacobian(residual, vec(f0))
    return all(pattern[findall(!iszero, M)])
end

"""
    is_linear(op)
    is_linear(term)

Whether the operator is linear in f (up to a constant source); false for flux-limiter
schemes, whose face values depend on f. For an [`EquationTerm`](@ref): whether the term
is linear in the state up to a constant (`false` by default; terms opt in), which lets
[`NewtonSolver`](@ref) accept a step of a linear system on the update test alone.
"""
is_linear(op::AdvectionDiffusion) = !(is_nonlinear_scheme(op.p_scheme) || is_nonlinear_scheme(op.xi_scheme) ||
                                      is_nonlinear_scheme(op.r_scheme))

"""
    operator_matrix(op; backend=sparse_ad_backend(pattern=stencil_pattern(op)), check_local=true)

Sparse matrix `M` with `vec(apply(op, f)) = M * vec(f) + c` for an operator `op` that is
linear in `f` up to a constant source `c`. The default pattern follows the operator's
schemes and boundary conditions ([`stencil_pattern`](@ref)). `check_local = true` refuses
operators that are not local ([`is_local`](@ref)); pass `false` only with a backend that
detects the pattern itself.
"""
function operator_matrix(op; backend=nothing, check_local::Bool=true)
    if check_local && !is_local(op)
        throw(ArgumentError("$(typeof(op)) is not a local operator; assembling it with the " *
                            "stencil pattern would silently corrupt the matrix. Apply it " *
                            "matrix-free instead."))
    end
    op isa AdvectionDiffusion && !is_linear(op) &&
        throw(ArgumentError("the operator uses a flux limiter and is nonlinear in f; it has no matrix " *
                            "(its Jacobian depends on f: use jacobian! of an EquationSystem)"))
    backend = something(backend, sparse_ad_backend(pattern=stencil_pattern(op)))
    residual! = (df, f) -> (fill!(df, zero(eltype(df))); apply!(df, op, f))
    sj = SparseJacobian(residual!, op.grid; backend)
    return jacobian(sj, zeros(eltype(op.advection_p), size(op.grid)))
end
