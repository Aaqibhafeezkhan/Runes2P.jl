using SparseArrays: nnz

@testset "sparse Jacobian" begin
    rng = Xoshiro(2)
    mom = PXiGrid(uniform_p_faces(12, 0.0, 2.0), uniform_xi_faces(7))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(5, 1.0)))
    op = AdvectionDiffusion(g; scheme=:central, rbc=:absorbing)
    for A in (op.Ap, op.Axi, op.Ar, op.Dpxi, op.Dxip)
        A .= randn(rng, size(A))
    end
    for D in (op.Dpp, op.Dxixi, op.Drr)
        D .= rand(rng, size(D))
    end
    M = operator_matrix(op)
    f = rand(rng, size(g)...)
    df = zeros(size(g))
    apply!(df, op, f)
    @test M * vec(f) ≈ vec(df)
    # 9-point momentum stencil plus 2 radial neighbours at most
    @test nnz(M) <= 11 * ncells(g)

    # jacobian! reuses the pattern for a nonlinear residual
    F! = (y, x) -> (fill!(y, 0); apply!(y, op, x); y .+= x .^ 2; y)
    sj = SparseJacobian(F!, g)
    J = jacobian(sj, f)
    f2 = rand(rng, size(g)...)
    jacobian!(J, sj, f2)
    @test J ≈ M + Diagonal(2 .* vec(f2))
end
