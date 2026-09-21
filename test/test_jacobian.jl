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

@testset "stencil pattern covers traced pattern" begin
    rng = Xoshiro(3)
    mom = PXiGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
    op = AdvectionDiffusion(g; scheme=:central, pbc=:prescribed, rbc=:absorbing)
    for A in (op.Ap, op.Axi, op.Ar, op.Dpxi, op.Dxip)
        A .= randn(rng, size(A))
    end
    for D in (op.Dpp, op.Dxixi, op.Drr)
        D .= rand(rng, size(D))
    end
    op.pflux_in .= rand(rng, size(op.pflux_in))
    M_stencil = operator_matrix(op)
    M_traced = operator_matrix(op; backend=sparse_ad_backend())
    @test M_stencil ≈ M_traced
    @test check_stencil_pattern(op)
end

@testset "non-local operators are refused" begin
    struct DenseTestOperator
        grid::KineticGrid{Float64}
    end
    g = cylindrical_grid(PXiGrid(uniform_p_faces(3, 0.0, 1.0), uniform_xi_faces(2)),
                         RadialGrid(uniform_r_faces(1, 1.0)))
    @test !is_local(DenseTestOperator(g))
    @test_throws ArgumentError operator_matrix(DenseTestOperator(g))
end

@testset "prescribed lower-p influx" begin
    mom = PXiGrid(uniform_p_faces(8, 0.5, 2.0), uniform_xi_faces(4))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
    @test_throws ArgumentError AdvectionDiffusion(
        cylindrical_grid(PXiGrid(uniform_p_faces(4, 0.0, 1.0), uniform_xi_faces(2)),
                         RadialGrid(uniform_r_faces(1, 1.0))); pbc=:prescribed)
    op = AdvectionDiffusion(g; pbc=:prescribed)
    op.pflux_in .= 0.7
    df = zeros(size(g))
    apply!(df, op, zeros(size(g)))
    gain = sum(g.Vp[i, j, ir] * mom.dp[i] * mom.dxi[j] * g.rad.dr[ir] * df[i, j, ir]
               for i in 1:n_p(g), j in 1:n_xi(g), ir in 1:n_r(g))
    influx = sum(g.Vp_fp[1, j, ir] * mom.dxi[j] * g.rad.dr[ir] * 0.7
                 for j in 1:n_xi(g), ir in 1:n_r(g))
    @test gain ≈ influx
end
