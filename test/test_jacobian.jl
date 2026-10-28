using Random: Xoshiro
using SparseArrays: findnz, nnz
using Runes2P: is_local

@testset "sparse Jacobian" begin
    rng = Xoshiro(2)
    mom = MomentumGrid(uniform_p_faces(12, 0.0, 2.0), uniform_xi_faces(7))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(5, 1.0)))
    op = AdvectionDiffusion(g; scheme = :central, r_boundary = :absorbing)
    for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
        A .= randn(rng, size(A))
    end
    for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
        D .= rand(rng, size(D))
    end
    M = operator_matrix(op)
    f = rand(rng, size(g)...)
    df = zeros(size(g))
    apply!(df, op, f)
    @test M * vec(f) ≈ vec(df)
    # 9-point momentum stencil plus 2 radial neighbours at most
    @test nnz(M) <= 11 * cell_count(g)

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
    mom = MomentumGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
    op = AdvectionDiffusion(g; scheme = :central, p_lower_boundary = :prescribed, r_boundary = :absorbing)
    for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
        A .= randn(rng, size(A))
    end
    for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
        D .= rand(rng, size(D))
    end
    op.p_flux_at_lower_boundary .= rand(rng, size(op.p_flux_at_lower_boundary))
    M_stencil = operator_matrix(op)
    M_traced = operator_matrix(op; backend = sparse_ad_backend())
    @test M_stencil ≈ M_traced
    @test check_stencil_pattern(op)
end

@testset "chunk limit of the dual numbers" begin
    rng = Xoshiro(5)
    mom = MomentumGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
    op = AdvectionDiffusion(g; scheme = :central, r_boundary = :absorbing)
    for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
        A .= randn(rng, size(A))
    end
    pattern = stencil_pattern(op)
    adaptive = sparse_ad_backend(; pattern, chunk_limit = nothing)
    @test adaptive.dense_ad isa Runes2P.AutoForwardDiff{nothing}
    reference = operator_matrix(op; backend = adaptive)
    # the values of the Jacobian do not depend on the chunk size
    for limit in (1, 4, 8)
        backend = sparse_ad_backend(; pattern, chunk_limit = limit)
        @test backend.dense_ad isa Runes2P.AutoForwardDiff{limit}
        @test operator_matrix(op; backend) == reference
    end
    # a limit above the number of colors leaves ForwardDiff's choice, and so does a
    # backend that detects its pattern
    @test sparse_ad_backend(; pattern, chunk_limit = 1000).dense_ad isa
          Runes2P.AutoForwardDiff{nothing}
    @test sparse_ad_backend(; chunk_limit = 4).dense_ad isa Runes2P.AutoForwardDiff{nothing}
    # the environment variable RUNES2P_JACOBIAN_CHUNK_SIZE sets the default
    withenv("RUNES2P_JACOBIAN_CHUNK_SIZE" => "4") do
        @test Runes2P.jacobian_chunk_limit() == 4
        @test sparse_ad_backend(; pattern).dense_ad isa Runes2P.AutoForwardDiff{4}
    end
    withenv("RUNES2P_JACOBIAN_CHUNK_SIZE" => nothing) do
        @test Runes2P.jacobian_chunk_limit() === nothing
        @test sparse_ad_backend(; pattern).dense_ad isa Runes2P.AutoForwardDiff{nothing}
    end
    withenv("RUNES2P_JACOBIAN_CHUNK_SIZE" => "zero") do
        @test_throws ArgumentError Runes2P.jacobian_chunk_limit()
    end
end

@testset "one stencil for the operator and the term" begin
    # stencil_pattern and the sparsity of FokkerPlanckTerm come from one function:
    # the term's entries are the operator's pattern, and the pattern covers the traced one
    mom = MomentumGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g)])
    for scheme in (:central, :quick, :tcdf),
        p_upper_boundary in (:f_zero, :extrapolated_flux, :extrapolated_flux_consistent)

        op = AdvectionDiffusion(g; scheme, p_upper_boundary, r_boundary = :absorbing)
        entries = Runes2P.sparsity!(Tuple{Int, Int}[], FokkerPlanckTerm(:f, op), layout, 1)
        rows, columns, _ = findnz(stencil_pattern(op))
        @test Set(entries) == Set(zip(rows, columns))
        @test check_stencil_pattern(op)
    end
end

@testset "non-local operators are refused" begin
    struct DenseTestOperator
        grid::KineticGrid{Float64}
    end
    g = cylindrical_grid(MomentumGrid(uniform_p_faces(3, 0.0, 1.0), uniform_xi_faces(2)),
        RadialGrid(uniform_r_faces(1, 1.0)))
    @test !is_local(DenseTestOperator(g))
    @test_throws ArgumentError operator_matrix(DenseTestOperator(g))
end

@testset "prescribed lower-p influx" begin
    mom = MomentumGrid(uniform_p_faces(8, 0.5, 2.0), uniform_xi_faces(4))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
    @test_throws ArgumentError AdvectionDiffusion(
        cylindrical_grid(MomentumGrid(uniform_p_faces(4, 0.0, 1.0), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 1.0))); p_lower_boundary = :prescribed)
    op = AdvectionDiffusion(g; p_lower_boundary = :prescribed)
    op.p_flux_at_lower_boundary .= 0.7
    df = zeros(size(g))
    apply!(df, op, zeros(size(g)))
    gain = sum(phase_space_jacobian(g, ir)[i, j] * mom.p_widths[i] * mom.xi_widths[j] *
               g.radial_grid.r_widths[ir] * df[i, j, ir]
    for i in 1:momentum_cell_count(g), j in 1:pitch_cell_count(g),
    ir in 1:radial_cell_count(g))
    influx = sum(phase_space_jacobian_p_faces(g, ir)[1, j] * mom.xi_widths[j] *
                 g.radial_grid.r_widths[ir] * 0.7
    for j in 1:pitch_cell_count(g), ir in 1:radial_cell_count(g))
    @test gain ≈ influx
end
