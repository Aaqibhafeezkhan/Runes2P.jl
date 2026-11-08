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

@testset "stencil of the declared coefficients" begin
    # a Fokker-Planck operator declares no radial transport and no cross diffusion: its
    # pattern couples neither the radii nor p and ξ off the axes, and it covers the traced one
    mom = MomentumGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g)])
    cells = CartesianIndices(size(g))
    plasma = PlasmaState(cold_electron_temperature = [100.0, 200.0, 300.0, 400.0],
        ions = [IonSpecies("D", 1, 1, fill(1e20, 4))],
        electric_field = fill(5.0, 4), magnetic_field = fill(5.0, 4))
    # the entries of a pattern that couple two radii, and those off the axes in p and ξ
    function couplings(pattern)
        rows, columns, _ = findnz(pattern)
        radial = count(k -> cells[rows[k]][3] != cells[columns[k]][3], eachindex(rows))
        off_axes = count(eachindex(rows)) do k
            a, b = cells[rows[k]], cells[columns[k]]
            a[1] != b[1] && a[2] != b[2]
        end
        return (; radial, off_axes)
    end
    for scheme in (:central, :quick),
        p_upper_boundary in (:f_zero, :constant_flux, :extrapolated_flux, :extrapolated_flux_consistent)

        op = fokker_planck_operator(g, plasma; scheme, p_upper_boundary)
        @test !op.radial_transport[] && !op.cross_diffusion[]
        @test couplings(stencil_pattern(op)) == (radial = 0, off_axes = 0)
        @test check_stencil_pattern(op)
        entries = Runes2P.sparsity!(Tuple{Int, Int}[], FokkerPlanckTerm(:f, op), layout, 1)
        rows, columns, _ = findnz(stencil_pattern(op))
        @test Set(entries) == Set(zip(rows, columns))
    end
    # an operator built by hand declares both, and a term with a plasma source clears them
    op = AdvectionDiffusion(g; scheme = :central, r_boundary = :absorbing)
    @test op.radial_transport[] && op.cross_diffusion[]
    full = couplings(stencil_pattern(op))
    @test full.radial > 0 && full.off_axes > 0
    @test couplings(stencil_pattern(op.grid)) == full
    term = prescribed_fokker_planck_term(:f, op, t -> plasma)
    @test !op.radial_transport[] && !op.cross_diffusion[]
    entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 1)
    @test Set(entries) == Set(zip(findnz(stencil_pattern(op))[1:2]...))
    @test couplings(stencil_pattern(op)) == (radial = 0, off_axes = 0)
    @test describe(op; coefficients = false)["radial_transport"] == false
    # each declaration alone, and the keyword form on a bare grid
    radial_only = couplings(stencil_pattern(AdvectionDiffusion(g; cross_diffusion = false)))
    @test radial_only.radial == full.radial && radial_only.off_axes == 0
    @test couplings(stencil_pattern(g; radial_transport = false)) ==
          (radial = 0, off_axes = full.off_axes)
end

@testset "undeclared coefficients are refused" begin
    rng = Xoshiro(7)
    mom = MomentumGrid(uniform_p_faces(6, 0.2, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(3, 1.0)))
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g)])
    system_of(op) = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
    # cross diffusion in an operator that declares none would fold onto declared entries
    op = AdvectionDiffusion(g; scheme = :central, cross_diffusion = false)
    op.diffusion_p_p .= rand(rng, size(op.diffusion_p_p))
    @test system_of(op) isa EquationSystem
    op.diffusion_xi_p .= rand(rng, size(op.diffusion_xi_p))
    @test_throws "FokkerPlanckTerm of f holds diffusion_xi_p" system_of(op)
    @test_throws "cross_diffusion = false" operator_matrix(op)
    @test !check_stencil_pattern(op)
    # radial transport: the writers require the declaration, the system refuses a coefficient
    # written by hand
    local_op = AdvectionDiffusion(g; scheme = :central, r_boundary = :absorbing, radial_transport = false)
    @test_throws "declares no radial transport" add_rechester_rosenbluth_diffusion!(
        local_op, 1e-3; major_radius = 1.0)
    @test_throws "declares no radial transport" RechesterRosenbluthUpdate(
        local_op, t -> 1e-3; major_radius = 1.0)
    local_op.diffusion_r_r .= 1.0
    @test_throws "holds diffusion_r_r" system_of(local_op)
    @test_throws "radial_transport = false" operator_matrix(local_op)
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
