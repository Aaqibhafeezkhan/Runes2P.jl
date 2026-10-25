# Kinetic grids with a momentum grid per radius: shared detection, accessors, operators
# that need the same cell counts refusing others, and the knock-on operator on different
# momentum grids at different radii.

# an assertion on allocations holds for optimised code alone: it is skipped, and reported
# as broken, in a development run at -O0 or -O1
const OPTIMIZED = Base.JLOptions().opt_level >= 2

@testset "momentum grid per radius" begin
    rad = RadialGrid(uniform_r_faces(3, 0.6))
    m1 = MomentumGrid(biuniform_p_faces(24, 8, 0.0, 0.6, 4.0), uniform_xi_faces(10))
    m2 = MomentumGrid(uniform_p_faces(18, 0.0, 3.5), -cos.(range(0, π; length = 13)))
    m3 = MomentumGrid(uniform_p_faces(30, 0.0, 5.0), uniform_xi_faces(8))

    @testset "shared grids" begin
        single = cylindrical_grid(m1, rad)
        same_object = cylindrical_grid([m1, m1, m1], rad)
        copied = MomentumGrid(copy(m1.p_faces), copy(m1.xi_faces))
        equal_faces = cylindrical_grid([m1, copied, m1], rad)
        for g in (single, same_object, equal_faces)
            @test has_shared_momentum_grid(g)
            @test size(g) == (24, 10, 3)
            @test cell_count(g) == 24 * 10 * 3
            @test momentum_grid(g) === momentum_grid(g, 2)
            for ir in 1:3
                @test phase_space_jacobian(g, ir) == phase_space_jacobian(single, ir)
                @test phase_space_jacobian_p_faces(g, ir) ==
                      phase_space_jacobian_p_faces(single, ir)
                @test phase_space_jacobian_xi_faces(g, ir) ==
                      phase_space_jacobian_xi_faces(single, ir)
            end
            @test phase_space_jacobian_r_faces(g) == phase_space_jacobian_r_faces(single)
        end
        @test_throws DimensionMismatch cylindrical_grid([m1, m1], rad)
    end

    g = cylindrical_grid([m1, m2, m3], rad)

    @testset "accessors on different grids" begin
        @test !has_shared_momentum_grid(g)
        @test [momentum_cell_count(g, ir) for ir in 1:3] == [24, 18, 30]
        @test [pitch_cell_count(g, ir) for ir in 1:3] == [10, 12, 8]
        @test cell_count(g) == 24 * 10 + 18 * 12 + 30 * 8
        @test radial_cell_count(g) == 3
        for (ir, m) in enumerate((m1, m2, m3))
            @test momentum_grid(g, ir) === m
            expected = [2π * p^2 * rad.spatial_jacobian[ir]
                        for p in m.p_centers, _ in m.xi_centers]
            @test phase_space_jacobian(g, ir) ≈ expected rtol = 1e-15
            np, nxi = momentum_cell_count(m), pitch_cell_count(m)
            @test size(phase_space_jacobian_p_faces(g, ir)) == (np + 1, nxi)
            @test size(phase_space_jacobian_xi_faces(g, ir)) == (np, nxi + 1)
            @test length(phase_space_jacobian_over_p2_at_zero(g, ir)) == pitch_cell_count(m)
        end
        @test_throws ArgumentError size(g)
        @test_throws ArgumentError momentum_grid(g)
        @test_throws ArgumentError momentum_cell_count(g)
        @test_throws ArgumentError phase_space_jacobian_r_faces(g)
    end

    @testset "operators that need the same cell counts refuse others" begin
        # the pitch faces may differ between radii (test_per_radius_pitch_grids.jl), the cell
        # counts and the p faces may not
        @test_throws ArgumentError AdvectionDiffusion(g)
        plasma = PlasmaState(cold_electron_temperature = fill(100.0, 3),
            ions = [IonSpecies("D", 1, 1, fill(1e20, 3))],
            electric_field = fill(1.0, 3), magnetic_field = fill(5.0, 3))
        @test_throws ArgumentError fokker_planck_operator(g, plasma)
        # the message names the reason, not the accessor that failed
        @test_throws "the same cell counts and the same p faces" AdvectionDiffusion(g)
        @test_throws "the same cell counts and the same p faces" fokker_planck_operator(g, plasma)
    end

    @testset "knock-on operator on different grids" begin
        # each radius gives the result of a shared-grid operator built from that radius's
        # grids; radii 1 and 3 have the same knock-on and primary grids
        gK = cylindrical_grid([m1, m2, m1], rad)
        gP = cylindrical_grid([m3, m2, m3], rad)
        n_tot = [1e20, 2e20, 3e20]
        cutoffs = [0.1, 0.15, 0.1]
        for method in (:loop, :blas)
            op = KnockOnOperator(gK, gP, cutoffs; method)
            for ir in 1:3
                mK, mP = momentum_grid(gK, ir), momentum_grid(gP, ir)
                f = [exp(-p / 0.8) * (1.2 + ξ) for p in mP.p_centers, ξ in mP.xi_centers]
                df = zeros(momentum_cell_count(mK), pitch_cell_count(mK))
                apply_at_radius!(df, op, f, n_tot[ir], ir; scale = 0.5)
                # reference: shared grids with the momentum grids of radius ir
                reference = KnockOnOperator(
                    cylindrical_grid(mK, rad), cylindrical_grid(mP, rad),
                    cutoffs[ir]; method)
                f3 = zeros(momentum_cell_count(mP), pitch_cell_count(mP), 3)
                f3[:, :, ir] .= f
                df3 = zeros(momentum_cell_count(mK), pitch_cell_count(mK), 3)
                apply!(df3, reference, f3, n_tot; scale = 0.5)
                @test df == df3[:, :, ir]
            end
            # radii 1 and 3 have the same grids and cutoff and share their kernels
            @test energy_kernel(op, 1) === energy_kernel(op, 3)
            @test energy_kernel(op, 1) !== energy_kernel(op, 2)
            @test length(op.cache) == 2
            @test_throws ArgumentError apply!(zeros(24, 10, 3), op, zeros(30, 8, 3), n_tot)
            @test_throws DimensionMismatch apply_at_radius!(
                zeros(3, 3), op, zeros(30, 8), 1e20, 1)
            df1, f1 = zeros(24, 10), ones(30, 8)
            apply_at_radius!(df1, op, f1, 1e20, 1)
            @test (@allocated apply_at_radius!(df1, op, f1, 1e20, 1)) == 0 skip = !OPTIMIZED
        end
    end
end
