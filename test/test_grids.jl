@testset "grids" begin
    @testset "momentum grid" begin
        g = MomentumGrid(biuniform_p_faces(30, 10, 0.0, 1.0, 5.0), uniform_xi_faces(8))
        @test momentum_cell_count(g) == 30
        @test pitch_cell_count(g) == 8
        @test sum(g.p_widths) ≈ 5.0
        @test sum(g.xi_widths) ≈ 2.0
        @test g.p_faces[11] ≈ 1.0
        @test all(g.p_centers .> 0)
        @test_throws ArgumentError MomentumGrid([0.0, 1.0, 1.0], uniform_xi_faces(2))
        @test_throws ArgumentError MomentumGrid([0.0, 1.0], [-1.5, 1.0])
        @test_throws ArgumentError MomentumGrid([0.0, 1.0], [-1.0, -6e-17, 0.0, 1.0])
    end

    @testset "biuniform_theta_xi_faces" begin
        n, n1, thsep = 15, 10, 0.5
        faces = biuniform_theta_xi_faces(n, n1, thsep)
        @test length(faces) == n + 1
        @test issorted(faces; lt = <=)
        @test faces[1] == -1.0
        @test faces[end] == 1.0
        # the separation face θ = thsep is near ξ = 1 (θ near 0), so in increasing-ξ order
        # it is the (n - n1 + 1)-th face, not the (n1 + 1)-th
        @test faces[n - n1 + 1] ≈ cos(thsep) rtol = 1e-14
        @test count(<=(cos(thsep)), faces) == n - n1 + 1  # the coarse (upper-θ) region

        # literal 1-indexed port of DREAM's XiBiUniformThetaGridGenerator::Rebuild loop
        # (thMin = 0, thMax = π), th[k] and xi[k] holding its 0-indexed th_f[k-1], xi_f[k-1]
        dth_lo = thsep / n1
        dth_up = (pi - thsep) / (n - n1)
        dth = [k <= n1 ? dth_lo : dth_up for k in 1:n]
        th = zeros(n + 1)
        th[1] = 0.0
        for k in 2:n
            th[k] = th[k - 1] + dth[k - 1]
        end
        xi = zeros(n + 1)
        xi[n + 1] = cos(0.0)
        xi[1] = cos(pi)
        for k in 2:n
            xi[n - k + 2] = cos(th[k])
        end
        @test faces ≈ xi rtol = 1e-14

        @test_throws ArgumentError biuniform_theta_xi_faces(2, 2, 0.5)   # upper region needs a cell
        @test_throws ArgumentError biuniform_theta_xi_faces(5, 2, 0.0)   # theta_separation must be in (0, π)
        @test_throws ArgumentError biuniform_theta_xi_faces(5, 2, pi)
    end

    @testset "cylindrical density of a Maxwellian-like f" begin
        mom = MomentumGrid(uniform_p_faces(400, 0.0, 1.0), uniform_xi_faces(4))
        rad = RadialGrid(uniform_r_faces(3, 1.0))
        g = cylindrical_grid(mom, rad)
        θ = 0.01  # T/(m c²)
        f = [exp(-mom.p_centers[i]^2 / (2θ))
             for i in 1:momentum_cell_count(g), _ in 1:pitch_cell_count(g),
        _ in 1:radial_cell_count(g)]
        # ∫ 2π p² exp(-p²/2θ) dp dξ over ξ ∈ [-1,1] = 2 · 2π · √(π/2) θ^(3/2)
        exact = 4π * sqrt(π / 2) * θ^1.5
        @test all(isapprox.(density(g, f), exact; rtol = 1e-4))
        @test total_particles(g, f) ≈ exact * sum(rad.spatial_jacobian .* rad.r_widths)
    end
end
