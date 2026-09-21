@testset "grids" begin
    @testset "momentum grid" begin
        g = PXiGrid(biuniform_p_faces(30, 10, 0.0, 1.0, 5.0), uniform_xi_faces(8))
        @test n_p(g) == 30
        @test n_xi(g) == 8
        @test sum(g.dp) ≈ 5.0
        @test sum(g.dxi) ≈ 2.0
        @test g.p_f[11] ≈ 1.0
        @test all(g.p .> 0)
        @test_throws ArgumentError PXiGrid([0.0, 1.0, 1.0], uniform_xi_faces(2))
        @test_throws ArgumentError PXiGrid([0.0, 1.0], [-1.5, 1.0])
        @test_throws ArgumentError PXiGrid([0.0, 1.0], [-1.0, -6e-17, 0.0, 1.0])
    end

    @testset "cylindrical density of a Maxwellian-like f" begin
        mom = PXiGrid(uniform_p_faces(400, 0.0, 1.0), uniform_xi_faces(4))
        rad = RadialGrid(uniform_r_faces(3, 1.0))
        g = cylindrical_grid(mom, rad)
        θ = 0.01  # T/(m c²)
        f = [exp(-mom.p[i]^2 / (2θ)) for i in 1:n_p(g), _ in 1:n_xi(g), _ in 1:n_r(g)]
        # ∫ 2π p² exp(-p²/2θ) dp dξ over ξ ∈ [-1,1] = 2 · 2π · √(π/2) θ^(3/2)
        exact = 4π * sqrt(π / 2) * θ^1.5
        @test all(isapprox.(density(g, f), exact; rtol=1e-4))
        @test total_particles(g, f) ≈ exact * sum(rad.VpVol .* rad.dr)
    end
end
