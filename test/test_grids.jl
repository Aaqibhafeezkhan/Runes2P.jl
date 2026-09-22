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

    @testset "cylindrical density of a Maxwellian-like f" begin
        mom = MomentumGrid(uniform_p_faces(400, 0.0, 1.0), uniform_xi_faces(4))
        rad = RadialGrid(uniform_r_faces(3, 1.0))
        g = cylindrical_grid(mom, rad)
        θ = 0.01  # T/(m c²)
        f = [exp(-mom.p_centers[i]^2 / (2θ)) for i in 1:momentum_cell_count(g), _ in 1:pitch_cell_count(g),
                                                _ in 1:radial_cell_count(g)]
        # ∫ 2π p² exp(-p²/2θ) dp dξ over ξ ∈ [-1,1] = 2 · 2π · √(π/2) θ^(3/2)
        exact = 4π * sqrt(π / 2) * θ^1.5
        @test all(isapprox.(density(g, f), exact; rtol=1e-4))
        @test total_particles(g, f) ≈ exact * sum(rad.spatial_jacobian .* rad.r_widths)
    end
end
