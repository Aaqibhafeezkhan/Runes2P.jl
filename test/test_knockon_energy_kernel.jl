# Møller energy kernel S[i, k]. DREAM's own conservation tests for this kernel
# (MollerEnergyKernel.cpp, KnockOn.cpp) use an absolute tolerance of 1e-14 on
# quantities of order 2π r₀² c ≈ 1.5e-20 m³/s and therefore cannot fail; the tests
# here use relative tolerances.
using QuadGK: quadgk

@testset "knock-on energy kernel" begin
    β(p) = p / sqrt(1 + p^2)
    scale = 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT

    knock = MomentumGrid(biuniform_p_faces(40, 15, 0.0, 0.5, 3.0), uniform_xi_faces(4))
    prim = MomentumGrid(uniform_p_faces(25, 0.0, 12.0), uniform_xi_faces(4))
    p_cut = knock.p_faces[3] + 0.3 * knock.p_widths[3]  # inside a cell
    K = MollerEnergyKernel(knock, prim, p_cut)

    @testset "cell sums equal the production onto the grid" begin
        for k in 1:momentum_cell_count(prim)
            @test sum(knock.p_widths .* K.rates[:, k]) ≈ K.total_rates[k] rtol = 1e-13 atol = 1e-14 * scale
        end
        @test any(>(0), K.total_rates)
    end

    @testset "production on and above the grid adds up to the total rate" begin
        for k in 1:momentum_cell_count(prim)
            rate = moller_total_rate(prim.p_centers[k], p_cut)
            @test K.total_rates[k] + K.rates_above_grid[k] ≈ rate rtol = 1e-12 atol = 1e-14 * scale
        end
        # primaries energetic enough to produce knock-ons above p = 3
        @test any(>(0), K.rates_above_grid)
    end

    @testset "cell averages against quadrature" begin
        for k in (5, 12, 25), i in 1:momentum_cell_count(knock)
            p1 = prim.p_centers[k]
            pL = max(knock.p_faces[i], p_cut)
            pU = min(knock.p_faces[i+1], max_knockon_momentum(p1))
            ref = pU > pL ?
                  β(p1) * quadgk(p -> β(p) * moller_cross_section(p, p1), pL, pU; rtol=1e-13)[1] /
                  knock.p_widths[i] : 0.0
            @test K.rates[i, k] ≈ ref rtol = 1e-10 atol = 1e-14 * scale
        end
    end

    @testset "support: nothing below the cutoff or above the kinematic limit" begin
        inside(i, k) = knock.p_faces[i+1] > p_cut &&
                       knock.p_faces[i] < max_knockon_momentum(prim.p_centers[k])
        @test all((K.rates[i, k] > 0) == inside(i, k)
                  for i in 1:momentum_cell_count(knock), k in 1:momentum_cell_count(prim))
        @test all(>=(0), K.rates)
    end

    @testset "cutoff above the knock-on grid gives an empty kernel" begin
        K2 = MollerEnergyKernel(knock, prim, 5.0)
        @test all(iszero, K2.rates)
        @test all(iszero, K2.total_rates)
        @test K2.rates_above_grid ≈ [moller_total_rate(p1, 5.0) for p1 in prim.p_centers]
        @test_throws ArgumentError MollerEnergyKernel(knock, prim, 0.0)
    end

    @testset "production from a smooth primary distribution converges at second order" begin
        # Σ_k Δp1_k p1_k² g(p1_k) total_k → ∫ p1² g(p1) β₁ ∫ Σ dγ dp1 (primary at cell centres).
        # The rate has kinks in p1 where max_knockon_momentum(p1) crosses p_cut and the grid
        # top, so the midpoint-rule error oscillates in sign but stays O(Δp1²).
        g(p1) = exp(-p1 / 2)
        kk = MomentumGrid(uniform_p_faces(30, 0.0, 3.0), uniform_xi_faces(2))
        pc = 0.2
        rate(p1) = moller_total_rate(p1, pc) - moller_total_rate(p1, 3.0)
        kink(p) = sqrt((2 * sqrt(1 + p^2) - 1)^2 - 1)
        edges = (0.0, kink(pc), kink(3.0), 20.0)
        ref = sum(quadgk(p1 -> p1^2 * g(p1) * rate(p1), edges[j], edges[j+1]; rtol=1e-13)[1]
                  for j in 1:3)
        ns = (80, 160, 320, 640)
        errs = map(ns) do n
            pg = MomentumGrid(uniform_p_faces(n, 0.0, 20.0), uniform_xi_faces(2))
            Kn = MollerEnergyKernel(kk, pg, pc)
            abs(sum(pg.p_widths .* pg.p_centers .^ 2 .* g.(pg.p_centers) .* Kn.total_rates) / ref - 1)
        end
        @test all(errs .* ns .^ 2 .< 3)  # observed ≤ 2.7
        @test errs[end] < 1e-5
    end
end
