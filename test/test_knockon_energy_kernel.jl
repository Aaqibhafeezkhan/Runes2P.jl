# Møller energy kernel S[i, k]. DREAM's own conservation tests for this kernel
# (MollerEnergyKernel.cpp, KnockOn.cpp) use an absolute tolerance of 1e-14 on
# quantities of order 2π r₀² c ≈ 1.5e-20 m³/s and therefore cannot fail; the tests
# here use relative tolerances.
using QuadGK: quadgk

@testset "knock-on energy kernel" begin
    β(p) = p / sqrt(1 + p^2)
    scale = 2π * Constants.r0^2 * Constants.c

    knock = PXiGrid(biuniform_p_faces(40, 15, 0.0, 0.5, 3.0), uniform_xi_faces(4))
    prim = PXiGrid(uniform_p_faces(25, 0.0, 12.0), uniform_xi_faces(4))
    p_cut = knock.p_f[3] + 0.3 * knock.dp[3]  # inside a cell
    K = MollerEnergyKernel(knock, prim, p_cut)

    @testset "cell sums equal the production onto the grid" begin
        for k in 1:n_p(prim)
            @test sum(knock.dp .* K.S[:, k]) ≈ K.total[k] rtol = 1e-13 atol = 1e-14 * scale
        end
        @test any(>(0), K.total)
    end

    @testset "production on and above the grid adds up to the total rate" begin
        for k in 1:n_p(prim)
            @test K.total[k] + K.above_grid[k] ≈ moller_total_rate(prim.p[k], p_cut) rtol = 1e-12 atol = 1e-14 * scale
        end
        # primaries energetic enough to produce knock-ons above p = 3
        @test any(>(0), K.above_grid)
    end

    @testset "cell averages against quadrature" begin
        for k in (5, 12, 25), i in 1:n_p(knock)
            p1 = prim.p[k]
            pL = max(knock.p_f[i], p_cut)
            pU = min(knock.p_f[i+1], max_knockon_momentum(p1))
            ref = pU > pL ?
                  β(p1) * quadgk(p -> β(p) * moller_cross_section(p, p1), pL, pU; rtol=1e-13)[1] /
                  knock.dp[i] : 0.0
            @test K.S[i, k] ≈ ref rtol = 1e-10 atol = 1e-14 * scale
        end
    end

    @testset "support: nothing below the cutoff or above the kinematic limit" begin
        inside(i, k) = knock.p_f[i+1] > p_cut && knock.p_f[i] < max_knockon_momentum(prim.p[k])
        @test all((K.S[i, k] > 0) == inside(i, k) for i in 1:n_p(knock), k in 1:n_p(prim))
        @test all(>=(0), K.S)
    end

    @testset "cutoff above the knock-on grid gives an empty kernel" begin
        K2 = MollerEnergyKernel(knock, prim, 5.0)
        @test all(iszero, K2.S)
        @test all(iszero, K2.total)
        @test K2.above_grid ≈ [moller_total_rate(p1, 5.0) for p1 in prim.p]
        @test_throws ArgumentError MollerEnergyKernel(knock, prim, 0.0)
    end

    @testset "production from a smooth primary distribution converges at second order" begin
        # Σ_k Δp1_k p1_k² g(p1_k) total_k → ∫ p1² g(p1) β₁ ∫ Σ dγ dp1 (primary at cell centres).
        # The rate has kinks in p1 where max_knockon_momentum(p1) crosses p_cut and the grid
        # top, so the midpoint-rule error oscillates in sign but stays O(Δp1²).
        g(p1) = exp(-p1 / 2)
        kk = PXiGrid(uniform_p_faces(30, 0.0, 3.0), uniform_xi_faces(2))
        pc = 0.2
        rate(p1) = moller_total_rate(p1, pc) - moller_total_rate(p1, 3.0)
        kink(p) = sqrt((2 * sqrt(1 + p^2) - 1)^2 - 1)
        edges = (0.0, kink(pc), kink(3.0), 20.0)
        ref = sum(quadgk(p1 -> p1^2 * g(p1) * rate(p1), edges[j], edges[j+1]; rtol=1e-13)[1]
                  for j in 1:3)
        ns = (80, 160, 320, 640)
        errs = map(ns) do n
            pg = PXiGrid(uniform_p_faces(n, 0.0, 20.0), uniform_xi_faces(2))
            Kn = MollerEnergyKernel(kk, pg, pc)
            abs(sum(pg.dp .* pg.p .^ 2 .* g.(pg.p) .* Kn.total) / ref - 1)
        end
        @test all(errs .* ns .^ 2 .< 3)  # observed ≤ 2.7
        @test errs[end] < 1e-5
    end
end
