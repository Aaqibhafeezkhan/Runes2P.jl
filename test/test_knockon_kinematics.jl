# Knock-on kinematics and Møller cross section. The first group ports the assertions of
# DREAM `tests/cxx/tests/DREAM/KnockOn.cpp` (with tighter tolerances where exact
# quadrature allows); the rest checks against independent relativistic kinematics and
# the textbook Møller cross section.
using QuadGK: quadgk

@testset "knock-on kinematics" begin
    γ(p) = sqrt(1 + p^2)
    β(p) = p / sqrt(1 + p^2)

    @testset "DREAM KnockOn.cpp assertions" begin
        # CheckMollerFluxIntegration: S(b) - S(a) = ∫ v Σ dp
        p1 = 7.0
        a, b = 0.2, 0.9
        num = quadgk(p -> β(p) * moller_cross_section(p, p1), a, b; rtol = 1e-14)[1]
        @test moller_flux(b, p1) - moller_flux(a, p1) ≈ num rtol = 1e-12
        # Check{MollerDifferential,MollerFlux,XiStar}ConvergesToInfiniteLimit
        for fun in (moller_cross_section, moller_flux, xi_star)
            @test fun(0.5, 1e8) ≈ fun(0.5, Inf) rtol = 1e-6
        end
    end

    @testset "literal DREAM formulas agree to round-off" begin
        # DREAM evaluates γ - 1 directly; Runes2P uses p²/(γ + 1). Over the momenta used
        # by knock-on operators (p ≳ 0.05) the results agree to round-off.
        pre = 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT
        function dream_sigma(p, p1)
            g, g1 = γ(p), γ(p1)
            t2 = -(g - 1) * (g1 - g) / g1^2 * (2g1^2 + 2g1 - 1 - (g - 1) * (g1 - g))
            return pre * ((g1 - 1)^2 + t2) * g1^2 / (p1^2 * (g - 1)^2 * (g1 - g)^2)
        end
        function dream_flux(p, p1)
            g, g1 = γ(p), γ(p1)
            return pre / p1^2 * (g1^2 * (1 / (g1 - g) - 1 / (g - 1)) + g -
                    (2g1 - 1) / (g1 - 1) * log((g - 1) / (g1 - g)))
        end
        dream_xi(p, p1) = sqrt((γ(p1) + 1) * (γ(p) - 1) / ((γ(p1) - 1) * (γ(p) + 1)))
        for p1 in (0.5, 2.0, 10.0, 100.0), frac in (0.1, 0.5, 0.99)

            p = frac * max_knockon_momentum(p1)
            @test moller_cross_section(p, p1) ≈ dream_sigma(p, p1) rtol = 1e-11
            @test moller_flux(p, p1) ≈ dream_flux(p, p1) rtol = 1e-11
            @test xi_star(p, p1) ≈ dream_xi(p, p1) rtol = 1e-13
        end
    end

    @testset "textbook Møller cross section" begin
        # dσ/dε = 2π r₀²/β₁² [1/ε² + 1/(T-ε)² + 1/(T+1)² - (2T+1)/((T+1)² ε (T-ε))],
        # ε = γ - 1, T = γ₁ - 1; symmetric under ε ↔ T - ε (exchange of the electrons).
        function dσdε(p, p1)
            ε, T = γ(p) - 1, γ(p1) - 1
            return 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 / β(p1)^2 *
                   (1 / ε^2 + 1 / (T - ε)^2 + 1 / (T + 1)^2 -
                    (2T + 1) / ((T + 1)^2 * ε * (T - ε)))
        end
        for (p, p1) in ((0.05, 0.5), (0.3, 2.0), (1.0, 10.0), (2.0, 100.0), (0.1, 1e4))
            @test moller_cross_section(p, p1) ≈ Constants.SPEED_OF_LIGHT * dσdε(p, p1) rtol = 1e-12
        end
        # the other outgoing electron carries energy T - ε (moderate p1, where γ₁ + 1 - γ
        # is computed without cancellation)
        for (p, p1) in ((0.05, 0.5), (0.3, 2.0), (1.0, 10.0))
            p_other = sqrt((γ(p1) - γ(p) + 1)^2 - 1)
            @test moller_cross_section(p_other, p1) ≈ moller_cross_section(p, p1) rtol = 1e-12
        end
        @test moller_cross_section(0.4, Inf) ≈
              2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT /
              (γ(0.4) - 1)^2
    end

    @testset "ξ* from energy and momentum conservation" begin
        # γ₁ + 1 = γ + γ', 𝐩₁ = 𝐩 + 𝐩'  ⇒  p'² = p₁² + p² - 2 p p₁ ξ*
        for p1 in (0.3, 1.0, 5.0, 50.0), frac in (0.01, 0.3, 0.9, 1.0)

            p = frac * max_knockon_momentum(p1)
            ξ = xi_star(p, p1)
            p_other2 = (γ(p1) + 1 - γ(p))^2 - 1
            @test p1^2+p^2-2p*p1*ξ≈p_other2 rtol=1e-12 atol=1e-14*p1^2
            @test ξ ≈ (γ(p1) + 1) / (γ(p) + 1) * p / p1 rtol = 1e-13
            @test 0 < ξ <= 1
        end
        # at the maximum knock-on momentum both electrons leave with equal energy and angle
        p1 = 3.0
        pm = max_knockon_momentum(p1)
        @test γ(pm) ≈ (γ(p1) + 1) / 2 rtol = 1e-14
        @test 2pm * xi_star(pm, p1) ≈ p1 rtol = 1e-13
        @test isinf(max_knockon_momentum(Inf))
        # accurate at small knock-on momentum (γ - 1 without cancellation)
        @test xi_star(1e-6, 2.0) ≈ (γ(2.0) + 1) / 2 * 1e-6 / 2.0 rtol = 1e-10
    end

    @testset "total production rate above a cutoff" begin
        for p1 in (0.8, 3.0, 20.0), p_cut in (0.05, 0.2)

            pmax = max_knockon_momentum(p1)
            p_cut < pmax || continue
            # β₁ ∫ Σ dγ over the knock-on energy range
            ref = β(p1) * quadgk(p -> β(p) * moller_cross_section(p, p1), p_cut, pmax;
                rtol = 1e-13)[1]
            @test moller_total_rate(p1, p_cut) ≈ ref rtol = 1e-11
        end
        # ultra-relativistic limit: 2π r₀² c / (γ_c - 1), approached from finite p1
        p_cut = 0.3
        rp = 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT /
             (γ(p_cut) - 1)
        @test moller_total_rate(Inf, p_cut) ≈ rp rtol = 1e-14
        @test moller_total_rate(1e6, p_cut) ≈ rp rtol = 1e-5
        # no knock-ons above the kinematic limit
        @test moller_total_rate(0.5, max_knockon_momentum(0.5)) == 0
        @test moller_total_rate(0.5, 1.0) == 0
        @test_throws ArgumentError moller_total_rate(1.0, 0.0)
    end
end
