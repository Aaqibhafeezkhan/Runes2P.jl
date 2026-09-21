# Physics checks of the small-angle operator that do not rely on DREAM.
import LinearAlgebra
using QuadGK: quadgk

function _single_species_plasma(; T=100.0, n=1e20, E=0.0, B=0.0, nr_=1)
    ions = [IonSpecies("D", 1, 1, fill(n, nr_))]
    return PlasmaState(T_cold=fill(T, nr_), ions=ions, E=fill(E, nr_), B=fill(B, nr_))
end

@testset "physics" begin
    @testset "friction balances acceleration at the critical field" begin
        # Ultra-relativistic, completely screened, constant lnΛ: p ν_s = 4π r₀² c lnΛ_c n,
        # so Aᵖ(ξ = 1) = (e/mc)(E - E_c) with E_c = 4π r₀² c lnΛ_c n mc/e.
        settings = CollisionSettings(mode=:ultra_relativistic, screening=:completely,
                                     lnlambda=:constant)
        n, T = 1e20, 100.0
        Ec = 4π * Constants.r0^2 * Constants.c * lnlambda_relativistic(T, n) * n *
             Constants.me * Constants.c / Constants.ec
        mom = PXiGrid(uniform_p_faces(20, 0.0, 50.0), [-1.0, 0.0, 0.999, 1.0])
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        for (factor, sgn) in ((0.999, -1), (1.001, 1))
            op = fokker_planck_operator(g, _single_species_plasma(; T, n, E=factor * Ec), settings)
            ξ = mom.xi[end]
            expected = Constants.ec / (Constants.me * Constants.c) * (factor * Ec * ξ - Ec)
            @test op.Ap[end, end, 1] ≈ expected rtol = 1e-12
            @test sign(op.Ap[end, end, 1]) == sgn
        end
    end

    @testset "relativistic Maxwellian is steady in full mode" begin
        # ν_∥ = Θγν_s makes Γᵖ = -pν_s f - ν_∥ ∂f/∂p vanish for f ∝ exp(-γ/Θ); the
        # discrete flux vanishes at second order and nothing leaves through p = 0.
        T = 1000.0
        Θ = T / Constants.mc2inEV
        plasma = _single_species_plasma(; T)
        settings = CollisionSettings(mode=:full, screening=:completely)
        cm = CollisionModel(plasma, settings)
        @test p3nu_s_at_zero(cm, 1) == 0
        errs = Float64[]
        for n1 in (50, 100, 200)
            mom = PXiGrid(uniform_p_faces(n1, 0.0, 0.5), uniform_xi_faces(4))
            g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
            op = fokker_planck_operator(g, plasma, settings)
            f = [exp(-(sqrt(1 + mom.p[i]^2) - 1) / Θ) for i in 1:n1, _ in 1:4, _ in 1:1]
            df = zeros(size(g))
            apply!(df, op, f)
            # relative to the size of the individual flux divergence terms
            fric = zeros(size(g))
            op.Dpp .= 0
            apply!(fric, op, f)
            bulk = 1:n1÷2
            push!(errs, maximum(abs, df[bulk, :, :]) / maximum(abs, fric[bulk, :, :]))
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test all(rates .> 1.8)
        @test errs[end] < 2e-4
    end

    @testset "collision frequency limits" begin
        plasma = _single_species_plasma(; T=10.0)
        # fully ionized hydrogen, superthermal, same thermal lnΛ for e-e and e-i:
        # ν_D/ν_s = (1 + Z)/γ
        cm = CollisionModel(plasma, CollisionSettings(mode=:superthermal,
                                                       screening=:completely, lnlambda=:thermal))
        for p in (0.1, 1.0, 10.0)
            @test nu_D(cm, 1, p) / nu_s(cm, 1, p) ≈ 2 / sqrt(1 + p^2) rtol = 1e-14
        end
        # partial screening lies between complete and no screening for an impurity
        ions = [IonSpecies("D", 1, 1, [1e20]), IonSpecies("Ar", 18, 2, [1e19])]
        pl = PlasmaState(T_cold=[10.0], ions=ions, E=[0.0], B=[0.0])
        ν(sc, fun, p) = fun(CollisionModel(pl, CollisionSettings(mode=:superthermal,
                                                                 screening=sc)), 1, p)
        for fun in (nu_s, nu_D), p in (0.5, 5.0)
            @test ν(:completely, fun, p) < ν(:partially, fun, p) < ν(:non, fun, p)
        end
    end

    @testset "conductivity in the Lorentz-gas limit" begin
        # Linear response f = f_M + f₁ with C f₁ = -E f_M. Without electron-electron
        # collisions (n_cold → 0) the operator reduces to a Lorentz gas, whose
        # conductivity is σ_L = (32/3π) n e² τ_e / m = 64√(2π) ε₀² T^{3/2} / (√m e² Z lnΛ)
        # (Braginskii, Z → ∞). The ξ moment with the cell-centre rule underestimates
        # ∫ ξ² dξ = 2/3 by the relative amount Δξ²/4; the ratio below divides out this
        # known ξ error, (3/2) Σ ξ_j² Δξ_j, so that only the p discretization is tested.
        function conductivity_ratio(; Z, np_, T=10.0, ne=1e20, lorentz=true, nxi=4)
            Θ = T / Constants.mc2inEV
            mom = PXiGrid(uniform_p_faces(np_, 0.0, 7 * sqrt(2Θ)), uniform_xi_faces(nxi))
            g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
            ions = [IonSpecies("X", Z, Z, [ne / Z])]
            kw = lorentz ? (; n_cold=[1e-10 * ne]) : (;)
            settings = CollisionSettings(mode=:full, screening=:completely, lnlambda=:thermal)
            plasma(E) = PlasmaState(; T_cold=[T], ions, E=[E], B=[0.0], kw...)
            MC = operator_matrix(fokker_planck_operator(g, plasma(0.0), settings))
            opE = AdvectionDiffusion(g; scheme=:central, pmax_bc=:f_zero)
            add_electric_field!(opE, plasma(1.0))
            fM = [exp(-(sqrt(1 + mom.p[i]^2) - 1) / Θ) for i in 1:np_, _ in 1:nxi, _ in 1:1]
            fM .*= ne / density(g, fM)[1]
            # C is singular on isotropic functions and f₁ is odd in ξ, so a shift far
            # below the smallest collision rate selects f₁ without perturbing it
            shift = 1e-10 * minimum(abs, filter(!iszero, LinearAlgebra.diag(MC))) *
                    LinearAlgebra.I
            f1 = reshape((shift - MC) \ (operator_matrix(opE) * vec(fM)), size(g))
            j = Constants.ec * Constants.c *
                sum(g.Vp[i, k, 1] / g.rad.VpVol[1] * mom.dp[i] * mom.dxi[k] *
                    mom.p[i] / sqrt(1 + mom.p[i]^2) * mom.xi[k] * f1[i, k, 1]
                    for i in 1:np_, k in 1:nxi)
            σL = 64 * sqrt(2π) * Constants.eps0^2 * (T * Constants.ec)^1.5 /
                 (sqrt(Constants.me) * Constants.ec^2 * Z * lnlambda_thermal(T, ne))
            return abs(j) / σL / (1.5 * sum(mom.xi .^ 2 .* mom.dxi))
        end
        # Second order in p; the extrapolated limit differs from the nonrelativistic
        # σ_L by -3.9 Θ (first relativistic correction, 8e-5 at 10 eV).
        errs = [conductivity_ratio(Z=1, np_=n) - 1 for n in (40, 80, 160, 320)]
        d = diff(errs)
        @test all(3.9 .< d[1:end-1] ./ d[2:end] .< 4.1)
        limit = errs[end] + d[end] / 3
        @test abs(limit) < 5 * 10.0 / Constants.mc2inEV
        # With electron-electron collisions, the test-particle operator approaches the
        # Lorentz limit as 1/Z (no momentum-conserving field-particle part, so Z = 1
        # gives about half of Spitzer's value).
        ratios = [conductivity_ratio(Z=Z, np_=80, lorentz=false) for Z in (1, 10, 80)]
        @test issorted(ratios)
        @test 0.9 < ratios[end] < 1
    end

    @testset "thermal Ψ functions against high-precision quadrature" begin
        # DREAM's asymptotic expansions (superthermal beyond 10 T, low energy for
        # p < 0.15 or Θ < 0.005) are accurate to about 5e-8 at their thresholds.
        for T in (10.0, 1000.0, 5000.0, 20000.0)
            Θ = T / Constants.mc2inEV
            pl = _single_species_plasma(; T)
            cm = Runes2P.CollisionModel(pl, CollisionSettings())
            err = 0.0
            for p in exp.(range(log(1e-3), log(20.0); length=60)), n in 0:2
                ref = quadgk(s -> exp(-(s^2 / (1 + sqrt(1 + s^2))) / Θ) * sqrt(1 + s^2)^(n - 1),
                             big(0.0), big(p); rtol=1e-20)[1]
                err = max(err, Float64(abs(Runes2P._psi(n, cm, 1, p) / ref - 1)))
            end
            @test err < 1e-7
        end
    end
end
