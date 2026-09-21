# Physics checks of the small-angle operator that do not rely on DREAM.

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
end
