# Physics checks of the small-angle operator that do not rely on DREAM.
# The analytic references (the critical field, the ratios of the collision frequencies, the
# Maxwellian) hold without radiation losses, so the settings leave both out.
import LinearAlgebra
using QuadGK: quadgk

function _single_species_plasma(; T = 100.0, n = 1e20, E = 0.0, B = 0.0, nr_ = 1)
    ions = [IonSpecies("D", 1, 1, fill(n, nr_))]
    return PlasmaState(cold_electron_temperature = fill(T, nr_),
        ions = ions, electric_field = fill(E, nr_),
        magnetic_field = fill(B, nr_))
end

@testset "physics" begin
    @testset "friction balances acceleration at the critical field" begin
        # Ultra-relativistic, completely screened, constant lnΛ: p ν_s = 4π r₀² c lnΛ_c n,
        # so Aᵖ(ξ = 1) = (e/mc)(E - E_c) with E_c = 4π r₀² c lnΛ_c n mc/e.
        settings = CollisionSettings(mode = :ultra_relativistic, screening = :complete,
            coulomb_logarithm = :relativistic, bremsstrahlung = :neglect)
        n, T = 1e20, 100.0
        Ec = 4π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT *
             relativistic_coulomb_logarithm(T, n) * n *
             Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT /
             Constants.ELEMENTARY_CHARGE
        mom = MomentumGrid(uniform_p_faces(20, 0.0, 50.0), [-1.0, 0.0, 0.999, 1.0])
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        for (factor, sgn) in ((0.999, -1), (1.001, 1))
            op = fokker_planck_operator(g, _single_species_plasma(; T, n, E = factor * Ec),
                settings; synchrotron = false)
            ξ = mom.xi_centers[end]
            expected = Constants.ELEMENTARY_CHARGE /
                       (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT) *
                       (factor * Ec * ξ - Ec)
            @test op.advection_p[end, end, 1] ≈ expected rtol = 1e-12
            @test sign(op.advection_p[end, end, 1]) == sgn
        end
    end

    @testset "relativistic Maxwellian is steady in full mode" begin
        # ν_∥ = Θγν_s makes Γᵖ = -pν_s f - ν_∥ ∂f/∂p vanish for f ∝ exp(-γ/Θ); the
        # discrete flux vanishes at second order and nothing leaves through p = 0.
        T = 1000.0
        Θ = T / Constants.ELECTRON_REST_ENERGY
        plasma = _single_species_plasma(; T)
        settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect)
        cm = CollisionModel(plasma, settings)
        @test p3_slowing_down_frequency_at_zero(cm, 1) == 0
        # centred interpolation, second order; the default van Albada limiter is below,
        # exponential fitting in test_exponential_fitting.jl
        errs = Float64[]
        errs_van_albada = Float64[]
        for n1 in (50, 100, 200)
            mom = MomentumGrid(uniform_p_faces(n1, 0.0, 0.5), uniform_xi_faces(4))
            g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
            f = [exp(-(sqrt(1 + mom.p_centers[i]^2) - 1) / Θ)
                 for i in 1:n1, _ in 1:4, _ in 1:1]
            op_va = fokker_planck_operator(
                g, plasma, settings; scheme = :van_albada, synchrotron = false)
            df = zeros(size(g))
            apply!(df, op_va, f)
            fric = zeros(size(g))
            op_va.diffusion_p_p .= 0
            apply!(fric, op_va, f)
            push!(errs_van_albada, maximum(abs, df[1:(n1 ÷ 2), :, :]) /
                                   maximum(abs, fric[1:(n1 ÷ 2), :, :]))
            op = fokker_planck_operator(
                g, plasma, settings; scheme = :central, synchrotron = false)
            f = [exp(-(sqrt(1 + mom.p_centers[i]^2) - 1) / Θ)
                 for i in 1:n1, _ in 1:4, _ in 1:1]
            df = zeros(size(g))
            apply!(df, op, f)
            # relative to the size of the individual flux divergence terms
            fric = zeros(size(g))
            op.diffusion_p_p .= 0
            apply!(fric, op, f)
            bulk = 1:(n1 ÷ 2)
            push!(errs, maximum(abs, df[bulk, :, :]) / maximum(abs, fric[bulk, :, :]))
        end
        rates = log2.(errs[1:(end - 1)] ./ errs[2:end])
        @test all(rates .> 1.8)
        @test errs[end] < 2e-4
        # van Albada: observed orders 1.75 and 1.75, 4.5e-4 at 200 cells, about 4 times
        # central's error; the limiter flattens the slope at the peak p = 0.
        # This is the residual in the bulk only (p < 0.25, about 5.6 p_th at 1 keV); the
        # steady state's tail is far less accurate.
        rates_van_albada = log2.(errs_van_albada[1:(end - 1)] ./ errs_van_albada[2:end])
        @test all(rates_van_albada .> 1.7)
        @test errs_van_albada[end] < 5e-4
    end

    @testset "collision frequency limits" begin
        plasma = _single_species_plasma(; T = 10.0)
        # fully ionized hydrogen, superthermal, same thermal lnΛ for e-e and e-i:
        # ν_D/ν_s = (1 + Z)/γ
        cm = CollisionModel(plasma,
            CollisionSettings(mode = :superthermal,
                screening = :complete, coulomb_logarithm = :thermal, bremsstrahlung = :neglect))
        for p in (0.1, 1.0, 10.0)
            @test deflection_frequency(cm, 1, p) / slowing_down_frequency(cm, 1, p) ≈
                  2 / sqrt(1 + p^2) rtol = 1e-14
        end
        # partial screening lies between complete and no screening for an impurity
        ions = [IonSpecies("D", 1, 1, [1e20]), IonSpecies("Ar", 18, 2, [1e19])]
        pl = PlasmaState(
            cold_electron_temperature = [10.0], ions = ions, electric_field = [0.0],
            magnetic_field = [0.0])
        ν(sc, fun, p) = fun(
            CollisionModel(pl,
                CollisionSettings(mode = :superthermal,
                    screening = sc, bremsstrahlung = :neglect)),
            1,
            p)
        for fun in (slowing_down_frequency, deflection_frequency), p in (0.5, 5.0)

            @test ν(:complete, fun, p) < ν(:partial, fun, p) < ν(:none, fun, p)
        end
    end

    @testset "partial screening window" begin
        # Window edges in momentum: kinetic energy I_j and 4 I_j. That gives
        # p ≈ 0.029 to 0.059 for Ar⁺ and 0.032 to 0.063 for Ar²⁺.
        momentum_at(energy) = sqrt((1 + energy / Constants.ELECTRON_REST_ENERGY)^2 - 1)
        for (charge, lower, upper) in ((1, 0.029, 0.059), (2, 0.032, 0.063))
            excitation = mean_excitation_energy(18, charge)
            @test round(momentum_at(excitation); digits = 3) == lower
            @test round(momentum_at(4excitation); digits = 3) == upper
        end
        # Argon in Ar⁺ to Ar³⁺ and Ar¹⁶⁺ next to deuterium
        densities = zeros(19, 1)
        densities[2:4, 1] .= 1e18
        densities[17, 1] = 1e18
        ions = [IonSpecies("D", 1, 1, [1e20]), IonSpecies("Ar", 18, densities)]
        for mode in (:superthermal, :full)
            plasma = PlasmaState(
                cold_electron_temperature = [10.0], ions = ions, electric_field = [0.0],
                magnetic_field = [0.0])
            model(window, screening = :partial) = CollisionModel(plasma,
                CollisionSettings(; mode, screening, screening_window = window, bremsstrahlung = :neglect))
            window, dream, complete = model(:excitation_energy), model(:dream),
            model(:dream, :complete)
            excitations = [mean_excitation_energy(18, j) for j in (1, 2, 3, 16)]
            p_below = 0.9momentum_at(minimum(excitations))
            p_above = 1.1momentum_at(4maximum(excitations))
            for fun in (slowing_down_frequency, deflection_frequency, parallel_diffusion_frequency)
                # above every window the frequencies are DREAM's to round-off, below every
                # window those of complete screening
                for p in (p_above, 0.5, 5.0)
                    @test fun(window, 1, p) == fun(dream, 1, p)
                end
                for p in (0.1p_below, p_below)
                    @test fun(window, 1, p) ≈ fun(complete, 1, p) rtol = 1e-15
                end
                # inside, between the two, and the ratio to DREAM's continuous (the
                # frequencies themselves fall as 1/p³); ν_∥ vanishes in :superthermal mode
                fun(dream, 1, p_above) == 0 && continue
                ps = range(p_below, p_above; length = 1000)
                values = [fun(window, 1, p) for p in ps]
                @test all(fun(complete, 1, p) * (1 - 1e-14) <= v <=
                          fun(dream, 1, p) * (1 + 1e-14)
                for (p, v) in zip(ps, values))
                ratios = values ./ [fun(dream, 1, p) for p in ps]
                @test maximum(abs, diff(ratios)) < 0.02
            end
            @test p3_slowing_down_frequency_at_zero(window, 1) ≈
                  p3_slowing_down_frequency_at_zero(complete, 1) rtol = 1e-15
        end
        @test_throws ArgumentError CollisionSettings(screening_window = :born, bremsstrahlung = :neglect)
        @test_throws ArgumentError CollisionSettings(screening_window_start = 4.0,
            screening_window_end = 1.0, bremsstrahlung = :neglect)
    end

    @testset "conductivity in the Lorentz-gas limit" begin
        # Linear response f = f_M + f₁ with C f₁ = -E f_M. Without electron-electron
        # collisions (n_cold → 0) the operator reduces to a Lorentz gas, whose
        # conductivity is σ_L = (32/3π) n e² τ_e / m = 64√(2π) ε₀² T^{3/2} / (√m e² Z lnΛ)
        # (Braginskii, Z → ∞). f₁ is exactly ∝ ξ_j on the discrete grid (ξ_j is an
        # eigenvector of the discrete pitch scattering), so the current with the
        # piecewise-linear pitch rule has no ξ error and the ratio tests the
        # p discretization alone; with the cell-centre rule (DREAM's) the ratio is short by
        # the known factor (3/2) Σ ξ_j² Δξ_j (Δξ²/4 relative, 6.25% at 4 pitch cells).
        function conductivity_ratio(;
                Z, np_, T = 10.0, ne = 1e20, lorentz = true, nxi = 4, pitch_rule = :linear)
            Θ = T / Constants.ELECTRON_REST_ENERGY
            mom = MomentumGrid(uniform_p_faces(np_, 0.0, 7 * sqrt(2Θ)), uniform_xi_faces(nxi))
            g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
            ions = [IonSpecies("X", Z, Z, [ne / Z])]
            kw = lorentz ? (; cold_electron_density = [1e-10 * ne]) : (;)
            settings = CollisionSettings(mode = :full, screening = :complete,
                coulomb_logarithm = :thermal, bremsstrahlung = :neglect)
            plasma(E) = PlasmaState(;
                cold_electron_temperature = [T], ions, electric_field = [E],
                magnetic_field = [0.0], kw...)
            MC = operator_matrix(fokker_planck_operator(
                g, plasma(0.0), settings; scheme = :central, synchrotron = false))
            opE = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
            add_electric_field!(opE, plasma(1.0))
            fM = [exp(-(sqrt(1 + mom.p_centers[i]^2) - 1) / Θ)
                  for i in 1:np_, _ in 1:nxi, _ in 1:1]
            fM .*= ne / density(g, fM)[1]
            # C is singular on isotropic functions and f₁ is odd in ξ, so a shift far
            # below the smallest collision rate selects f₁ without perturbing it
            shift = 1e-10 * minimum(abs, filter(!iszero, LinearAlgebra.diag(MC))) *
                    LinearAlgebra.I
            f1 = reshape((shift - MC) \ (operator_matrix(opE) * vec(fM)), size(g))
            j = current_density(g, f1; pitch_rule)[1]
            σL = 64 * sqrt(2π) * Constants.VACUUM_PERMITTIVITY^2 *
                 (T * Constants.ELEMENTARY_CHARGE)^1.5 /
                 (sqrt(Constants.ELECTRON_MASS) * Constants.ELEMENTARY_CHARGE^2 * Z *
                  thermal_coulomb_logarithm(T, ne))
            return abs(j) / σL
        end
        # the cell-centre rule reproduces the linear rule once its ξ error is divided out
        mom4 = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(4))
        xi_factor = 1.5 * sum(mom4.xi_centers .^ 2 .* mom4.xi_widths)
        @test conductivity_ratio(Z = 1, np_ = 40, pitch_rule = :cell_center) / xi_factor ≈
              conductivity_ratio(Z = 1, np_ = 40) rtol = 1e-12
        # Second order in p; the extrapolated limit differs from the nonrelativistic
        # σ_L by -3.9 Θ (first relativistic correction, 8e-5 at 10 eV).
        errs = [conductivity_ratio(Z = 1, np_ = n) - 1 for n in (40, 80, 160, 320)]
        d = diff(errs)
        @test all(3.9 .< d[1:(end - 1)] ./ d[2:end] .< 4.1)
        limit = errs[end] + d[end] / 3
        @test abs(limit) < 5 * 10.0 / Constants.ELECTRON_REST_ENERGY
        # With electron-electron collisions, the test-particle operator approaches the
        # Lorentz limit as 1/Z (no momentum-conserving field-particle part, so Z = 1
        # gives about half of Spitzer's value).
        ratios = [conductivity_ratio(Z = Z, np_ = 80, lorentz = false) for Z in (1, 10, 80)]
        @test issorted(ratios)
        @test 0.9 < ratios[end] < 1
    end

    @testset "thermal Ψ functions against high-precision quadrature" begin
        # DREAM's asymptotic expansions (superthermal beyond 10 T, low energy for
        # p < 0.15 or Θ < 0.005) are accurate to about 5e-8 at their thresholds.
        for T in (10.0, 1000.0, 5000.0, 20000.0)
            Θ = T / Constants.ELECTRON_REST_ENERGY
            pl = _single_species_plasma(; T)
            cm = Runes2P.CollisionModel(pl, CollisionSettings(bremsstrahlung = :neglect))
            err = 0.0
            for p in exp.(range(log(1e-3), log(20.0); length = 60)), n in 0:2

                ref = quadgk(
                    s -> exp(-(s^2 / (1 + sqrt(1 + s^2))) / Θ) * sqrt(1 + s^2)^(n - 1),
                    big(0.0), big(p); rtol = 1e-20)[1]
                err = max(err, Float64(abs(Runes2P._psi(n, cm, 1, p) / ref - 1)))
            end
            @test err < 1e-7
        end
    end
end
