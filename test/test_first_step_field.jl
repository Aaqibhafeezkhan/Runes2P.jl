# The field of the first short step of a self-consistent system against Braginskii: the field
# that holds the current of a shifted Maxwellian against the friction of the collisions.
#
# Reference. S. I. Braginskii, "Transport processes in a plasma", Reviews of Plasma Physics
# 1, 205 (Consultants Bureau, New York, 1965). A Maxwellian shifted to first order,
# f = f⁰ (1 + m U·v/T), loses momentum on the ions at R = −m n U/τ_e (eq. (4.6), p. 240),
# with the collision time of eq. (2.5e), p. 215, in Gaussian units,
#   τ_e = 3 √m T^{3/2} / (4 √(2π) λ e⁴ Z² n_i).
# In SI units e² becomes e²/(4π ε₀), and Z² n_i = Z n with n = n_e (p. 214):
#   τ_e = 6 √2 π^{3/2} ε₀² √m T^{3/2} / (lnΛ e⁴ Z n),   T in J,
# which is 3.44e5 T^{3/2}/(n λ) s with T in eV and n in cm⁻³ (A. S. Richardson, NRL Plasma
# Formulary, NRL/PU/6770--19-652 (2019), p. 36). The current j = −e n U is held over a short
# step, so the field balances the friction, e n E = m n U/τ_e:
#   E = m j / (n e² τ_e) = j/σ_⊥,   σ_⊥ = e² n τ_e/m (eq. (2.7), p. 216).
# The factor 0.51 of eq. (2.6) is the steady state, not the first step.
#
# Electron-electron collisions. The test-particle operator alone (C(f₁, f⁰), with the
# Maxwellian f⁰ as target) slows the drifting electrons as well. A test particle slows down as
# dv/dt = −ν_s v with ν_s = (1 + m_α/m_β) ψ(x) ν₀, ν₀ ∝ n_β e_β²/v³, x = m_β v²/(2T_β) and
# ψ(x) = (2/√π) ∫₀ˣ t^{1/2} e^{−t} dt (NRL Plasma Formulary, p. 31). On the ions
# (m_e/m_i → 0, ψ → 1) ν_s = ν₀ with n_i Z², on the electrons ν_s = 2ψ(x) ν₀ with n_e.
# Averaged over f₁ ∝ v_∥ f⁰ the momentum loss is ∝ ∫ ν_s v⁴ f⁰ dv ∝ ∫ (ν_s v³) e^{−x} dx, so
# the electron-electron loss over the ion loss is
#   (2/Z) ∫₀^∞ ψ(x) e^{−x} dx = (2/Z) (2/√π) ∫₀^∞ x^{1/2} e^{−2x} dx = (2/Z) 2^{−3/2} = 1/(√2 Z)
# (integrated by parts; it is the ratio M⁰⁰ = −1/√2 of the like-species test-particle element
# of test_field_particle.jl). With the test-particle term alone the field of the first step is
# (1 + 1/(√2 Z)) = 1.7071 times that of the ions alone, at Z = 1. With the field-particle term
# (FieldParticleTerm) the electron-electron collisions conserve the parallel momentum, and the
# shapes of its heat-flux modes, v_∥ L_a^{(3/2)}(x²) f⁰ with a ≥ 1, carry no current
# (orthogonal to L_0): the field stays at that of the ions alone.
#
# The case: the axis of reference/dream/cases/self_consistent_field.py, T = 120 eV,
# n_e = n_D = 5e19 m⁻³ (its argon left out: pure deuterium) and j = 1e6 A/m², complete
# screening, the thermal Coulomb logarithm lnΛ = 14.9 − ½ ln(n/1e20) + ln(T/keV) = 13.13 for
# both the electron-ion and the electron-electron collisions, no synchrotron loss and no
# bremsstrahlung. Then τ_e = 6.89e-7 s and E = 1.030 V/m. The system holds the current by a
# constraint on the current moment, with the field as its unknown, and takes one backward
# Euler step. The ion scattering alone is the operator without cold electrons (1e-10 of the
# density), as in the Lorentz limit of test_field_particle.jl: the electron-electron terms are
# proportional to the cold density, the ion terms and lnΛ to the density of the ions.
#
# The shift is to first order, as in eq. (4.6): the system is then linear in the shift, so E/j
# does not depend on j. What limits the agreement (measured at 120 eV):
# - the step: the ion scattering ν_D ∝ p⁻³ relaxes the cells of small p within the step, which
#   then hold less of the current. With 80 momentum cells the field falls short of that of
#   dt/τ_e = 1e-8 by 2.5e-2, 6.9e-3, 1.3e-3, 1.4e-4 and 1.4e-5 at dt/τ_e = 1e-3 to 1e-7, by a
#   factor that grows from 3.6 to 9 per decade; the test takes 1e-8.
# - the momentum grid: second order, −2.1e-3, −2.5e-4, +2.2e-4 and +3.4e-4 with 40, 80, 160
#   and 320 cells to 7 thermal momenta (the observed order between 40, 80 and 160 is 2.00).
# - the relativistic operator, Θ = T/(m c²). For the Maxwellian exp(−(γ − 1)/Θ) the current of
#   the shift a p ξ f⁰ is e c a Θ n exactly (⟨p²/γ⟩ = 3Θ), the ion loss with ν_D ∝ γ/p³ is
#   ∝ ⟨1/p⟩ = √(2/(πΘ)) (1 + Θ)/(1 + 15Θ/8), and the response of the current to the field is
#   ∝ ⟨1/γ − p²/(3γ³)⟩ = (1 − 5Θ/8)/(1 + 15Θ/8), to first order in Θ (checked by quadrature).
#   So E = (1 + 13Θ/8) m j/(n e² τ_e): 1 + 3.82e-4 at 120 eV, 1 + 9.54e-5 at 30 eV. The values
#   extrapolated in the momentum grid are 6.3e-6 and 3.3e-6 below these.
# An exactly shifted Maxwellian adds the nonlinear friction of the ions: the field of the drift
# u falls as 3√π ψ(x²)/(4x³) = 1 − (3/5) x² + (3/14) x⁴ − …, x = u/√(2T/m). The friction of
# ions at rest is ∝ ∫ (v/v³) f d³v, the field of a Gaussian charge cloud at the distance u,
# ∝ ψ(x²)/u², with ψ(x²) → (4/(3√π)) x³ as x → 0. Its test compares two currents, so that
# the O(Θ) difference between the nonrelativistic shift and the relativistic Maxwellian cancels,
# on 32 pitch cells, since the exact shift has all odd Legendre orders where the first-order
# shift has only the first, which the pitch grid represents exactly. For the same reason the
# field-particle term on an exact shift needs the current moment with the pitch weights of its
# momentum conservation (`pitch_rule = :cell_center`): with the linear weights, the default of
# CurrentMomentTerm, the current and the conserved momentum differ at the pitch resolution, and
# the field is 2.2e-3, 7.8e-4, 1.9e-4 and 4.2e-5 above that of the ions alone with 4, 8, 16 and
# 32 pitch cells at u/√(2T/m) = 0.19; with matching weights it is within 5e-6 on every grid.
using QuadGK: quadgk

@testset "first-step field against Braginskii" begin
    e = Constants.ELEMENTARY_CHARGE
    me = Constants.ELECTRON_MASS
    eps0 = Constants.VACUUM_PERMITTIVITY
    settings = CollisionSettings(mode = :full, screening = :complete,
        coulomb_logarithm = :thermal, bremsstrahlung = :neglect)
    # the axis of reference/dream/cases/self_consistent_field.py, without its argon
    n, j0 = 5e19, 1e6
    function collision_time(T)
        temperature = T * e
        return 6 * sqrt(2) * π^1.5 * eps0^2 * sqrt(me) * temperature^1.5 /
               (thermal_coulomb_logarithm(n, T) * e^4 * n)
    end
    # E = m j/(n e² τ_e), Braginskii's j/σ_⊥
    braginskii_field(T, j) = me * j / (n * e^2 * collision_time(T))

    # the field after one step dt from the shifted Maxwellian that carries j, with the current
    # held. electron_electron: :none (ions alone), :test_particle, or :field_particle (both)
    function first_step_field(; T = 120.0, np_ = 80, nxi = 4, dt_over_tau = 1e-8,
            electron_electron = :none, j = j0, exact = false, pitch_rule = :linear)
        g = _field_particle_grid(; np_, nxi, T)
        cold_fraction = electron_electron === :none ? 1e-10 : 1.0
        plasma = PlasmaState(; cold_electron_temperature = [T],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [0.0],
            magnetic_field = [0.0], cold_electron_density = [cold_fraction * n])
        operator = fokker_planck_operator(g, plasma, settings; synchrotron = false,
            scheme = :central, field_in_coefficients = false)
        m = momentum_grid(g)
        f = if exact
            # the nonrelativistic Maxwellian shifted by the drift momentum δ = u/c
            theta = T / Constants.ELECTRON_REST_ENERGY
            delta = j / (e * Constants.SPEED_OF_LIGHT * n)
            shifted = [exp(-(p^2 - 2p * xi * delta + delta^2) / (2theta))
                       for p in m.p_centers, xi in m.xi_centers, _ in 1:1]
            shifted .* (n / density(g, shifted)[1])
        else
            # f⁰ (1 + m U·v/T): the shift is p ξ f⁰ times a constant, scaled to carry j
            maxwellian = _maxwellian_on(g, T)
            maxwellian .*= n / density(g, maxwellian)[1]
            shift = [p * xi for p in m.p_centers, xi in m.xi_centers, _ in 1:1] .*
                    maxwellian
            maxwellian .+ shift .* (j / current_density(g, shift; pitch_rule)[1])
        end
        current = current_density(g, f; pitch_rule)[1]
        if !exact
            @test current ≈ j rtol = 1e-12
        end
        layout = UnknownLayout(g.radial_grid,
            [
                RadialUnknown(:electric_field; scale = 1.0), KineticUnknown(:f, g; scale = 1e20)])
        terms = Any[FokkerPlanckTerm(:f, operator; electric_field = :electric_field)]
        if electron_electron === :field_particle
            push!(terms,
                FieldParticleTerm(:f,
                    FieldParticleOperator(g, plasma, settings; scheme = :central)))
        end
        system = EquationSystem(layout,
            [
                Equation(
                    :electric_field, Constraint(), CurrentMomentTerm(:f, g; pitch_rule),
                    PrescribedValueTerm([-current]); scale = 1e6),
                Equation(:f, TimeDerivative(), terms...)])
        u0 = zeros(length(layout))
        kinetic_view(u0, layout, unknown_id(layout, :f), 1) .= f
        step = integrate(system, u0, [0.0, dt_over_tau * collision_time(T)];
            solver = NewtonSolver(residual_tolerance = 1e-12))
        @test all(step.statistics.accepted)
        field = only(unknown_values(step.states[:, end], layout,
            unknown_id(layout, :electric_field)))
        return field / braginskii_field(T, current)
    end

    @testset "the collision time" begin
        T = 120.0
        lnL = thermal_coulomb_logarithm(n, T)
        @test lnL ≈ 13.126 atol = 1e-3
        # NRL Plasma Formulary, p. 36: 3.44e5 T^{3/2}/(n λ) s, T in eV, n in cm⁻³
        @test collision_time(T) ≈ 3.44e5 * T^1.5 / (n * 1e-6 * lnL) rtol = 2e-3
        @test collision_time(T) ≈ 6.892e-7 rtol = 1e-3
        @test braginskii_field(T, j0) ≈ 1.030 rtol = 1e-3
    end

    @testset "ion scattering alone" begin
        # the step: the shortfall falls by more than 3 per decade of dt down to 1e-6 τ_e,
        # and the steps of 1e-7 and 1e-8 τ_e agree to 1.4e-5
        by_step = [first_step_field(; dt_over_tau = 10.0^-k) for k in 3:8]
        shortfall = 1 .- by_step[1:4] ./ by_step[end]
        @test all(shortfall .> 0)
        @test all(shortfall[1:3] ./ shortfall[2:4] .> 3)
        @test by_step[5] ≈ by_step[6] rtol = 5e-5
        # the momentum grid: second order
        by_grid = [first_step_field(; np_) for np_ in (40, 80, 160)]
        order = log2((by_grid[2] - by_grid[1]) / (by_grid[3] - by_grid[2]))
        @test 1.8 < order < 2.2
        extrapolated = by_grid[3] + (by_grid[3] - by_grid[2]) / 3
        # the extrapolated field against Braginskii with the relativistic correction 13Θ/8
        # (measured 6.3e-6 below it at 120 eV, 3.3e-6 at 30 eV)
        relativistic(T) = 1 + 13 / 8 * T / Constants.ELECTRON_REST_ENERGY
        @test extrapolated ≈ relativistic(120.0) atol = 2e-5
        at_30 = [first_step_field(; T = 30.0, np_) for np_ in (80, 160)]
        extrapolated_30 = at_30[2] + (at_30[2] - at_30[1]) / 3
        @test extrapolated_30 ≈ relativistic(30.0) atol = 2e-5
        # the grid of 80 cells within 1e-3
        @test by_grid[2] ≈ 1 rtol = 1e-3
    end

    @testset "electron-electron terms" begin
        for np_ in (80, 160)
            ions_alone = first_step_field(; np_)
            # the test-particle term adds the loss 1/(√2 Z) of the ion loss
            # (measured 1 + 0.707149 with 80 cells, 1 + 0.707068 with 160)
            @test first_step_field(; np_, electron_electron = :test_particle) / ions_alone ≈
                  1 + 1 / sqrt(2) rtol = 1e-4
            # the momentum-conserving operator leaves the field of the ions alone (measured
            # 2.3e-6 apart with 160 cells)
            @test first_step_field(; np_, electron_electron = :field_particle) ≈
                  ions_alone rtol = 1e-5
        end
    end

    @testset "exactly shifted Maxwellian" begin
        # E/j of the drift x = u/v_T against that of a 10 times smaller drift, against the
        # ratio of the Lorentz friction on a drifting Maxwellian, 3√π ψ(x²)/(4x³)
        psi(y) = 2 / sqrt(π) * quadgk(t -> sqrt(t) * exp(-t), 0, y; rtol = 1e-13)[1]
        friction(x) = 3 * sqrt(π) * psi(x^2) / (4 * x^3)
        thermal_speed = sqrt(2 * 120.0 * e / me)
        small, large = 1e6, 1e7
        x(j) = j / (e * n) / thermal_speed
        # the shifted Maxwellian has all odd Legendre orders, so the pitch grid matters: the
        # ratio is 1.5e-3, 2.2e-4, 9e-5 and 5.7e-5 above with 4, 8, 16 and 32 pitch cells
        ratio = first_step_field(; j = large, exact = true, nxi = 32) /
                first_step_field(; j = small, exact = true, nxi = 32)
        expected = friction(x(large)) / friction(x(small))
        # a departure of 2.2e-2 from 1
        @test 1 - expected > 0.02
        @test ratio ≈ expected rtol = 1e-4
        # the momentum-conserving operator leaves the field of the ions alone on the exact
        # shift as well, with the current moment of the pitch weights of its conservation
        # (measured 4.1e-6 apart), and with the linear weights as the pitch grid is refined
        # (2.2e-3 and 1.9e-4 above with 4 and 16 pitch cells)
        ions_alone = first_step_field(; j = large, exact = true, pitch_rule = :cell_center)
        @test first_step_field(; j = large, exact = true, pitch_rule = :cell_center,
            electron_electron = :field_particle) ≈ ions_alone rtol = 1e-5
        departure = map((4, 16)) do nxi
            first_step_field(; j = large, exact = true, nxi,
                electron_electron = :field_particle) /
            first_step_field(; j = large, exact = true, nxi) - 1
        end
        @test departure[1] > 1e-3
        @test 0 < departure[2] < departure[1] / 8
    end
end
