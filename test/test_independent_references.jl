# Independent references for results that rested on DREAM alone: the energy of a closed
# plasma with ionizing charge states, the time integration of the energy of the cold
# electrons against a closed form, the split of a Maxwellian at the threshold of the
# kinetic ionization against the continuous integrals, the coronal balance of argon against
# published balances with other atomic data, the cross sections of the kinetic ionization
# against measurements, Gould's relativistic correction against the 4BN cross section, and the
# 4BN stopping power against the 3BN cross section and ESTAR.
using QuadGK: quadgk

@testset "independent references of 5.9 and 5.10" begin
    # the complementary error function by quadrature (1 − erf loses its digits at x = 25)
    erfc(z) = 2 / sqrt(π) * quadgk(t -> exp(-t^2), z, Inf; rtol=1e-13)[1]
    e = Constants.ELEMENTARY_CHARGE
    minus(name) = LinearTerm(name; coefficient=-1)

    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the closed neon plasma is skipped"
        @test_skip isdir(adas_cache_directory())
    else
    @testset "the energy of a closed neon plasma (item 1)" begin
        # Neutral neon deposited into a deuterium plasma at 30 eV, no field and no transport:
        # the charge states ionize, the cold electrons pay the binding energy and the
        # radiation, and the ions exchange energy with them. The energy of the plasma, the
        # thermal energies of the electrons and the ions and the potential energy of the
        # charge states E_pot = Σ n_Z0 Σ_{k<Z0} e I(k) (NIST, summed here), changes only by
        # what the radiation takes out: W_cold + W_i + E_pot + ∫ P_rad dt is constant. Independent
        # of the formula of each term, it catches a sign or an index of the binding energy and
        # of the exchange.
        composition = IonComposition(["D" => 1, "Ne" => 10])
        neon = zeros(11, 1)
        neon[1, 1] = 3e18
        prescribed = PlasmaState(; cold_electron_temperature=[30.0], electric_field=[0.0], magnetic_field=[0.0],
                                 ions=[IonSpecies("D", 1, 1, [5e19]), IonSpecies("Ne", 10, neon)])
        radial = RadialGrid(uniform_r_faces(1, 0.5))
        layout = UnknownLayout(radial, [ion_densities_unknown(composition), RadialUnknown(:cold_electron_density; scale=1e19),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0),
                                        RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e20),
                                        RadialUnknown(:ion_thermal_energy; components=2, scale=THERMAL_ENERGY_FACTOR * 1e20,
                                                      nonnegative=true)])
        ions = (ions=:ion_densities,)
        radiation = RadiatedPowerTerm(prescribed; radiation=:recombination, ions...)
        equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(composition)), population_equations(composition)...,
                     thermal_energy_equation(),
                     Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), radiation,
                              IonEnergyExchangeTerm(prescribed; side=:electrons, ions...)),
                     Equation(:ion_thermal_energy, TimeDerivative(), IonEnergyExchangeTerm(prescribed; side=:ions, ions...))]
        u0 = zeros(length(layout))
        set_ion_densities!(u0, layout, prescribed)
        radial_values(u0, layout, 2) .= free_electron_density(prescribed, 1)
        radial_values(u0, layout, 3) .= 30.0
        set_ion_thermal_energy!(u0, layout, prescribed, 3.0; ions=:ion_densities)
        system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u0)
        times = collect(range(0.0, 2e-6; length=41))
        solution = integrate(system, u0, times; stepper=FixedSteps(), solver=NewtonSolver(residual_tolerance=1e-12))
        @test all(solution.statistics.accepted)
        potential(u) = sum(e * sum(ionization_energy("Ne", k) for k in 0:Z0-1; init=0.0) * radial_view(u, layout, 1, 1)[2 + Z0 + 1]
                           for Z0 in 0:10) + e * ionization_energy("D", 0) * radial_view(u, layout, 1, 1)[2]
        energy(u) = only(radial_values(u, layout, 4)) + sum(radial_view(u, layout, 5, 1)) + potential(u)
        radiated = 0.0
        balance = Float64[]
        for m in 2:length(times)
            u = solution.states[:, m]
            # backward Euler: the radiation of the step at the state of its end
            radiated += (times[m] - times[m-1]) * only(radiated_power(radiation, u, layout, times[m]).radiation)
            push!(balance, energy(u) + radiated - energy(solution.states[:, 1]))
        end
        # the neon ionized (its potential energy grew by 15 %) and radiated 8.3 J/m³, both far
        # above the residual of the balance
        @test potential(solution.states[:, end]) > 1.1 * potential(solution.states[:, 1])
        @test maximum(abs, balance) < 1e-8 * energy(solution.states[:, 1])
        @test radiated > 1e4 * maximum(abs, balance)
    end
    end

    @testset "Gould's bremsstrahlung cools a plasma as the closed form (item 2)" begin
        # Fully ionized deuterium with a fixed density and the radiation of DREAM's mode, which
        # here is Gould's bremsstrahlung alone (no bound electrons, no line radiation):
        # (3/2) e n dT/dt = −C √T n² (1 + a T), a = (11/24 + κ)/mₑc² (GOULD_ION_CORRECTION,
        # GOULD_ELECTRON_CORRECTION). With k = C n/((3/2) e) its solution is
        # √a T(t)^{1/2} = tan(atan(√(a T₀)) − k √a t/2). Backward Euler converges to it at first
        # order: the error falls by about two when the step halves.
        if !isdir(adas_cache_directory())
            @info "no ADF11 cache: the cooling by Gould's bremsstrahlung is skipped"
            @test_skip isdir(adas_cache_directory())
        else
            n, T0 = 1e20, 20.0
            prescribed = PlasmaState(; cold_electron_temperature=[T0], electric_field=[0.0], magnetic_field=[0.0],
                                     ions=[IonSpecies("D", 1, 1, [n])], cold_electron_density=[n])
            radial = RadialGrid(uniform_r_faces(1, 0.5))
            layout = UnknownLayout(radial, [RadialUnknown(:cold_electron_density; scale=1e20),
                                            RadialUnknown(:cold_electron_temperature; scale=10.0),
                                            RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e21)])
            equations = [Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n]), minus(:cold_electron_density)),
                         thermal_energy_equation(),
                         Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                                  RadiatedPowerTerm(prescribed; radiation=:dream))]
            u0 = zeros(length(layout))
            radial_values(u0, layout, 1) .= n
            radial_values(u0, layout, 2) .= T0
            system = EquationSystem(layout, equations; time_scale=1e-3, initial_state=u0)
            mc2 = Constants.ELECTRON_REST_ENERGY
            a = (Runes2P.GOULD_ION_CORRECTION + Runes2P.GOULD_ELECTRON_CORRECTION) / mc2
            k = Runes2P.GOULD_BREMSSTRAHLUNG_PREFACTOR * n / (THERMAL_ENERGY_FACTOR)
            t_end = sqrt(T0) / k                      # to about a quarter of the temperature
            exact(t) = tan(atan(sqrt(a * T0)) - k * sqrt(a) * t / 2)^2 / a
            errors = map((25, 50, 100)) do steps
                solution = integrate(system, u0, collect(range(0.0, t_end; length=steps + 1)); stepper=FixedSteps(),
                                     solver=NewtonSolver(residual_tolerance=1e-12))
                abs(only(radial_values(solution.states[:, end], layout, 2)) - exact(t_end)) / exact(t_end)
            end
            @test exact(t_end) < 0.5T0
            @test errors[3] < 1e-2
            @test 1.8 < errors[1] / errors[2] < 2.2 && 1.8 < errors[2] / errors[3] < 2.2
        end
    end

    @testset "the split of a Maxwellian at the threshold (item 5)" begin
        # f_> of the grid against the quadrature of the continuous relativistic Maxwellian
        # exp(−(γ − 1)/Θ) that the split takes, and A_> against that of the Maxwellian times v σ
        # above the threshold: both separate the quadrature of the grid from the method of the
        # split. At 1 eV f_> is also the nonrelativistic closed form erfc(√x) + 2√(x/π) e^{−x},
        # x = E_c/T, to the relativistic correction p⁴/(8Θ) of the exponent, 6e-4 at five
        # thermal momenta (0.6 % at 10 eV).
        for T in (1.0, 10.0)
        theta = T / Constants.ELECTRON_REST_ENERGY
        composition = IonComposition(["Ar" => 18])
        radial = RadialGrid(uniform_r_faces(1, 0.5))
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(4000, 0.0, 12 * sqrt(2theta)), uniform_xi_faces(1)), radial)
        layout = UnknownLayout(radial, [ion_densities_unknown(composition), RadialUnknown(:cold_electron_temperature),
                                        KineticUnknown(:f, grid)])
        u = zeros(length(layout))
        radial_values(u, layout, 2) .= T
        for factor in (3.0, 5.0)
            kinetic = KineticIonizationTerm(composition, :f, grid;
                                            region=MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor)))
            split = threshold_split(kinetic, 1, u, layout, 0.0)
            p0 = factor * sqrt(2theta)
            x = (sqrt(1 + p0^2) - 1) / theta        # E_c/T
            maxwellian(p) = p^2 * exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)
            speed(p) = Constants.SPEED_OF_LIGHT * p / sqrt(1 + p^2)
            total = quadgk(maxwellian, 0, 40sqrt(2theta); rtol=1e-12)[1]
            @test split.fraction ≈ quadgk(maxwellian, p0, 40sqrt(2theta); rtol=1e-12)[1] / total rtol = 2e-4
            T == 1.0 && @test split.fraction ≈ erfc(sqrt(x)) + 2sqrt(x / π) * exp(-x) rtol = 2e-3
            # A_> of Ar⁺ (component 2)
            above = quadgk(p -> maxwellian(p) * speed(p) * ionization_cross_section("Ar", 1, p), p0, 40sqrt(2theta);
                           rtol=1e-10)[1]
            @test split.coefficients[2] ≈ above / total rtol = 2e-3
        end
        end
    end

    @testset "the coronal balance of argon against published balances (item 3)" begin
        # The coronal equilibrium of the ADAS rates (coronal_equilibrium) against two ionization
        # balances with other atomic data, in the zero-density limit: Mazzotta, Mazzitelli,
        # Colafrancesco, Vittorio, Astron. Astrophys. Suppl. Ser. 133 (1998) 403, Table 2, and
        # Bryans, Landi, Savin, Astrophys. J. 691 (2009) 1540, doi:10.1088/0004-637X/691/2/1540,
        # machine-readable Table 2. Both tabulate −log₁₀ of the fraction on a grid of 0.1 in
        # log₁₀ T [K]; the peaks below are the vertex of the parabola through the largest log
        # fraction and its neighbours, the mean charges interpolate the log fractions linearly
        # in log T (1 eV = 11604.518 K). Values of (Mazzotta, Bryans):
        # Ar⁸⁺ peaks at (54.8, 59.3) eV, Ar¹⁶⁺ at (865, 873) eV, the mean charge is (3.03, 2.97)
        # at 10 eV, (8.56, 8.48) at 100 eV and (15.972, 15.970) at 1 keV. ADAS (1989) is older
        # data and the three differ by the spread of the atomic data, so the bands catch a wrong
        # block, index or sign of the rates (a whole charge state, a factor two in temperature),
        # not the data: Runes2P at 1e16 m⁻³, the lowest density of the tables, has the peaks at
        # 48.4 and 847 eV and the mean charges 3.26, 8.55 and 15.975 (2026-10-12).
        if !isdir(adas_cache_directory())
            @info "no ADF11 cache: the coronal balance of argon is skipped"
            @test_skip isdir(adas_cache_directory())
        else
            term = IonRateTerm(IonComposition(["D" => 1, "Ar" => 18]))
            n_e = 1e16
            fraction(T) = coronal_equilibrium(term, 2, n_e, T; density=1.0)
            mean_charge(T) = sum((0:18) .* fraction(T))
            function peak(z)
                grid = 10 .^ range(log10(5.0), log10(3000.0); length=1201)
                return grid[argmax([fraction(T)[z+1] for T in grid])]
            end
            for (z, references) in ((8, (54.8, 59.3)), (16, (865.0, 873.0)))
                for reference in references
                    @test 0.75 < peak(z) / reference < 1.33
                end
            end
            for (T, references) in ((10.0, (3.03, 2.97)), (100.0, (8.56, 8.48)), (1000.0, (15.972, 15.970)))
                for reference in references
                    @test abs(mean_charge(T) - reference) < (T == 10.0 ? 0.5 : 0.2)
                end
            end
        end
    end

    @testset "the cross sections of the kinetic ionization against measurements (item 4)" begin
        # Measured cross sections of the neutral atoms [m²]:
        # - hydrogen, M. B. Shah, D. S. Elliott, H. B. Gilbody, J. Phys. B 20, 3501 (1987),
        #   Table 1 (atomic H, ±6.8 % in the absolute scale), at its rows of 102.0, 198.2,
        #   508.2 and 998.2 eV;
        # - argon, single ionization σ⁺ of H. C. Straub et al., Phys. Rev. A 52, 1115 (1995),
        #   Table II (±3.5 %), at 100 eV.
        # Above 100 keV the relativistic Bethe asymptote σ = 4π(ħ/mc)² [M² (β⁻² ln(β²/(1 − β²)) − 1)
        # + C β⁻²] of F. F. Rieke, W. Prepejchal, Phys. Rev. A 6, 1507 (1972), eq. (3), with
        # M² = 4.22, C = 37.93 for argon (their Table II, pure gas) and for atomic hydrogen
        # M² = 0.2823, C = 4.007 from the fit σE = 1.30 log₁₀E + 2.85 (πa₀², hartree) of Shah et
        # al., eqs. (4)-(5) (C = M²(ln c + 11.2268), M. Inokuti, Rev. Mod. Phys. 43, 297 (1971),
        # eq. (4.27)).
        # Asserted: both forms fit hydrogen to 10 % at 0.1 to 1 keV and argon at 100 eV, and the
        # form with 1/β² has the energy dependence of the asymptote from 0.5 to 2 MeV, where
        # DREAM's grows away from it. Departures that the fits carry and the test documents:
        # argon falls to 0.75 of σ⁺ at 0.5 to 1 keV in both forms; at 1 MeV DREAM's form is
        # 1.72 times the asymptote for hydrogen (and grows), and 0.92 for argon by a
        # coincidence of its slope; the form with 1/β² is 1.43 and 0.74.
        mc2 = Constants.ELECTRON_REST_ENERGY
        momentum(E) = sqrt((1 + E / mc2)^2 - 1)
        for (E, measured) in ((102.0, 5.40e-21), (198.2, 3.98e-21), (508.2, 2.00e-21), (998.2, 1.13e-21)), form in (:dream, :bethe)
            @test 0.9 < ionization_cross_section("H", 0, momentum(E); form) / measured < 1.1
        end
        for form in (:dream, :bethe)
            @test 0.9 < ionization_cross_section("Ar", 0, momentum(100.0); form) / 2.64e-20 < 1.15
        end
        compton = 4π * 3.8615926796e-13^2       # 4π(ħ/mc)² [m²], the reduced Compton wavelength of CODATA 2018
        function asymptote(M2, C, E)
            p = momentum(E)
            β2 = p^2 / (1 + p^2)
            return compton * (M2 * (log(β2 / (1 - β2)) / β2 - 1) + C / β2)
        end
        energies = (0.5e6, 1e6, 2e6)
        for (element, M2, C) in (("H", 0.2823, 4.007), ("Ar", 4.22, 37.93))
            ratio(form) = [ionization_cross_section(element, 0, momentum(E); form) / asymptote(M2, C, E) for E in energies]
            @test maximum(ratio(:bethe)) / minimum(ratio(:bethe)) < 1.08
            element == "H" && @test maximum(ratio(:dream)) / minimum(ratio(:dream)) > 1.3
        end
    end

    @testset "Gould's relativistic correction from the 4BN cross section (item 6)" begin
        # The power of electron-ion bremsstrahlung of a Jüttner distribution, n_e n_i Z² α r₀² m c³
        # ⟨β γ φ⟩ with φ = bremsstrahlung_cross_section (the 4BN of Koch and Motz), over its
        # nonrelativistic limit (16/3)√(8θ/π): (R − 1)/θ tends to the coefficient of θ in
        # Gould's power, 11/24 after his erratum (R. J. Gould, Astrophys. J. 238, 1026 (1980),
        # eq. (18); Astrophys. J. 243, 677 (1981)), not the 19/24 of the paper and of DREAM.
        # The slope, linear in θ, is extrapolated from θ = 1e-3 and 2e-3.
        function excess(θ)
            jüttner(p) = p^2 * exp(-p^2 / (1 + sqrt(1 + p^2)) / θ)
            top = 40 * sqrt(2θ)
            norm = quadgk(jüttner, 0, top; rtol=1e-13)[1]
            average = quadgk(p -> jüttner(p) * p * bremsstrahlung_cross_section(p), 0, top; rtol=1e-13)[1] / norm
            return (average / (16 / 3 * sqrt(8θ / π)) - 1) / θ
        end
        slope = 2 * excess(1e-3) - excess(2e-3)
        @test slope ≈ Runes2P.GOULD_ION_CORRECTION atol = 1e-4
        @test Runes2P.GOULD_ION_CORRECTION == 11 / 24
        @test abs(slope - Runes2P.DREAM_GOULD_ION_CORRECTION) > 0.3
    end

    @testset "the 4BN stopping power against 3BN and ESTAR (item 7)" begin
        # 4BN, the radiative energy-loss cross section φ = (1/E₀)∫k dσ_k of H. W. Koch and J. W.
        # Motz, Rev. Mod. Phys. 31, 920 (1959), against the integral of their differential cross
        # section 3BN (Table I, p. 925; Born, no screening) over the photon energy, in units of
        # α r₀² Z² and of mₑc (energies E, momenta p of the electron before (₀) and after).
        function bn3(E0, p0, E, p)
            k = E0 - E
            ε0, ε = log((E0 + p0) / (E0 - p0)), 2atanh(p / E)
            L = 2log((E0 * E + p0 * p - 1) / k)
            return (p / p0) / k * (4 / 3 - 2 * E0 * E * (p^2 + p0^2) / (p^2 * p0^2) + ε0 * E / p0^3 + ε * E0 / p^3 -
                                   ε * ε0 / (p0 * p) +
                                   L * (8 * E0 * E / (3p0 * p) + k^2 * (E0^2 * E^2 + p0^2 * p^2) / (p0^3 * p^3) +
                                        k / (2p0 * p) * ((E0 * E + p0^2) / p0^3 * ε0 - (E0 * E + p^2) / p^3 * ε +
                                                         2k * E0 * E / (p^2 * p0^2))))
        end
        for p0 in (0.3, 1.0, 3.0)
            E0 = sqrt(1 + p0^2)
            # over the final momentum p = p₀(1 − s²), with k = E₀ − E(p) and dk = (p/E) dp: the
            # logarithm of k at k → 0 becomes s ln s, smooth for the quadrature
            integrand(s) = (p = p0 * (1 - s^2); E = sqrt(1 + p^2); (E0 - E) * bn3(E0, p0, E, p) * p / E * 2p0 * s)
            integral = quadgk(integrand, 0, 1; rtol=1e-12)[1] / E0
            @test integral ≈ bremsstrahlung_cross_section(p0) rtol = 1e-9
            @test !(integral ≈ bremsstrahlung_cross_section(p0; constant=:dream))
        end
        # the radiative stopping power of hydrogen of NIST ESTAR (M. J. Berger et al., ESTAR,
        # https://physics.nist.gov/Star, exported 2026-10-12) at 10 and 22.4 keV, where the
        # electron-electron bremsstrahlung that ESTAR adds is small and the Born cross section
        # lies below the exact one (Koch and Motz, p. 926): ESTAR is at most a few percent above
        # 4BN with 4/3, and lies below the form with DREAM's 1
        mc2 = 0.51099895                                   # MeV
        to_estar = 6.02214076e23 / 1.00794 * Constants.FINE_STRUCTURE_CONSTANT * (Constants.CLASSICAL_ELECTRON_RADIUS * 100)^2 * mc2
        for (T, estar) in ((0.010, 9.702e-4), (0.0224, 1.012e-3))
            p = sqrt((1 + T / mc2)^2 - 1)
            stopping(constant) = to_estar * sqrt(1 + p^2) * bremsstrahlung_cross_section(p; constant)
            @test 1.0 < estar / stopping(:koch_motz) < 1.06
            @test estar / stopping(:dream) < 1.0
        end
    end
end
