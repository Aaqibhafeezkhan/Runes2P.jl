# Independent references for results that rested on DREAM alone: the energy of a closed
# plasma with ionizing charge states, the time integration of the energy of the cold
# electrons against a closed form, and the split of a Maxwellian at the threshold of the
# kinetic ionization against the continuous integrals.
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
        # (3/2) e n dT/dt = −C √T n² (1 + a T), a = (19/24 + κ)/mₑc² (GOULD_ION_CORRECTION,
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
end
