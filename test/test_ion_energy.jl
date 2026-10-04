# The thermal energy of the ions and its exchange with the cold electrons:
# the masses, the exchange against Spitzer's two-temperature relaxation of the NRL
# formulary, the conservation of the energy, the relaxation of a run to the common
# temperature, the Jacobian, and the refusal of one side without the other.

@testset "ion energy" begin
    e = Constants.ELEMENTARY_CHARGE
    u_mass = Runes2P.ATOMIC_MASS_UNIT

    @testset "masses" begin
        @test ion_mass(IonSpecies("D", 1, 1, [1e19])) == 3.3435837724e-27
        @test ion_mass(IonSpecies("H", 1, 1, [1e19])) == 1.67262192369e-27
        @test ion_mass(IonSpecies("T", 1, 1, [1e19])) == 5.00826765e-27
        @test ion_mass(IonSpecies("Ar", 18, 1, [1e19])) == 39.95 * u_mass
        @test ion_mass(IonSpecies("W", 74, 1, [1e19])) ≈ 2.3 * 74 * u_mass rtol = 1e-14
    end

    # one radius, deuterium and singly ionized neon
    radial = RadialGrid(uniform_r_faces(1, 0.5))
    n_D, n_Ne = 8e19, 2e18
    neon = zeros(11, 1)
    neon[2, 1] = n_Ne
    plasma = PlasmaState(; cold_electron_temperature=[1000.0], ions=[IonSpecies("D", 1, 1, [n_D]), IonSpecies("Ne", 10, neon)],
                         electric_field=[0.0], magnetic_field=[0.0])
    n_e = n_D + n_Ne
    layout = UnknownLayout(radial, [RadialUnknown(:cold_electron_temperature; scale=1e3),
                                    RadialUnknown(:cold_electron_density; scale=1e20),
                                    RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e23),
                                    RadialUnknown(:ion_thermal_energy; components=2, scale=THERMAL_ENERGY_FACTOR * 1e22,
                                                  nonnegative=true)])
    function state(T_e, T_i)
        u = zeros(length(layout))
        radial_values(u, layout, 1) .= T_e
        radial_values(u, layout, 2) .= n_e
        radial_values(u, layout, 3) .= THERMAL_ENERGY_FACTOR * n_e * T_e
        set_ion_thermal_energy!(u, layout, plasma, T_i)
        return u
    end
    ions = IonEnergyExchangeTerm(plasma; side=:ions)
    electrons = IonEnergyExchangeTerm(plasma; side=:electrons)
    exchange(term, u, row) = (out = zeros(length(u)); Runes2P.rhs!(out, term, u, layout, row, 0.0); out)

    @testset "against Spitzer's relaxation" begin
        # Q = (3/2) e N_i ν (T_e − T_i), ν = c √(m_e m_i) Z² n_e lnΛ / (m_i T_e + m_e T_i)^{3/2} in
        # cgs units (g, cm⁻³, eV, s⁻¹), with c = 8√(2π) e⁴/(3 e_eV^{3/2}) = 1.754e-19 exactly and
        # 1.8e-19 in the NRL formulary; DREAM's form gave 0.9746 of the NRL value in a first
        # estimate, the ratio of the two coefficients
        T_e, T_i = 1000.0, 500.0
        u = state(T_e, T_i)
        only_deuterium = IonEnergyExchangeTerm(plasma; side=:ions, ion_ion=false)
        Q = exchange(only_deuterium, u, 4)[Runes2P.unknown_range(layout, 4, 1)][1]
        m_D, m_e = 3.3435837724e-24, Constants.ELECTRON_MASS * 1e3      # g
        lnL = thermal_coulomb_logarithm(T_e, n_e)
        e_cgs = 4.80320471e-10
        exact = 8 * sqrt(2π) * e_cgs^4 / (3 * (1.602176634e-12)^1.5)
        @test exact ≈ 1.754e-19 rtol = 1e-3
        rate(c) = c * sqrt(m_D * m_e) * (n_e * 1e-6) * lnL / (m_D * T_e + m_e * T_i)^1.5
        @test Q ≈ 1.5 * e * n_D * rate(exact) * (T_e - T_i) rtol = 2e-3
        @test Q ≈ 1.5 * e * n_D * rate(1.8e-19) * (T_e - T_i) rtol = 3e-2
    end

    @testset "conservation" begin
        u = state(800.0, [300.0, 1500.0])
        to_ions = exchange(ions, u, 4)[Runes2P.unknown_range(layout, 4, 1)]
        from_electrons = exchange(electrons, u, 1)[1]
        without_ion_ion = exchange(IonEnergyExchangeTerm(plasma; side=:ions, ion_ion=false), u, 4)[Runes2P.unknown_range(layout, 4, 1)]
        # the electrons lose what the ions receive from them, and the ion-ion exchange sums to zero
        @test sum(without_ion_ion) ≈ -from_electrons rtol = 1e-13
        @test sum(to_ions) ≈ sum(without_ion_ion) rtol = 1e-10
        @test to_ions[2] - without_ion_ion[2] < 0     # the hot neon gives energy to the deuterium
        # nothing flows at one temperature
        @test maximum(abs, exchange(ions, state(700.0, 700.0), 4)) < 1e-12 * abs(from_electrons)
    end

    @testset "relaxation" begin
        system = EquationSystem(layout, [Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), electrons),
                                         Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n_e]),
                                                  LinearTerm(:cold_electron_density; coefficient=-1)),
                                         thermal_energy_equation(),
                                         Equation(:ion_thermal_energy, TimeDerivative(), ions)]; initial_state=state(1000.0, 100.0))
        u0 = state(1000.0, [100.0, 2000.0])
        energy(u) = u[3] + sum(u[Runes2P.unknown_range(layout, 4, 1)])
        times = collect(range(0.0, 0.2; length=81))
        solution = integrate(system, u0, times; stepper=FixedSteps())
        for k in eachindex(times)
            @test energy(solution.states[:, k]) ≈ energy(u0) rtol = 1e-9
        end
        T_final = (n_e * 1000.0 + n_D * 100.0 + n_Ne * 2000.0) / (n_e + n_D + n_Ne)
        final = solution.states[:, end]
        @test final[1] ≈ T_final rtol = 1e-3
        @test ion_temperatures(final, layout, plasma)[:, 1] ≈ [T_final, T_final] rtol = 1e-3
    end

    @testset "Jacobian" begin
        system = EquationSystem(layout, [Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), electrons),
                                         Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n_e]),
                                                  LinearTerm(:cold_electron_density; coefficient=-1)),
                                         thermal_energy_equation(),
                                         Equation(:ion_thermal_energy, TimeDerivative(), ions)]; initial_state=state(900.0, 200.0))
        u = state(900.0, [200.0, 1300.0])
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        # the temperature, the density, and the energy of each species (a column of the other
        # species than the row)
        for column in (1, 2, Runes2P.unknown_range(layout, 4, 1)...)
            h = 1e-6 * abs(u[column])
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(u)), zeros(length(u))
            rhs!(F_up, up, system, 0.0)
            rhs!(F_down, down, system, 0.0)
            @test Vector(J[:, column]) ≈ (F_up .- F_down) ./ (2h) rtol = 1e-6 atol = 1e-12 * maximum(abs, F_up)
        end
    end

    @testset "refusals, guard and record" begin
        lonely = [Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), PrescribedValueTerm([0.0])),
                  Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n_e]), LinearTerm(:cold_electron_density; coefficient=-1)),
                  thermal_energy_equation(),
                  Equation(:ion_thermal_energy, TimeDerivative(), ions)]
        @test_throws "both sides" EquationSystem(layout, lonely; initial_state=state(900.0, 200.0))
        @test_throws ArgumentError IonEnergyExchangeTerm(plasma; side=:other)
        @test_throws DimensionMismatch IonEnergyExchangeTerm(plasma; side=:ions, masses=[1.0])
        three = UnknownLayout(radial, [RadialUnknown(:cold_electron_temperature), RadialUnknown(:cold_electron_density),
                                       RadialUnknown(:ion_thermal_energy; components=3)])
        @test_throws "one component per ion species" Runes2P.validate(ions, three, 3)
        @test Runes2P._temperature_source(ions) == (:unknown, :cold_electron_temperature)
        record = describe(electrons, Runes2P.DescriptionContext())
        @test record["side"] == "electrons" && record["ion_ion"] == true
    end
end
