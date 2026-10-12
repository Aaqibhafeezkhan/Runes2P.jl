# The thermal energy of the ions and its exchange with the cold electrons:
# the masses, the exchange against Spitzer's two-temperature relaxation of the NRL
# formulary, between the electrons and the ions and among the ions with the mixed ion-ion
# logarithm, the conservation of the energy, the relaxation of a
# run to the common temperature, the Jacobian, and the refusal of one side without the other.

@testset "ion energy" begin
    e = Constants.ELEMENTARY_CHARGE
    u_mass = Runes2P.ATOMIC_MASS_UNIT

    @testset "masses" begin
        # the nuclei of CODATA 2018 and the standard atomic weights of CIAAW
        @test ion_mass(IonSpecies("D", 1, 1, [1e19])) == 3.3435837724e-27
        @test ion_mass(IonSpecies("H", 1, 1, [1e19])) == 1.67262192369e-27
        @test ion_mass(IonSpecies("T", 1, 1, [1e19])) == 5.0073567446e-27
        @test ion_mass(IonSpecies("Ar", 18, 1, [1e19])) == 39.95 * u_mass
        @test ion_mass(IonSpecies("W", 74, 1, [1e19])) == 183.84 * u_mass
        @test ion_mass(IonSpecies("Xe", 54, 1, [1e19])) == 131.293 * u_mass
        @test_throws ArgumentError ion_mass(IonSpecies("Tc", 43, 1, [1e19]))
        # DREAM's: the atomic mass of ³H, and 2.3 Z beyond Z = 40
        @test ion_mass(IonSpecies("T", 1, 1, [1e19]); table=:dream) == 5.00826765e-27
        @test ion_mass(IonSpecies("T", 1, 1, [1e19]); table=:dream) - ion_mass(IonSpecies("T", 1, 1, [1e19])) ≈
              Constants.ELECTRON_MASS rtol = 1e-4
        @test ion_mass(IonSpecies("W", 74, 1, [1e19]); table=:dream) ≈ 2.3 * 74 * u_mass rtol = 1e-14
        @test ion_mass(IonSpecies("Ar", 18, 1, [1e19]); table=:dream) == 39.95 * u_mass
        @test_throws ArgumentError ion_mass(IonSpecies("D", 1, 1, [1e19]); table=:other)
        @test_throws ArgumentError IonEnergyExchangeTerm(PlasmaState(; cold_electron_temperature=[1.0],
                                                                    ions=[IonSpecies("D", 1, 1, [1e19])],
                                                                    electric_field=[0.0], magnetic_field=[0.0]);
                                                         side=:ions, ion_coulomb_logarithm=:other)
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

    @testset "among the ions against Spitzer's relaxation (issue #183, item 8)" begin
        # The exchange between deuterium and Ne⁺ that ion_ion adds, against the rate of the NRL
        # formulary for two ion species, Q = (3/2) e N_D ν (T_Ne − T_D),
        # ν = c √(m_D m_Ne) Z_D² Z_Ne² n_Ne λ / (m_D T_Ne + m_Ne T_D)^{3/2} (cgs), with the mixed
        # ion-ion logarithm of the formulary (2019, p. 34) written out here, screened by the two
        # ions and the electrons, λ = 23 − ln[Z Z′ (μ + μ′)/(μ T′ + μ′ T) (n Z²/T + n′ Z′²/T′ + n_e/T_e)^{1/2}]
        T_e, T_D, T_Ne = 10.0, 1000.0, 300.0
        u = state(T_e, [T_D, T_Ne])
        both = exchange(ions, u, 4)[Runes2P.unknown_range(layout, 4, 1)]
        alone = exchange(IonEnergyExchangeTerm(plasma; side=:ions, ion_ion=false), u, 4)[Runes2P.unknown_range(layout, 4, 1)]
        Q = both[1] - alone[1]
        m_D, m_Ne = 3.3435837724e-24, 20.180 * Runes2P.ATOMIC_MASS_UNIT * 1e3      # g
        μ_D, μ_Ne = m_D / 1.67262192369e-24, m_Ne / 1.67262192369e-24
        λ = 23 - log((μ_D + μ_Ne) / (μ_D * T_Ne + μ_Ne * T_D) * sqrt(n_D * 1e-6 / T_D + n_Ne * 1e-6 / T_Ne + n_e * 1e-6 / T_e))
        e_cgs = 4.80320471e-10
        c = 8 * sqrt(2π) * e_cgs^4 / (3 * (1.602176634e-12)^1.5)
        ν = c * sqrt(m_D * m_Ne) * (n_Ne * 1e-6) * λ / (m_D * T_Ne + m_Ne * T_D)^1.5
        @test Q ≈ 1.5 * e * n_D * ν * (T_Ne - T_D) rtol = 2e-3
        @test Q ≈ -(both[2] - alone[2]) rtol = 1e-12
        # DREAM's logarithm takes T_e and n_free, the same for every pair
        dream = exchange(IonEnergyExchangeTerm(plasma; side=:ions, ion_coulomb_logarithm=:dream), u, 4)[Runes2P.unknown_range(layout, 4, 1)]
        λ_dream = 17.3 - 0.5 * log(n_e / 1e20) + 1.5 * log(T_e / 1e3)
        @test (dream[1] - alone[1]) / Q ≈ λ_dream / λ rtol = 1e-10
    end

    @testset "the logarithm among the ions after the thermal quench (issue #183, item 8)" begin
        # deuterium and injected deuterium, 5e19 m⁻³ each, at 1 keV and 900 eV in a plasma at
        # T_e = 10 eV: the NRL logarithm 23 − ln[2/(T_a + T_b) √(n_a/T_a + n_b/T_b + n_e/T_e)] (cgs)
        # is 14.9, DREAM's 17.3 + 3/2 ln(10 eV/1 keV) = 10.39, so DREAM's exchange among them is
        # 1.43 times too slow (an estimate screened by the ions alone gives 1.6)
        cold = PlasmaState(; cold_electron_temperature=[10.0], ions=[IonSpecies("D", 1, 1, [5e19]), IonSpecies("D_inj", 1, 1, [5e19])],
                           electric_field=[0.0], magnetic_field=[0.0])
        small = UnknownLayout(radial, [RadialUnknown(:cold_electron_temperature), RadialUnknown(:cold_electron_density),
                                       RadialUnknown(:ion_thermal_energy; components=2, nonnegative=true)])
        v = zeros(length(small))
        radial_values(v, small, 1) .= 10.0
        radial_values(v, small, 2) .= 1e20
        set_ion_thermal_energy!(v, small, cold, [1000.0, 900.0])
        received(kind; ion_ion=true) =
            Runes2P.rhs!(zeros(length(v)), IonEnergyExchangeTerm(cold; side=:ions, ion_ion, ion_coulomb_logarithm=kind),
                      v, small, 3, 0.0)[Runes2P.unknown_range(small, 3, 1)][1]
        to_electrons = received(:nrl; ion_ion=false)
        λ_nrl = 23 - log(2 / (1000.0 + 900.0) * sqrt(5e13 / 1000.0 + 5e13 / 900.0 + 1e14 / 10.0))
        λ_dream = 17.3 + 1.5 * log(10.0 / 1e3)
        @test λ_nrl ≈ 14.88 atol = 0.01
        @test λ_dream ≈ 10.39 atol = 0.01
        @test (received(:nrl) - to_electrons) / (received(:dream) - to_electrons) ≈ λ_nrl / λ_dream rtol = 1e-12
        @test 1.4 < λ_nrl / λ_dream < 1.5
        # cold, dense, highly charged ions take the formula below zero (Ar¹⁶⁺ with Ar¹⁶⁺ at 0.5 eV
        # and 10¹⁹ m⁻³, screened by the two alone: −1.7), where it is cut so that the exchange
        # keeps its sign
        m_Ar = 39.95 * u_mass
        μ = m_Ar / Constants.PROTON_MASS
        species(n, nZ, nZ2, T, m) = (n, nZ2, THERMAL_ENERGY_FACTOR * n * T, m, nZ)
        argon(T) = species(1e19, 16e19, 256e19, T, m_Ar)
        screening(T) = 1e-6 * 2 * 256e19 / T
        @test 23 - log(256 * 2μ / (2μ * 0.5) * sqrt(screening(0.5))) ≈ -1.67 atol = 0.01
        @test Runes2P._nrl_ion_logarithm(argon(0.5), argon(0.5), screening(0.5)) == 0
        @test Runes2P._nrl_ion_logarithm(argon(1e3), argon(1e3), screening(1e3)) ≈
              23 - log(256 * 2μ / (2μ * 1e3) * sqrt(screening(1e3))) rtol = 1e-14
        # a species of several charge states takes its mean charge (nZ²)/(nZ): Ar⁺ and Ar³⁺,
        # 1e19 m⁻³ each, give 10/4 = 2.5; a species of neutrals alone takes 1; a temperature
        # below the floor of 0.5 eV takes the floor
        μ_D = 3.3435837724e-27 / Constants.PROTON_MASS
        deuterium = species(1e20, 1e20, 1e20, 100.0, 3.3435837724e-27)
        hand(Z, T) = 23 - log(Z * (μ + μ_D) / (μ * 100.0 + μ_D * T) * sqrt(1e13))
        @test Runes2P._nrl_ion_logarithm(species(2e19, 4e19, 10e19, 50.0, m_Ar), deuterium, 1e13) ≈ hand(2.5, 50.0) rtol = 1e-14
        @test Runes2P._nrl_ion_logarithm(species(2e19, 0.0, 0.0, 50.0, m_Ar), deuterium, 1e13) ≈ hand(1.0, 50.0) rtol = 1e-14
        @test Runes2P._nrl_ion_logarithm(species(2e19, 4e19, 10e19, 0.01, m_Ar), deuterium, 1e13) ≈ hand(2.5, 0.5) rtol = 1e-14
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
