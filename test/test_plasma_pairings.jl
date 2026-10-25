# Terms that must agree take their plasma from the same source and in the same treatment,
# explicit with explicit and implicit with implicit: one temperature for the
# terms of the cold bulk, the Fokker-Planck and the field-particle term of one unknown
# lagged with lagged, a corrected conductivity with an implicit Fokker-Planck term, and the
# knock-on cutoff reading the temperature of the state.

@testset "plasma pairings" begin
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    g = cylindrical_grid(MomentumGrid(uniform_p_faces(12, 0.0, 0.1), uniform_xi_faces(2)), rad)
    settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :thermal)
    base(t) = PlasmaState(; cold_electron_temperature = [10.0, 8.0] .* exp(-t / 1e-3),
        ions = [IonSpecies("D", 1, 1, [1e19, 5e18])], electric_field = zeros(2), magnetic_field = zeros(2))
    from_unknowns = PlasmaFromUnknowns(base; density = nothing)
    operator() = AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
    fp(lagged = :ions; plasma = from_unknowns) = FokkerPlanckTerm(
        :f, operator(); plasma, settings, lagged)
    field_particle(plasma) = FieldParticleTerm(:f, FieldParticleOperator(g, base(0.0), settings); plasma)
    minus(name) = LinearTerm(name; coefficient = -1)
    check(terms...) = Runes2P._check_plasma_pairings([Equation(:f, TimeDerivative(), terms...)])

    @testset "Fokker-Planck and field-particle term" begin
        @test check(fp(:all), field_particle(from_unknowns)) === nothing
        @test_throws "misses the change of the electron-electron part" check(
            fp(:ions), field_particle(from_unknowns))
        # a prescribed plasma on both: the same kind of source
        @test check(PrescribedFokkerPlanckTerm(:f, operator(), base; settings), field_particle(base)) ===
              nothing
        @test_throws "different sources" check(
            PrescribedFokkerPlanckTerm(:f, operator(), base; settings),
            field_particle(nothing))
        @test_throws "different sources" check(FokkerPlanckTerm(:f, operator(); settings), field_particle(base))
        # one of the two following the unknowns, the other one fixed
        @test_throws ArgumentError check(fp(:all), field_particle(nothing))
        # through the system
        layout = UnknownLayout(rad,
            [
                KineticUnknown(:f, g; scale = 1e19), RadialUnknown(:cold_electron_temperature)])
        @test_throws "misses the change of the electron-electron part" EquationSystem(
            layout,
            [Equation(:f, TimeDerivative(), fp(), field_particle(from_unknowns)),
                Equation(:cold_electron_temperature, Algebraic(),
                    PrescribedValueTerm([10.0, 8.0]),
                    minus(:cold_electron_temperature))])
    end

    @testset "one temperature for the cold bulk" begin
        @test check(fp(), ParticleSourceTerm(:particle_source, g, :cold_electron_temperature)) ===
              nothing
        @test_throws "prescribed or fixed temperature" check(
            fp(), ParticleSourceTerm(:particle_source, g, [10.0, 8.0]))
        @test_throws "one temperature" check(fp(), ParticleSourceTerm(:particle_source, g, :other_temperature))
        thermal(temperature) = DensityMomentTerm(:f, g; region = MomentumRegion(ThermalMomentum(temperature)))
        @test check(fp(), thermal(:cold_electron_temperature)) === nothing
        @test_throws "prescribed or fixed temperature" check(fp(), thermal([10.0, 8.0]))
        # an operator of radial transport alone reads no temperature, a frozen collision
        # operator a fixed one
        @test check(fp(), FokkerPlanckTerm(:f, AdvectionDiffusion(g))) === nothing
        @test_throws "prescribed or fixed temperature" check(
            fp(), FokkerPlanckTerm(:f, fokker_planck_operator(g, base(0.0),
                settings)))
        # the moments of one hot region count the same electrons
        hot(factor; side = :above) = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor); side)
        @test check(fp(), DensityMomentTerm(:f, g; region = hot(5)),
            EnergyMomentTerm(:f, g; region = hot(5; side = :below))) === nothing
        @test_throws "one region" check(fp(), DensityMomentTerm(:f, g; region = hot(5)),
            EnergyMomentTerm(:f, g; region = hot(7)))
        # a moment without a region reads no temperature
        @test check(fp(), DensityMomentTerm(:f, g)) === nothing
        # without a temperature unknown, fixed values stay free
        @test check(FokkerPlanckTerm(:f, operator(); settings),
            ParticleSourceTerm(:particle_source, g, [10.0, 8.0])) === nothing
        # a conductivity with a prescribed plasma next to a bulk that follows the unknown
        @test_throws "prescribed or fixed temperature" check(
            fp(), ConductivityTerm(:electric_field, ConductivityModel(base)))
    end

    @testset "corrected conductivity with an implicit Fokker-Planck term" begin
        corrected = ConductivityTerm(:electric_field, ConductivityModel(from_unknowns; kinetic = :dream_fit))
        plain = ConductivityTerm(:electric_field, ConductivityModel(from_unknowns))
        @test check(fp(:ions), corrected) === nothing
        @test_throws "lag no more than the ions" check(fp(:all), corrected)
        @test check(fp(:all), plain) === nothing
    end

    @testset "knock-on cutoff from the temperature of the state" begin
        hot = cylindrical_grid(MomentumGrid(biuniform_p_faces(20, 8, 0.0, 0.6, 2.5), uniform_xi_faces(4)), rad)
        cutoff = momentum_grid(hot).p_faces[5]
        knockon_plasma(t) = PlasmaState(;
            cold_electron_temperature = [2000.0, 1000.0] .* exp(-t / 1e-3),
            ions = [IonSpecies("D", 1, 1, [1e20, 3e19])], electric_field = [3.0, 1.0],
            magnetic_field = [5.0, 5.0])
        layout = UnknownLayout(rad, [RadialUnknown(:cold_electron_temperature; scale = 1000.0)])
        prescribed = KnockOnCutoffUpdate(KnockOnOperator(hot, hot, cutoff), knockon_plasma)
        following = KnockOnCutoffUpdate(KnockOnOperator(hot, hot, cutoff),
            PlasmaFromUnknowns(knockon_plasma; density = nothing))
        # the state holds the temperature of the prescribed plasma at a later time, which the
        # update reads at the time 0 of its prescribed rest
        seen = Vector{Float64}[]
        for t in (0.0, 3e-3)
            u = copy(knockon_plasma(t).cold_electron_temperature)
            Runes2P.apply_update!(prescribed, u, layout, t)
            Runes2P.apply_update!(following, u, layout, 0.0)
            @test following.operator.p_cutoff == prescribed.operator.p_cutoff
            push!(seen, copy(following.operator.p_cutoff))
        end
        @test seen[1] != seen[2]   # the cutoff followed the temperature of the state
        record = describe(following, DescriptionContext(; times = [0.0]))
        @test record["plasma_unknowns"]["temperature"] == "cold_electron_temperature"
        # a system whose bulk reads the temperature unknown refuses the prescribed cutoff
        system = (; equations = [Equation(:f, TimeDerivative(), fp())])
        @test_throws "knock-on cutoff" Runes2P.check_update(prescribed, system)
        @test Runes2P.check_update(following, system) === nothing
        @test Runes2P.check_update(prescribed,
            (;
                equations = [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, operator()))])) ===
              nothing
    end

    # the terms of the ion charge states and their energies (one numbered test per check)
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the pairings of the ion terms are skipped"
        @test_skip isdir(adas_cache_directory())
    else
        @testset "ion terms" begin
            c = IonComposition(["D" => 1, "Ar" => 18])
            argon = zeros(19, 2)
            argon[2, :] .= 1e17
            prescribed = PlasmaState(; cold_electron_temperature = [10.0, 8.0],
                electric_field = zeros(2), magnetic_field = zeros(2),
                ions = [IonSpecies("D", 1, 1, [1e19, 1e19]), IonSpecies("Ar", 18, argon)])
            region(factor; temperature = :cold_electron_temperature) = MomentumRegion(ThermalMomentum(
                temperature; factor))
            # the split at 7 thermal momenta, the minimum with CHIANTI's cross sections
            k = KineticIonizationTerm(c, :f, g; region = region(7))
            rate = IonRateTerm(c; kinetic = k)
            radiation = RadiatedPowerTerm(prescribed; ions = :ion_densities, kinetic = k)
            closure = population_equations(c)
            ions(terms...) = Equation(:ion_densities, TimeDerivative(), terms...)
            temperature(terms...) = Equation(:cold_electron_temperature, TimeDerivative(), terms...)
            paired(equations...) = Runes2P._check_plasma_pairings([equations...;
                                                                   closure...])
            @test paired(ions(rate, k), temperature(radiation)) === nothing
            # 1: the kinetic part of the split is a term of the system, of the same ions
            @test_throws "not a term of the system" paired(ions(rate), temperature(radiation))
            @test_throws "same ion unknown" Runes2P._check_plasma_pairings([
                ions(IonRateTerm(c; ions = :other_ions, kinetic = k), k), population_equations(
                    c; ions = :other_ions)..., closure...])
            # 2: the binding energy at the rate of the rate equations
            @test_throws "split the ionization differently" paired(ions(rate, k),
                temperature(RadiatedPowerTerm(prescribed; ions = :ion_densities)))
            @test_throws "split the ionization differently" paired(
                ions(IonRateTerm(c), k), temperature(radiation))
            # 3: the threshold of the split and of the hot density count the same electrons
            @test_throws "different thresholds" paired(
                ions(rate, k), temperature(radiation),
                Equation(:n_hot, Algebraic(), DensityMomentTerm(:f, g; region = region(5))))
            # 4: the threshold reads the temperature of the rates
            hot = KineticIonizationTerm(c, :f, g; region = region(7; temperature = :other_temperature))
            @test_throws ArgumentError paired(ions(IonRateTerm(c; kinetic = hot), hot))
            @test_throws ArgumentError paired(ions(rate, k),
                temperature(radiation, SecondaryHeatingTerm(k; temperature = :other)))
            # 5: no kinetic ionization without the closure of the cold density
            @test_throws "no closure" Runes2P._check_plasma_pairings([
                ions(rate, k), temperature(radiation)])
            # 6: one source of ions where the system evolves them, and both sides of the exchange on it
            fp(plasma) = FokkerPlanckTerm(
                :f, AdvectionDiffusion(g; scheme = :exponential_fitting); plasma, settings)
            @test_throws "prescribed ions" paired(ions(rate, k), temperature(radiation),
                Equation(:f, TimeDerivative(), fp(PlasmaFromUnknowns(prescribed))))
            @test paired(ions(rate, k), temperature(radiation),
                Equation(:f, TimeDerivative(), fp(PlasmaFromUnknowns(prescribed; ions = :ion_densities)))) ===
                  nothing
            @test_throws "prescribed ions" paired(
                ions(rate, k), temperature(radiation, RadiatedPowerTerm(prescribed)))
            # a conductivity on the prescribed ions, the temperature of the unknowns
            @test_throws "prescribed ions" paired(ions(rate, k), temperature(radiation),
                Equation(:electric_field, Algebraic(),
                    ConductivityTerm(:electric_field,
                        ConductivityModel(PlasmaFromUnknowns(prescribed)))))
            @test paired(ions(rate, k), temperature(radiation),
                Equation(:electric_field, Algebraic(),
                    ConductivityTerm(:electric_field,
                        ConductivityModel(PlasmaFromUnknowns(prescribed; ions = :ion_densities))))) ===
                  nothing
            exchange(side, ions) = IonEnergyExchangeTerm(prescribed; side, ions)
            @test_throws "both sides" paired(
                ions(rate, k), temperature(radiation, exchange(:electrons, :ion_densities)),
                Equation(:ion_thermal_energy, TimeDerivative(), exchange(:ions, :other_ions)))
            # 7: the ion energies from the densities of the state after a deposit
            layout = UnknownLayout(rad,
                [
                    ion_densities_unknown(c), RadialUnknown(:ion_thermal_energy; components = 2)])
            u = zeros(length(layout))
            set_ion_densities!(u, layout, prescribed)
            deposit_neutrals!(u, layout, c, "Ar" => 3e17)
            set_ion_thermal_energy!(u, layout, prescribed, 2.0; ions = :ion_densities)
            @test [radial_view(u, layout, 2, ir)[2] for ir in 1:2] ≈
                  THERMAL_ENERGY_FACTOR .* [4e17, 4e17] .* 2.0
            @test ion_temperatures(u, layout, prescribed; ions = :ion_densities) ≈
                  fill(2.0, 2, 2)
            @test ion_temperatures(u, layout, prescribed)[2, :] ≈ [8.0, 8.0]   # the prescribed 1e17 do not hold
            # 8: a temperature that only fluid terms read gets the default floor, unless it declares a limit
            fluid = [ions(IonRateTerm(c)),
                temperature(RadiatedPowerTerm(prescribed; ions = :ion_densities))]
            @test Runes2P.fluid_temperature_floors(fluid) ==
                  [:cold_electron_temperature => Runes2P.DEFAULT_TEMPERATURE_FLOOR]
            @test Runes2P.DEFAULT_TEMPERATURE_FLOOR == 0.5
            @test isempty(Runes2P.fluid_temperature_floors([fluid;
                                                            Equation(:f, TimeDerivative(),
                                                                fp(PlasmaFromUnknowns(
                                                                    prescribed;
                                                                    ions = :ion_densities)))]))
            # the workspace of a fluid-only system: the floor on every entry of the temperature, a
            # declared limit kept as it is, and a step whose solution lies below the floor refused
            let d = IonComposition(["D" => 1]), nr = 2
                rad = RadialGrid(uniform_r_faces(nr, 0.5))
                function workspace(limit; target = 5.0)
                    kw = limit === nothing ? (; scale = 10.0) :
                         (; scale = 10.0, nonnegative = limit)
                    lay = UnknownLayout(rad,
                        [ion_densities_unknown(d),
                            RadialUnknown(:cold_electron_density; scale = 1e19),
                            RadialUnknown(:cold_electron_temperature; kw...)])
                    eqs = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(d)),
                        population_equations(d)...,
                        Equation(:cold_electron_temperature, Algebraic(),
                            PrescribedValueTerm(fill(target, nr)),
                            LinearTerm(:cold_electron_temperature; coefficient = -1))]
                    u0 = zeros(length(lay))
                    set_ion_densities!(u0, lay, [IonSpecies("D", 1, 1, fill(1e19, nr))])
                    radial_values(u0, lay, 2) .= 1e19
                    radial_values(u0, lay, 3) .= 5.0
                    sys = EquationSystem(lay, eqs; time_scale = 1e-7, initial_state = u0)
                    ws = NewtonWorkspace(sys)
                    entries = reduce(vcat,
                        [unknown_range(lay, unknown_id(lay, :cold_electron_temperature), ir)
                         for ir in 1:nr])
                    limits = [ws.limit_value[findfirst(==(i), ws.limited)] for i in entries]
                    return sys, u0, limits
                end
                @test workspace(nothing)[3] == fill(0.5, nr)
                @test workspace(true)[3] == fill(0.0, nr)
                @test workspace(LowerLimit(value = 0.2))[3] == fill(0.2, nr)
                @test workspace(LowerLimit(value = 0.8))[3] == fill(0.8, nr)
                # the prescribed value 0.3 eV lies below the floor: refused; with the limit 0 it is reached
                sys, u0, _ = workspace(nothing; target = 0.3)
                @test newton_solve!(copy(u0), u0, NewtonWorkspace(sys), 0.0, 1e-7).status ===
                      :below_lower_limit
                sys, u0, _ = workspace(true; target = 0.3)
                u = copy(u0)
                @test newton_solve!(u, u0, NewtonWorkspace(sys), 0.0, 1e-7).status ===
                      :converged
                @test radial_values(u, sys.layout, 3) ≈ fill(0.3, nr)
            end
            # 9: the fluid terms say what to do with a plasma from the unknowns
            @test_throws "keywords temperature, density and ions" RadiatedPowerTerm(PlasmaFromUnknowns(prescribed))
            @test_throws "keywords temperature, density and ions" IonEnergyExchangeTerm(
                PlasmaFromUnknowns(prescribed);
                side = :ions)
            @test_throws "keywords temperature, density and ions" RunawayFluidEnergyTransferTerm(PlasmaFromUnknowns(prescribed))
            # 10: one composition per ion unknown
            swapped = IonComposition(["Ar" => 18, "D" => 1])
            @test_throws "one composition" paired(
                ions(rate, k, IonDiffusionTerm(swapped; diffusion = 1.0)),
                temperature(radiation))
            # 11: the sinks of W_hot over its region
            loss_source = fp(PlasmaFromUnknowns(prescribed; ions = :ion_densities))
            energy = total_energy_equation(:f, g; region = region(7))
            sinks(region) = Equation(:total_electron_energy, TimeDerivative(),
                KineticEnergyLossTerm(loss_source; part = :sinks,
                    region))
            base_system = [ions(rate, k), temperature(radiation),
                Equation(:f, TimeDerivative(), loss_source), energy]
            @test paired(base_system..., sinks(region(7))) === nothing
            @test_throws "region of the energy moment" paired(base_system..., sinks(nothing))
            @test_throws "different thresholds" paired(base_system..., sinks(region(5)))
            # 12: the rates and the binding energy from the same extrapolation of the tables
            @test_throws "extrapolations" paired(
                ions(IonRateTerm(c; kinetic = k, extrapolation = :clamp_density), k),
                temperature(radiation))
        end
    end
end
