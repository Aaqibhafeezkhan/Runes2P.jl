# Terms that must agree take their plasma from the same source and in the same treatment:
# one temperature for the
# terms of the cold bulk, the Fokker-Planck and the field-particle term of one unknown
# lagged with lagged, a corrected conductivity with an implicit Fokker-Planck term, and the
# knock-on cutoff reading the temperature of the state.

@testset "plasma pairings" begin
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    g = cylindrical_grid(MomentumGrid(uniform_p_faces(12, 0.0, 0.1), uniform_xi_faces(2)), rad)
    settings = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal)
    base(t) = PlasmaState(; cold_electron_temperature=[10.0, 8.0] .* exp(-t / 1e-3),
                          ions=[IonSpecies("D", 1, 1, [1e19, 5e18])], electric_field=zeros(2), magnetic_field=zeros(2))
    from_unknowns = PlasmaFromUnknowns(base; density=nothing)
    operator() = AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
    fp(coefficients=:implicit; plasma=from_unknowns) = FokkerPlanckTerm(:f, operator(); plasma, settings, coefficients)
    field_particle(plasma) = FieldParticleTerm(:f, FieldParticleOperator(g, base(0.0), settings); plasma)
    minus(name) = LinearTerm(name; coefficient=-1)
    check(terms...) = Runes2P._check_plasma_pairings([Equation(:f, TimeDerivative(), terms...)])

    @testset "Fokker-Planck and field-particle term (#70)" begin
        @test check(fp(:lagged), field_particle(from_unknowns)) === nothing
        @test_throws "issue #70" check(fp(:implicit), field_particle(from_unknowns))
        # a prescribed plasma on both: the same kind of source
        @test check(PrescribedFokkerPlanckTerm(:f, operator(), base; settings), field_particle(base)) === nothing
        @test_throws "different sources" check(PrescribedFokkerPlanckTerm(:f, operator(), base; settings),
                                               field_particle(nothing))
        @test_throws "different sources" check(FokkerPlanckTerm(:f, operator(); settings), field_particle(base))
        # one of the two following the unknowns, the other one fixed
        @test_throws ArgumentError check(fp(:lagged), field_particle(nothing))
        # through the system
        layout = UnknownLayout(rad, [KineticUnknown(:f, g; scale=1e19), RadialUnknown(:cold_electron_temperature)])
        @test_throws "issue #70" EquationSystem(layout, [Equation(:f, TimeDerivative(), fp(), field_particle(from_unknowns)),
                                                         Equation(:cold_electron_temperature, Algebraic(),
                                                                  PrescribedValueTerm([10.0, 8.0]),
                                                                  minus(:cold_electron_temperature))])
    end

    @testset "one temperature for the cold bulk" begin
        @test check(fp(), ParticleSourceTerm(:particle_source, g, :cold_electron_temperature)) === nothing
        @test_throws "prescribed or fixed temperature" check(fp(), ParticleSourceTerm(:particle_source, g, [10.0, 8.0]))
        @test_throws "one temperature" check(fp(), ParticleSourceTerm(:particle_source, g, :other_temperature))
        thermal(temperature) = DensityMomentTerm(:f, g; region=MomentumRegion(ThermalMomentum(temperature)))
        @test check(fp(), thermal(:cold_electron_temperature)) === nothing
        @test_throws "prescribed or fixed temperature" check(fp(), thermal([10.0, 8.0]))
        # an operator of radial transport alone reads no temperature, a frozen collision
        # operator a fixed one
        @test check(fp(), FokkerPlanckTerm(:f, AdvectionDiffusion(g))) === nothing
        @test_throws "prescribed or fixed temperature" check(fp(), FokkerPlanckTerm(:f, fokker_planck_operator(g, base(0.0),
                                                                                                             settings)))
        # the moments of one hot region count the same electrons
        hot(factor; side=:above) = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor); side)
        @test check(fp(), DensityMomentTerm(:f, g; region=hot(5)), EnergyMomentTerm(:f, g; region=hot(5; side=:below))) === nothing
        @test_throws "one region" check(fp(), DensityMomentTerm(:f, g; region=hot(5)), EnergyMomentTerm(:f, g; region=hot(7)))
        # a moment without a region reads no temperature
        @test check(fp(), DensityMomentTerm(:f, g)) === nothing
        # without a temperature unknown, fixed values stay free
        @test check(FokkerPlanckTerm(:f, operator(); settings), ParticleSourceTerm(:particle_source, g, [10.0, 8.0])) === nothing
        # a conductivity with a prescribed plasma next to a bulk that follows the unknown
        @test_throws "prescribed or fixed temperature" check(fp(), ConductivityTerm(:electric_field, ConductivityModel(base)))
    end

    @testset "corrected conductivity with an implicit Fokker-Planck term" begin
        corrected = ConductivityTerm(:electric_field, ConductivityModel(from_unknowns; kinetic=:dream_fit))
        plain = ConductivityTerm(:electric_field, ConductivityModel(from_unknowns))
        @test check(fp(:implicit), corrected) === nothing
        @test_throws "coefficients = :implicit" check(fp(:lagged), corrected)
        @test check(fp(:lagged), plain) === nothing
    end

    @testset "knock-on cutoff from the temperature of the state" begin
        hot = cylindrical_grid(MomentumGrid(biuniform_p_faces(20, 8, 0.0, 0.6, 2.5), uniform_xi_faces(4)), rad)
        cutoff = momentum_grid(hot).p_faces[5]
        knockon_plasma(t) = PlasmaState(; cold_electron_temperature=[2000.0, 1000.0] .* exp(-t / 1e-3),
                                        ions=[IonSpecies("D", 1, 1, [1e20, 3e19])], electric_field=[3.0, 1.0],
                                        magnetic_field=[5.0, 5.0])
        layout = UnknownLayout(rad, [RadialUnknown(:cold_electron_temperature; scale=1000.0)])
        prescribed = KnockOnCutoffUpdate(KnockOnOperator(hot, hot, cutoff), knockon_plasma)
        following = KnockOnCutoffUpdate(KnockOnOperator(hot, hot, cutoff),
                                        PlasmaFromUnknowns(knockon_plasma; density=nothing))
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
        record = describe(following, DescriptionContext(; times=[0.0]))
        @test record["plasma_unknowns"]["temperature"] == "cold_electron_temperature"
        # a system whose bulk reads the temperature unknown refuses the prescribed cutoff
        system = (; equations=[Equation(:f, TimeDerivative(), fp())])
        @test_throws "knock-on cutoff" Runes2P.check_update(prescribed, system)
        @test Runes2P.check_update(following, system) === nothing
        @test Runes2P.check_update(prescribed, (; equations=[Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, operator()))])) === nothing
    end
end
