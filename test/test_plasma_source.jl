# The plasma as the source of the coefficients: the prescribed plasma
# source, the caches keyed on the plasma values of each radius, and the field-particle term
# in a plasma prescribed in time.
using Random: Xoshiro

const SOURCE_SETTINGS = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal)

function _source_plasma(T; ne=1e20, E=0.0, nr=2)
    temperatures = T isa Real ? fill(T, nr) : T
    return PlasmaState(; cold_electron_temperature=temperatures, ions=[IonSpecies("D", 1, 1, fill(ne, nr))],
                       electric_field=fill(E, nr), magnetic_field=fill(5.0, nr), cold_electron_density=fill(ne, nr))
end

function _source_grid(; np=40, nxi=4, nr=2, T=10.0)
    mom = MomentumGrid(uniform_p_faces(np, 0.0, 7 * sqrt(2T / Constants.ELECTRON_REST_ENERGY)), uniform_xi_faces(nxi))
    return cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
end

_source_maxwellian(g, T) = [exp(-(sqrt(1 + p^2) - 1) / (T / Constants.ELECTRON_REST_ENERGY))
                            for p in momentum_grid(g).p_centers, _ in 1:pitch_cell_count(g), _ in 1:radial_cell_count(g)]

# a cooling from 10 to 5 eV over 1e-5 s, with the time of each call counted
function _cooling(calls=Ref(0); E=0.0)
    return t -> (calls[] += 1; _source_plasma(10.0 * 2.0^(-min(t, 1e-5) / 1e-5); E))
end

@testset "plasma source" begin
    @testset "prescribed plasma and keys" begin
        constant = _source_plasma(10.0)
        @test plasma_values(PrescribedPlasma(constant), nothing, nothing, 1.0) === constant
        calls = Ref(0)
        source = PrescribedPlasma(_cooling(calls))
        first_state = plasma_values(source, nothing, nothing, 0.0)
        @test plasma_values(source, nothing, nothing, 0.0) === first_state
        @test calls[] == 1
        @test plasma_values(source, nothing, nothing, 5e-6).cold_electron_temperature ≈ fill(10 / sqrt(2), 2)
        @test calls[] == 2
        @test PrescribedPlasma(source) === source

        key = Runes2P.PlasmaKey()
        @test Runes2P.update_key!(key, constant)
        @test !Runes2P.update_key!(key, _source_plasma(10.0))
        # the field counts only for coefficients that hold it
        @test Runes2P.update_key!(key, _source_plasma(10.0; E=1.0))
        field_free = Runes2P.PlasmaKey()
        @test Runes2P.update_key!(field_free, _source_plasma(10.0; E=1.0); electric_field=false)
        @test !Runes2P.update_key!(field_free, _source_plasma(10.0; E=2.0); electric_field=false)
        @test Runes2P.update_key!(key, _source_plasma([10.0, 9.0]); electric_field=false)
        @test Runes2P.update_key!(key, _source_plasma(10.0; nr=3); electric_field=false)
        values = Float64[]
        @test Runes2P.update_key!(values, [1.0, 2.0]) && !Runes2P.update_key!(values, [1.0, 2.0])
        @test Runes2P.update_key!(values, [1.0, 3.0]) && values == [1.0, 3.0]
    end

    @testset "Fokker-Planck term rebuilds when the plasma values change" begin
        g = _source_grid()
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g)])
        u = rand(Xoshiro(71), length(layout))
        steady = PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(g; p_upper_boundary=:f_zero),
                                            t -> _source_plasma(10.0); settings=SOURCE_SETTINGS)
        @test steady isa FokkerPlanckTerm && Runes2P.has_plasma_source(steady)
        for t in (0.0, 1e-6, 2e-6)
            Runes2P.rhs!(zeros(length(layout)), steady, u, layout, 1, t)
        end
        @test steady.rebuilds[] == 1
        cooling = PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting,
                                                                    p_upper_boundary=:f_zero),
                                             _cooling(); settings=SOURCE_SETTINGS)
        for (k, t) in enumerate((0.0, 1e-6, 1e-6, 2e-6))
            out = Runes2P.rhs!(zeros(length(layout)), cooling, u, layout, 1, t)
            fresh = fokker_planck_operator(g, _cooling()(t), SOURCE_SETTINGS)
            expected = zeros(size(g))
            Runes2P.apply!(expected, fresh, reshape(u, size(g)))
            @test out ≈ vec(expected) rtol = 1e-14
        end
        @test cooling.rebuilds[] == 3
        # the same term through the keyword of FokkerPlanckTerm, with a constant state
        fixed = FokkerPlanckTerm(:f, AdvectionDiffusion(g; p_upper_boundary=:f_zero); plasma=_source_plasma(10.0),
                                 settings=SOURCE_SETTINGS)
        Runes2P.rhs!(zeros(length(layout)), fixed, u, layout, 1, 0.0)
        Runes2P.rhs!(zeros(length(layout)), fixed, u, layout, 1, 1.0)
        @test fixed.rebuilds[] == 1
        @test !Runes2P.has_plasma_source(FokkerPlanckTerm(:f, AdvectionDiffusion(g)))
    end

    @testset "caches of the source shape, the region and the conductivity" begin
        g = _source_grid(; nr=3)
        # a temperature constant in time: the shape is built once
        source = ParticleSourceTerm(:source, g, t -> fill(10.0, 3))
        shape = Runes2P.source_shape(source, 1, 0.0)
        shape[1] = -1.0    # a mark that a rebuild would overwrite
        @test Runes2P.source_shape(source, 1, 1e-6)[1] == -1.0
        following = ParticleSourceTerm(:source, g, t -> fill(10.0 - 1e6t, 3))
        Runes2P.source_shape(following, 1, 0.0)[1] = -1.0
        @test Runes2P.source_shape(following, 1, 1e-6) ≈ Runes2P.source_shape(ParticleSourceTerm(:source, g, fill(9.0, 3)), 1)

        # a thermal limit is evaluated once per time, not once per radius
        calls = Ref(0)
        region = MomentumRegion(ThermalMomentum(t -> (calls[] += 1; fill(10.0, 3)); factor=3))
        envelopes = Runes2P.RegionEnvelopes(region, g)
        for ir in 1:3
            Runes2P._envelope_at(envelopes, g, ir, nothing, nothing, 0.0)
        end
        @test calls[] == 1
        envelopes.values[2][1] = -1.0
        Runes2P._envelope_at(envelopes, g, 2, nothing, nothing, 1e-6)
        @test calls[] == 2 && envelopes.values[2][1] == -1.0
        @test envelopes.values[3] == momentum_envelope(region, g, 3, 0.0)

        # the conductivity of the discrete operator: one evaluation while neither the plasma
        # nor the operator changes
        nr2 = _source_grid()
        operator = PrescribedFokkerPlanckTerm(:f, fokker_planck_operator(nr2, _source_plasma(10.0), SOURCE_SETTINGS),
                                              t -> _source_plasma(10.0); settings=SOURCE_SETTINGS)
        model = ConductivityModel(t -> _source_plasma(10.0); kinetic=:discrete_operator, operator)
        term = ConductivityTerm(:E, model)
        first_values = copy(conductivity_at(term, 0.0))
        term.values[1] = -1.0
        @test conductivity_at(term, 1e-6)[1] == -1.0
        cooling = ConductivityTerm(:E, ConductivityModel(_cooling(); kinetic=:discrete_operator,
                                                         operator=PrescribedFokkerPlanckTerm(:f,
                                                             fokker_planck_operator(nr2, _source_plasma(10.0),
                                                                                    SOURCE_SETTINGS),
                                                             _cooling(); settings=SOURCE_SETTINGS)))
        @test conductivity_at(cooling, 0.0) ≈ first_values rtol = 1e-14
        @test conductivity_at(cooling, 1e-5) ≈ ConductivityModel(_cooling()(1e-5); kinetic=:discrete_operator,
                                                                 operator=fokker_planck_operator(nr2, _cooling()(1e-5),
                                                                                                 SOURCE_SETTINGS))(0.0) rtol = 1e-12
    end

    @testset "field-particle term in a cooling plasma (issue #87)" begin
        g = _source_grid()
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e20)])
        fM = _source_maxwellian(g, 10.0)
        fM .*= 1e20 ./ reshape(density(g, fM), 1, 1, :)
        E = 0.05
        function cooling_run(linear_solver)
            fp = PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(g; p_upper_boundary=:f_zero), _cooling(; E);
                                            settings=SOURCE_SETTINGS)
            op = FieldParticleOperator(g, _source_plasma(10.0), SOURCE_SETTINGS)
            field_particle = FieldParticleTerm(:f, op; plasma=_cooling(; E))
            system = EquationSystem(layout, [Equation(:f, TimeDerivative(), fp, field_particle)]; time_scale=1e-6)
            times = range(0.0, 1e-5; length=11)
            return integrate(system, vec(fM), times; solver=NewtonSolver(; linear_solver)), field_particle, times
        end
        solution, term, times = cooling_run(GMRES())
        # at every step the term has the coefficients of the plasma of its time, and with the
        # electron-electron test-particle operator of that time it conserves the parallel
        # momentum per radius to round-off
        # (the first state is the isotropic Maxwellian, which loses no momentum)
        for (k, t) in collect(enumerate(times))[2:end]
            f = reshape(solution.states[:, k], size(g))
            dfp = zeros(length(layout))
            Runes2P.rhs!(dfp, term, solution.states[:, k], layout, 1, t)
            plasma = _cooling()(t)
            fresh = FieldParticleOperator(g, plasma, SOURCE_SETTINGS)
            @test term.operator.loss_functionals == fresh.loss_functionals
            @test term.operator.restoring_shapes == fresh.restoring_shapes
            dee = zeros(size(g))
            Runes2P.apply!(dee, electron_electron_operator(g, plasma, SOURCE_SETTINGS), f)
            loss = parallel_momentum(fresh, dee)
            @test all(abs.(loss) .> 0)
            @test maximum(abs, parallel_momentum(fresh, dee .+ reshape(dfp, size(g))) ./ loss) < 1e-11
        end
        # the direct solve takes the Woodbury factors of the plasma of each step: the
        # linear system then has the same solution as the Krylov solve on the exact product
        direct, _, _ = cooling_run(SparseLU())
        @test direct.states ≈ solution.states rtol = 1e-8
        # the steady-state conductivity at a fixed time is that of a term built from the
        # plasma of that time: the term holds the same functionals and shapes (above), and
        # its record names the plasma in place of the coefficients of the last rebuild
        record = describe(term, Runes2P.DescriptionContext(times=collect(times)))
        @test !haskey(record["operator"], "coefficient_source")
        @test record["collision_settings"]["mode"] == "full"
        @test haskey(record, "plasma_table")
    end
end
