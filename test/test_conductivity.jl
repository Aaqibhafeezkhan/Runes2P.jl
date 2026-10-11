# Conductivity: the conductivity of Braams and Karney as DREAM evaluates
# it against the Lorentz gas and Spitzer and Härm, the conductivity of the discrete kinetic
# operator by its linear response, DREAM's fit of it, and the corrected conductivity of the
# fully kinetic mode in the steady state, in the transient after a step of the field, and in
# the resistive decay of a current profile (the plasma current test of DREAM #240 in a
# cylinder).

# Spitzer and Härm (1953), Table III: σ/σ_L = γ_E(Z)
const CONDUCTIVITY_SPITZER_HARM = Dict(1 => 0.5816, 2 => 0.6833, 4 => 0.7849, 16 => 0.9225)

# Lorentz conductivity σ_L = 64√(2π) ε₀² T^{3/2} / (√m e² Z lnΛ) (Braginskii, Z → ∞)
lorentz_conductivity(T, ne, Z) =
    64 * sqrt(2π) * Constants.VACUUM_PERMITTIVITY^2 * (T * Constants.ELEMENTARY_CHARGE)^1.5 /
    (sqrt(Constants.ELECTRON_MASS) * Constants.ELEMENTARY_CHARGE^2 * Z * thermal_coulomb_logarithm(T, ne))

function conductivity_plasma(; T=10.0, ne=1e20, Z=1, radii=1, cold_fraction=1.0, E=0.0)
    ions = [IonSpecies("X", Z, Z, fill(ne / Z, radii))]
    return PlasmaState(; cold_electron_temperature=fill(T, radii), ions, electric_field=fill(E, radii),
                       magnetic_field=zeros(radii), cold_electron_density=fill(cold_fraction * ne, radii))
end

function conductivity_grid(; cells=160, pitch_cells=4, T=10.0, radii=1, minor_radius=1.0,
                           xi_faces=uniform_xi_faces(pitch_cells))
    theta = T / Constants.ELECTRON_REST_ENERGY
    momentum = MomentumGrid(uniform_p_faces(cells, 0.0, 7 * sqrt(2theta)), xi_faces)
    return cylindrical_grid(momentum, RadialGrid(uniform_r_faces(radii, minor_radius)))
end

const CONDUCTIVITY_SETTINGS = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal)

@testset "conductivity" begin
    @testset "table of Braams and Karney" begin
        @test normalized_conductivity(0.0, 0.0) == 12.7661
        @test normalized_conductivity(100.0, 1.0) == 0.3
        @test normalized_conductivity(0.05, 0.5) == 6.7381
        @test normalized_conductivity(0.01, 1 / 3) ≈ 8.5328 rtol = 1e-12
        # bilinear between the points
        @test normalized_conductivity(0.005, 0.0) ≈ (12.7661 + 12.2972) / 2 rtol = 1e-14
        @test normalized_conductivity(0.0, 0.75) ≈ (7.4290 + 3.7599) / 2 rtol = 1e-14
        @test normalized_conductivity(0.015, 0.25) ≈
              (10.0778 + 9.7896 + 8.5328 + 8.3265) / 4 rtol = 1e-12
        @test_throws DomainError normalized_conductivity(101.0, 0.5)
        @test_throws DomainError normalized_conductivity(0.1, 1.1)
        @test_throws DomainError normalized_conductivity(-0.1, 0.5)
        # the normalization is the Lorentz gas: σ̄(0, 0) = 32/√(2π)
        @test normalized_conductivity(0.0, 0.0) ≈ 32 / sqrt(2π) rtol = 1e-5
        # the factors of Spitzer and Härm at the points of the table (Z = 1, 2, 5, 10)
        lorentz = normalized_conductivity(0.0, 0.0)
        @test normalized_conductivity(0.0, 1 / 2) / lorentz ≈ CONDUCTIVITY_SPITZER_HARM[1] rtol = 1e-3
        @test normalized_conductivity(0.0, 1 / 3) / lorentz ≈ CONDUCTIVITY_SPITZER_HARM[2] rtol = 4e-3
        # between the points the interpolation in 1/(1 + Z) is 1 % off at Z = 4 and 16
        @test normalized_conductivity(0.0, 1 / 5) / lorentz ≈ CONDUCTIVITY_SPITZER_HARM[4] rtol = 1.5e-2
        @test normalized_conductivity(0.0, 1 / 17) / lorentz ≈ CONDUCTIVITY_SPITZER_HARM[16] rtol = 1.5e-2
    end

    @testset "effective charge and the conductivity" begin
        ions = [IonSpecies("D", 1, 1, [1e19, 0.0]), IonSpecies("Ar", 18, 2, [2e17, 0.0])]
        @test effective_charge(ions, 1) ≈ (1e19 + 4 * 2e17) / (1e19 + 2 * 2e17) rtol = 1e-15
        @test effective_charge(ions, 2) == 1
        neutral = [IonSpecies("D", 1, 1, [1e19]), IonSpecies("Ar", 18, 0, [5e19])]
        @test effective_charge(neutral, 1) == 1
        plasma = conductivity_plasma(; Z=4, T=100.0, ne=5e19)
        @test effective_charge(plasma, 1) ≈ 4 rtol = 1e-15
        sigma = braams_karney_conductivity(plasma, 1)
        @test sigma == braams_karney_conductivity(100.0, effective_charge(plasma, 1), 5e19)
        theta = 100.0 * Constants.ELEMENTARY_CHARGE / (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT^2)
        @test sigma ≈ lorentz_conductivity(100.0, 5e19, 4) * normalized_conductivity(theta, 1 / 5) /
                      (32 / sqrt(2π)) rtol = 1e-5
        # Spitzer and Härm at low temperature
        for Z in (1, 2)
            @test braams_karney_conductivity(1.0, Z, 1e20) / lorentz_conductivity(1.0, 1e20, Z) ≈
                  CONDUCTIVITY_SPITZER_HARM[Z] rtol = 4e-3
        end
        # it falls with the temperature over mc² at equal T^{3/2}/lnΛ
        ratio(T) = braams_karney_conductivity(T, 1.0, 1e20) / lorentz_conductivity(T, 1e20, 1)
        @test ratio(1e4) < ratio(1e3) < ratio(10.0)
        @test ratio(1e3) / ratio(10.0) ≈ 1 - (7.4290 - 7.2736) / 7.4290 * (1e3 - 10) / 5109.9895 rtol = 1e-6
        @test_throws DomainError braams_karney_conductivity(-1.0, 1.0, 1e20)
        @test_throws DomainError braams_karney_conductivity(10.0, 0.0, 1e20)
        # DREAM's fit of the conductivity of its kinetic operator
        @test test_particle_conductivity_fraction(1) ≈ 0.5132 atol = 1e-4
        @test test_particle_conductivity_fraction(2) ≈ 0.6384 atol = 1e-4
        @test test_particle_conductivity_fraction(16) ≈ 0.9214 atol = 1e-4
    end

    function discrete(; Z=1, T=10.0, cells=160, pitch_cells=4, scheme=:central, cold_fraction=1.0,
                      p_upper_boundary=:f_zero, settings=CONDUCTIVITY_SETTINGS, kwargs...)
        grid = conductivity_grid(; cells, pitch_cells, T)
        plasma = conductivity_plasma(; Z, T, cold_fraction)
        operator = fokker_planck_operator(grid, plasma, settings; scheme, p_upper_boundary,
                                          electric_field=false)
        return only(kinetic_conductivity(operator; density=get(kwargs, :density, [1e20]),
                                         pitch_rule=get(kwargs, :pitch_rule, :linear)))
    end

    @testset "conductivity of the discrete operator" begin
        # the Lorentz gas: without electron-electron collisions the conductivity is σ_L, which
        # has no bremsstrahlung. Its stopping power is a drag without a diffusion,
        # so without the electron-electron collisions the Maxwellian is no steady state of the
        # operator and the linear response has no meaning: the bremsstrahlung is neglected.
        lorentz_settings = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal,
                                             bremsstrahlung=:neglect)
        lorentz = [discrete(; cells, cold_fraction=1e-10, settings=lorentz_settings) / lorentz_conductivity(10.0, 1e20, 1)
                   for cells in (40, 80, 160)]
        errors = lorentz .- 1
        @test abs(errors[3]) < 2e-3
        @test 3.5 < (errors[1] - errors[2]) / (errors[2] - errors[3]) < 4.5
        # the test-particle operator: DREAM's fit to CODE, and its ratio to the Lorentz gas
        values = Dict(Z => discrete(; Z) for Z in (1, 2, 4, 16))
        for Z in (1, 2, 4, 16)
            braams_karney = braams_karney_conductivity(10.0, Z, 1e20)
            @test values[Z] / braams_karney ≈ test_particle_conductivity_fraction(Z) rtol = 2.5e-2
        end
        @test 0.29 < values[1] / lorentz_conductivity(10.0, 1e20, 1) < 0.30
        # the response is proportional to the density of the equilibrium at fixed collisions
        @test discrete(; density=[2e20]) ≈ 2 * values[1] rtol = 1e-10
        # the schemes and the pitch grid
        @test discrete(; scheme=:exponential_fitting) ≈ values[1] rtol = 2e-3
        @test discrete(; scheme=:tcdf) ≈ values[1] rtol = 2e-2
        @test discrete(; pitch_cells=8) ≈ values[1] rtol = 1e-6
        # DREAM's flux boundaries at p_max make the bordered systems singular: the response
        # takes f = 0 beyond p_max in their place, and the operator of the run
        # keeps its own boundary
        for boundary in (:constant_flux, :extrapolated_flux, :extrapolated_flux_consistent)
            @test discrete(; p_upper_boundary=boundary) ≈ values[1] rtol = 1e-14
        end
        @test discrete(; scheme=:exponential_fitting, p_upper_boundary=:constant_flux) ≈
              discrete(; scheme=:exponential_fitting) rtol = 1e-14
        constant_flux = fokker_planck_operator(conductivity_grid(), conductivity_plasma(), CONDUCTIVITY_SETTINGS;
                                               p_upper_boundary=:constant_flux, electric_field=false)
        kinetic_conductivity(constant_flux; density=[1e20])
        @test constant_flux.p_upper_boundary === :constant_flux
        @test discrete(; pitch_rule=:cell_center) / values[1] ≈ 1.5 * sum(abs2, [-0.75, -0.25, 0.25, 0.75]) * 0.5 rtol = 1e-8
        # the equilibrium is that of the operator: the collision term of the run vanishes on it
        grid = conductivity_grid()
        with_field = fokker_planck_operator(grid, conductivity_plasma(; E=1.0), CONDUCTIVITY_SETTINGS)
        @test_throws "hold an electric field" kinetic_conductivity(with_field)
        bare = AdvectionDiffusion(grid)
        @test_throws "does not know its plasma" kinetic_conductivity(bare)
        # several radii at once
        radii = conductivity_grid(; radii=2)
        plasma = PlasmaState(; cold_electron_temperature=[10.0, 10.0],
                             ions=[IonSpecies("X", 1, 1, [1e20, 0.0]), IonSpecies("Y", 2, 2, [0.0, 0.5e20])],
                             electric_field=zeros(2), magnetic_field=zeros(2))
        both = kinetic_conductivity(fokker_planck_operator(radii, plasma, CONDUCTIVITY_SETTINGS; scheme=:central,
                                                           electric_field=false))
        @test both ≈ [values[1], values[2]] rtol = 1e-8
    end

    @testset "partial screening only above the excitation energy (issue #97)" begin
        # Deuterium with 5 % argon in Ar⁺ to Ar³⁺ (one third each), synthetic.
        # The thermal bulk lies below the window of partial screening, so the operator with
        # partial screening has the conductivity of complete screening, within 1 % of DREAM's
        # fit (0.9941 of it). DREAM's option applies partial screening to the bulk and gives
        # 0.954, 0.930 and 0.897 of the fit at 5, 10 and 20 eV.
        dream_ratios = Dict(5.0 => 0.954, 10.0 => 0.930, 20.0 => 0.897)
        for T in (5.0, 10.0, 20.0)
            argon = zeros(19, 1)
            argon[2:4, 1] .= 0.05e20 / 3
            plasma = PlasmaState(; cold_electron_temperature=[T],
                                 ions=[IonSpecies("D", 1, 1, [1e20]), IonSpecies("Ar", 18, argon)],
                                 electric_field=[0.0], magnetic_field=[0.0])
            fit = braams_karney_conductivity(plasma, 1) *
                  test_particle_conductivity_fraction(effective_charge(plasma, 1))
            ratio(settings) = only(kinetic_conductivity(fokker_planck_operator(conductivity_grid(; T), plasma, settings;
                                                                               scheme=:central,
                                                                               electric_field=false))) / fit
            window = ratio(CollisionSettings(mode=:full, screening=:partial, coulomb_logarithm=:thermal))
            dream = ratio(CollisionSettings(mode=:full, screening=:partial, coulomb_logarithm=:thermal,
                                            screening_window=:dream))
            @test abs(window - 1) < 1e-2
            @test window ≈ ratio(CONDUCTIVITY_SETTINGS) rtol = 1e-5
            @test dream ≈ dream_ratios[T] atol = 1e-3
        end
    end

    @testset "model and term" begin
        radial = RadialGrid(uniform_r_faces(2, 1.0))
        plasma = PlasmaState(; cold_electron_temperature=[10.0, 20.0],
                             ions=[IonSpecies("X", 1, 1, [1e20, 0.0]), IonSpecies("Y", 2, 2, [0.0, 0.5e20])],
                             electric_field=zeros(2), magnetic_field=zeros(2))
        reference = [braams_karney_conductivity(plasma, ir) for ir in 1:2]
        @test ConductivityModel(plasma)(0.0) == reference
        fit = ConductivityModel(plasma; kinetic=:dream_fit)
        @test fit(0.0) ≈ reference .* (1 .- test_particle_conductivity_fraction.([1, 2])) rtol = 1e-14
        cooling(t) = PlasmaState(; cold_electron_temperature=[10.0, 20.0] .* exp(-t), ions=plasma.ions,
                                 electric_field=zeros(2), magnetic_field=zeros(2))
        following = ConductivityModel(cooling; kinetic=:dream_fit)
        @test following(0.0) == fit(0.0)
        @test following(1.0) ≈ [braams_karney_conductivity(cooling(1.0), ir) for ir in 1:2] .*
                                (1 .- test_particle_conductivity_fraction.([1, 2])) rtol = 1e-14
        @test_throws ArgumentError ConductivityModel(plasma; reference=:spitzer)
        @test_throws ArgumentError ConductivityModel(plasma; kinetic=:fit)
        @test_throws ArgumentError ConductivityModel(plasma; kinetic=:discrete_operator)
        @test_throws ArgumentError ConductivityModel(plasma; kinetic=:dream_fit, operator=AdvectionDiffusion(conductivity_grid()))
        # a constant plasma of the model with an operator that is rebuilt in time: the
        # conductivity of the discrete operator follows the operator
        let grid = conductivity_grid(; cells=40, T=10.0), single = conductivity_plasma(; T=10.0)
            bulk(t) = conductivity_plasma(; T=10.0 * exp(-t))
            prescribed = PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(grid), bulk; settings=CONDUCTIVITY_SETTINGS)
            model = ConductivityModel(single; kinetic=:discrete_operator, operator=prescribed)
            @test Runes2P.is_time_dependent(model)
            # the operator of the term as its rebuild makes it at t = 0.5
            later = set_fokker_planck_coefficients!(AdvectionDiffusion(grid), bulk(0.5), CONDUCTIVITY_SETTINGS)
            @test model(0.5) ≈ [braams_karney_conductivity(single, 1)] .- kinetic_conductivity(later) rtol = 1e-12
            term = ConductivityTerm(:electric_field, model)
            @test copy(conductivity_at(term, 0.5)) != conductivity_at(term, 0.0)
            # its record takes the first time point alone: every time costs a factorization
            record = describe(model, DescriptionContext(; times=[0.5, 1.0]))
            @test record["conductivity"]["sampled"] == "first time point" && record["conductivity"]["time"] == 0.5
            @test record["conductivity"]["values"] ≈ model(0.5) rtol = 1e-14
            @test describe(model)["conductivity"]["described"] === false
        end
        # the pitch rule of the model is that of the current moment of its equation
        let grid = conductivity_grid(; cells=20, T=10.0), single = conductivity_plasma(; T=10.0),
            operator = fokker_planck_operator(grid, single, CONDUCTIVITY_SETTINGS; electric_field=false),
            layout = UnknownLayout(grid.radial_grid, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density),
                                                      KineticUnknown(:f, grid)])
            system_of(model_rule, moment_rule) = EquationSystem(layout, [
                Equation(:electric_field, Algebraic(), PrescribedValueTerm(1.0), LinearTerm(:electric_field; coefficient=-1)),
                Equation(:ohmic_current_density, Algebraic(), CurrentMomentTerm(:f, grid; pitch_rule=moment_rule),
                         ConductivityTerm(:electric_field, ConductivityModel(single; kinetic=:discrete_operator, operator,
                                                                             pitch_rule=model_rule)),
                         LinearTerm(:ohmic_current_density; coefficient=-1)),
                Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, operator))])
            @test_throws "pitch rule" system_of(:linear, :cell_center)
            @test system_of(:cell_center, :cell_center) isa EquationSystem
        end
        # a plasma defined on the interval of the run alone: the system, the run and its
        # record evaluate the model from the initial time on
        let late(t) = t < 1.0 ? error("no plasma before the run") : conductivity_plasma(; T=10.0 + t, radii=2)
            model = ConductivityModel(late)
            layout = UnknownLayout(RadialGrid(uniform_r_faces(2, 1.0)),
                                   [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale=1e4)])
            system = EquationSystem(layout, [
                Equation(:electric_field, Algebraic(), PrescribedValueTerm([0.5, 2.0]), LinearTerm(:electric_field; coefficient=-1)),
                Equation(:ohmic_current_density, Algebraic(), ConductivityTerm(:electric_field, model),
                         LinearTerm(:ohmic_current_density; coefficient=-1))]; initial_time=1.0)
            solution = integrate(system, zeros(length(layout)), [1.0, 1.5, 2.0])
            current = radial_values(solution.states[:, end], layout, unknown_id(layout, :ohmic_current_density))
            @test current ≈ model(2.0) .* [0.5, 2.0] rtol = 1e-12
            record = describe(system, DescriptionContext(; times=[1.0, 1.5, 2.0]))
            @test record["equations"]["ohmic_current_density"]["terms"]["term_1"]["model"]["conductivity"]["values"][:, 3] ≈
                  model(2.0) rtol = 1e-14
        end

        layout = UnknownLayout(radial, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale=1e4),
                                        ScalarUnknown(:plasma_current)])
        term = ConductivityTerm(:electric_field, following; coefficient=2)
        row = unknown_id(layout, :ohmic_current_density)
        Runes2P.validate(term, layout, row)
        @test dependencies(term) == [:electric_field] && is_linear(term)
        u = [0.5, 0.0, 2.0, 0.0, 0.0]
        radial_values(u, layout, unknown_id(layout, :electric_field)) .= [0.5, 2.0]
        du = zeros(length(layout))
        for t in (0.0, 1.0, 0.0)
            fill!(du, 0)
            rhs!(du, term, u, layout, row, t)
            @test radial_values(du, layout, row) ≈ 2 .* following(t) .* [0.5, 2.0] rtol = 1e-14
            @test conductivity_at(term, t) == following(t)
        end
        @test (@allocated rhs!(du, term, u, layout, row, 0.0)) == 0
        @test_throws ArgumentError Runes2P.validate(term, layout, unknown_id(layout, :plasma_current))
        @test_throws DimensionMismatch Runes2P.validate(ConductivityTerm(:electric_field,
                                                                      ConductivityModel(conductivity_plasma())),
                                                     layout, row)
        system = EquationSystem(layout, [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm([0.5, 2.0]), LinearTerm(:electric_field; coefficient=-1)),
            Equation(:ohmic_current_density, Algebraic(), term, LinearTerm(:ohmic_current_density; coefficient=-1)),
            Equation(:plasma_current, Algebraic(), CrossSectionIntegralTerm(:ohmic_current_density),
                     LinearTerm(:plasma_current; coefficient=-1))])
        times = [0.0, 0.5, 1.0]
        solution = integrate(system, zeros(length(layout)), times)
        final = solution.states[:, end]
        @test radial_values(final, layout, row) ≈ 2 .* following(1.0) .* [0.5, 2.0] rtol = 1e-12
        record = describe(system, DescriptionContext(; times))
        term_record = record["equations"]["ohmic_current_density"]["terms"]["term_1"]
        @test term_record["electric_field"] == "electric_field" && term_record["coefficient"] == 2
        @test term_record["model"]["reference"] == "braams_karney" && term_record["model"]["kinetic"] == "dream_fit"
        @test term_record["model"]["conductivity"]["values"][:, 3] ≈ following(1.0) rtol = 1e-14
        constant = describe(ConductivityTerm(:electric_field, fit), DescriptionContext())
        @test constant["model"]["conductivity"] ≈ fit(0.0) rtol = 1e-14
        fields = settings_fields(ConductivityTerm)
        @test Set(union(fields.described, fields.derived)) == Set(fieldnames(ConductivityTerm))
    end

    # One radius with a small prescribed field, the density held by the particle source.
    # The steady state of the corrected ohmic current against σ E, and the time after a step
    # of the field from the Maxwellian.
    @testset "steady state and transient of the corrected conductivity" begin
        T, ne = 10.0, 1e20
        grid = conductivity_grid(; cells=120, T, minor_radius=0.5)
        radial = grid.radial_grid
        plasma = conductivity_plasma(; T, ne)
        operator = fokker_planck_operator(grid, plasma, CONDUCTIVITY_SETTINGS; electric_field=false)
        kinetic_term = FokkerPlanckTerm(:hot_tail_distribution, operator; electric_field=:electric_field)
        braams_karney = braams_karney_conductivity(plasma, 1)
        sigma_kinetic = only(kinetic_conductivity(operator))
        collision_time = thermal_collision_time(ne, T, thermal_coulomb_logarithm(T, ne))
        # a ten-thousandth of the Dreicer field, 5 critical fields at 10 eV: the nonlinear
        # response adds 1.7e-7 of σ E to the steady current, 1.7e-5 at a thousandth
        field = 1e-4 * dreicer_field(ne, T, thermal_coulomb_logarithm(T, ne))
        minus(name) = LinearTerm(name; coefficient=-1)
        function run(model, times; field_particle=false)
            layout = UnknownLayout(radial, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale=1e4),
                                            RadialUnknown(:particle_source; scale=ne / collision_time),
                                            KineticUnknown(:hot_tail_distribution, grid; scale=1e26)])
            system = EquationSystem(layout, [
                Equation(:electric_field, Algebraic(), PrescribedValueTerm(field), minus(:electric_field)),
                Equation(:ohmic_current_density, Algebraic(), CurrentMomentTerm(:hot_tail_distribution, grid),
                         ConductivityTerm(:electric_field, model), minus(:ohmic_current_density)),
                Equation(:particle_source, Constraint(), DensityMomentTerm(:hot_tail_distribution),
                         PrescribedValueTerm(-ne); scale=ne),
                Equation(:hot_tail_distribution, TimeDerivative(), kinetic_term,
                         ParticleSourceTerm(:particle_source, grid, [T]))])
            u0 = zeros(length(layout))
            id = unknown_id(layout, :hot_tail_distribution)
            kinetic_view(u0, layout, id, 1) .= ne .* source_shape(system.equations[4].terms[2], 1)
            solution = integrate(system, u0, times)
            @test all(solution.statistics.accepted)
            return [only(unknown_view(view(solution.states, :, k), layout, :ohmic_current_density, 1))
                    for k in eachindex(times)] ./ (braams_karney * field)
        end
        # steady state: large steps of 100 collision times
        steady = collect(range(0, 2000 * collision_time; length=21))
        exact = run(ConductivityModel(plasma; kinetic=:discrete_operator, operator), steady)
        fitted = run(ConductivityModel(plasma; kinetic=:dream_fit), steady)
        # with the conductivity of the discrete operator the steady current is σ E
        @test exact[end] ≈ 1 rtol = 1e-6
        @test exact[end] > 1
        # with DREAM's fit it is off by the difference of the fit to this operator
        expected = (sigma_kinetic + (1 - test_particle_conductivity_fraction(1)) * braams_karney) / braams_karney
        @test fitted[end] ≈ expected rtol = 1e-6
        @test 0.005 < abs(fitted[end] - 1) < 0.03
        # the transient after the step of the field: the correction acts at once, the current
        # of the distribution within collision times. At the start the corrected current is
        # the share 1 − σ_kinetic/σ of the steady one.
        transient = vcat(0.0, collision_time .* [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
        history = run(ConductivityModel(plasma; kinetic=:discrete_operator, operator), transient)
        @test history[2] ≈ 1 - sigma_kinetic / braams_karney rtol = 0.05
        @test issorted(history[2:end])
        @test history[end] ≈ 1 rtol = 1e-3
        @test 0.6 < history[6] < 0.95      # one collision time after the step
        # observed: 0.496 of σ E at 0.01 collision times, 0.711 after one, 0.968 after ten,
        # 0.99987 after a hundred; σ_kinetic/σ = 0.508, and 0.9945 with DREAM's fit

        # two grids: the hot-tail grid below a runaway grid to p = 1. The
        # conductivity of the hot-tail operator takes f = 0 beyond its p_max, and the steady
        # state of the two grids has σ E as that of one grid.
        hot_operator = fokker_planck_operator(grid, plasma, CONDUCTIVITY_SETTINGS; electric_field=false,
                                              p_upper_boundary=:interface)
        @test only(kinetic_conductivity(hot_operator)) ≈ sigma_kinetic rtol = 1e-14
        runaway = runaway_grid(grid, uniform_p_faces(4, last(momentum_grid(grid, 1).p_faces), 1.0)[2:end],
                               uniform_xi_faces(4))
        runaway_operator = fokker_planck_operator(runaway, plasma, CONDUCTIVITY_SETTINGS; electric_field=false,
                                                  p_lower_boundary=:interface)
        paired_hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator; electric_field=:electric_field)
        paired_interface = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, paired_hot_term, runaway_operator)
        paired_layout = UnknownLayout(radial, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale=1e4),
                                        RadialUnknown(:particle_source; scale=ne / collision_time),
                                        KineticUnknown(:hot_tail_distribution, grid; scale=1e26),
                                        KineticUnknown(:runaway_distribution, runaway; scale=1e26)])
        paired_source = ParticleSourceTerm(:particle_source, grid, [T])
        paired_model = ConductivityModel(plasma; kinetic=:discrete_operator, operator=hot_operator)
        paired_system = EquationSystem(paired_layout, [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(field), minus(:electric_field)),
            Equation(:ohmic_current_density, Algebraic(), CurrentMomentTerm(:hot_tail_distribution, grid),
                     ConductivityTerm(:electric_field, paired_model), minus(:ohmic_current_density)),
            Equation(:particle_source, Constraint(), DensityMomentTerm(:hot_tail_distribution),
                     PrescribedValueTerm(-ne); scale=ne),
            Equation(:hot_tail_distribution, TimeDerivative(), paired_hot_term, paired_interface, paired_source),
            Equation(:runaway_distribution, TimeDerivative(),
                     FokkerPlanckTerm(:runaway_distribution, runaway_operator; electric_field=:electric_field), paired_interface)])
        paired_u0 = zeros(length(paired_layout))
        kinetic_view(paired_u0, paired_layout, unknown_id(paired_layout, :hot_tail_distribution), 1) .= ne .* source_shape(paired_source, 1)
        paired_solution = integrate(paired_system, paired_u0, steady)
        @test all(paired_solution.statistics.accepted)
        two_grids = only(unknown_view(view(paired_solution.states, :, length(steady)), paired_layout, :ohmic_current_density, 1)) /
                    (braams_karney * field)
        @test two_grids ≈ 1 rtol = 1e-5
        @test two_grids ≈ exact[end] rtol = 1e-6
    end

    # The plasma current of a kinetic run with the corrected conductivity against the fluid
    # run with the conductivity of Braams and Karney, in the resistive decay of a current
    # profile: the rate of the discrete system of each, by the ratio of two states of
    # backward Euler (DREAM #240, where the currents of the two kinds of run differ).
    @testset "resistive decay, kinetic against fluid" begin
        T, ne, radii = 10.0, 1e20, 5
        grid = conductivity_grid(; cells=80, T, radii, minor_radius=0.22)
        radial = grid.radial_grid
        plasma = conductivity_plasma(; T, ne, radii)
        braams_karney = braams_karney_conductivity(plasma, 1)
        minus(name) = LinearTerm(name; coefficient=-1)
        rate_scale = 2.404825557695773^2 / (Constants.VACUUM_PERMEABILITY * braams_karney * 0.22^2)
        steps = 120
        times = collect(range(0, 3 / rate_scale; length=steps + 1))
        dt = times[2] - times[1]
        current_scale = 1e4
        function decay(kind)
            unknowns = induction_unknowns(radial; electric_field_scale=1.0, current_density_scale=current_scale)
            equations = induction_equations(radial; current_density_scale=current_scale)
            if kind === :fluid
                push!(equations, Equation(:total_current_density, Algebraic(),
                                          ConductivityTerm(:electric_field, ConductivityModel(plasma)),
                                          minus(:total_current_density)))
            else
                operator = fokker_planck_operator(grid, plasma, CONDUCTIVITY_SETTINGS; electric_field=false)
                model = kind === :discrete ? ConductivityModel(plasma; kinetic=:discrete_operator, operator) :
                        ConductivityModel(plasma; kinetic=:dream_fit)
                append!(unknowns, [RadialUnknown(:particle_source; scale=ne * rate_scale),
                                   KineticUnknown(:hot_tail_distribution, grid; scale=1e26)])
                append!(equations, [
                    Equation(:total_current_density, Algebraic(), CurrentMomentTerm(:hot_tail_distribution, grid),
                             ConductivityTerm(:electric_field, model), minus(:total_current_density)),
                    Equation(:particle_source, Constraint(), DensityMomentTerm(:hot_tail_distribution),
                             PrescribedValueTerm(-ne); scale=ne),
                    Equation(:hot_tail_distribution, TimeDerivative(),
                             FokkerPlanckTerm(:hot_tail_distribution, operator; electric_field=:electric_field),
                             ParticleSourceTerm(:particle_source, grid, fill(T, radii)))])
            end
            layout = UnknownLayout(radial, unknowns)
            system = EquationSystem(layout, equations)
            # start: the lowest mode of the field, the current of the fluid, and for the
            # kinetic run the Maxwellian, which takes the current within collision times
            profile = [1 - (r / 0.22)^2 for r in radial.r_centers]
            field = 1e-2 .* profile
            initial = initial_poloidal_flux(radial, braams_karney .* field)
            u0 = zeros(length(layout))
            radial_values(u0, layout, unknown_id(layout, :electric_field)) .= field
            radial_values(u0, layout, unknown_id(layout, :total_current_density)) .= braams_karney .* field
            radial_values(u0, layout, unknown_id(layout, :poloidal_flux)) .= initial.poloidal_flux
            for name in (:plasma_current, :edge_poloidal_flux, :wall_poloidal_flux)
                u0[only(unknown_range(layout, unknown_id(layout, name)))] = getproperty(initial, name)
            end
            if kind !== :fluid
                shape = source_shape(system.equations[end].terms[2], 1)
                for ir in 1:radii
                    kinetic_view(u0, layout, unknown_id(layout, :hot_tail_distribution), ir) .= ne .* shape
                end
            end
            solution = integrate(system, u0, times)
            @test all(solution.statistics.accepted)
            position = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
            current = solution.states[position, :]
            return (current[steps] / current[steps+1] - 1) / dt
        end
        fluid = decay(:fluid)
        discrete = decay(:discrete)
        fitted = decay(:dream_fit)
        # the decay time is 1e5 collision times: the kinetic run is in its steady state
        # observed: the rate with the conductivity of the discrete operator is that of the
        # fluid to 1.2e-5, with DREAM's fit 0.58 % above it, the rate of the fluid 0.987 of the
        # lowest mode of the continuum at 5 radii
        @test discrete ≈ fluid rtol = 1e-4
        @test fitted ≈ fluid rtol = 3e-2
        @test abs(fitted / fluid - 1) > 3e-3
        @test 0.9 < fluid / rate_scale < 1.0
    end

    @testset "correction and field-particle term" begin
        grid = conductivity_grid(; cells=40)
        plasma = conductivity_plasma()
        operator = fokker_planck_operator(grid, plasma, CONDUCTIVITY_SETTINGS; electric_field=false)
        restoring = FieldParticleTerm(:hot_tail_distribution, FieldParticleOperator(grid, plasma, CONDUCTIVITY_SETTINGS))
        layout = UnknownLayout(grid.radial_grid, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density),
                                                  KineticUnknown(:hot_tail_distribution, grid)])
        equations(model) = [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(1.0), LinearTerm(:electric_field; coefficient=-1)),
            Equation(:ohmic_current_density, Algebraic(), CurrentMomentTerm(:hot_tail_distribution, grid),
                     ConductivityTerm(:electric_field, model), LinearTerm(:ohmic_current_density; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(),
                     FokkerPlanckTerm(:hot_tail_distribution, operator; electric_field=:electric_field), restoring)]
        @test_throws "Take one of the two" EquationSystem(layout, equations(ConductivityModel(plasma; kinetic=:dream_fit)))
        @test_throws "Take one of the two" EquationSystem(layout, equations(ConductivityModel(plasma; kinetic=:discrete_operator,
                                                                                         operator)))
        @test EquationSystem(layout, equations(ConductivityModel(plasma))) isa EquationSystem
    end
end
