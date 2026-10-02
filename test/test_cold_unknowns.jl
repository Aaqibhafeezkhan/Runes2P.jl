# The cold-electron temperature and density as unknowns of the system: the
# plasma source that reads them, the implicit and the lagged coefficients, and the oracle
# runs against a plasma prescribed in time.
using ForwardDiff: ForwardDiff
using Random: Xoshiro

const COLD_SETTINGS = CollisionSettings(mode=:full)

_cold_temperature(t) = [100.0, 80.0] .* (1 - 0.4 * t / 1e-4)
_cold_density(t) = [1e20, 8e19] .* (1 + 0.3 * t / 1e-4)

function _cold_base(t)
    return PlasmaState(; cold_electron_temperature=_cold_temperature(t), cold_electron_density=_cold_density(t),
                       ions=[IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field=[20.0, 15.0],
                       magnetic_field=[5.0, 5.0])
end

function _cold_grid()
    mom = MomentumGrid(uniform_p_faces(30, 0.0, 0.2), uniform_xi_faces(4))
    return cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
end

# f with T and n as unknowns held at the prescribed values by algebraic rows
function _cold_system(g, u0; coefficients=:implicit, density=:cold_electron_density, scheme=:exponential_fitting,
                      field=false, jacobian=:exact, base=_cold_base)
    unknowns = [KineticUnknown(:f, g; scale=1e20), RadialUnknown(:cold_electron_temperature; scale=100.0)]
    density === nothing || push!(unknowns, RadialUnknown(:cold_electron_density; scale=1e20))
    field && push!(unknowns, RadialUnknown(:electric_field; scale=10.0))
    layout = UnknownLayout(g.radial_grid, unknowns)
    plasma = PlasmaFromUnknowns(base; density)
    term = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme, p_upper_boundary=:f_zero);
                            plasma, settings=COLD_SETTINGS, coefficients, jacobian,
                            electric_field=field ? :electric_field : nothing)
    minus(name) = LinearTerm(name; coefficient=-1)
    equations = [Equation(:f, TimeDerivative(), term),
                 Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm(_cold_temperature),
                          minus(:cold_electron_temperature))]
    field && push!(equations, Equation(:electric_field, Algebraic(), PrescribedValueTerm([20.0, 15.0]),
                                       minus(:electric_field)))
    density === nothing ||
        push!(equations, Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(_cold_density),
                                  minus(:cold_electron_density)))
    return EquationSystem(layout, equations; time_scale=1e-6, initial_state=u0(layout)), term
end

function _cold_initial_state(g, f0)
    return layout -> begin
        u = zeros(length(layout))
        kinetic_array(u, layout, unknown_id(layout, :f)) .= f0
        radial_values(u, layout, unknown_id(layout, :cold_electron_temperature)) .= _cold_temperature(0.0)
        names = [x.name for x in layout.unknowns]
        :cold_electron_density in names &&
            (radial_values(u, layout, unknown_id(layout, :cold_electron_density)) .= _cold_density(0.0))
        :electric_field in names && (radial_values(u, layout, unknown_id(layout, :electric_field)) .= [20.0, 15.0])
        u
    end
end

# the same distribution with the plasma prescribed, shifted in time by `lag`
function _cold_reference(g, f0, times; lag=0.0, plasma_at=t -> _cold_base(max(t - lag, 0.0)))
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e20)])
    term = PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:f_zero),
                                      plasma_at; settings=COLD_SETTINGS)
    system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)]; time_scale=1e-6)
    return integrate(system, vec(f0), times)
end

@testset "cold temperature and density as unknowns" begin
    g = _cold_grid()
    theta = 100.0 / Constants.ELECTRON_REST_ENERGY
    f0 = [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta) for p in momentum_grid(g).p_centers, _ in 1:4, _ in 1:2]
    u0 = _cold_initial_state(g, f0)
    times = collect(range(0.0, 1e-4; length=11))

    @testset "the source and the term" begin
        system, term = _cold_system(g, u0)
        layout = system.layout
        u = u0(layout)
        plasma = plasma_values(term.plasma, u, layout, 0.0)
        @test plasma.cold_electron_temperature == _cold_temperature(0.0)
        @test plasma.cold_electron_density == _cold_density(0.0)
        @test plasma.ions[1].charge_state_densities == _cold_base(0.0).ions[1].charge_state_densities
        @test is_state_dependent(term.plasma) && !is_linear(term)
        @test Runes2P.jacobian_kind(term) === :exact && Runes2P.has_plasma_columns(term)
        @test Set(dependencies(term)) == Set([:f, :cold_electron_temperature, :cold_electron_density])
        @test !system.linear
        # the coefficients follow the unknowns: a new temperature rebuilds them, the same does not
        Runes2P.rhs!(zeros(length(layout)), term, u, layout, 1, 0.0)
        count = term.rebuilds[]
        Runes2P.rhs!(zeros(length(layout)), term, u, layout, 1, 0.0)
        @test term.rebuilds[] == count
        hotter = copy(u)
        radial_values(hotter, layout, unknown_id(layout, :cold_electron_temperature)) .*= 1.1
        Runes2P.rhs!(zeros(length(layout)), term, hotter, layout, 1, 0.0)
        @test term.rebuilds[] == count + 1
        # a temperature at the floor is a domain error, and the zero state needs the initial state
        cold = copy(u)
        radial_values(cold, layout, unknown_id(layout, :cold_electron_temperature)) .= 0.0
        @test_throws DomainError plasma_values(term.plasma, cold, layout, 0.0)
        @test_throws ArgumentError EquationSystem(layout, system.equations; time_scale=1e-6)
        lagged, lagged_term = _cold_system(g, u0; coefficients=:lagged)
        @test Runes2P.jacobian_kind(lagged_term) === :exact && !is_linear(lagged_term)
        @test !Runes2P.has_plasma_columns(lagged_term)
        @test_throws ArgumentError FokkerPlanckTerm(:f, AdvectionDiffusion(g); plasma=term.plasma, coefficients=:other)
        # the field-particle term takes the plasma from the unknowns lagged
        @test !is_linear(FieldParticleTerm(:f, FieldParticleOperator(g, _cold_base(0.0), COLD_SETTINGS);
                                           plasma=term.plasma))
        record = describe(term, Runes2P.DescriptionContext(times=times))
        @test record["coefficients"] == "implicit"
        @test record["plasma_unknowns"]["temperature"] == "cold_electron_temperature"
        @test record["plasma"] == "_cold_base"
    end

    @testset "implicit coefficients: the run with the plasma prescribed" begin
        reference = _cold_reference(g, f0, times)
        for density in (:cold_electron_density, nothing)
            system, _ = _cold_system(g, u0; density)
            solution = integrate(system, u0(system.layout), times)
            @test all(solution.statistics.accepted)
            f = reduce(hcat, vec(kinetic_array(solution.states[:, k], system.layout, 1)) for k in eachindex(times))
            # without the density unknown the base plasma holds the same n(t)
            @test f ≈ reference.states rtol = 1e-8
        end
    end

    @testset "lagged coefficients: the plasma of the start of each step" begin
        dt = times[2] - times[1]
        reference = _cold_reference(g, f0, times; lag=dt)
        system, _ = _cold_system(g, u0; coefficients=:lagged)
        solution = integrate(system, u0(system.layout), times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], system.layout, 1)) for k in eachindex(times))
        @test f ≈ reference.states rtol = 1e-8
        # and it differs from the implicit run by the lag
        implicit = _cold_reference(g, f0, times)
        @test !isapprox(f, implicit.states; rtol=1e-6)
        # the prescribed part of the plasma at the time of the evaluation, the unknowns at
        # the start of the step: a field that grows in time tells the two apart
        rising(t) = PlasmaState(; cold_electron_temperature=_cold_temperature(t), cold_electron_density=_cold_density(t),
                                ions=[IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field=[20.0, 15.0] .* (1 + t / 1e-4),
                                magnetic_field=[5.0, 5.0])
        mixed(t) = PlasmaState(; cold_electron_temperature=_cold_temperature(max(t - dt, 0.0)),
                               cold_electron_density=_cold_density(max(t - dt, 0.0)),
                               ions=[IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field=[20.0, 15.0] .* (1 + t / 1e-4),
                               magnetic_field=[5.0, 5.0])
        system, _ = _cold_system(g, u0; coefficients=:lagged, base=rising)
        solution = integrate(system, u0(system.layout), times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], system.layout, 1)) for k in eachindex(times))
        @test f ≈ _cold_reference(g, f0, times; plasma_at=mixed).states rtol = 1e-8
        @test !isapprox(f, _cold_reference(g, f0, times; plasma_at=t -> rising(max(t - dt, 0.0))).states; rtol=1e-6)
    end

    @testset "the energy row: T through W_cold" begin
        dt = times[2] - times[1]
        thermal(t) = THERMAL_ENERGY_FACTOR .* _cold_density(t) .* _cold_temperature(t)
        # the heating that backward Euler turns into the prescribed T(t) on the step grid
        heating(t) = (thermal(t) .- thermal(t - dt)) ./ dt
        function energy_system(power; limits=true)
            layout = UnknownLayout(g.radial_grid,
                                   [KineticUnknown(:f, g; scale=1e20),
                                    RadialUnknown(:cold_electron_temperature; scale=100.0, nonnegative=limits),
                                    RadialUnknown(:cold_electron_density; scale=1e20),
                                    RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e22,
                                                  nonnegative=limits)])
            term = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:f_zero);
                                    plasma=PlasmaFromUnknowns(_cold_base), settings=COLD_SETTINGS)
            equations = [Equation(:f, TimeDerivative(), term),
                         Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                                  PrescribedValueTerm(power)),
                         Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(_cold_density),
                                  LinearTerm(:cold_electron_density; coefficient=-1)),
                         thermal_energy_equation()]
            u = u0(layout)
            radial_values(u, layout, unknown_id(layout, :cold_electron_density)) .= _cold_density(0.0)
            set_thermal_energy!(u, layout)
            return EquationSystem(layout, equations; time_scale=1e-6, initial_state=u), u
        end
        system, u_start = energy_system(heating)
        layout = system.layout
        W_id, T_id = unknown_id(layout, :cold_thermal_energy), unknown_id(layout, :cold_electron_temperature)
        @test radial_values(u_start, layout, W_id) == thermal(0.0)
        solution = integrate(system, u_start, times; stepper=FixedSteps())
        @test all(solution.statistics.accepted)
        reference = _cold_reference(g, f0, times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], layout, 1)) for k in eachindex(times))
        @test f ≈ reference.states rtol = 1e-8
        for k in eachindex(times)
            state = solution.states[:, k]
            @test radial_values(state, layout, T_id) ≈ _cold_temperature(times[k]) rtol = 1e-10
            @test radial_values(state, layout, W_id) ≈ thermal(times[k]) rtol = 1e-10
        end
        # the zero state has no temperature: the prototype needs the state
        @test_throws ArgumentError EquationSystem(layout, system.equations; time_scale=1e-6)
        # a sink that would empty the energy in one step: the lower limits keep T and W
        # positive, the temperature floor stops the iterate before the collision model
        # fails, and the step fails in place of a negative temperature
        sink, u_sink = energy_system(t -> -10 .* thermal(0.0) ./ dt)
        @test_throws TimeStepFailure integrate(sink, u_sink, times[1:2]; stepper=FixedSteps(; max_halvings=0))
    end

    @testset "the plasma columns against central differences" begin
        # a state away from the prescribed one, with an anisotropic f
        rng = Xoshiro(58)
        # with the flux limiters as well
        for scheme in (:exponential_fitting, :central, :van_albada, :tcdf), density in (:cold_electron_density, nothing),
            field in (false, true)
            (field && scheme !== :exponential_fitting) && continue
            system, term = _cold_system(g, u0; scheme, density, field)
            layout = system.layout
            u = u0(layout)
            kinetic_array(u, layout, 1) .*= 1 .+ 0.2 .* rand(rng, size(f0)...)
            T_id = unknown_id(layout, :cold_electron_temperature)
            radial_values(u, layout, T_id) .*= [1.1, 0.9]
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 3e-5)
            names = Runes2P.source_dependencies(term.plasma)
            for name in names, ir in 1:2
                column = first(Runes2P.unknown_range(layout, unknown_id(layout, name), ir))
                h = 1e-6 * abs(u[column])
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                F_up, F_down = zeros(length(u)), zeros(length(u))
                rhs!(F_up, up, system, 3e-5)
                rhs!(F_down, down, system, 3e-5)
                difference = (F_up .- F_down) ./ (2h)
                @test Vector(J[:, column]) ≈ difference rtol = 1e-6 atol = 1e-7 * maximum(abs, difference)
            end
        end
    end

    @testset "Newton counts: implicit against lagged coefficients" begin
        # the oracle run against the prescribed plasma, on steps five times longer than in its test
        counts = Dict{Symbol,Int}()
        for coefficients in (:implicit, :lagged)
            system, _ = _cold_system(g, u0; coefficients)
            solution = integrate(system, u0(system.layout), collect(range(0.0, 1e-4; length=3));
                                 stepper=FixedSteps())
            @test all(solution.statistics.accepted)
            counts[coefficients] = sum(solution.statistics.newton_iterations)
        end
        @info "Newton iterations on 2 steps of 5e-5 s" counts
        @test counts[:implicit] <= 4 * 2
    end

    @testset "the exact operator with another scheme's block in f" begin
        # the stored block in f is upwind's, the residual exponential fitting's: GMRES with the
        # exact operator adds the difference, and the plasma columns are the residual's own
        system, term = _cold_system(g, u0; jacobian=:upwind)
        @test Runes2P.jacobian_kind(term) === :approximate && Runes2P.has_plasma_columns(term)
        layout = system.layout
        u = u0(layout)
        rng = Xoshiro(59)
        v = 1e-3 .* abs.(u) .* (rand(rng, length(u)) .- 0.5)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 3e-5)
        F_up, F_down = zeros(length(u)), zeros(length(u))
        rhs!(F_up, u .+ 1e-3 .* v, system, 3e-5)
        rhs!(F_down, u .- 1e-3 .* v, system, 3e-5)
        difference = (F_up .- F_down) ./ 2e-3
        product = jacobian_operator(J, system, u, 3e-5) * v
        @test product ≈ difference rtol = 1e-6
        @test !isapprox(J * v, difference; rtol=1e-3)
    end

    @testset "the conductivity follows the temperature" begin
        layout = UnknownLayout(g.radial_grid, [RadialUnknown(:electric_field; scale=10.0),
                                               RadialUnknown(:cold_electron_temperature; scale=100.0),
                                               RadialUnknown(:ohmic_current_density; scale=1e6)])
        minus(name) = LinearTerm(name; coefficient=-1)
        for kinetic in (:none, :dream_fit)
            model = ConductivityModel(PlasmaFromUnknowns(_cold_base; density=nothing); kinetic)
            term = ConductivityTerm(:electric_field, model)
            @test !is_linear(term) && Set(dependencies(term)) == Set([:electric_field, :cold_electron_temperature])
            equations = [Equation(:electric_field, Algebraic(), PrescribedValueTerm([20.0, 15.0]), minus(:electric_field)),
                         Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm(_cold_temperature),
                                  minus(:cold_electron_temperature)),
                         Equation(:ohmic_current_density, Algebraic(), term, minus(:ohmic_current_density))]
            u = zeros(length(layout))
            radial_values(u, layout, 1) .= [20.0, 15.0]
            radial_values(u, layout, 2) .= [100.0, 60.0]
            system = EquationSystem(layout, equations; initial_state=u)
            F = zeros(length(u))
            rhs!(F, u, system, 0.0)
            base = _cold_base(0.0)
            expected = [braams_karney_conductivity(T, effective_charge(base, ir), free_electron_density(base, ir)) *
                        (kinetic === :dream_fit ? 1 - test_particle_conductivity_fraction(effective_charge(base, ir)) : 1)
                        for (ir, T) in enumerate([100.0, 60.0])] .* [20.0, 15.0]
            @test radial_values(F, layout, 3) ≈ expected rtol = 1e-14
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            for ir in 1:2
                column = first(Runes2P.unknown_range(layout, 2, ir))
                h = 1e-6 * u[column]
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                F_up, F_down = zeros(length(u)), zeros(length(u))
                rhs!(F_up, up, system, 0.0)
                rhs!(F_down, down, system, 0.0)
                @test Vector(J[:, column]) ≈ (F_up .- F_down) ./ (2h) rtol = 1e-6
            end
            record = describe(model, Runes2P.DescriptionContext(times=[0.0]))
            @test record["conductivity"]["described"] === false
            # Newton steps with the column of the temperature in the fluid block
            solution = integrate(system, u, [0.0, 1e-6, 2e-6])
            @test all(solution.statistics.accepted)
            @test radial_values(solution.states[:, end], layout, 3) ≈
                  [braams_karney_conductivity(T, effective_charge(base, ir), free_electron_density(base, ir)) *
                   (kinetic === :dream_fit ? 1 - test_particle_conductivity_fraction(effective_charge(base, ir)) : 1)
                   for (ir, T) in enumerate(_cold_temperature(2e-6))] .* [20.0, 15.0] rtol = 1e-10
        end
        @test_throws ArgumentError ConductivityModel(PlasmaFromUnknowns(_cold_base); kinetic=:discrete_operator,
                                                     operator=AdvectionDiffusion(g))
    end

    @testset "the flux terms refuse coefficients that follow unknowns" begin
        system, term = _cold_system(g, u0)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g), RadialUnknown(:cold_electron_temperature),
                                               RadialUnknown(:cold_electron_density), RadialUnknown(:runaway_density)])
        flux = MomentumBoundaryFluxTerm(:f, term)
        @test_throws "issue #143" Runes2P.validate(flux, layout, unknown_id(layout, :runaway_density))
        _, lagged = _cold_system(g, u0; coefficients=:lagged)
        @test Runes2P.validate(MomentumBoundaryFluxTerm(:f, lagged), layout, unknown_id(layout, :runaway_density)) === nothing
        # the interface between the hot-tail and the runaway grid
        runaway = runaway_grid(g, [0.4, 0.8])
        upper = AdvectionDiffusion(runaway; scheme=:exponential_fitting, p_lower_boundary=:interface)
        two_grids = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g), KineticUnknown(:runaway, runaway),
                                                  RadialUnknown(:cold_electron_temperature),
                                                  RadialUnknown(:cold_electron_density)])
        for (coefficients, refused) in ((:implicit, true), (:lagged, false))
            lower = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:interface);
                                     plasma=PlasmaFromUnknowns(_cold_base), settings=COLD_SETTINGS, coefficients)
            interface = MomentumInterfaceTerm(:f, :runaway, lower, upper)
            if refused
                @test_throws "issue #143" Runes2P.validate(interface, two_grids, 1)
            else
                @test Runes2P.validate(interface, two_grids, 1) === nothing
            end
        end
    end
end
