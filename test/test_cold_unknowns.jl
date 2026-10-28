# The cold-electron temperature and density as unknowns of the system: the plasma source
# that reads them, the implicit and the lagged coefficients, and the oracle runs against a
# plasma prescribed in time.
using ForwardDiff: ForwardDiff
using Random: Xoshiro

const COLD_SETTINGS = CollisionSettings(mode = :full)

_cold_temperature(t) = [100.0, 80.0] .* (1 - 0.4 * t / 1e-4)
_cold_density(t) = [1e20, 8e19] .* (1 + 0.3 * t / 1e-4)

function _cold_base(t)
    return PlasmaState(; cold_electron_temperature = _cold_temperature(t),
        cold_electron_density = _cold_density(t),
        ions = [IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field = [20.0, 15.0],
        magnetic_field = [5.0, 5.0])
end

function _cold_grid()
    mom = MomentumGrid(uniform_p_faces(30, 0.0, 0.2), uniform_xi_faces(4))
    return cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
end

# f with T and n as unknowns held at the prescribed values by algebraic rows
function _cold_system(g, u0; lagged = :ions, density = :cold_electron_density,
        scheme = :exponential_fitting,
        field = false, jacobian = :exact, base = _cold_base)
    unknowns = [KineticUnknown(:f, g; scale = 1e20),
        RadialUnknown(:cold_electron_temperature; scale = 100.0)]
    density === nothing ||
        push!(unknowns, RadialUnknown(:cold_electron_density; scale = 1e20))
    field && push!(unknowns, RadialUnknown(:electric_field; scale = 10.0))
    layout = UnknownLayout(g.radial_grid, unknowns)
    plasma = PlasmaFromUnknowns(base; density)
    term = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme, p_upper_boundary = :f_zero);
        plasma, settings = COLD_SETTINGS, lagged, jacobian,
        electric_field = field ? :electric_field : nothing)
    minus(name) = LinearTerm(name; coefficient = -1)
    equations = [Equation(:f, TimeDerivative(), term),
        Equation(
            :cold_electron_temperature, Algebraic(), PrescribedValueTerm(_cold_temperature),
            minus(:cold_electron_temperature))]
    field && push!(equations,
        Equation(:electric_field, Algebraic(), PrescribedValueTerm([20.0, 15.0]),
            minus(:electric_field)))
    density === nothing ||
        push!(equations,
            Equation(
                :cold_electron_density, Algebraic(), PrescribedValueTerm(_cold_density),
                minus(:cold_electron_density)))
    return EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0(layout)),
    term
end

function _cold_initial_state(g, f0)
    return layout -> begin
        u = zeros(length(layout))
        kinetic_array(u, layout, unknown_id(layout, :f)) .= f0
        #! format: off
        unknown_values(u, layout, unknown_id(layout, :cold_electron_temperature)) .= _cold_temperature(0.0)
        names = [x.name for x in layout.unknowns]
        :cold_electron_density in names &&
            (unknown_values(u, layout, unknown_id(layout, :cold_electron_density)) .= _cold_density(0.0))
        :electric_field in names && (unknown_values(u, layout, unknown_id(layout, :electric_field)) .= [20.0, 15.0])
        #! format: on
        u
    end
end

# the same distribution with the plasma prescribed, shifted in time by `lag`
function _cold_reference(
        g, f0, times; lag = 0.0, plasma_at = t -> _cold_base(max(t - lag, 0.0)))
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1e20)])
    term = PrescribedFokkerPlanckTerm(:f,
        AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero),
        plasma_at; settings = COLD_SETTINGS)
    system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)]; time_scale = 1e-6)
    return integrate(system, vec(f0), times)
end

@testset "cold temperature and density as unknowns" begin
    g = _cold_grid()
    theta = 100.0 / Constants.ELECTRON_REST_ENERGY
    f0 = [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta)
          for p in momentum_grid(g).p_centers, _ in 1:4, _ in 1:2]
    u0 = _cold_initial_state(g, f0)
    times = collect(range(0.0, 1e-4; length = 11))

    @testset "the source and the term" begin
        system, term = _cold_system(g, u0)
        layout = system.layout
        u = u0(layout)
        plasma = plasma_values(term.plasma, u, layout, 0.0)
        @test plasma.cold_electron_temperature == _cold_temperature(0.0)
        @test plasma.cold_electron_density == _cold_density(0.0)
        @test plasma.ions[1].charge_state_densities ==
              _cold_base(0.0).ions[1].charge_state_densities
        @test is_state_dependent(term.plasma) && !is_linear(term)
        @test Runes2P.jacobian_kind(term) === :exact && Runes2P.has_plasma_columns(term)
        @test Set(dependencies(term)) ==
              Set([:f, :cold_electron_temperature, :cold_electron_density])
        @test !system.linear
        # the coefficients follow the unknowns: a new temperature rebuilds them, the same does not
        Runes2P.rhs!(zeros(length(layout)), term, u, layout, 1, 0.0)
        count = term.rebuilds[]
        Runes2P.rhs!(zeros(length(layout)), term, u, layout, 1, 0.0)
        @test term.rebuilds[] == count
        hotter = copy(u)
        unknown_values(hotter, layout, unknown_id(layout, :cold_electron_temperature)) .*= 1.1
        Runes2P.rhs!(zeros(length(layout)), term, hotter, layout, 1, 0.0)
        @test term.rebuilds[] == count + 1
        # a temperature at the floor is a domain error, and the zero state needs the initial state
        cold = copy(u)
        unknown_values(cold, layout, unknown_id(layout, :cold_electron_temperature)) .= 0.0
        @test_throws DomainError plasma_values(term.plasma, cold, layout, 0.0)
        # the default floor is 0.5 eV: a temperature at it is refused, one above
        # it is taken, and a source with a lower floor follows DREAM below it
        @test term.plasma.temperature_floor == 0.5
        unknown_values(cold, layout, unknown_id(layout, :cold_electron_temperature)) .= 0.5
        @test_throws DomainError plasma_values(term.plasma, cold, layout, 0.0)
        unknown_values(cold, layout, unknown_id(layout, :cold_electron_temperature)) .= 0.51
        @test all(==(0.51), plasma_values(term.plasma, cold, layout, 0.0).cold_electron_temperature)
        unknown_values(cold, layout, unknown_id(layout, :cold_electron_temperature)) .= 0.1
        low = PlasmaFromUnknowns(
            term.plasma.prescribed; temperature = term.plasma.temperature,
            density = term.plasma.density,
            temperature_floor = 0.01)
        @test all(==(0.1), plasma_values(low, cold, layout, 0.0).cold_electron_temperature)
        @test_throws ArgumentError EquationSystem(layout, system.equations; time_scale = 1e-6)
        lagged, lagged_term = _cold_system(g, u0; lagged = :all)
        @test Runes2P.jacobian_kind(lagged_term) === :exact && !is_linear(lagged_term)
        @test !Runes2P.has_plasma_columns(lagged_term)
        @test_throws ArgumentError FokkerPlanckTerm(
            :f, AdvectionDiffusion(g); plasma = term.plasma, lagged = :other)
        # the field-particle term takes the plasma from the unknowns lagged
        @test !is_linear(FieldParticleTerm(
            :f, FieldParticleOperator(g, _cold_base(0.0), COLD_SETTINGS);
            plasma = term.plasma))
        record = describe(term, Runes2P.DescriptionContext(times = times))
        @test !haskey(record, "lagged") && !haskey(record, "coefficients")
        @test describe(lagged_term, Runes2P.DescriptionContext(times = times))["lagged"] ==
              "all"
        @test record["plasma_unknowns"]["temperature"] == "cold_electron_temperature"
        @test record["plasma"] == "_cold_base"
    end

    @testset "implicit coefficients: the run with the plasma prescribed" begin
        reference = _cold_reference(g, f0, times)
        for density in (:cold_electron_density, nothing)
            system, _ = _cold_system(g, u0; density)
            solution = integrate(system, u0(system.layout), times)
            @test all(solution.statistics.accepted)
            f = reduce(hcat,
                vec(kinetic_array(solution.states[:, k], system.layout, 1))
                for k in eachindex(times))
            # without the density unknown the base plasma holds the same n(t)
            @test f ≈ reference.states rtol = 1e-8
        end
    end

    @testset "lagged coefficients: the plasma of the start of each step" begin
        dt = times[2] - times[1]
        reference = _cold_reference(g, f0, times; lag = dt)
        system, _ = _cold_system(g, u0; lagged = :all)
        solution = integrate(system, u0(system.layout), times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], system.layout, 1))
        for k in eachindex(times))
        @test f ≈ reference.states rtol = 1e-8
        # and it differs from the implicit run by the lag
        implicit = _cold_reference(g, f0, times)
        @test !isapprox(f, implicit.states; rtol = 1e-6)
        # the prescribed part of the plasma at the time of the evaluation, the unknowns at
        # the start of the step: a field that grows in time tells the two apart
        rising(t) = PlasmaState(; cold_electron_temperature = _cold_temperature(t),
            cold_electron_density = _cold_density(t),
            ions = [IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field = [20.0, 15.0] .*
                                                                           (1 + t / 1e-4),
            magnetic_field = [5.0, 5.0])
        mixed(t) = PlasmaState(;
            cold_electron_temperature = _cold_temperature(max(t - dt, 0.0)),
            cold_electron_density = _cold_density(max(t - dt, 0.0)),
            ions = [IonSpecies("D", 1, 1, [1e20, 8e19])], electric_field = [20.0, 15.0] .*
                                                                           (1 + t / 1e-4),
            magnetic_field = [5.0, 5.0])
        system, _ = _cold_system(g, u0; lagged = :all, base = rising)
        solution = integrate(system, u0(system.layout), times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], system.layout, 1))
        for k in eachindex(times))
        @test f ≈ _cold_reference(g, f0, times; plasma_at = mixed).states rtol = 1e-8
        @test !isapprox(f,
            _cold_reference(g, f0, times; plasma_at = t -> rising(max(t - dt, 0.0))).states;
            rtol = 1e-6)
    end

    @testset "the energy row: T through W_cold" begin
        dt = times[2] - times[1]
        thermal(t) = THERMAL_ENERGY_FACTOR .* _cold_density(t) .* _cold_temperature(t)
        # the heating that backward Euler turns into the prescribed T(t) on the step grid
        heating(t) = (thermal(t) .- thermal(t - dt)) ./ dt
        function energy_system(power; limits = true)
            layout = UnknownLayout(g.radial_grid,
                [KineticUnknown(:f, g; scale = 1e20),
                    RadialUnknown(:cold_electron_temperature; scale = 100.0, nonnegative = limits),
                    RadialUnknown(:cold_electron_density; scale = 1e20),
                    RadialUnknown(
                        :cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e22,
                        nonnegative = limits)])
            term = FokkerPlanckTerm(:f,
                AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
                plasma = PlasmaFromUnknowns(_cold_base), settings = COLD_SETTINGS)
            equations = [Equation(:f, TimeDerivative(), term),
                Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                    PrescribedValueTerm(power)),
                Equation(
                    :cold_electron_density, Algebraic(), PrescribedValueTerm(_cold_density),
                    LinearTerm(:cold_electron_density; coefficient = -1)),
                thermal_energy_equation()]
            u = u0(layout)
            unknown_values(u, layout, unknown_id(layout, :cold_electron_density)) .= _cold_density(0.0)
            set_thermal_energy!(u, layout)
            return EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u),
            u
        end
        system, u_start = energy_system(heating)
        layout = system.layout
        W_id, T_id = unknown_id(layout, :cold_thermal_energy),
        unknown_id(layout, :cold_electron_temperature)
        @test unknown_values(u_start, layout, W_id) == thermal(0.0)
        solution = integrate(system, u_start, times; stepper = FixedSteps())
        @test all(solution.statistics.accepted)
        reference = _cold_reference(g, f0, times)
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], layout, 1))
        for k in eachindex(times))
        @test f ≈ reference.states rtol = 1e-8
        for k in eachindex(times)
            state = solution.states[:, k]
            @test unknown_values(state, layout, T_id) ≈ _cold_temperature(times[k]) rtol = 1e-10
            @test unknown_values(state, layout, W_id) ≈ thermal(times[k]) rtol = 1e-10
        end
        # the zero state has no temperature: the prototype needs the state
        @test_throws ArgumentError EquationSystem(layout, system.equations; time_scale = 1e-6)
        # a sink that would empty the energy in one step: the lower limits keep T and W
        # positive, the temperature floor stops the iterate before the collision model
        # fails, and the step fails in place of a negative temperature
        sink, u_sink = energy_system(t -> -10 .* thermal(0.0) ./ dt)
        @test_throws TimeStepFailure integrate(
            sink, u_sink, times[1:2]; stepper = FixedSteps(; max_halvings = 0))
        # the floor of the plasma source is the lower limit of its temperature in the Newton
        # iteration, with and without a limit declared on the unknown
        for limits in (true, false)
            floored, _ = energy_system(heating; limits)
            @test Runes2P.temperature_floors(floored) == [:cold_electron_temperature => 0.5]
            ws = NewtonWorkspace(floored)
            entries = unknown_range(floored.layout, unknown_id(floored.layout, :cold_electron_temperature), 1)
            @test all(i -> ws.limit_value[findfirst(==(i), ws.limited)] == 0.5, entries)
        end
        # the iterates of the sink stop at the floor: the failed solve leaves T above 0.5 eV
        u_trial = copy(u_sink)
        result = newton_solve!(u_trial, u_sink, NewtonWorkspace(sink), times[2], dt)
        @test result.status !== :converged
        @test all(>=(0.5),
            unknown_values(u_trial, sink.layout, unknown_id(sink.layout, :cold_electron_temperature)))
    end

    @testset "the plasma columns against central differences" begin
        # a state away from the prescribed one, with an anisotropic f
        rng = Xoshiro(58)
        # with the flux limiters van Albada and TCDF as well
        for scheme in (:exponential_fitting, :central, :van_albada, :tcdf),
            density in (:cold_electron_density, nothing),
            field in (false, true)
            (field && scheme !== :exponential_fitting) && continue
            system, term = _cold_system(g, u0; scheme, density, field)
            layout = system.layout
            u = u0(layout)
            kinetic_array(u, layout, 1) .*= 1 .+ 0.2 .* rand(rng, size(f0)...)
            T_id = unknown_id(layout, :cold_electron_temperature)
            unknown_values(u, layout, T_id) .*= [1.1, 0.9]
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
                @test Vector(J[:, column])≈difference rtol=1e-6 atol=1e-7*maximum(abs, difference)
            end
        end
    end

    @testset "Newton counts: implicit against lagged coefficients" begin
        # the oracle runs above on steps five times longer
        counts = Dict{Symbol, Int}()
        for lagged in (:ions, :all)
            system, _ = _cold_system(g, u0; lagged)
            solution = integrate(
                system, u0(system.layout), collect(range(0.0, 1e-4; length = 3));
                stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            counts[lagged] = sum(solution.statistics.newton_iterations)
        end
        @info "Newton iterations on 2 steps of 5e-5 s" counts
        @test counts[:ions] <= 4 * 2
    end

    @testset "the exact operator with another scheme's block in f" begin
        # the stored block in f is upwind's, the residual exponential fitting's: GMRES with the
        # exact operator adds the difference, and the plasma columns are the residual's own
        system, term = _cold_system(g, u0; jacobian = :upwind)
        @test Runes2P.jacobian_kind(term) === :approximate &&
              Runes2P.has_plasma_columns(term)
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
        @test !isapprox(J * v, difference; rtol = 1e-3)
    end

    @testset "the conductivity follows the temperature" begin
        layout = UnknownLayout(g.radial_grid,
            [RadialUnknown(:electric_field; scale = 10.0),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:ohmic_current_density; scale = 1e6)])
        minus(name) = LinearTerm(name; coefficient = -1)
        for kinetic in (:none, :dream_fit)
            model = ConductivityModel(PlasmaFromUnknowns(_cold_base; density = nothing); kinetic)
            term = ConductivityTerm(:electric_field, model)
            @test !is_linear(term) &&
                  Set(dependencies(term)) ==
                  Set([:electric_field, :cold_electron_temperature])
            equations = [
                Equation(:electric_field, Algebraic(),
                    PrescribedValueTerm([20.0, 15.0]), minus(:electric_field)),
                Equation(:cold_electron_temperature, Algebraic(),
                    PrescribedValueTerm(_cold_temperature),
                    minus(:cold_electron_temperature)),
                Equation(:ohmic_current_density, Algebraic(), term, minus(:ohmic_current_density))]
            u = zeros(length(layout))
            unknown_values(u, layout, 1) .= [20.0, 15.0]
            unknown_values(u, layout, 2) .= [100.0, 60.0]
            system = EquationSystem(layout, equations; initial_state = u)
            F = zeros(length(u))
            rhs!(F, u, system, 0.0)
            base = _cold_base(0.0)
            expected = [braams_karney_conductivity(T, effective_charge(base, ir),
                            free_electron_density(base, ir)) *
                        (kinetic === :dream_fit ?
                         1 -
                         test_particle_conductivity_fraction(effective_charge(base, ir)) :
                         1)
                        for (ir, T) in enumerate([100.0, 60.0])] .* [20.0, 15.0]
            @test unknown_values(F, layout, 3) ≈ expected rtol = 1e-14
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
            record = describe(model, Runes2P.DescriptionContext(times = [0.0]))
            @test record["conductivity"]["described"] === false
            # Newton steps with the column of the temperature in the fluid block
            solution = integrate(system, u, [0.0, 1e-6, 2e-6])
            @test all(solution.statistics.accepted)
            @test unknown_values(solution.states[:, end], layout, 3) ≈
                  [braams_karney_conductivity(T, effective_charge(base, ir), free_electron_density(base, ir)) *
                   (kinetic === :dream_fit ?
                    1 - test_particle_conductivity_fraction(effective_charge(base, ir)) :
                    1)
                   for (ir, T) in enumerate(_cold_temperature(2e-6))] .* [20.0, 15.0] rtol = 1e-10
        end
        @test_throws ArgumentError ConductivityModel(
            PlasmaFromUnknowns(_cold_base); kinetic = :discrete_operator,
            operator = AdvectionDiffusion(g))
    end

    @testset "the boundary flux follows the plasma unknowns" begin
        # the flux through p_max into a density and an energy, with the temperature, the
        # density and the field as unknowns, exponential fitting and f = 0 beyond p_max (the
        # boundary of DREAM's disruption case): its columns of the plasma against
        # central differences of the residual at the default optimisation level
        system, term = _cold_system(g, u0; field = true)
        @test Runes2P.has_plasma_columns(term)
        layout = UnknownLayout(g.radial_grid,
            [system.layout.unknowns; RadialUnknown(:runaway_density; scale = 1e15);
             RadialUnknown(:lost_energy; scale = 1.0)])
        flux = MomentumBoundaryFluxTerm(:f, term)
        energy = MomentumBoundaryFluxTerm(:f, term; weight = :energy, coefficient = -1)
        @test Runes2P.has_plasma_columns(flux) && !is_linear(flux)
        @test :cold_electron_temperature in dependencies(flux) &&
              :cold_electron_density in dependencies(flux)
        equations = [system.equations...;
                     Equation(:runaway_density, TimeDerivative(), flux);
                     Equation(:lost_energy, TimeDerivative(), energy)]
        full = EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0(layout))
        u = u0(layout)
        J = jacobian_prototype(full)
        jacobian!(J, full, u, 0.0)
        rhs_of(x) = (out = zeros(length(x)); rhs!(out, x, full, 0.0); out)
        rows = [first(unknown_range(layout, unknown_id(layout, name), ir))
                for name in (:runaway_density, :lost_energy)
                for ir in 1:2]
        for name in (:cold_electron_temperature, :cold_electron_density, :electric_field),
            ir in 1:2

            column = first(unknown_range(layout, unknown_id(layout, name), ir))
            h = 1e-5 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (rhs_of(up)[rows] - rhs_of(down)[rows]) / 2h
            @test maximum(abs, difference) > 0
            @test maximum(abs, Vector(J[rows, column]) - difference) <=
                  1e-6 * maximum(abs, difference)
        end
        # lagged, the flux has no plasma columns and stays linear in f
        _, lagged = _cold_system(g, u0; lagged = :all)
        @test !Runes2P.has_plasma_columns(MomentumBoundaryFluxTerm(:f, lagged))
        # the interface between the hot-tail and the runaway grid
        runaway = runaway_grid(g, [0.4, 0.8])
        upper = AdvectionDiffusion(runaway; scheme = :exponential_fitting, p_lower_boundary = :interface)
        two_grids = UnknownLayout(g.radial_grid,
            [KineticUnknown(:f, g), KineticUnknown(:runaway, runaway),
                RadialUnknown(:cold_electron_temperature),
                RadialUnknown(:cold_electron_density)])
        for (lagged, refused) in ((:ions, true), (:all, false))
            lower = FokkerPlanckTerm(:f,
                AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :interface);
                plasma = PlasmaFromUnknowns(_cold_base), settings = COLD_SETTINGS, lagged)
            interface = MomentumInterfaceTerm(:f, :runaway, lower, upper)
            if refused
                @test_throws "no Jacobian columns for them" Runes2P.validate(interface, two_grids, 1)
            else
                @test Runes2P.validate(interface, two_grids, 1) === nothing
            end
        end
    end
end
