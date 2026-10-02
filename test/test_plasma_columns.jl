# The collision coefficients built from a plasma in dual numbers, the source
# of the Jacobian columns of the cold-electron temperature and density: their partials
# against central differences of the coefficients, in the full and superthermal modes and
# in the three screening models.
using ForwardDiff: ForwardDiff

@testset "plasma columns" begin
    @testset "dual coefficients against central differences" begin
        # partially ionized argon, so that the screening models differ
        function screened(T, n)
            ions = [IonSpecies("D", 1, 1, [1e20, 8e19]), IonSpecies("Ar", 18, 2, [5e18, 4e18])]
            return PlasmaState(; cold_electron_temperature=T, cold_electron_density=n, ions,
                               electric_field=[20.0, 15.0], magnetic_field=[5.0, 5.0])
        end
        mom = MomentumGrid(uniform_p_faces(25, 0.0, 1.5), uniform_xi_faces(4))
        wide = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
        T, n = [100.0, 30.0], [1.2e20, 9e19]
        fields = (:advection_p, :diffusion_p_p, :advection_xi, :diffusion_xi_xi, :advection_p_times_p2_at_zero)
        for mode in (:full, :superthermal), screening in (:partial, :complete, :none)
            settings = CollisionSettings(; mode, screening)
            build(plasma) = fokker_planck_operator(wide, plasma, settings; synchrotron=true)
            reference = build(screened(T, n))
            dual = Runes2P.similar_coefficients(reference, ForwardDiff.Dual{Runes2P.PlasmaColumnTag,Float64,2})
            set_fokker_planck_coefficients!(dual, Runes2P.dual_plasma(screened(T, n), true), settings; synchrotron=true)
            for field in fields
                @test ForwardDiff.value.(getfield(dual, field)) ≈ getfield(reference, field) rtol = 1e-13
            end
            for (k, (dT, dn)) in enumerate(((1e-5, 0.0), (0.0, 1e-5)))
                up = build(screened(T .* (1 + dT), n .* (1 + dn)))
                down = build(screened(T .* (1 - dT), n .* (1 - dn)))
                for field in fields
                    scale = k == 1 ? reshape(2dT .* T, 1, 1, :) : reshape(2dn .* n, 1, 1, :)
                    field === :advection_p_times_p2_at_zero && (scale = dropdims(scale; dims=1))
                    difference = (getfield(up, field) .- getfield(down, field)) ./ scale
                    partial = ForwardDiff.partials.(getfield(dual, field), k)
                    @test partial ≈ difference rtol = 1e-6 atol = 1e-6 * maximum(abs, difference)
                end
            end
        end
    end


    # a two-radius grid down to the bulk at a few eV
    mom = MomentumGrid(uniform_p_faces(30, 0.0, 0.06), uniform_xi_faces(4))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
    theta(T) = T / Constants.ELECTRON_REST_ENERGY
    maxwellian(T) = [1e19 * exp(-(sqrt(1 + p^2) - 1) / theta(T)) for p in mom.p_centers, _ in 1:4]
    minus(name) = LinearTerm(name; coefficient=-1)
    # the column of `name` at radius ir against a central difference of the right-hand side
    function column_difference(system, u, column, t)
        h = 1e-6 * abs(u[column])
        up, down = copy(u), copy(u)
        up[column] += h
        down[column] -= h
        F_up, F_down = zeros(length(u)), zeros(length(u))
        rhs!(F_up, up, system, t)
        rhs!(F_down, down, system, t)
        return (F_up .- F_down) ./ (2h)
    end

    @testset "particle source with the temperature as an unknown" begin
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e19), RadialUnknown(:particle_source; scale=1e22),
                                               RadialUnknown(:cold_electron_temperature; scale=10.0)])
        term = ParticleSourceTerm(:particle_source, g, :cold_electron_temperature)
        @test !is_linear(term) && Set(dependencies(term)) == Set([:particle_source, :cold_electron_temperature])
        for temperatures in ([1.0, 1.5], [8.0, 5.0])
            equations = [Equation(:f, TimeDerivative(), term),
                         Equation(:particle_source, Algebraic(), PrescribedValueTerm([1e22, 2e22]), minus(:particle_source)),
                         Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm(temperatures),
                                  minus(:cold_electron_temperature))]
            u = zeros(length(layout))
            for ir in 1:2
                kinetic_view(u, layout, 1, ir) .= maxwellian(temperatures[ir])
            end
            radial_values(u, layout, 2) .= [1e22, 2e22]
            radial_values(u, layout, 3) .= temperatures
            system = EquationSystem(layout, equations; initial_state=u)
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            # the direct block against the AD of the residual, and the column against a difference
            @test Matrix(J) ≈ ForwardDiff.jacobian(v -> rhs!(similar(v), v, system, 0.0), u) rtol = 1e-12
            for ir in 1:2
                column = first(Runes2P.unknown_range(layout, 3, ir))
                @test Vector(J[:, column]) ≈ column_difference(system, u, column, 0.0) rtol = 1e-6
                # the source adds S particles at every temperature: the column moves none
                change = reshape(Vector(J[Runes2P.unknown_range(layout, 1, ir), column]), size(maxwellian(1.0)))
                f_change = zeros(size(g)); f_change[:, :, ir] .= change
                @test abs(density(g, f_change)[ir]) < 1e-12 * density(g, abs.(f_change))[ir]
            end
        end
    end

    @testset "hot threshold with the temperature as an unknown" begin
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e19),
                                               RadialUnknown(:cold_electron_temperature; scale=10.0),
                                               RadialUnknown(:hot_electron_density; scale=1e17),
                                               RadialUnknown(:hot_current_density; scale=1e4)])
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=5))
        density_term = DensityMomentTerm(:f, g; region)
        current_term = CurrentMomentTerm(:f, g; region)
        @test !is_linear(density_term) && :cold_electron_temperature in dependencies(current_term)
        # a temperature that puts the limit inside a cell, and one that puts it on a face
        face = mom.p_faces[20]
        for temperatures in ([3.0, 2.0], [(face / 5)^2 * Constants.ELECTRON_REST_ENERGY / 2, 2.0])
            equations = [Equation(:f, TimeDerivative(), LinearTerm(:f; coefficient=-1e3)),
                         Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm(temperatures),
                                  minus(:cold_electron_temperature)),
                         Equation(:hot_electron_density, Algebraic(), density_term, minus(:hot_electron_density)),
                         Equation(:hot_current_density, Algebraic(), current_term, minus(:hot_current_density))]
            u = zeros(length(layout))
            for ir in 1:2
                kinetic_view(u, layout, 1, ir) .= maxwellian(temperatures[ir]) .* (1 .+ 0.2 .* (1:4)')
            end
            radial_values(u, layout, 2) .= temperatures
            system = EquationSystem(layout, equations; initial_state=u)
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            @test Matrix(J) ≈ ForwardDiff.jacobian(v -> rhs!(similar(v), v, system, 0.0), u) rtol = 1e-12
            for ir in 1:2
                column = first(Runes2P.unknown_range(layout, 2, ir))
                rows = [first(Runes2P.unknown_range(layout, 3, ir)), first(Runes2P.unknown_range(layout, 4, ir))]
                if ir == 1 && temperatures[1] != 3.0
                    # on a face the envelope has a kink: the column is one of the one-sided derivatives
                    h = 1e-7 * u[column]
                    up, down = copy(u), copy(u)
                    up[column] += h
                    down[column] -= h
                    F, F_up, F_down = zeros(length(u)), zeros(length(u)), zeros(length(u))
                    rhs!(F, u, system, 0.0); rhs!(F_up, up, system, 0.0); rhs!(F_down, down, system, 0.0)
                    forward, backward = (F_up .- F)[rows] ./ h, (F .- F_down)[rows] ./ h
                    @test isapprox(Vector(J[rows, column]), forward; rtol=1e-4) ||
                          isapprox(Vector(J[rows, column]), backward; rtol=1e-4)
                else
                    @test Vector(J[rows, column]) ≈ column_difference(system, u, column, 0.0)[rows] rtol = 1e-6
                end
            end
        end
        @test_throws ArgumentError momentum_envelope(region, g, 1, 0.0)
    end

    @testset "field-particle term lagged with the plasma from the unknowns (issue #70)" begin
        settings = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal)
        temperature(t) = [5.0, 4.0] .* (1 - 0.3 * t / 1e-5)
        base(t) = PlasmaState(; cold_electron_temperature=temperature(t), cold_electron_density=[1e19, 8e18],
                              ions=[IonSpecies("D", 1, 1, [1e19, 8e18])], electric_field=[0.5, 0.4],
                              magnetic_field=[5.0, 5.0])
        f0 = zeros(size(g))
        for ir in 1:2
            f0[:, :, ir] .= maxwellian(temperature(0.0)[ir])
        end
        times = collect(range(0.0, 1e-5; length=6))
        dt = times[2] - times[1]
        # T an unknown held at T(t), both terms lagged
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e19),
                                               RadialUnknown(:cold_electron_temperature; scale=10.0)])
        plasma = PlasmaFromUnknowns(base; density=nothing)
        fp = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:f_zero);
                              plasma, settings, coefficients=:lagged)
        field_particle = FieldParticleTerm(:f, FieldParticleOperator(g, base(0.0), settings); plasma)
        @test !is_linear(field_particle) && :cold_electron_temperature in dependencies(field_particle)
        u0 = zeros(length(layout))
        kinetic_array(u0, layout, 1) .= f0
        radial_values(u0, layout, 2) .= temperature(0.0)
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), fp, field_particle),
                                         Equation(:cold_electron_temperature, Algebraic(),
                                                  PrescribedValueTerm(temperature), minus(:cold_electron_temperature))];
                                time_scale=1e-6, initial_state=u0)
        solution = integrate(system, u0, times; solver=NewtonSolver(; linear_solver=GMRES()))
        # the same with the plasma prescribed one step earlier
        reference_layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e19)])
        shifted(t) = base(max(t - dt, 0.0))
        reference_system = EquationSystem(reference_layout, [Equation(:f, TimeDerivative(),
            PrescribedFokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting, p_upper_boundary=:f_zero),
                                       shifted; settings),
            FieldParticleTerm(:f, FieldParticleOperator(g, base(0.0), settings); plasma=shifted))]; time_scale=1e-6)
        reference = integrate(reference_system, vec(f0), times; solver=NewtonSolver(; linear_solver=GMRES()))
        f = reduce(hcat, vec(kinetic_array(solution.states[:, k], layout, 1)) for k in eachindex(times))
        @test f ≈ reference.states rtol = 1e-8
    end
end
