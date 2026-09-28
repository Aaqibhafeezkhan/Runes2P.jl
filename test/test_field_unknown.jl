# The electric field as an unknown of the kinetic equations: the operator
# with the field as an argument against the operator that holds a prescribed field, the
# column of the field in the stored Jacobian, the Jacobian modes of the limiters, the flux
# between the hot-tail and the runaway grid, and the runaway threshold of the knock-on cutoff.
using Random: Xoshiro

# directional derivative of the right-hand side by central differences
function directional_derivative(system, u, v, t; step=1e-6)
    up, down = zeros(length(u)), zeros(length(u))
    h = step * maximum(abs, u) / maximum(abs, v)
    rhs!(up, u .+ h .* v, system, t)
    rhs!(down, u .- h .* v, system, t)
    return (up .- down) ./ (2h)
end

@testset "electric field as an unknown" begin
    rng = Xoshiro(52)
    rad = RadialGrid(uniform_r_faces(3, 0.6))
    momentum = MomentumGrid(biuniform_p_faces(14, 6, 0.0, 0.5, 3.0), uniform_xi_faces(8))
    grid = cylindrical_grid(momentum, rad)
    field = [1.5, -0.7, 3.0]
    plasma = PlasmaState(cold_electron_temperature=[100.0, 300.0, 1000.0],
                         ions=[IonSpecies("D", 1, 1, [1e20, 5e19, 2e19])], electric_field=field,
                         magnetic_field=fill(5.0, 3))
    settings = CollisionSettings(mode=:full)

    @testset "unit field coefficients" begin
        op = AdvectionDiffusion(grid)
        @test all(==(0), op.unit_field_advection_p) && all(==(0), op.unit_field_advection_xi)
        set_unit_field_coefficients!(op)
        a = Constants.ELEMENTARY_CHARGE / (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT)
        for ir in 1:3, (j, xi) in enumerate(momentum.xi_centers), k in 1:15
            @test op.unit_field_advection_p[k, j, ir] == a * xi
        end
        for ir in 1:3, (k, xi) in enumerate(momentum.xi_faces), (i, p) in enumerate(momentum.p_centers)
            @test op.unit_field_advection_xi[i, k, ir] == a * (1 - xi^2) / p
        end
        # they are those of add_electric_field! at 1 V/m
        unit = AdvectionDiffusion(grid)
        add_electric_field!(unit, with_electric_field(plasma, ones(3)))
        @test unit.advection_p == op.unit_field_advection_p
        @test unit.advection_xi == op.unit_field_advection_xi
        # a rebuild of the coefficients leaves them as they are
        zero_coefficients!(op)
        @test op.unit_field_advection_p == unit.advection_p
        @test with_electric_field(plasma, [1.0, 2.0, 3.0]).electric_field == [1.0, 2.0, 3.0]
        @test_throws DimensionMismatch with_electric_field(plasma, [1.0, 2.0])
    end

    boundaries = (:f_zero, :outflow, :constant_flux, :extrapolated_flux_consistent)
    @testset "operator with the field as an argument ($scheme, $p_upper_boundary)" for
            scheme in (:central, :exponential_fitting, :upwind, :tcdf, :quick), p_upper_boundary in boundaries
        prescribed = fokker_planck_operator(grid, plasma, settings; scheme, p_upper_boundary, synchrotron=true)
        free = fokker_planck_operator(grid, plasma, settings; scheme, p_upper_boundary, synchrotron=true,
                                      electric_field=false)
        # the operator without the field holds the collisions and the radiation alone
        without_field = fokker_planck_operator(grid, with_electric_field(plasma, zeros(3)), settings; scheme,
                                               p_upper_boundary, synchrotron=true)
        @test free.advection_p == without_field.advection_p
        @test free.advection_xi == without_field.advection_xi
        f = rand(rng, size(grid)...)
        expected = apply!(zeros(size(grid)), prescribed, f)
        with_argument = apply!(zeros(size(grid)), free, f; electric_field=field)
        @test with_argument ≈ expected rtol = 1e-12
        @test apply!(zeros(size(grid)), free, f) != expected
        for ir in 1:3, face in (1, 7, 15)
            @test momentum_flux(free, f, face, ir; electric_field=field) ≈ momentum_flux(prescribed, f, face, ir) rtol = 1e-11 atol = 1e-14 * maximum(abs, expected)
        end
        @test (@allocated apply!(with_argument, free, f; electric_field=field)) == 0
        @test_throws DimensionMismatch apply!(with_argument, free, f; electric_field=[1.0, 2.0])
    end

    # the field as an unknown that an algebraic equation holds at prescribed values
    function field_system(operator; jacobian=:exact, values=field)
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field; scale=1.0),
                                     KineticUnknown(:hot_tail_distribution, grid; scale=1e16)])
        term = FokkerPlanckTerm(:hot_tail_distribution, operator; jacobian, electric_field=:electric_field)
        system = EquationSystem(layout, [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(values),
                     LinearTerm(:electric_field; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(), term)])
        return layout, system, term
    end
    initial(p, xi) = 1e16 * exp(-p / 0.3) * (1 + 0.5 * xi)
    f0 = [initial(p, xi) for p in momentum.p_centers, xi in momentum.xi_centers, _ in 1:3]

    @testset "term and stored Jacobian ($scheme)" for scheme in (:central, :exponential_fitting, :tcdf)
        free = fokker_planck_operator(grid, plasma, settings; scheme, electric_field=false)
        prescribed = fokker_planck_operator(grid, plasma, settings; scheme)
        layout, system, term = field_system(free)
        @test dependencies(term) == [:hot_tail_distribution, :electric_field]
        @test !is_linear(term)
        @test is_linear(FokkerPlanckTerm(:hot_tail_distribution, prescribed)) == (scheme !== :tcdf)
        f_id, field_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :electric_field)
        u = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u, layout, f_id, ir) .= f0[:, :, ir] .* (0.5 .+ rand(rng, 14, 8))
            radial_view(u, layout, field_id, ir) .= field[ir]
        end
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        expected = apply!(zeros(size(grid)), prescribed, kinetic_array(u, layout, f_id))
        @test kinetic_array(du, layout, f_id) ≈ expected rtol = 1e-12
        @test all(abs.(radial_values(du, layout, field_id)) .< 1e-15)
        @test radial_values(u, layout, field_id) == field
        @test (@allocated rhs!(du, u, system, 0.0)) == 0

        # the stored Jacobian holds the column of the field: one more color, and the
        # products agree with central differences in the distribution and in the field
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        for ir in 1:3
            column = only(unknown_range(layout, field_id, ir))
            rows = unknown_range(layout, f_id, ir)
            @test count(!iszero, J[rows, column]) > 0.9 * length(rows)
            other_rows = setdiff(1:length(layout), rows, column)
            @test all(iszero, J[other_rows, column])
        end
        alone = EquationSystem(UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, grid)]),
                               [Equation(:hot_tail_distribution, TimeDerivative(),
                                         FokkerPlanckTerm(:hot_tail_distribution, prescribed))])
        @test jacobian_colors(system) <= jacobian_colors(alone) + 1
        # central differences of a limiter miss the derivative by up to 1e-4 where a limiter
        # switches within the step (measured against the dense Jacobian of forward-mode AD)
        scales = unknown_scales(layout)
        for _ in 1:3
            v = scales .* randn(rng, length(layout))
            reference = directional_derivative(system, u, v, 0.0)
            @test J * v ≈ reference rtol = (scheme === :tcdf ? 1e-3 : 1e-6)
        end
        # the field alone: the derivative is the operator of the unit field applied to f
        v = zeros(length(layout))
        radial_values(v, layout, field_id) .= [1.0, 0.0, 0.0]
        unit = AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero)
        if scheme !== :tcdf && scheme !== :exponential_fitting
            add_electric_field!(unit, with_electric_field(plasma, [1.0, 0.0, 0.0]))
            by_field = apply!(zeros(size(grid)), unit, kinetic_array(u, layout, f_id))
            # the diffusive part of the f_zero boundary cancels in the difference of two operators
            without = apply!(zeros(size(grid)), AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero),
                             kinetic_array(u, layout, f_id))
            @test kinetic_array(J * v, layout, f_id) ≈ by_field .- without rtol = 1e-10
        end
    end

    @testset "approximate Jacobian modes with the field ($jacobian)" for jacobian in (:frozen_limiter, :upwind)
        free = fokker_planck_operator(grid, plasma, settings; scheme=:tcdf, electric_field=false)
        layout, system, term = field_system(free; jacobian)
        @test jacobian_kind(term) === :approximate
        f_id, field_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :electric_field)
        u = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u, layout, f_id, ir) .= f0[:, :, ir] .* (0.5 .+ rand(rng, 14, 8))
            radial_view(u, layout, field_id, ir) .= field[ir]
        end
        # the residual is the limiter's in every mode
        exact_layout, exact_system, _ = field_system(free)
        du, du_exact = zeros(length(layout)), zeros(length(layout))
        rhs!(du, u, system, 0.0)
        rhs!(du_exact, u, exact_system, 0.0)
        @test du == du_exact
        # the stored block with the correction is the exact Jacobian, which the system of the
        # exact mode stores
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        operator = jacobian_operator(J, system, u, 0.0)
        J_exact = jacobian_prototype(exact_system)
        jacobian!(J_exact, exact_system, u, 0.0)
        scales = unknown_scales(layout)
        for _ in 1:2
            v = scales .* randn(rng, length(layout))
            @test operator * v ≈ J_exact * v rtol = 1e-12
            @test !isapprox(J * v, J_exact * v; rtol=1e-4)
        end
        # the column of the field in the stored block: that of the stored scheme
        field_column = only(unknown_range(layout, field_id, 2))
        @test count(!iszero, J[:, field_column]) > 100
    end

    @testset "a run with the field as an unknown" begin
        times = collect(range(0, 2e-5; length=11))
        prescribed = fokker_planck_operator(grid, plasma, settings)
        free = fokker_planck_operator(grid, plasma, settings; electric_field=false)
        alone = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, grid; scale=1e16)])
        reference = integrate(EquationSystem(alone, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                              FokkerPlanckTerm(:hot_tail_distribution, prescribed))]),
                              vec(f0), times)
        layout, system, _ = field_system(free)
        f_id, field_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :electric_field)
        u0 = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u0, layout, f_id, ir) .= f0[:, :, ir]
            radial_view(u0, layout, field_id, ir) .= field[ir]
        end
        for solver in (NewtonSolver(), NewtonSolver(linear_solver=GMRES()))
            solution = integrate(system, u0, times; solver)
            @test all(solution.statistics.accepted)
            final = kinetic_array(solution.states[:, end], layout, f_id)
            @test final ≈ reshape(reference.states[:, end], size(grid)) rtol = 1e-10
            @test radial_values(solution.states[:, end], layout, field_id) ≈ field rtol = 1e-13
        end
        # a field that changes in time, as a function
        ramp(t) = field .* (1 + t / 1e-5)
        layout, system, _ = field_system(free; values=ramp)
        solution = integrate(system, u0, times)
        @test radial_values(solution.states[:, end], layout, field_id) ≈ ramp(times[end]) rtol = 1e-13
        ramped = kinetic_array(solution.states[:, end], layout, f_id)
        @test !isapprox(ramped, reshape(reference.states[:, end], size(grid)); rtol=1e-6)
        record = describe(system, DescriptionContext(; times))
        @test record["equations"]["hot_tail_distribution"]["terms"]["term_1"]["electric_field"] == "electric_field"
        @test record["equations"]["electric_field"]["terms"]["term_1"]["values"]["values"][:, 1] == field
        operator_record = record["equations"]["hot_tail_distribution"]["terms"]["term_1"]["operator"]
        @test operator_record["coefficient_source"]["electric_field"] === false
    end

    @testset "prescribed plasma with the field as an unknown" begin
        plasma_at(t) = PlasmaState(cold_electron_temperature=[100.0, 300.0, 1000.0] .* exp(-t / 1e-4),
                                   ions=[IonSpecies("D", 1, 1, [1e20, 5e19, 2e19])],
                                   electric_field=fill(1e3, 3), magnetic_field=fill(5.0, 3))
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field), KineticUnknown(:hot_tail_distribution, grid)])
        operator = AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
        term = PrescribedFokkerPlanckTerm(:hot_tail_distribution, operator, plasma_at; settings,
                                          electric_field=:electric_field)
        system = EquationSystem(layout, [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(field), LinearTerm(:electric_field; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(), term)])
        f_id, field_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :electric_field)
        u = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u, layout, f_id, ir) .= f0[:, :, ir]
            radial_view(u, layout, field_id, ir) .= field[ir]
        end
        du = zeros(length(layout))
        t = 3e-5
        rhs!(du, u, system, t)
        # the field of the prescribed plasma (1 kV/m) is left out, that of the state counts
        expected = apply!(zeros(size(grid)), fokker_planck_operator(grid, with_electric_field(plasma_at(t), field),
                                                                   settings), f0)
        @test kinetic_array(du, layout, f_id) ≈ expected rtol = 1e-12
        @test dependencies(term) == [:hot_tail_distribution, :electric_field]
    end

    @testset "the operator must not hold the field as well" begin
        with_field = fokker_planck_operator(grid, plasma, settings)
        @test_throws "electric_field = false" FokkerPlanckTerm(:hot_tail_distribution, with_field;
                                                                electric_field=:electric_field)
        # a plasma without a field, or coefficients set by hand, are accepted
        quiet = fokker_planck_operator(grid, with_electric_field(plasma, zeros(3)), settings)
        @test FokkerPlanckTerm(:hot_tail_distribution, quiet; electric_field=:electric_field) isa FokkerPlanckTerm
        by_hand = AdvectionDiffusion(grid)
        term = FokkerPlanckTerm(:hot_tail_distribution, by_hand; electric_field=:electric_field)
        @test any(!iszero, by_hand.unit_field_advection_p)
        # the field is a radial unknown with one component
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field; components=2),
                                     KineticUnknown(:hot_tail_distribution, grid)])
        @test_throws ArgumentError EquationSystem(layout, [
            Equation(:electric_field, Algebraic(), LinearTerm(:electric_field; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(), term)])
        @test_throws ArgumentError EquationSystem(
            UnknownLayout(rad, [ScalarUnknown(:electric_field), KineticUnknown(:hot_tail_distribution, grid)]),
            [Equation(:electric_field, Algebraic(), LinearTerm(:electric_field; coefficient=-1)),
             Equation(:hot_tail_distribution, TimeDerivative(), term)])
    end

    @testset "prescribed values" begin
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field), ScalarUnknown(:plasma_current),
                                     RadialUnknown(:ion_densities; components=2)])
        hold(name, values) = Equation(name, Algebraic(), PrescribedValueTerm(values), LinearTerm(name; coefficient=-1))
        free(name) = Equation(name, Algebraic(), LinearTerm(name; coefficient=-1))
        system = EquationSystem(layout, [hold(:electric_field, t -> [1.0, 2.0, 3.0] .* t), hold(:plasma_current, 4.0),
                                         free(:ion_densities)])
        du = zeros(length(layout))
        rhs!(du, zeros(length(layout)), system, 2.0)
        @test radial_values(du, layout, unknown_id(layout, :electric_field)) == [2.0, 4.0, 6.0]
        @test du[only(unknown_range(layout, unknown_id(layout, :plasma_current)))] == 4.0
        @test is_linear(PrescribedValueTerm(1.0)) && isempty(dependencies(PrescribedValueTerm(1.0)))
        @test_throws ArgumentError EquationSystem(layout, [hold(:electric_field, [1.0, 2.0]), free(:plasma_current),
                                                           free(:ion_densities)])
        @test_throws ArgumentError EquationSystem(layout, [free(:electric_field), hold(:plasma_current, [1.0, 2.0, 3.0]),
                                                           free(:ion_densities)])
        @test_throws ArgumentError EquationSystem(layout, [free(:electric_field), free(:plasma_current),
                                                           hold(:ion_densities, 1.0)])
        @test describe(PrescribedValueTerm([1.0, 2.0, 3.0]))["values"] == [1.0, 2.0, 3.0]
        values = radial_values(collect(1.0:length(layout)), layout, unknown_id(layout, :electric_field))
        @test length(values) == 3 && values isa LayoutRadialValues
        @test_throws ArgumentError radial_values(zeros(length(layout)), layout, unknown_id(layout, :ion_densities))
    end

    @testset "flux between the grids with the field as an unknown ($mapping)" for mapping in (:overlap, :dream)
        hot = cylindrical_grid(MomentumGrid(biuniform_p_faces(9, 4, 0.0, 0.4, 2.5), uniform_xi_faces(9)), rad)
        runaway = runaway_grid(hot, uniform_p_faces(5, 2.5, 12.0)[2:end])
        function operators(; electric_field)
            (fokker_planck_operator(hot, plasma, settings; p_upper_boundary=:interface, electric_field),
             fokker_planck_operator(runaway, plasma, settings; p_lower_boundary=:interface, electric_field))
        end
        function system_of(; unknown_field)
            hot_operator, runaway_operator = operators(; electric_field=!unknown_field)
            name = unknown_field ? :electric_field : nothing
            hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator; electric_field=name)
            # names of their own: a closure shares the variables of the test set
            interface_here = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, hot_term,
                                                   runaway_operator; pitch_mapping=mapping)
            unknowns = Any[KineticUnknown(:hot_tail_distribution, hot; scale=1e16),
                           KineticUnknown(:runaway_distribution, runaway; scale=1e16)]
            equations = Any[
                Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface_here),
                Equation(:runaway_distribution, TimeDerivative(),
                         FokkerPlanckTerm(:runaway_distribution, runaway_operator; electric_field=name),
                         interface_here)]
            if unknown_field
                pushfirst!(unknowns, RadialUnknown(:electric_field))
                push!(equations, Equation(:electric_field, Algebraic(), PrescribedValueTerm(field),
                                          LinearTerm(:electric_field; coefficient=-1)))
            end
            layout_here = UnknownLayout(rad, unknowns)
            return layout_here, EquationSystem(layout_here, equations), interface_here
        end
        layout, system, interface = system_of(; unknown_field=true)
        reference_layout, reference_system, reference_interface = system_of(; unknown_field=false)
        @test dependencies(interface) == [:hot_tail_distribution, :runaway_distribution, :electric_field]
        @test !is_linear(interface) && is_linear(reference_interface)
        u = zeros(length(layout))
        reference_u = 1e16 .* rand(rng, length(reference_layout))
        copy_unknowns!(u, layout, reference_u, reference_layout)
        field_id = unknown_id(layout, :electric_field)
        radial_values(u, layout, field_id) .= field
        du, reference_du = zeros(length(layout)), zeros(length(reference_layout))
        rhs!(du, u, system, 0.0)
        rhs!(reference_du, reference_u, reference_system, 0.0)
        for name in (:hot_tail_distribution, :runaway_distribution), ir in 1:3
            @test unknown_view(du, layout, name, ir) ≈ unknown_view(reference_du, reference_layout, name, ir) rtol = 1e-11
        end
        for ir in 1:3
            @test interface_flux(interface, u, layout, ir) ≈
                  interface_flux(reference_interface, reference_u, reference_layout, ir) rtol = 1e-11
        end
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        scales = unknown_scales(layout)
        for _ in 1:2
            v = scales .* randn(rng, length(layout))
            @test J * v ≈ directional_derivative(system, u, v, 0.0) rtol = 1e-6
        end
    end

    @testset "runaway threshold of the knock-on cutoff" begin
        plasma_at(t) = with_electric_field(plasma, fill(0.05, 3))
        knockon = KnockOnOperator(grid, grid, momentum.p_faces[4])
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field), KineticUnknown(:hot_tail_distribution, grid)])
        from_plasma = KnockOnCutoffUpdate(knockon, plasma_at)
        u = zeros(length(layout))
        radial_values(u, layout, unknown_id(layout, :electric_field)) .= [30.0, 30.0, 30.0]
        Runes2P.apply_update!(from_plasma, u, layout, 0.0)
        low_field = copy(knockon.p_cutoff)
        other = KnockOnOperator(grid, grid, momentum.p_faces[4])
        from_state = KnockOnCutoffUpdate(other, plasma_at; electric_field=:electric_field)
        Runes2P.apply_update!(from_state, u, layout, 0.0)
        # the field of the state is 600 times that of the plasma: a lower threshold and cutoff
        expected = KnockOnOperator(grid, grid, momentum.p_faces[4])
        Runes2P.apply_update!(KnockOnCutoffUpdate(expected, t -> with_electric_field(plasma, fill(30.0, 3))), u, layout, 0.0)
        @test other.p_cutoff == expected.p_cutoff
        @test other.p_cutoff != low_field
        @test all(other.p_cutoff .<= low_field)
        record = describe(from_state, DescriptionContext(; times=[0.0]))
        @test record["electric_field"] == "electric_field"
        @test !haskey(describe(from_plasma, DescriptionContext(; times=[0.0])), "electric_field")
        fields = settings_fields(KnockOnCutoffUpdate)
        @test Set(union(fields.described, fields.derived)) == Set(fieldnames(KnockOnCutoffUpdate))
    end
end
