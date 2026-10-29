# The electron populations around the two distributions: the particle
# source with its magnitude as an unknown, the constraint it enforces, the flux through the
# upper momentum boundary into the electrons above the grid, and the particle budget of a
# run with the field as an unknown, the flux between the grids and the knock-on source.
# The cases need a net flux out through the upper momentum boundary, which the radiation
# losses reverse at 5 T, so the operators leave them out.
using Random: Xoshiro

# (1/V') Σ 𝒱' Δp Δξ g at the radius ir
function cell_sum(grid, values, ir)
    m = momentum_grid(grid, ir)
    V = phase_space_jacobian(grid, ir)
    return sum(V[i, j] * m.p_widths[i] * m.xi_widths[j] * values[i, j]
    for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)) /
           grid.radial_grid.spatial_jacobian[ir]
end

@testset "electron populations" begin
    rng = Xoshiro(525)
    rad = RadialGrid(uniform_r_faces(2, 0.5))

    @testset "particle source" begin
        grids = [MomentumGrid(biuniform_p_faces(30 + 4ir, 20, 0.0, 0.5, 2.5), uniform_xi_faces(6 +
                                                                                               ir))
                 for ir in 1:2]
        grid = cylindrical_grid(grids, rad)
        temperature = [300.0, 1000.0]
        for shape in (:relativistic_maxwellian, :maxwellian, :lowest_cell)
            term = ParticleSourceTerm(:particle_source, grid, temperature; shape)
            for ir in 1:2
                values = source_shape(term, ir)
                @test size(values) ==
                      (momentum_cell_count(grid, ir), pitch_cell_count(grid, ir))
                @test cell_sum(grid, values, ir) ≈ 1 rtol = 1e-14
                @test all(>=(0), values)
                p = grids[ir].p_centers
                theta = temperature[ir] / Constants.ELECTRON_REST_ENERGY
                if shape === :relativistic_maxwellian
                    @test values[5, 2] / values[1, 1] ≈
                          exp(-(sqrt(1 + p[5]^2) - sqrt(1 + p[1]^2)) / theta) rtol = 1e-10
                elseif shape === :maxwellian
                    @test values[5, 2] / values[1, 1] ≈ exp(-(p[5]^2 - p[1]^2) / (2theta)) rtol = 1e-12
                else
                    @test all(==(0), values[2:end, :]) &&
                          all(==(values[1, 1]), values[1, :])
                end
            end
        end
        # the two Maxwellians agree at a low temperature
        cold = ParticleSourceTerm(:particle_source, grid, [5.0, 5.0])
        classical = ParticleSourceTerm(:particle_source, grid, [5.0, 5.0]; shape = :maxwellian)
        @test source_shape(cold, 1) ≈ source_shape(classical, 1) rtol = 1e-3
        @test source_shape(cold, 1) != source_shape(classical, 1)
        # the shape follows the temperature in time
        cooling(t) = temperature .* exp(-t / 1e-3)
        following = ParticleSourceTerm(:particle_source, grid, cooling)
        for t in (0.0, 1e-3, 0.0)
            fixed = ParticleSourceTerm(:particle_source, grid, cooling(t))
            @test source_shape(following, 2, t) == source_shape(fixed, 2)
        end
        plasma_at(t) = PlasmaState(cold_electron_temperature = cooling(t),
            ions = [IonSpecies("D", 1, 1, fill(1e19, 2))],
            electric_field = zeros(2), magnetic_field = fill(5.0, 2))
        @test source_shape(ParticleSourceTerm(:particle_source, grid, plasma_at), 1, 1e-3) ==
              source_shape(following, 1, 1e-3)
        @test_throws ArgumentError ParticleSourceTerm(:particle_source, grid, temperature; shape = :delta)
        @test_throws DimensionMismatch ParticleSourceTerm(:particle_source, grid, [
            1.0, 2.0, 3.0])
        @test_throws DomainError source_shape(
            ParticleSourceTerm(:particle_source, grid, [
                100.0, -1.0]), 1)
    end

    # a distribution that loses particles at a rate per radius, and the source that holds its
    # density at a prescribed value: the magnitude is the loss
    @testset "source held by a constraint" begin
        grid = cylindrical_grid(
            MomentumGrid(biuniform_p_faces(30, 20, 0.0, 0.5, 2.5), uniform_xi_faces(6)),
            rad)
        m = momentum_grid(grid)
        layout = UnknownLayout(rad,
            [RadialUnknown(:particle_source; scale = 1e22),
                KineticUnknown(:hot_tail_distribution, grid; scale = 1e20)])
        target = [1e19, 3e19]
        loss = [100.0, 300.0]
        source = ParticleSourceTerm(:particle_source, grid, [300.0, 1000.0])
        kinetic = Equation(:hot_tail_distribution, TimeDerivative(),
            LinearTerm(:hot_tail_distribution; coefficient = -loss), source)
        held(kind) = Equation(
            :particle_source, kind, DensityMomentTerm(:hot_tail_distribution),
            PrescribedValueTerm(-target); scale = 1e19)
        @test dependencies(source) == [:particle_source] && is_linear(source)
        @test jacobian_assembly(source) === :direct
        # the equation of the magnitude does not hold the magnitude
        @test_throws "cannot determine it" EquationSystem(layout, [
            held(Algebraic()), kinetic])
        system = EquationSystem(layout, [held(Constraint()), kinetic])
        @test all(ir -> iszero(mass_matrix(system)[only(unknown_range(layout, 1, ir)), :]), 1:2)
        f_id = unknown_id(layout, :hot_tail_distribution)
        u0 = zeros(length(layout))
        # the start has the density of the target and another shape than the source
        for ir in 1:2
            f = kinetic_view(u0, layout, f_id, ir)
            f .= source_shape(source, ir) .*
                 (0.8 .+ 0.4 .* rand(rng, momentum_cell_count(m), pitch_cell_count(m)))
            f .*= target[ir] / cell_sum(grid, f, ir)
        end
        # the stored Jacobian of the linear system
        J = jacobian_prototype(system)
        jacobian!(J, system, u0, 0.0)
        v = unknown_scales(layout) .* randn(rng, length(layout))
        du, du0 = zeros(length(layout)), zeros(length(layout))
        rhs!(du, v, system, 0.0)
        rhs!(du0, zeros(length(layout)), system, 0.0)
        @test J * v ≈ du .- du0 rtol = 1e-12
        @test jacobian_colors(system) <= 2
        times = collect(range(0, 5e-3; length = 11))
        for solver in (NewtonSolver(), NewtonSolver(linear_solver = GMRES()))
            solution = integrate(system, u0, times; solver)
            @test all(solution.statistics.accepted)
            for k in 2:length(times), ir in 1:2

                state = view(solution.states, :, k)
                f = kinetic_view(state, layout, f_id, ir)
                @test cell_sum(grid, f, ir) ≈ target[ir] rtol = 1e-10
                @test only(unknown_view(state, layout, :particle_source, ir)) ≈
                      loss[ir] * target[ir] rtol = 1e-8
            end
            # the distribution approaches the shape of the source
            final = kinetic_view(view(solution.states, :, length(times)), layout, f_id, 2)
            @test final ≈ target[2] .* source_shape(source, 2) rtol = 0.2
        end
        # a constraint names the scale of what it balances
        @test_throws "needs the scale" EquationSystem(layout,
            [
                Equation(:particle_source, Constraint(),
                    DensityMomentTerm(:hot_tail_distribution),
                    PrescribedValueTerm(-target)), kinetic])
        # a constraint on an unknown that no other equation holds
        @test_throws "nothing determines it" EquationSystem(layout,
            [
                held(Constraint()), Equation(:hot_tail_distribution, TimeDerivative(),
                    LinearTerm(:hot_tail_distribution; coefficient = -loss))])
        record = describe(system, DescriptionContext(; times))
        @test record["equations"]["particle_source"]["kind"] == "constraint"
        source_record = record["equations"]["hot_tail_distribution"]["terms"]["term_2"]
        @test source_record["shape"] == "relativistic_maxwellian" &&
              source_record["source"] == "particle_source"
        @test source_record["temperature"] == [300.0, 1000.0]
        fields = settings_fields(ParticleSourceTerm)
        @test Set(union(fields.described, fields.derived)) ==
              Set(fieldnames(ParticleSourceTerm))
    end

    temperature, bulk_density = 1000.0, 1e19
    plasma = PlasmaState(cold_electron_temperature = fill(temperature, 2),
        cold_electron_density = fill(bulk_density, 2),
        ions = [IonSpecies("D", 1, 1, fill(bulk_density, 2))], electric_field = zeros(2),
        magnetic_field = fill(5.0, 2))
    settings = CollisionSettings(mode = :full, screening = :complete,
        coulomb_logarithm = :relativistic, bremsstrahlung = :neglect)
    hot = cylindrical_grid(MomentumGrid(biuniform_p_faces(56, 40, 0.0, 0.5, 2.5), uniform_xi_faces(8)), rad)
    runaway = runaway_grid(hot, uniform_p_faces(12, 2.5, 26.5)[2:end], uniform_xi_faces(8))
    hot_m, runaway_m = momentum_grid(hot), momentum_grid(runaway)
    field = [0.3, 0.2]

    @testset "flux through the upper momentum boundary ($scheme, $boundary)" for (scheme, boundary) in ((
        :exponential_fitting, :f_zero), (:central, :outflow), (:upwind, :constant_flux),
        (:central, :extrapolated_flux), (
        :exponential_fitting, :extrapolated_flux_consistent),
        (:tcdf, :extrapolated_flux_consistent))
        operator = fokker_planck_operator(
            runaway, plasma, settings; scheme, p_upper_boundary = boundary,
            field_in_coefficients = false, synchrotron = false)
        term = FokkerPlanckTerm(:runaway_distribution, operator; electric_field = :electric_field)
        layout = UnknownLayout(rad,
            [RadialUnknown(:electric_field),
                RadialUnknown(:runaway_density_above_grid; scale = 1e10),
                RadialUnknown(:runaway_current_above_grid; scale = 1e10 * 4.8e-11),
                KineticUnknown(:runaway_distribution, runaway; scale = 1e10)])
        above = MomentumBoundaryFluxTerm(:runaway_distribution, term)
        current = MomentumBoundaryFluxTerm(:runaway_distribution, term; weight = :current, coefficient = 2)
        system = EquationSystem(layout,
            [
                Equation(:electric_field, Algebraic(), PrescribedValueTerm(field),
                    LinearTerm(:electric_field; coefficient = -1)),
                Equation(:runaway_density_above_grid, TimeDerivative(), above),
                Equation(:runaway_current_above_grid, TimeDerivative(), current),
                Equation(:runaway_distribution, TimeDerivative(), term)])
        @test dependencies(above) == [:runaway_distribution, :electric_field]
        @test !is_linear(above) && jacobian_assembly(above) === :direct
        @test is_linear(MomentumBoundaryFluxTerm(:runaway_distribution, operator)) ==
              (scheme !== :tcdf)
        f_id = unknown_id(layout, :runaway_distribution)
        u = zeros(length(layout))
        f = [1e10 * exp(-(p - 22)^2 / 30) * exp(2 * (xi - 1)) * (0.7 + 0.6 * rand(rng))
             for p in runaway_m.p_centers, xi in runaway_m.xi_centers, _ in 1:2]
        for ir in 1:2
            kinetic_view(u, layout, f_id, ir) .= f[:, :, ir]
            radial_view(u, layout, unknown_id(layout, :electric_field), ir) .= field[ir]
        end
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        np = momentum_cell_count(runaway_m)
        charge = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT
        for ir in 1:2
            flux = momentum_flux(operator, f, np + 1, ir; electric_field = field)
            @test flux > 0
            @test only(unknown_view(du, layout, :runaway_density_above_grid, ir)) ≈ flux rtol = 1e-13
            weighted = sum(runaway_m.xi_widths[j] * runaway_m.xi_centers[j] *
                           Runes2P._p_flux(operator, f, np + 1, j, ir, nothing, field)
            for j in 1:8) / rad.spatial_jacobian[ir]
            @test only(unknown_view(du, layout, :runaway_current_above_grid, ir)) ≈
                  2 * charge * weighted rtol = 1e-13
            # what leaves the grid is what the density above it gains
            lower = momentum_flux(operator, f, 1, ir; electric_field = field)
            @test lower == 0
            @test cell_sum(runaway, kinetic_view(du, layout, f_id, ir), ir) ≈ -flux rtol = 1e-10
        end
        @test (@allocated rhs!(du, u, system, 0.0)) == 0
        # the rows assembled directly against central differences
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        scales = unknown_scales(layout)
        rows = [only(unknown_range(layout, unknown_id(layout, name), ir))
                for name in (:runaway_density_above_grid, :runaway_current_above_grid),
        ir in 1:2]
        for _ in 1:3
            v = scales .* randn(rng, length(layout))
            h = 1e-6 * maximum(abs, u) / maximum(abs, v)
            up, down = zeros(length(layout)), zeros(length(layout))
            rhs!(up, u .+ h .* v, system, 0.0)
            rhs!(down, u .- h .* v, system, 0.0)
            reference = (up .- down) ./ (2h)
            @test (J * v)[rows] ≈ reference[rows] rtol = (scheme === :tcdf ? 1e-3 : 1e-7)
        end
        # the field alone
        v = zeros(length(layout))
        unknown_values(v, layout, unknown_id(layout, :electric_field)) .= [1.0, 1.0]
        up, down = zeros(length(layout)), zeros(length(layout))
        rhs!(up, u .+ 1e-5 .* v, system, 0.0)
        rhs!(down, u .- 1e-5 .* v, system, 0.0)
        @test (J * v)[rows] ≈ ((up .- down) ./ 2e-5)[rows] rtol = (scheme === :tcdf ? 1e-3 :
                                                                   1e-7)
        @test all(!iszero, (J * v)[rows])
    end

    @testset "checks of the boundary flux term" begin
        operator = fokker_planck_operator(
            hot, plasma, settings; p_upper_boundary = :interface, synchrotron = false)
        @test_throws ArgumentError MomentumBoundaryFluxTerm(:hot_tail_distribution, operator)
        operator = fokker_planck_operator(hot, plasma, settings; synchrotron = false)
        @test_throws ArgumentError MomentumBoundaryFluxTerm(:hot_tail_distribution, operator; weight = :other)
        @test_throws ArgumentError MomentumBoundaryFluxTerm(:hot_tail_distribution, :operator)
        term = MomentumBoundaryFluxTerm(:hot_tail_distribution, operator)
        layout = UnknownLayout(rad,
            [RadialUnknown(:runaway_density), ScalarUnknown(:plasma_current),
                KineticUnknown(:hot_tail_distribution, hot),
                KineticUnknown(:runaway_distribution, runaway)])
        Runes2P.validate(term, layout, 1)
        @test_throws ArgumentError Runes2P.validate(term, layout, 2)
        @test_throws DimensionMismatch Runes2P.validate(
            MomentumBoundaryFluxTerm(:runaway_distribution, operator), layout, 1)
        record = describe(term, DescriptionContext())
        @test record["kinetic"] == "hot_tail_distribution" && record["weight"] == "density"
        fields = settings_fields(MomentumBoundaryFluxTerm)
        @test Set(union(fields.described, fields.derived)) ==
              Set(fieldnames(MomentumBoundaryFluxTerm))
    end

    # Both grids with the field as an unknown, the flux between the grids, the knock-on
    # source, the electrons above the runaway grid and the particle source. The budget of a
    # step of backward Euler with all terms at the new state:
    #   Δ[n(f_hot) + n(f_re) + n_above]/Δt = S + K + Φ₀,
    # with the magnitude S of the source, the knock-on production K on both grids and the
    # flux Φ₀ through p = 0 (negative: a loss).
    @testset "particle budget" begin
        coulomb_logarithm = relativistic_coulomb_logarithm(bulk_density, temperature)
        critical_electric_field = critical_field(bulk_density, coulomb_logarithm)
        critical = critical_momentum(maximum(field), critical_electric_field)
        window = cutoff_window(thermal_momentum(temperature), critical)
        cutoff = choose_cutoff(hot_m.p_faces, window.lower, window.upper).p_cutoff
        hot_operator = fokker_planck_operator(
            hot, plasma, settings; p_upper_boundary = :interface,
            field_in_coefficients = false, synchrotron = false)
        runaway_operator = fokker_planck_operator(
            runaway, plasma, settings; p_lower_boundary = :interface,
            field_in_coefficients = false, synchrotron = false)
        hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator; electric_field = :electric_field)
        runaway_term = FokkerPlanckTerm(
            :runaway_distribution, runaway_operator; electric_field = :electric_field)
        interface = MomentumInterfaceTerm(
            :hot_tail_distribution, :runaway_distribution, hot_term, runaway_operator)
        hot_from_hot = KnockOnOperator(hot, hot, cutoff)
        hot_from_runaway = KnockOnOperator(hot, runaway, cutoff)
        runaway_from_runaway = KnockOnOperator(runaway, runaway, cutoff)
        # a knock-on takes at most half of the kinetic energy of its primary: primaries on the
        # grids produce nothing above the runaway grid
        @test all(ir -> all(==(0), energy_kernel(runaway_from_runaway, ir).rates_above_grid), 1:2)
        @test all(ir -> all(==(0), energy_kernel(hot_from_hot, ir).rates_above_grid), 1:2)
        knockon(primary, operator) = KnockOnTerm(
            primary, operator; target_density = fill(bulk_density, 2),
            treatment = :implicit)
        region = MomentumRegion(ThermalMomentum(fill(temperature, 2)))
        source = ParticleSourceTerm(:particle_source, hot, fill(temperature, 2))
        free_electron_density = fill(bulk_density, 2)
        minus(name) = LinearTerm(name; coefficient = -1)
        layout = UnknownLayout(rad,
            [
                RadialUnknown(:electric_field), RadialUnknown(:hot_electron_density; scale = 1e10),
                RadialUnknown(:runaway_density; scale = 1e10), RadialUnknown(
                    :runaway_density_above_grid; scale = 1e9),
                RadialUnknown(:cold_electron_density; scale = bulk_density), RadialUnknown(
                    :particle_source; scale = 1e20),
                KineticUnknown(:hot_tail_distribution, hot; scale = 1e19),
                KineticUnknown(:runaway_distribution, runaway; scale = 1e10)])
        system = EquationSystem(layout,
            [
                Equation(:electric_field, Algebraic(), PrescribedValueTerm(field), minus(:electric_field)),
                Equation(:hot_electron_density, Algebraic(),
                    DensityMomentTerm(:hot_tail_distribution, hot; region), minus(:hot_electron_density)),
                Equation(
                    :runaway_density, Algebraic(), DensityMomentTerm(:runaway_distribution),
                    LinearTerm(:runaway_density_above_grid), minus(:runaway_density)),
                Equation(:runaway_density_above_grid, TimeDerivative(),
                    MomentumBoundaryFluxTerm(:runaway_distribution, runaway_term)),
                Equation(:cold_electron_density, Algebraic(),
                    PrescribedValueTerm(free_electron_density),
                    minus(:hot_electron_density), minus(:runaway_density), minus(:cold_electron_density)),
                Equation(:particle_source, Constraint(),
                    DensityMomentTerm(:hot_tail_distribution),
                    minus(:cold_electron_density), minus(:hot_electron_density); scale = bulk_density),
                Equation(
                    :hot_tail_distribution, TimeDerivative(), hot_term, interface, source,
                    knockon(:hot_tail_distribution, hot_from_hot), knockon(
                        :runaway_distribution, hot_from_runaway)),
                Equation(:runaway_distribution, TimeDerivative(), runaway_term, interface,
                    knockon(:runaway_distribution, runaway_from_runaway))])
        hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution),
        unknown_id(layout, :runaway_distribution)
        value(state, name, ir) = only(unknown_view(state, layout, name, ir))
        # start: the bulk of the source, a seed at the interface and one near the upper boundary
        u0 = zeros(length(layout))
        seed(m, centre) = [1e10 * exp(-(p - centre)^2 / 0.5) * exp(3 * (xi - 1))
                           for p in m.p_centers, xi in m.xi_centers]
        for ir in 1:2
            f_hot = kinetic_view(u0, layout, hot_id, ir)
            f_hot .= seed(hot_m, 2.0)
            f_runaway = kinetic_view(u0, layout, runaway_id, ir)
            f_runaway .= seed(runaway_m, 3.0) .+ seed(runaway_m, 24.0)
            n_runaway = cell_sum(runaway, f_runaway, ir)
            f_hot .+= (bulk_density - n_runaway - cell_sum(hot, seed(hot_m, 2.0), ir)) .*
                      source_shape(source, ir)
            radial_view(u0, layout, unknown_id(layout, :electric_field), ir) .= field[ir]
            radial_view(u0, layout, unknown_id(layout, :runaway_density), ir) .= n_runaway
        end
        for ir in 1:2
            n_hot = density(hot, kinetic_array(u0, layout, hot_id); region)[ir]
            radial_view(u0, layout, unknown_id(layout, :hot_electron_density), ir) .= n_hot
            radial_view(u0, layout, unknown_id(layout, :cold_electron_density), ir) .= bulk_density -
                                                                                       n_hot -
                                                                                       value(
                u0, :runaway_density, ir)
        end
        times = collect(range(0, 0.04; length = 21))
        solution = integrate(system, u0, times; solver = NewtonSolver(linear_solver = GMRES()))
        @test all(solution.statistics.accepted)
        content(state, ir) = cell_sum(hot, kinetic_view(state, layout, hot_id, ir), ir) +
                             cell_sum(runaway, kinetic_view(state, layout, runaway_id, ir), ir) +
                             value(state, :runaway_density_above_grid, ir)
        produced = [zeros(size(hot)[1:2]), zeros(size(runaway)[1:2])]
        worst = 0.0
        for k in 2:length(times), ir in 1:2

            state, before = view(solution.states, :, k), view(solution.states, :, k - 1)
            f_hot, f_runaway = kinetic_view(state, layout, hot_id, ir),
            kinetic_view(state, layout, runaway_id, ir)
            n_hot, n_runaway = value(state, :hot_electron_density, ir),
            value(state, :runaway_density, ir)
            n_cold = value(state, :cold_electron_density, ir)
            # the populations
            @test n_hot ≈ density(hot, kinetic_array(state, layout, hot_id); region)[ir] rtol = 1e-6
            @test n_runaway ≈
                  cell_sum(runaway, f_runaway, ir) +
                  value(state, :runaway_density_above_grid, ir) rtol = 1e-6
            @test n_cold + n_hot + n_runaway ≈ bulk_density rtol = 1e-9
            @test cell_sum(hot, f_hot, ir) ≈ n_cold + n_hot rtol = 1e-9
            @test content(state, ir) ≈ bulk_density rtol = 1e-9
            # the budget of the step
            fill!.(produced, 0)
            apply_at_radius!(produced[1], hot_from_hot, f_hot, bulk_density, ir)
            apply_at_radius!(produced[1], hot_from_runaway, f_runaway, bulk_density, ir)
            apply_at_radius!(produced[2], runaway_from_runaway, f_runaway, bulk_density, ir)
            knockons = cell_sum(hot, produced[1], ir) + cell_sum(runaway, produced[2], ir)
            through_zero = momentum_flux(
                hot_operator, kinetic_array(state, layout, hot_id), 1, ir;
                electric_field = unknown_values(state, layout, unknown_id(layout, :electric_field)))
            magnitude = value(state, :particle_source, ir)
            # the full collision mode has no flux through p = 0
            @test knockons > 0 && through_zero <= 0
            dt = times[k] - times[k - 1]
            gained = dt * (magnitude + knockons + through_zero)
            size_of_terms = dt * (abs(magnitude) + knockons + abs(through_zero))
            worst = max(worst, abs(content(state, ir) - content(before, ir) - gained) /
                               size_of_terms)
        end
        # observed: the largest error of a step is 1.5e-10 of its terms
        @test worst < 1e-8
        # the seeds reach the runaway grid and the region above it
        final = view(solution.states, :, length(times))
        for ir in 1:2
            # observed: 0.48 of the seed at p = 24 has left the grid, the electrons of the seed
            # with a negative pitch stay
            @test value(final, :runaway_density_above_grid, ir) >
                  0.3 * cell_sum(runaway, seed(runaway_m, 24.0), ir)
            @test value(final, :runaway_density, ir) > 1.5 * value(u0, :runaway_density, ir)
            @test interface_flux(interface, final, layout, ir) > 0
            @test value(final, :hot_electron_density, ir) > 0
        end
        @test jacobian_colors(system) <= 19
    end

    # The Dreicer rate with the self-consistent source: the
    # case of test_runaway_physics.jl, 1 keV and 1e20 m⁻³ with complete screening and the
    # relativistic Coulomb logarithm, centred interpolation, 110 × 14 cells to p = 2. There a
    # source holds the density below p = 0.12 at its initial value, which is what the
    # published rates assume. Here the magnitude of the source holds the density of the grid
    # at n_cold + n_hot, with n_cold = n_free − n_hot − n_re and n_re fed by the flux through
    # p_max, as in DREAM: the electrons that ran away are missing from the bulk.
    @testset "Dreicer rate with the self-consistent source" begin
        temperature, free_density = 1000.0, 1e20
        single = RadialGrid(uniform_r_faces(1, 0.5))
        grid = cylindrical_grid(
            MomentumGrid(biuniform_p_faces(110, 80, 0.0, 0.4, 2.0), uniform_xi_faces(14)),
            single)
        m = momentum_grid(grid)
        coulomb_logarithm = relativistic_coulomb_logarithm(free_density, temperature)
        critical_electric_field = critical_field(free_density, coulomb_logarithm)
        collision_time = thermal_collision_time(free_density, temperature,
            thermal_coulomb_logarithm(free_density, temperature))
        settings = CollisionSettings(mode = :full, screening = :complete,
            coulomb_logarithm = :relativistic, bremsstrahlung = :neglect)
        theta = temperature / Constants.ELECTRON_REST_ENERGY
        maxwellian = [free_density * (2pi * theta)^(-1.5) * exp(-p^2 / (2theta))
                      for p in m.p_centers, _ in m.xi_centers]
        region = MomentumRegion(ThermalMomentum([temperature]))
        minus(name) = LinearTerm(name; coefficient = -1)

        function rate_of(over_critical, collision_times; self_consistent)
            field = over_critical * critical_electric_field
            plasma = PlasmaState(cold_electron_temperature = [temperature],
                cold_electron_density = [free_density],
                ions = [IonSpecies("D", 1, 1, [free_density])], electric_field = [field],
                magnetic_field = [5.0])
            operator = fokker_planck_operator(
                grid, plasma, settings; scheme = :central, synchrotron = false)
            times = collect(range(0, collision_times * collision_time; length = 61))
            term = FokkerPlanckTerm(:hot_tail_distribution, operator)
            if self_consistent
                layout = UnknownLayout(single,
                    [
                        RadialUnknown(:hot_electron_density; scale = 1e15), RadialUnknown(
                            :runaway_density; scale = 1e15),
                        RadialUnknown(:cold_electron_density; scale = free_density),
                        RadialUnknown(:particle_source; scale = free_density /
                                                                collision_time),
                        KineticUnknown(:hot_tail_distribution, grid; scale = free_density)])
                system = EquationSystem(layout,
                    [
                        Equation(:hot_electron_density, Algebraic(),
                            DensityMomentTerm(:hot_tail_distribution, grid; region), minus(:hot_electron_density)),
                        Equation(:runaway_density, TimeDerivative(),
                            MomentumBoundaryFluxTerm(:hot_tail_distribution, operator)),
                        Equation(:cold_electron_density, Algebraic(),
                            PrescribedValueTerm(free_density),
                            minus(:hot_electron_density), minus(:runaway_density), minus(:cold_electron_density)),
                        Equation(:particle_source, Constraint(),
                            DensityMomentTerm(:hot_tail_distribution),
                            minus(:cold_electron_density), minus(:hot_electron_density); scale = free_density),
                        Equation(:hot_tail_distribution, TimeDerivative(), term,
                            ParticleSourceTerm(:particle_source, grid, [temperature]))])
            else
                layout = UnknownLayout(single,
                    [KineticUnknown(:hot_tail_distribution, grid; scale = free_density)])
                bulk = sum(phase_space_jacobian(grid, 1)[i, j] * m.p_widths[i] *
                           m.xi_widths[j] * maxwellian[i, j]
                for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)
                if m.p_centers[i] <= 0.12) /
                       single.spatial_jacobian[1]
                system = EquationSystem(layout,
                    [
                        Equation(:hot_tail_distribution, TimeDerivative(), term,
                        MaxwellianSourceTerm(:hot_tail_distribution, grid,
                            temperature; target_density = bulk,
                            rate = 1 / (times[2] - times[1]), bulk_momentum = 0.12))])
            end
            id = unknown_id(layout, :hot_tail_distribution)
            u0 = zeros(length(layout))
            kinetic_view(u0, layout, id, 1) .= maxwellian
            if self_consistent
                n_hot = density(grid, reshape(maxwellian, size(grid)); region)[1]
                radial_view(u0, layout, unknown_id(layout, :hot_electron_density), 1) .= n_hot
                radial_view(u0, layout, unknown_id(layout, :cold_electron_density), 1) .= free_density -
                                                                                          n_hot
            end
            solution = integrate(system, u0, times)
            @test all(solution.statistics.accepted)
            final = view(solution.states, :, length(times))
            f = reshape(collect(kinetic_view(final, layout, id, 1)), size(grid))
            critical = critical_momentum(field, critical_electric_field)
            fluxes = [momentum_flux(operator, f, findfirst(>=(x * critical), m.p_faces))
                      for x in (1.5, 2.5, 4.0)]
            populations = self_consistent ?
                          (
                runaway = only(unknown_view(final, layout, :runaway_density, 1)),
                cold = only(unknown_view(final, layout, :cold_electron_density, 1)),
                hot = only(unknown_view(final, layout, :hot_electron_density, 1)),
                on_grid = density(grid, f)[1]) : nothing
            return (;
                rate = fluxes[1], spread = maximum(fluxes) / minimum(fluxes), populations)
        end

        # E/E_c = 10 (E/E_D = 0.024): the electrons that run away in 1600 collision times
        # are 1e-7 of the bulk, and the two sources give the same rate
        held = rate_of(10.0, 1600; self_consistent = false)
        closed = rate_of(10.0, 1600; self_consistent = true)
        @test held.spread < 1.05 && closed.spread < 1.05
        @test closed.rate ≈ held.rate rtol = 0.01
        @test closed.populations.runaway < 1e-6 * free_density
        # E/E_c = 25 (E/E_D = 0.059, the low-temperature, high-E/E_D regime of the CODE
        # benchmark of DREAM's runaway rate): the bulk loses electrons to the runaway density
        # during the 600 collision times the tail needs to settle, and the rate falls with the
        # bulk
        held = rate_of(25.0, 600; self_consistent = false)
        closed = rate_of(25.0, 600; self_consistent = true)
        lost = closed.populations.runaway / free_density
        # observed: 0.78 % of the electrons have run away, and the rate is 0.962 of the rate
        # with the held bulk
        @test 0.005 < lost < 0.02
        @test 0.93 < closed.rate / held.rate < 0.99
        for run in (closed,)
            n = run.populations
            @test n.cold + n.hot + n.runaway ≈ free_density rtol = 1e-10
            @test n.on_grid ≈ n.cold + n.hot rtol = 1e-9
            @test n.on_grid + n.runaway ≈ free_density rtol = 1e-9
        end
    end
end
