# The induction equations in cylindrical geometry: Ampère's law against analytic profiles,
# the initial flux, the plasma current, and the resistive decay of a current profile with a
# uniform conductivity against the analytic rates, with the wall at the edge of the plasma and
# at a distance.

# J₀ and J₁ by their power series (arguments up to a few units)
function bessel_j(order, x)
    term = (x / 2)^order / factorial(order)
    total = term
    for k in 1:40
        term *= -(x / 2)^2 / (k * (k + order))
        total += term
    end
    return total
end

const FIRST_ZERO_OF_J0 = 2.404825557695773

# the lowest root of J₀(x) − x ln(b/a) J₁(x): the decay with a perfectly conducting wall at b
function lowest_root_with_wall(log_ratio)
    g(x) = bessel_j(0, x) - x * log_ratio * bessel_j(1, x)
    low, high = 1e-6, FIRST_ZERO_OF_J0
    for _ in 1:200
        middle = (low + high) / 2
        g(middle) > 0 ? (low = middle) : (high = middle)
    end
    return (low + high) / 2
end

# the entries of a radial unknown with one component over the radii
function rows_of(layout, name)
    [only(unknown_range(layout, unknown_id(layout, name), ir))
     for ir in 1:radial_cell_count(layout)]
end

@testset "induction equations" begin
    mu0 = Constants.VACUUM_PERMEABILITY
    minor_radius = 0.5

    @testset "Ampère's law on analytic profiles" begin
        rad = RadialGrid(uniform_r_faces(12, minor_radius))
        nr = radial_cell_count(rad)
        layout = UnknownLayout(rad, induction_unknowns(rad))
        names = InductionNames()
        term = PoloidalFluxDiffusionTerm(:poloidal_flux, :edge_poloidal_flux; coefficient = 2)
        row = unknown_id(layout, :poloidal_flux)
        Runes2P.validate(term, layout, row)
        @test dependencies(term) == [:poloidal_flux, :edge_poloidal_flux] && is_linear(term)
        # ψ = c r²: (1/r) ∂_r (r ∂_r ψ) = 4c, which the interior rows give exactly on a uniform
        # grid. The gradient at the edge is one-sided: c (a + r_N) in place of 2 c a
        c = 0.37
        u = zeros(length(layout))
        radial_values(u, layout, row) .= c .* rad.r_centers .^ 2
        u[only(unknown_range(layout, unknown_id(layout, :edge_poloidal_flux)))] = c *
                                                                                  minor_radius^2
        du = zeros(length(layout))
        rhs!(du, term, u, layout, row, 0.0)
        values = radial_values(du, layout, row)
        @test values[1:(nr - 1)] ≈ fill(2 * 4c, nr - 1) rtol = 1e-12
        edge_gradient = c * (minor_radius + rad.r_centers[nr])
        expected = (minor_radius * edge_gradient - rad.r_faces[nr] * 2c * rad.r_faces[nr]) /
                   (rad.r_centers[nr] * rad.r_widths[nr])
        @test values[nr] ≈ 2 * expected rtol = 1e-12
        @test all(iszero, du[setdiff(1:length(layout), rows_of(layout, :poloidal_flux))])
        entries = Tuple{Int, Int}[]
        Runes2P.sparsity!(entries, term, layout, row)
        @test length(entries) == 3nr - 2 + 1
        @test (last(rows_of(layout, :poloidal_flux)),
            only(unknown_range(layout, unknown_id(layout, :edge_poloidal_flux)))) in entries

        # the current through the cross-section
        current = CrossSectionIntegralTerm(:total_current_density; coefficient = -1)
        current_row = unknown_id(layout, :plasma_current)
        Runes2P.validate(current, layout, current_row)
        @test jacobian_assembly(current) === :direct
        fill!(u, 0)
        radial_values(u, layout, unknown_id(layout, :total_current_density)) .= 2e6
        fill!(du, 0)
        rhs!(du, current, u, layout, current_row, 0.0)
        @test du[only(unknown_range(layout, current_row))] ≈ -π * minor_radius^2 * 2e6 rtol = 1e-13
        @test sum(cross_section_weights(rad)) ≈ π * minor_radius^2 rtol = 1e-13
        @test_throws ArgumentError Runes2P.validate(current, layout, row)
        @test_throws ArgumentError Runes2P.validate(term, layout, current_row)
        @test_throws ArgumentError Runes2P.validate(
            PoloidalFluxDiffusionTerm(:poloidal_flux, :electric_field), layout, row)

        @test edge_to_wall_inductance(0.5, 0.5) == 0
        @test edge_to_wall_inductance(0.5, 1.0) ≈ mu0 * log(2) rtol = 1e-15
        @test_throws ArgumentError edge_to_wall_inductance(0.5, 0.4)
    end

    @testset "initial flux" begin
        for cells in (8, 16, 32), wall_radius in (minor_radius, 0.8)

            rad = RadialGrid(uniform_r_faces(cells, minor_radius))
            j = fill(1.5e6, cells)
            initial = initial_poloidal_flux(rad, j; wall_radius)
            plasma_current = π * minor_radius^2 * 1.5e6
            @test initial.plasma_current ≈ plasma_current rtol = 1e-13
            @test initial.wall_poloidal_flux == 0
            @test initial.edge_poloidal_flux≈-mu0*log(wall_radius/minor_radius)*plasma_current rtol=1e-13 atol=1e-20
            # a uniform current: ψ = ψ_edge − π μ₀ j (a² − r²)/2, exact in the discrete law up to
            # the gradient at the edge
            exact = initial.edge_poloidal_flux .-
                    π * mu0 * 1.5e6 .* (minor_radius^2 .- rad.r_centers .^ 2) ./ 2
            scale = π * mu0 * 1.5e6 * minor_radius^2 / 2
            @test maximum(abs, initial.poloidal_flux .- exact) / scale < 0.6 / cells^2
            # the discrete law holds to round-off
            layout = UnknownLayout(rad, induction_unknowns(rad))
            system = EquationSystem(layout,
                vcat(induction_equations(rad; wall_radius),
                    [Equation(:total_current_density, Algebraic(),
                        LinearTerm(:electric_field; coefficient = 1e6),
                        LinearTerm(:total_current_density; coefficient = -1))]))
            u = zeros(length(layout))
            radial_values(u, layout, unknown_id(layout, :electric_field)) .= 1.5
            radial_values(u, layout, unknown_id(layout, :total_current_density)) .= j
            radial_values(u, layout, unknown_id(layout, :poloidal_flux)) .= initial.poloidal_flux
            for name in (:plasma_current, :edge_poloidal_flux, :wall_poloidal_flux)
                u[only(unknown_range(layout, unknown_id(layout, name)))] = getproperty(initial, name)
            end
            du = zeros(length(layout))
            rhs!(du, u, system, 0.0)
            scales = residual_scales(system)
            for name in (:poloidal_flux, :total_current_density)
                rows = rows_of(layout, name)
                @test maximum(abs, du[rows] ./ scales[rows]) < 1e-12
            end
            for name in (:plasma_current, :edge_poloidal_flux)
                rows = unknown_range(layout, unknown_id(layout, name))
                @test maximum(abs, du[rows] ./ scales[rows]) < 1e-12
            end
        end
        @test_throws DimensionMismatch initial_poloidal_flux(RadialGrid(uniform_r_faces(4, 0.5)), ones(3))
    end

    @testset "system of the induction equations" begin
        rad = RadialGrid(uniform_r_faces(6, minor_radius))
        unknowns = induction_unknowns(rad; electric_field_scale = 2.0, current_density_scale = 3e6)
        @test [u.name for u in unknowns] ==
              [:electric_field, :poloidal_flux, :total_current_density,
            :plasma_current, :edge_poloidal_flux, :wall_poloidal_flux]
        @test unknowns[1].scale == 2 && unknowns[3].scale == 3e6
        @test unknowns[4].scale ≈ π * minor_radius^2 * 3e6 &&
              unknowns[2].scale ≈ mu0 * unknowns[4].scale
        equations = induction_equations(rad; wall_radius = 1.0, current_density_scale = 3e6)
        @test [e.unknown for e in equations] ==
              [:electric_field, :poloidal_flux, :plasma_current,
            :edge_poloidal_flux, :wall_poloidal_flux]
        @test equations[1].kind == TimeDerivative(:poloidal_flux)
        @test equations[2].kind isa Algebraic && equations[2].scale ≈ 2π * mu0 * 3e6
        layout = UnknownLayout(rad, unknowns)
        ohm = Equation(:total_current_density, Algebraic(),
            LinearTerm(:electric_field; coefficient = fill(1e6, 6)),
            LinearTerm(:total_current_density; coefficient = -1))
        system = EquationSystem(layout, vcat(equations, [ohm]))
        # the mass matrix: the rows of the field hold the derivative of the flux, the row of
        # the flux at the wall its own
        M = mass_matrix(system)
        field, flux = unknown_id(layout, :electric_field),
        unknown_id(layout, :poloidal_flux)
        wall = only(unknown_range(layout, unknown_id(layout, :wall_poloidal_flux)))
        for ir in 1:6
            @test M[only(unknown_range(layout, field, ir)), only(unknown_range(layout, flux, ir))] ==
                  1
        end
        @test M[wall, wall] == 1
        @test sum(M) == 7
        scales = residual_scales(system)
        @test scales[only(unknown_range(layout, flux, 3))] ≈ 2π * mu0 * 3e6
        @test scales[only(unknown_range(layout, field, 3))] ≈
              unknowns[2].scale / system.time_scale
        # other names
        names = InductionNames(electric_field = :E, poloidal_flux = :psi)
        renamed = induction_equations(rad; names)
        @test renamed[1].unknown === :E && renamed[1].kind == TimeDerivative(:psi)
        @test induction_unknowns(rad; names)[2].name === :psi
        record = describe(system, DescriptionContext(; times = [0.0, 1.0]))
        @test record["equations"]["electric_field"]["time_derivative_of"] == "poloidal_flux"
        @test record["equations"]["poloidal_flux"]["terms"]["term_1"]["edge_poloidal_flux"] ==
              "edge_poloidal_flux"
        @test record["equations"]["plasma_current"]["terms"]["term_1"]["unknown"] ==
              "total_current_density"
        @test record["equations"]["total_current_density"]["terms"]["term_1"]["coefficient"] ==
              fill(1e6, 6)
    end

    # The field obeys μ₀ σ ∂E/∂t = (1/r) ∂_r (r ∂_r E). With the flux at the edge held by a
    # perfectly conducting wall at b, the lowest mode is J₀(k r) with J₀(ka) = ka ln(b/a) J₁(ka)
    # and decays at k²/(μ₀ σ): j₀₁²/(μ₀ σ a²) for the wall at the edge. Backward Euler
    # multiplies a mode of the discrete operator by 1/(1 + γ Δt) per step, which gives the
    # rate γ of the discrete operator from two states without the error of the time step.
    @testset "resistive decay, wall at $(wall_radius / minor_radius) minor radii" for wall_radius in (minor_radius, 2 *
                                                                                                                    minor_radius)
        conductivity = 1e6
        k = lowest_root_with_wall(log(wall_radius / minor_radius)) / minor_radius
        wall_radius == minor_radius &&
            @test k * minor_radius ≈ FIRST_ZERO_OF_J0 rtol = 1e-12
        rate = k^2 / (mu0 * conductivity)
        errors = Float64[]
        for cells in (10, 20, 40),
            solver in (NewtonSolver(), NewtonSolver(linear_solver = GMRES()))

            cells == 10 || solver.linear_solver isa SparseLU || continue
            rad = RadialGrid(uniform_r_faces(cells, minor_radius))
            layout = UnknownLayout(rad, induction_unknowns(rad))
            ohm = Equation(:total_current_density, Algebraic(),
                LinearTerm(:electric_field; coefficient = fill(conductivity, cells)),
                LinearTerm(:total_current_density; coefficient = -1))
            system = EquationSystem(layout, vcat(induction_equations(rad; wall_radius), [ohm]))
            field = [bessel_j(0, k * r) for r in rad.r_centers]
            initial = initial_poloidal_flux(rad, conductivity .* field; wall_radius)
            u0 = zeros(length(layout))
            radial_values(u0, layout, unknown_id(layout, :electric_field)) .= field
            radial_values(u0, layout, unknown_id(layout, :total_current_density)) .= conductivity .*
                                                                                     field
            radial_values(u0, layout, unknown_id(layout, :poloidal_flux)) .= initial.poloidal_flux
            for name in (:plasma_current, :edge_poloidal_flux, :wall_poloidal_flux)
                u0[only(unknown_range(layout, unknown_id(layout, name)))] = getproperty(initial, name)
            end
            # four decay times: the higher modes of the start are gone
            steps = 200
            times = collect(range(0, 4 / rate; length = steps + 1))
            dt = times[2] - times[1]
            solution = integrate(system, u0, times; solver)
            @test all(solution.statistics.accepted)
            current_at(n) = solution.states[only(unknown_range(layout, unknown_id(layout, :plasma_current))), n]
            field_at(n) = solution.states[only(unknown_range(layout, unknown_id(layout, :electric_field), 1)), n]
            discrete_rate = (current_at(steps) / current_at(steps + 1) - 1) / dt
            @test (field_at(steps) / field_at(steps + 1) - 1) / dt ≈ discrete_rate rtol = 1e-6
            if solver.linear_solver isa SparseLU
                push!(errors, abs(discrete_rate / rate - 1))
            else
                @test abs(discrete_rate / rate - 1) ≈ errors[1] rtol = 1e-4
            end
            # the state keeps the algebraic equations
            final = solution.states[:, end]
            @test radial_values(final, layout, unknown_id(layout, :total_current_density)) ≈
                  conductivity .*
                  radial_values(final, layout, unknown_id(layout, :electric_field)) rtol = 1e-9
            @test current_at(steps + 1) ≈
                  sum(cross_section_weights(rad) .*
                      radial_values(final, layout, unknown_id(layout, :total_current_density))) rtol = 1e-9
            wall_flux = final[only(unknown_range(layout, unknown_id(layout, :wall_poloidal_flux)))]
            edge_flux = final[only(unknown_range(layout, unknown_id(layout, :edge_poloidal_flux)))]
            @test wall_flux == 0
            @test edge_flux≈-edge_to_wall_inductance(minor_radius, wall_radius)*current_at(steps+1) rtol=1e-9 atol=1e-20
            # the current falls by the decay: exp(-4) with the error of backward Euler
            @test current_at(steps + 1) / current_at(1) ≈ (1 + discrete_rate * dt)^(-steps) rtol = 2e-2
        end
        # second order in the radial grid. Observed errors of the rate at 10, 20 and 40 radii:
        # 3.2e-3, 7.9e-4 and 2.0e-4 with the wall at the edge, 2.1e-3, 5.2e-4 and 1.3e-4 with
        # the wall at two minor radii
        @test errors[3] < 1e-3
        @test 3.5 < errors[1] / errors[2] < 4.5
        @test 3.5 < errors[2] / errors[3] < 4.5
    end

    @testset "loop voltage at the wall" begin
        # a wall that is not perfect, with a prescribed loop voltage: the flux at the wall
        # rises at that voltage, and the field in a plasma without current follows it
        rad = RadialGrid(uniform_r_faces(8, minor_radius))
        layout = UnknownLayout(rad, induction_unknowns(rad))
        loop_voltage = 0.3
        no_current = Equation(:total_current_density, Algebraic(),
            LinearTerm(:total_current_density; coefficient = -1))
        system = EquationSystem(layout,
            vcat(
                induction_equations(rad; wall_radius = 0.7, loop_voltage_at_wall = loop_voltage),
                [no_current]))
        times = collect(range(0, 1e-3; length = 11))
        solution = integrate(system, zeros(length(layout)), times)
        final = solution.states[:, end]
        @test final[only(unknown_range(layout, unknown_id(layout, :wall_poloidal_flux)))] ≈
              loop_voltage * 1e-3 rtol = 1e-12
        @test radial_values(final, layout, unknown_id(layout, :poloidal_flux)) ≈
              fill(loop_voltage * 1e-3, 8) rtol = 1e-10
        @test radial_values(final, layout, unknown_id(layout, :electric_field)) ≈
              fill(loop_voltage / 2π, 8) rtol = 1e-9
        # a loop voltage that rises in time: backward Euler takes the voltage at
        # the end of each step, so the flux at the wall is Σ V(t_k) Δt
        rising(t) = 0.3 + 200 * t
        ramp = EquationSystem(layout,
            vcat(
                induction_equations(rad; wall_radius = 0.7, loop_voltage_at_wall = rising),
                [no_current]))
        solution = integrate(ramp, zeros(length(layout)), times)
        wall = [only(view(solution.states, unknown_range(layout, unknown_id(layout, :wall_poloidal_flux)), k))
                for k in eachindex(times)]
        @test wall ≈ cumsum(vcat(0.0, rising.(times[2:end]) .* diff(times))) rtol = 1e-12
        @test wall[end] > loop_voltage * 1e-3
    end
end
