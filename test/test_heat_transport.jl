# Heat transport of the cold electrons: the decay of the lowest Bessel mode
# of a uniform diffusivity in a cylinder, the energy balance with the edge flux, the
# Rechester-Rosenbluth heat flux against the energy moment of the kinetic Rechester-
# Rosenbluth operator on Maxwellians, the same against the energy term of the kinetic
# transport on a temperature profile and down a density gradient, the Jacobian, and the
# switch times of a level that changes in time.

# J₀ by its series, enough for 0 ≤ x ≤ 3
_bessel_j0(x) = sum((-1)^k * (x / 2)^(2k) / factorial(big(k))^2 for k in 0:30) |> Float64
const BESSEL_J0_FIRST_ZERO = 2.404825557695773

@testset "heat transport" begin
    e = Constants.ELEMENTARY_CHARGE
    function heat_layout(nr; a = 0.5)
        UnknownLayout(RadialGrid(uniform_r_faces(nr, a)),
            [RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19)])
    end
    function heat_rhs(term, layout, T, n; t = 0.0)
        u = zeros(length(layout))
        unknown_values(u, layout, 1) .= T
        unknown_values(u, layout, 2) .= n
        Runes2P.validate(term, layout, 1)
        out = zeros(length(u))
        Runes2P.rhs!(out, term, u, layout, 1, t)
        return unknown_values(out, layout, 1), u
    end

    @testset "the lowest Bessel mode" begin
        # dT/dt = D ∇²T at a uniform density: T = J₀(j₀₁ r/a) decays at D j₀₁²/a²
        D, a, n = 3.0, 0.5, 1e19
        exact = D * BESSEL_J0_FIRST_ZERO^2 / a^2
        function rate(nr, edge)
            layout = heat_layout(nr; a)
            rg = layout.radial_grid
            T = [_bessel_j0(BESSEL_J0_FIRST_ZERO * r / a) for r in rg.r_centers]
            rhs, _ = heat_rhs(HeatTransportTerm(; diffusion = D, edge), layout, T, fill(n, nr))
            volume = rg.spatial_jacobian .* rg.r_widths
            # the Rayleigh quotient of the discrete operator
            return -sum(volume .* T .* rhs) / sum(volume .* 1.5 .* e .* n .* T .^ 2)
        end
        errors = [abs(rate(nr, :absorbing) / exact - 1) for nr in (20, 40)]
        @test errors[2] < 2e-3
        @test errors[1] / errors[2] > 3     # second order
        # DREAM's edge sits half a cell outside: first order
        @test abs(rate(40, :absorbing_dream) / exact - 1) > errors[2]
    end

    @testset "energy balance" begin
        layout = heat_layout(6)
        rg = layout.radial_grid
        volume = rg.spatial_jacobian .* rg.r_widths
        T, n = [800.0, 700.0, 520.0, 300.0, 120.0, 40.0],
        [3e19, 2.8e19, 2.5e19, 2e19, 1.5e19, 1e19]
        for model in ((; diffusion = 2.0), (; delta_b_over_b = 1e-3, major_radius = 1.7)),
            edge in (:zero_flux, :absorbing, :absorbing_dream, :constant_gradient_dream)

            term = HeatTransportTerm(; model..., edge)
            rhs, u = heat_rhs(term, layout, T, n)
            lost = heat_flux_through_edge(term, u, layout, 0.0)
            @test sum(volume .* rhs)≈-lost rtol=1e-12 atol=1e-12*sum(volume .* abs.(rhs))
            edge === :zero_flux ? (@test lost == 0) : (@test lost > 0)
        end
    end

    @testset "Rechester-Rosenbluth against the kinetic operator" begin
        # The kinetic diffusion π q R₀ |v_∥| (δB/B)² on Maxwellians at T(r) with a uniform
        # density carries the energy flux −3√(2π) e c √Θ q R₀ (δB/B)² n ∂_r T (1 − 5Θ/8): the
        # pitch average of |ξ| is 1/2 and the energy moment ∫ x²(x − 3/2) e⁻ˣ dx is 3. The two
        # discretizations agree to the square of the relative step of T between the cells.
        n, delta, R0 = 1e19, 2e-3, 1.7
        for T0 in (1e3, 2e4)
            T = T0 .* [1.0, 1.01, 1.02]
            theta = maximum(T) / Constants.ELECTRON_REST_ENERGY
            rad = RadialGrid(uniform_r_faces(3, 0.5))
            grid = cylindrical_grid(
                MomentumGrid(uniform_p_faces(400, 0.0, 12 * sqrt(2theta)), uniform_xi_faces(8)),
                rad)
            op = AdvectionDiffusion(grid)
            add_rechester_rosenbluth_diffusion!(op, delta; major_radius = R0)
            np, nxi, nr = size(grid)
            f = zeros(np, nxi, nr)
            for ir in 1:nr, (i, p) in enumerate(momentum_grid(grid).p_centers)

                f[i, :, ir] .= exp(-p^2 / (1 + sqrt(1 + p^2)) /
                                   (T[ir] / Constants.ELECTRON_REST_ENERGY))
            end
            f .*= reshape(n ./ density(grid, f), 1, 1, :)
            df = zeros(size(f))
            Runes2P.apply!(df, op, f)
            m = momentum_grid(grid)
            energies = [Constants.ELECTRON_REST_ENERGY * e * p^2 / (1 + sqrt(1 + p^2))
                        for p in m.p_centers]
            kinetic = [sum(energies[i] * phase_space_jacobian(grid, ir)[i, j] *
                           m.p_widths[i] * m.xi_widths[j] * df[i, j, ir]
                       for i in 1:np, j in 1:nxi) / rad.spatial_jacobian[ir] for ir in 1:nr]
            fluid, _ = heat_rhs(
                HeatTransportTerm(; delta_b_over_b = delta, major_radius = R0),
                heat_layout(3), T, fill(n, 3))
            # at 20 keV the expansion 1 − 5Θ/8 misses terms of order Θ² ≈ 2e-3
            tolerance = T0 < 1e4 ? 1e-3 : 5e-3
            @test fluid[1] ≈ kinetic[1] rtol = tolerance
            @test fluid[3] ≈ kinetic[3] rtol = tolerance
        end
    end

    @testset "the energy term of kinetic transport against the Rechester-Rosenbluth heat transport" begin
        # The reason why a row of the total energy refuses the pair: on a Maxwellian bulk of
        # uniform density at a temperature profile, the energy that KineticEnergyChangeTerm of
        # the kinetic Rechester-Rosenbluth transport moves is the heat transport of
        # HeatTransportTerm at the same level, so a row with both moves it twice. Measured:
        # the ratio of the two is 1.0000 to 0.9944 at the six inner radii and 0.868 at the
        # seventh, near the edge
        n, delta, R0, a, nr = 1e19, 2e-3, 1.7, 0.5, 8
        rad = RadialGrid(uniform_r_faces(nr, a))
        T = [20 + 380 * (1 - (r / a)^2) for r in rad.r_centers]
        theta = maximum(T) / Constants.ELECTRON_REST_ENERGY
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(400, 0.0, 12 * sqrt(2theta)), uniform_xi_faces(8)), rad)
        op = AdvectionDiffusion(grid; cross_diffusion = false)
        add_rechester_rosenbluth_diffusion!(op, delta; major_radius = R0)
        np, nxi, _ = size(grid)
        layout = UnknownLayout(rad,
            [KineticUnknown(:f, grid; scale = n),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = n),
                RadialUnknown(:total_electron_energy; scale = 1e3)])
        u = zeros(length(layout))
        f = kinetic_array(u, layout, 1)
        for ir in 1:nr, (i, p) in enumerate(momentum_grid(grid).p_centers)

            f[i, :, ir] .= exp(-p^2 / (1 + sqrt(1 + p^2)) /
                               (T[ir] / Constants.ELECTRON_REST_ENERGY))
        end
        f .*= reshape(n ./ density(grid, f), 1, 1, :)
        unknown_values(u, layout, 2) .= T
        unknown_values(u, layout, 3) .= n
        rows = [first(Runes2P.unknown_range(layout, 4, ir)) for ir in 1:nr]
        kinetic = Runes2P.rhs!(
            zeros(length(u)), KineticEnergyChangeTerm(FokkerPlanckTerm(:f, op)), u, layout, 4,
            0.0)[rows]
        fluid, _ = heat_rhs(
            HeatTransportTerm(; delta_b_over_b = delta, major_radius = R0),
            heat_layout(nr; a), T, fill(n, nr))
        ratio = kinetic ./ fluid
        @test all(r -> isapprox(r, 1; atol = 1e-2), ratio[1:(nr - 2)])

        # Under a density gradient at a uniform temperature of 100 eV the heat transport of
        # the temperature is zero, while the energy term of the kinetic transport carries an
        # energy of 2 times the temperature per transported electron. That is the analytic
        # ratio of the energy flux to the particle flux for a flux proportional to v over a
        # Maxwellian. The term is negative at the inner radii, where the density is high, and
        # positive at the outermost one. A radial diffusion independent of p carries 1.5 times
        # the temperature. Measured on n = 1e19 (3 − 2 r²/a²) m⁻³ at 8 radii:
        # - Rechester-Rosenbluth at δB/B = 2e-3: 2.0002 per electron, −7.329e5 W/m³ at radii
        #   1 to 7 and +2.394e6 W/m³ at radius 8.
        # - D = 1 m²/s: 1.5004 per electron, −7.692e3 W/m³ at radii 1 to 7 and +2.513e4 W/m³
        #   at radius 8.
        flat = fill(100.0, nr)
        profile = [n * (1 + 2 * (1 - (r / a)^2)) for r in rad.r_centers]
        for ir in 1:nr, (i, p) in enumerate(momentum_grid(grid).p_centers)

            f[i, :, ir] .= exp(-p^2 / (1 + sqrt(1 + p^2)) /
                               (flat[ir] / Constants.ELECTRON_REST_ENERGY))
        end
        f .*= reshape(profile ./ density(grid, f), 1, 1, :)
        unknown_values(u, layout, 2) .= flat
        unknown_values(u, layout, 3) .= profile
        function moved_energy(transport)
            local power = Runes2P.rhs!(zeros(length(u)),
                KineticEnergyChangeTerm(FokkerPlanckTerm(:f, transport)), u, layout, 4, 0.0)[rows]
            local df = zeros(size(f))
            Runes2P.apply!(df, transport, f)
            return power, power ./ (e .* flat .* density(grid, df))
        end
        rr_power, rr_per_electron = moved_energy(op)
        constant = AdvectionDiffusion(grid; cross_diffusion = false)
        constant.diffusion_r_r .= 1.0
        constant_power, constant_per_electron = moved_energy(constant)
        uniform_t, _ = heat_rhs(
            HeatTransportTerm(; delta_b_over_b = delta, major_radius = R0),
            heat_layout(nr; a), flat, profile)
        @info "kinetic transport down a density gradient at 100 eV: Rechester-Rosenbluth $(rr_per_electron) T per electron, $(rr_power) W/m³; constant diffusion $(constant_per_electron) T per electron, $(constant_power) W/m³; heat transport of T $(uniform_t) W/m³"
        @test all(r -> isapprox(r, 2; rtol = 1e-2), rr_per_electron)
        @test all(r -> isapprox(r, 1.5; rtol = 1e-2), constant_per_electron)
        for power in (rr_power, constant_power)
            @test all(<(0), power[1:(nr - 1)]) && power[nr] > 0
        end
        @test all(isapprox(-7.329e5; rtol = 1e-2), rr_power[1:(nr - 1)])
        @test rr_power[nr] ≈ 2.394e6 rtol = 1e-2
        @test all(iszero, uniform_t)
    end

    @testset "Jacobian" begin
        layout = UnknownLayout(RadialGrid(uniform_r_faces(4, 0.5)),
            [RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        T, n = [900.0, 600.0, 250.0, 60.0], [3e19, 2.5e19, 2e19, 1e19]
        u = zeros(length(layout))
        unknown_values(u, layout, 1) .= T
        unknown_values(u, layout, 2) .= n
        unknown_values(u, layout, 3) .= THERMAL_ENERGY_FACTOR .* n .* T
        for term in (HeatTransportTerm(; diffusion = r -> 1 + 4r, edge = :constant_gradient_dream),
            HeatTransportTerm(;
            delta_b_over_b = 1e-3, major_radius = 1.7, safety_factor = r -> 1 + 3r^2,
            edge = :absorbing))
            system = EquationSystem(layout,
                [
                    Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), term),
                    Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(n),
                        LinearTerm(:cold_electron_density; coefficient = -1)),
                    thermal_energy_equation()];
                initial_state = u)
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            # columns of T and n at a radius and its neighbours
            for id in 1:2, ir in (2, 3)

                column = first(Runes2P.unknown_range(layout, id, ir))
                h = 1e-6 * u[column]
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                F_up, F_down = zeros(length(u)), zeros(length(u))
                rhs!(F_up, up, system, 0.0)
                rhs!(F_down, down, system, 0.0)
                @test Vector(J[:, column]) ≈ (F_up .- F_down) ./ (2h) rtol = 1e-6
            end
            # a profile is recorded by its values on the faces
            record = Runes2P.describe(term, Runes2P.DescriptionContext())
            level = record[term.model === :prescribed ? "diffusion" : "delta_b_over_b"]
            term.model === :prescribed &&
                @test level["faces"] ≈ 1 .+ 4 .* layout.radial_grid.r_faces
            term.model === :prescribed || @test record["safety_factor"]["faces"] ≈
                  1 .+ 3 .* layout.radial_grid.r_faces .^ 2
        end
    end

    @testset "a level that changes in time" begin
        layout = UnknownLayout(RadialGrid(uniform_r_faces(3, 0.5)),
            [RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        # the radial unknowns are stored radius by radius: set them by name
        start = zeros(length(layout))
        unknown_values(start, layout, 1) .= [500.0, 300.0, 100.0]
        unknown_values(start, layout, 2) .= 1e19
        history = PiecewiseConstant([1e-3, 1e-5], [1e-4])
        term = HeatTransportTerm(; delta_b_over_b_at = history, major_radius = 1.7, edge = :absorbing)
        @test forced_time_points(term) == [1e-4]
        system = EquationSystem(layout,
            [
                Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), term),
                Equation(
                    :cold_electron_density, Algebraic(), PrescribedValueTerm(fill(1e19, 3)),
                    LinearTerm(:cold_electron_density; coefficient = -1)),
                thermal_energy_equation()];
            initial_state = start)
        u0 = start
        @test_throws "not a time point" integrate(system, u0, [0.0, 1.5e-4, 3e-4]; stepper = FixedSteps())
        solution = integrate(system, u0, [0.0, 1e-4, 2e-4]; stepper = FixedSteps())
        T_end = unknown_values(solution.states[:, end], layout, 1)
        @test T_end[3] < 100 && T_end[1] < 500     # the heat flows out through the edge
        # the record names the history
        record = describe(term, Runes2P.DescriptionContext())
        @test record["model"] == "rechester_rosenbluth" && record["edge"] == "absorbing"
        @test record["delta_b_over_b"]["switch_times"] == [1e-4]
        # the step that ends at the switch time keeps the level before it, the next one takes
        # the level after it
        @test history(1e-4) == 1e-5
        @test Runes2P.level_over_step(history, 1e-4) == 1e-3
        @test Runes2P.level_over_step(history, 1.5e-4) == 1e-5
        @test Runes2P.level_over_step(history, 5e-5) == 1e-3
    end

    @testset "refusals and the guard" begin
        @test_throws ArgumentError HeatTransportTerm()
        @test_throws ArgumentError HeatTransportTerm(;
            diffusion = 1.0, delta_b_over_b = 1e-3, major_radius = 1.0)
        @test_throws ArgumentError HeatTransportTerm(; delta_b_over_b = 1e-3)
        @test_throws ArgumentError HeatTransportTerm(; diffusion = 1.0, edge = :other)
        @test_throws "two radii" Runes2P.validate(
            HeatTransportTerm(; diffusion = 1.0, edge = :constant_gradient_dream),
            heat_layout(1), 1)
        @test_throws "two radii" Runes2P.validate(
            HeatTransportTerm(; diffusion = 1.0, edge = :absorbing_dream),
            heat_layout(1), 1)
        @test_throws "must not be negative" Runes2P.validate(
            HeatTransportTerm(; diffusion = -1.0), heat_layout(2), 1)
        @test Runes2P._temperature_source(HeatTransportTerm(; diffusion = 1.0)) ==
              (:unknown, :cold_electron_temperature)
    end
end
