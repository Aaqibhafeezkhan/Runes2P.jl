# Heat transport of the cold electrons: the decay of the lowest Bessel mode
# of a uniform diffusivity in a cylinder, the energy balance with the edge flux, the
# Rechester-Rosenbluth heat flux against the energy moment of the kinetic Rechester-
# Rosenbluth operator on Maxwellians, the Jacobian, and the switch times of a level that
# changes in time.

# J₀ by its series, enough for 0 ≤ x ≤ 3
_bessel_j0(x) = sum((-1)^k * (x / 2)^(2k) / factorial(big(k))^2 for k in 0:30) |> Float64
const BESSEL_J0_FIRST_ZERO = 2.404825557695773

@testset "heat transport" begin
    e = Constants.ELEMENTARY_CHARGE
    function heat_layout(nr; a=0.5)
        UnknownLayout(RadialGrid(uniform_r_faces(nr, a)), [RadialUnknown(:cold_electron_temperature; scale=100.0),
                                                           RadialUnknown(:cold_electron_density; scale=1e19)])
    end
    function heat_rhs(term, layout, T, n; t=0.0)
        u = zeros(length(layout))
        radial_values(u, layout, 1) .= T
        radial_values(u, layout, 2) .= n
        Runes2P.validate(term, layout, 1)
        out = zeros(length(u))
        Runes2P.rhs!(out, term, u, layout, 1, t)
        return radial_values(out, layout, 1), u
    end

    @testset "the lowest Bessel mode" begin
        # dT/dt = D ∇²T at a uniform density: T = J₀(j₀₁ r/a) decays at D j₀₁²/a²
        D, a, n = 3.0, 0.5, 1e19
        exact = D * BESSEL_J0_FIRST_ZERO^2 / a^2
        function rate(nr, edge)
            layout = heat_layout(nr; a)
            rg = layout.radial_grid
            T = [_bessel_j0(BESSEL_J0_FIRST_ZERO * r / a) for r in rg.r_centers]
            rhs, _ = heat_rhs(HeatTransportTerm(; diffusion=D, edge), layout, T, fill(n, nr))
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
        T, n = [800.0, 700.0, 520.0, 300.0, 120.0, 40.0], [3e19, 2.8e19, 2.5e19, 2e19, 1.5e19, 1e19]
        for model in ((; diffusion=2.0), (; delta_b_over_b=1e-3, major_radius=1.7)),
            edge in (:zero_flux, :absorbing, :absorbing_dream, :constant_gradient_dream)
            term = HeatTransportTerm(; model..., edge)
            rhs, u = heat_rhs(term, layout, T, n)
            lost = heat_flux_through_edge(term, u, layout, 0.0)
            @test sum(volume .* rhs) ≈ -lost rtol = 1e-12 atol = 1e-12 * sum(volume .* abs.(rhs))
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
            grid = cylindrical_grid(MomentumGrid(uniform_p_faces(400, 0.0, 12 * sqrt(2theta)), uniform_xi_faces(8)), rad)
            op = AdvectionDiffusion(grid)
            add_rechester_rosenbluth_diffusion!(op, delta; major_radius=R0)
            np, nxi, nr = size(grid)
            f = zeros(np, nxi, nr)
            for ir in 1:nr, (i, p) in enumerate(momentum_grid(grid).p_centers)
                f[i, :, ir] .= exp(-p^2 / (1 + sqrt(1 + p^2)) / (T[ir] / Constants.ELECTRON_REST_ENERGY))
            end
            f .*= reshape(n ./ density(grid, f), 1, 1, :)
            df = zeros(size(f))
            Runes2P.apply!(df, op, f)
            m = momentum_grid(grid)
            energies = [Constants.ELECTRON_REST_ENERGY * e * p^2 / (1 + sqrt(1 + p^2)) for p in m.p_centers]
            kinetic = [sum(energies[i] * phase_space_jacobian(grid, ir)[i, j] * m.p_widths[i] * m.xi_widths[j] * df[i, j, ir]
                           for i in 1:np, j in 1:nxi) / rad.spatial_jacobian[ir] for ir in 1:nr]
            fluid, _ = heat_rhs(HeatTransportTerm(; delta_b_over_b=delta, major_radius=R0), heat_layout(3), T, fill(n, 3))
            # at 20 keV the expansion 1 − 5Θ/8 misses terms of order Θ² ≈ 2e-3
            tolerance = T0 < 1e4 ? 1e-3 : 5e-3
            @test fluid[1] ≈ kinetic[1] rtol = tolerance
            @test fluid[3] ≈ kinetic[3] rtol = tolerance
        end
    end

    @testset "Jacobian" begin
        layout = UnknownLayout(RadialGrid(uniform_r_faces(4, 0.5)),
                               [RadialUnknown(:cold_electron_temperature; scale=100.0),
                                RadialUnknown(:cold_electron_density; scale=1e19),
                                RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e21)])
        T, n = [900.0, 600.0, 250.0, 60.0], [3e19, 2.5e19, 2e19, 1e19]
        u = zeros(length(layout))
        radial_values(u, layout, 1) .= T
        radial_values(u, layout, 2) .= n
        radial_values(u, layout, 3) .= THERMAL_ENERGY_FACTOR .* n .* T
        for term in (HeatTransportTerm(; diffusion=r -> 1 + 4r, edge=:constant_gradient_dream),
                     HeatTransportTerm(; delta_b_over_b=1e-3, major_radius=1.7, safety_factor=r -> 1 + 3r^2,
                                       edge=:absorbing))
            system = EquationSystem(layout, [Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), term),
                                             Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(n),
                                                      LinearTerm(:cold_electron_density; coefficient=-1)),
                                             thermal_energy_equation()]; initial_state=u)
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
        end
    end

    @testset "a level that changes in time" begin
        layout = UnknownLayout(RadialGrid(uniform_r_faces(3, 0.5)),
                               [RadialUnknown(:cold_electron_temperature; scale=100.0),
                                RadialUnknown(:cold_electron_density; scale=1e19),
                                RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e21)])
        # the radial unknowns are stored radius by radius: set them by name
        start = zeros(length(layout))
        radial_values(start, layout, 1) .= [500.0, 300.0, 100.0]
        radial_values(start, layout, 2) .= 1e19
        history = PiecewiseConstant([1e-3, 1e-5], [1e-4])
        term = HeatTransportTerm(; delta_b_over_b_at=history, major_radius=1.7, edge=:absorbing)
        @test forced_time_points(term) == [1e-4]
        system = EquationSystem(layout, [Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), term),
                                         Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(fill(1e19, 3)),
                                                  LinearTerm(:cold_electron_density; coefficient=-1)),
                                         thermal_energy_equation()]; initial_state=start)
        u0 = start
        @test_throws "not a time point" integrate(system, u0, [0.0, 1.5e-4, 3e-4]; stepper=FixedSteps())
        solution = integrate(system, u0, [0.0, 1e-4, 2e-4]; stepper=FixedSteps())
        T_end = radial_values(solution.states[:, end], layout, 1)
        @test T_end[3] < 100 && T_end[1] < 500     # the heat flows out through the edge
        # the record names the history
        record = describe(term, Runes2P.DescriptionContext())
        @test record["model"] == "rechester_rosenbluth" && record["edge"] == "absorbing"
    end

    @testset "refusals and the guard" begin
        @test_throws ArgumentError HeatTransportTerm()
        @test_throws ArgumentError HeatTransportTerm(; diffusion=1.0, delta_b_over_b=1e-3, major_radius=1.0)
        @test_throws ArgumentError HeatTransportTerm(; delta_b_over_b=1e-3)
        @test_throws ArgumentError HeatTransportTerm(; diffusion=1.0, edge=:other)
        @test_throws "two radii" Runes2P.validate(HeatTransportTerm(; diffusion=1.0, edge=:constant_gradient_dream),
                                               heat_layout(1), 1)
        @test_throws "must not be negative" Runes2P.validate(HeatTransportTerm(; diffusion=-1.0), heat_layout(2), 1)
        @test Runes2P._temperature_source(HeatTransportTerm(; diffusion=1.0)) == (:unknown, :cold_electron_temperature)
    end
end
