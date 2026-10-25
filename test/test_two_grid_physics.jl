# The avalanche on a hot-tail and a runaway grid against the same case on one grid and
# against the kinetic rate of the literature: the growth rate must not
# depend on where the interface between the grids lies, and the runaway grid gains what
# passes the interface and what the knock-on collisions produce on it.
#
# The case is that of "avalanche growth rate against the kinetic rate of the literature" in
# test_runaway_physics.jl: 100 eV, 1e20 m⁻³, E/E_c = 20, a seed at p = 2, 120 × 14 cells to
# p = 100, 192 steps to 80 ms, the rate read over the last third of the steps.

@testset "avalanche on two grids" begin
    temperature, density, over_critical = 100.0, 1e20, 20.0
    coulomb_log = relativistic_coulomb_logarithm(temperature, density)
    critical_electric_field = critical_field(density, coulomb_log)
    field = over_critical * critical_electric_field
    critical = critical_momentum(field, critical_electric_field)
    p_faces = biuniform_p_faces(120, 60, 0.0, 0.6, 100.0)
    pitch = uniform_xi_faces(14)
    radial = RadialGrid(uniform_r_faces(1, 0.5))
    plasma = PlasmaState(
        cold_electron_temperature = [temperature], cold_electron_density = [density],
        ions = [IonSpecies("D", 1, 1, [density])], electric_field = [field], magnetic_field = [5.0])
    settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :relativistic)
    window = cutoff_window(thermal_momentum(temperature), critical)
    cutoff = choose_cutoff(p_faces, window.lower, window.upper).p_cutoff
    times = collect(range(0, 0.08; length = 193))
    third = (length(times) - 1) ÷ 3
    seed(m) = [1e12 * exp(-(p - 2.0)^2 / 0.08) * exp(4 * (xi - 1))
               for p in m.p_centers, xi in m.xi_centers]
    weights(grid) = [phase_space_jacobian(grid, 1)[i, j] * momentum_grid(grid).p_widths[i] *
                     momentum_grid(grid).xi_widths[j] /
                     grid.radial_grid.spatial_jacobian[1]
                     for i in 1:momentum_cell_count(grid), j in 1:pitch_cell_count(grid)]
    cumulative(rates) = cumsum([0.0;
                                [(rates[k] + rates[k + 1]) / 2 * (times[k + 1] - times[k])
                                 for k in 1:(length(times) - 1)]])
    rate_of(counts) = log(counts[end] / counts[end - third]) /
                      (times[end] - times[end - third])

    function one_grid()
        grid = cylindrical_grid(MomentumGrid(p_faces, pitch), radial)
        m = momentum_grid(grid)
        operator = fokker_planck_operator(grid, plasma, settings)
        layout = UnknownLayout(radial, [KineticUnknown(:hot_tail_distribution, grid; scale = 1e12)])
        system = EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(),
                FokkerPlanckTerm(:hot_tail_distribution, operator),
                KnockOnTerm(:hot_tail_distribution, KnockOnOperator(grid, grid, cutoff);
                    target_density = [density], treatment = :explicit))])
        u0 = zeros(length(layout))
        kinetic_view(u0, layout, 1, 1) .= seed(m)
        solution = integrate(system, u0, times)
        weight = weights(grid)
        state(k) = kinetic_view(view(solution.states, :, k), layout, 1, 1)
        counts = [sum(weight[i, j] * state(k)[i, j]
                  for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)
                  if m.p_centers[i] > critical) for k in eachindex(times)]
        lost = cumulative([momentum_flux(operator, reshape(state(k), size(grid)), momentum_cell_count(m) +
                                                                                  1)
                           for k in eachindex(times)])
        return rate_of(counts .+ lost)
    end

    function two_grids(interface_momentum, runaway_pitch; pitch_mapping = :overlap)
        cut = argmin(abs.(p_faces .- interface_momentum)) - 1
        hot = cylindrical_grid(MomentumGrid(p_faces[1:(cut + 1)], pitch), radial)
        runaway = runaway_grid(hot, p_faces[(cut + 2):end], runaway_pitch)
        hot_m, runaway_m = momentum_grid(hot), momentum_grid(runaway)
        hot_operator = fokker_planck_operator(hot, plasma, settings; p_upper_boundary = :interface)
        runaway_operator = fokker_planck_operator(runaway, plasma, settings; p_lower_boundary = :interface)
        hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator)
        interface = MomentumInterfaceTerm(
            :hot_tail_distribution, :runaway_distribution, hot_term,
            runaway_operator; pitch_mapping)
        runaway_from_runaway = KnockOnOperator(runaway, runaway, cutoff)
        layout = UnknownLayout(radial,
            [KineticUnknown(:hot_tail_distribution, hot; scale = 1e12),
                KineticUnknown(:runaway_distribution, runaway; scale = 1e12)])
        knockon(primary, operator) = KnockOnTerm(
            primary, operator; target_density = [density], treatment = :explicit)
        system = EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface,
                    knockon(:hot_tail_distribution, KnockOnOperator(hot, hot, cutoff)),
                    knockon(:runaway_distribution, KnockOnOperator(hot, runaway, cutoff))),
                Equation(:runaway_distribution, TimeDerivative(),
                    FokkerPlanckTerm(:runaway_distribution, runaway_operator), interface,
                    knockon(:runaway_distribution, runaway_from_runaway))])
        hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution),
        unknown_id(layout, :runaway_distribution)
        u0 = zeros(length(layout))
        kinetic_view(u0, layout, hot_id, 1) .= seed(hot_m)
        kinetic_view(u0, layout, runaway_id, 1) .= seed(runaway_m)
        solution = integrate(system, u0, times)
        @test all(solution.statistics.accepted)
        state(k) = view(solution.states, :, k)
        f_hot(k) = kinetic_view(state(k), layout, hot_id, 1)
        f_runaway(k) = kinetic_view(state(k), layout, runaway_id, 1)
        hot_weight, runaway_weight = weights(hot), weights(runaway)
        on_runaway_grid = [sum(runaway_weight .* f_runaway(k)) for k in eachindex(times)]
        counts = [sum(
                      hot_weight[i, j] * f_hot(k)[i, j]
                      for i in 1:momentum_cell_count(hot_m),
                          j in 1:pitch_cell_count(hot_m) if hot_m.p_centers[i] > critical;
                      init = 0.0) +
                  on_runaway_grid[k] for k in eachindex(times)]
        outflux = [momentum_flux(runaway_operator, reshape(f_runaway(k), size(runaway)),
                       momentum_cell_count(runaway_m) + 1) for k in eachindex(times)]
        # budget of the runaway grid with the fluxes of backward Euler (those of the new
        # state) and the explicit knock-on production (that of the old state)
        source = zeros(size(runaway)[1:2])
        gained = 0.0
        for k in 2:length(times)
            fill!(source, 0)
            apply_at_radius!(source, runaway_from_runaway, f_runaway(k - 1), density, 1)
            gained += (times[k] - times[k - 1]) *
                      (interface_flux(interface, state(k), layout, 1) +
                       sum(runaway_weight .* source) - outflux[k])
        end
        budget = (on_runaway_grid[end] - on_runaway_grid[1] - gained) / on_runaway_grid[end]
        return (; rate = rate_of(counts .+ cumulative(outflux)),
            budget, negative = count(<(0), solution.states))
    end

    single = one_grid()
    # CODE with the field-particle Boltzmann operator, Embréus, Stahl, Fülöp 2018, Fig. 3:
    # 0.915 Γ₀ at E/E_c = 20 (test_runaway_physics.jl), 26.0 1/s
    gamma_0 = (over_critical - 1) * 2pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 *
              Constants.SPEED_OF_LIGHT * density
    reference = 0.915 * gamma_0
    @test single ≈ reference rtol = 0.02

    # the two grids with one pitch grid are the one grid cut in two. Observed: the rates
    # differ by 1.2e-6 of the one-grid rate, from the ξ* tables of the knock-on kernels,
    # which cover the pairs of each pairing
    same = two_grids(2.5, pitch)
    @test same.rate ≈ single rtol = 1e-5
    @test abs(same.budget) < 1e-12
    @test same.negative == 0

    # the pitch grid of the runaway grid (12 cells to 0.9 rad, 8 to π) with the interface at
    # p = 0.6, 2.26 and 10.5. Observed: 0.26 %, 0.27 % and 0.17 % above the one-grid rate
    runaway_pitch = biuniform_theta_xi_faces(20, 12, 0.9)
    results = [two_grids(momentum, runaway_pitch) for momentum in (0.8, 2.5, 10.0)]
    for result in results
        @test result.rate ≈ reference rtol = 0.02
        @test result.rate ≈ single rtol = 5e-3
        @test abs(result.budget) < 1e-12
        @test result.negative == 0
    end
    rates = [result.rate for result in results]
    @test maximum(rates) - minimum(rates) < 2e-3 * single

    # DREAM's pitch mapping gives the same rate and leaves negative values where the pitch
    # grids differ (152 in this run), the overlap none
    dream = two_grids(2.5, runaway_pitch; pitch_mapping = :dream)
    @test dream.rate ≈ results[2].rate rtol = 1e-4
    @test abs(dream.budget) < 1e-12
    @test dream.negative > 0
end
