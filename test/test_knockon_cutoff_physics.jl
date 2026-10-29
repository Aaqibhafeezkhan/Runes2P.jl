# A knock-on cutoff that changes during a run: the runaways of an avalanche in
# a cooling plasma with a rising field do not depend on where the cutoff lies, as long as it
# stays inside its window, between the thermal and the critical momentum. The cutoff that
# follows the plasma (KnockOnCutoffUpdate) moves over several faces in this run. Fixed
# cutoffs that lie inside the window at all times give the same runaways. A cutoff above
# the critical momentum loses the secondaries born between the two.
#
# The plasma of the avalanche test of test_runaway_physics.jl (deuterium at 1e20 m⁻³,
# complete screening, a seed at p = 2 on a grid to p = 100), with the temperature falling
# from 100 eV to 10 eV and the field rising from 10 to 30 critical fields over 40 ms.

@testset "runaways with a knock-on cutoff that follows the plasma" begin
    density_, t_end = 1e20, 0.04
    temperature(t) = 100.0 * 0.1^(t / t_end)
    logarithm = relativistic_coulomb_logarithm(density_, 100.0)
    critical_field_ = critical_field(density_, logarithm)
    field(t) = critical_field_ * (10 + 20 * t / t_end)
    plasma_at(t) = PlasmaState(
        cold_electron_temperature = [temperature(t)], cold_electron_density = [density_],
        ions = [IonSpecies("D", 1, 1, [density_])], electric_field = [field(t)],
        magnetic_field = [5.0])
    threshold(plasma, ir) = critical_momentum(plasma.electric_field[ir], critical_field_)
    mom = MomentumGrid(biuniform_p_faces(120, 60, 0.0, 0.6, 100.0), uniform_xi_faces(14))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 0.5)))
    settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :relativistic)
    times = collect(range(0, t_end; length = 41))

    # the window at the start and at the end, and the faces inside it at all times
    first_window = cutoff_window(thermal_momentum(temperature(0.0)), threshold(plasma_at(0.0), 1))
    last_window = cutoff_window(thermal_momentum(temperature(t_end)), threshold(plasma_at(t_end), 1))
    @test last_window.upper < first_window.upper
    inside = filter(p -> first_window.lower < p < last_window.upper, mom.p_faces)
    @test length(inside) >= 4

    function runaways(; cutoff, update = false)
        operator = AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        knockon = KnockOnOperator(g, g, cutoff)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1e12)])
        system = EquationSystem(layout,
            [Equation(:f, TimeDerivative(),
                prescribed_fokker_planck_term(:f, operator, plasma_at; settings),
                KnockOnTerm(:f, knockon; target_density = [density_],
                    treatment = :explicit))])
        u0 = zeros(length(layout))
        seed = kinetic_view(u0, layout, 1, 1)
        for i in 1:momentum_cell_count(mom), j in 1:pitch_cell_count(mom)

            seed[i, j] = 1e12 * exp(-(mom.p_centers[i] - 2.0)^2 / 0.08) *
                         exp(4 * (mom.xi_centers[j] - 1))
        end
        updates = update ?
                  (KnockOnCutoffUpdate(knockon, plasma_at; critical_momentum_at = threshold),) :
                  ()
        solution = integrate(system, u0, times; updates)
        @test all(solution.statistics.accepted)
        probe = AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        lost = 0.0
        for k in 2:length(times)
            set_fokker_planck_coefficients!(probe, plasma_at(times[k]), settings)
            lost += (times[k] - times[k - 1]) *
                    momentum_flux(probe, reshape(solution.states[:, k], size(g)), momentum_cell_count(mom) +
                                                                                  1)
        end
        final = reshape(solution.states[:, end], size(g))
        jacobian = phase_space_jacobian(g, 1)
        # above the largest critical momentum of the run, so that the count is of runaways at every time
        above = sum(jacobian[i, j] * mom.p_widths[i] * mom.xi_widths[j] * final[i, j, 1]
        for i in 1:momentum_cell_count(mom), j in 1:pitch_cell_count(mom)
        if mom.p_centers[i] > first_window.upper) / g.radial_grid.spatial_jacobian[1]
        cutoffs = update ? vec(solution.update_states["knockon_cutoff"]) :
                  fill(Float64(cutoff), length(times))
        return (count = above + lost, cutoffs = cutoffs)
    end

    followed = runaways(cutoff = inside[2], update = true)
    # the cutoff moves with the plasma, downwards, over several faces, and stays below the threshold
    @test length(unique(followed.cutoffs)) >= 3
    @test issorted(followed.cutoffs; rev = true)
    @test all(followed.cutoffs[k] < threshold(plasma_at(times[k]), 1)
    for k in eachindex(times))

    without = runaways(cutoff = 50.0)       # no secondary above this cutoff stays on the grid for long
    fixed = [runaways(cutoff = p).count
             for p in (first(inside), inside[length(inside) ÷ 2], last(inside))]
    # the knock-on collisions multiply the seed, threefold in this run
    @test followed.count > 2.5 * without.count
    # measured: the cutoff goes over the faces 0.09 to 0.04, and the runaways of the fixed
    # cutoffs 0.08, 0.13 and 0.18 differ from its runaways by 4e-7, 5e-9 and 6e-8. A
    # secondary born below the critical momentum does not run away, wherever the cutoff is.
    for count in fixed
        @test count ≈ followed.count rtol = 1e-5
    end
    # a cutoff above the critical momentum loses the secondaries born between the two.
    # Measured shares of the runaways: 0.9998 at 0.2, 0.972 at 0.25, 0.879 at 0.3, 0.669 at
    # 0.4 and 0.471 at 0.6, with the critical momentum falling from 0.333 to 0.186.
    shares = [runaways(cutoff = mom.p_faces[argmin(abs.(mom.p_faces .- p))]).count /
              followed.count
              for p in (0.25, 0.4, 0.6)]
    @test issorted(shares; rev = true)
    @test 0.95 < shares[1] < 0.99
    @test 0.6 < shares[2] < 0.75
    @test 0.4 < shares[3] < 0.55
end
