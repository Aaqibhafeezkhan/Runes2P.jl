# The avalanche of the test of test_runaway_physics.jl as a run of DREAM's Boltzmann mode
# (reference/dream/cases/avalanche_kinetic.py, AVALANCHE_MODE_BOLTZMANN_KNOCK_ON): fully
# ionized deuterium at 100 eV and 1e20 m⁻³, E = 20 E_c, complete screening, the constant
# Coulomb logarithm, hot-tail grid only to p = 200 (180 × 14 cells), centred interpolation,
# a seed of runaways at p = 2, 192 steps to 80 ms. Runes2P takes DREAM's first distribution,
# its time points, central interpolation and DREAM's knock-on kernel.
#
# With a hot-tail grid only, DREAM adds the Rosenbluth-Putvinski source of the electrons
# beyond p_max (n_re) to the Boltzmann operator. Runes2P has no such source. The
# grid ends at p = 200, where no electron arrives within the run (the field moves the seed
# to p = 83), so that source is zero. With p_max = 100 the two codes differ by 3e-5 in the
# rate of the last window.
#
# This is the same method in two codes. The independent reference of the growth rate is in
# test_runaway_physics.jl (Embréus et al. 2018).
using HDF5

const DREAM_AVALANCHE_KINETIC_DATA = joinpath(@__DIR__, "data", "dream_avalanche_kinetic.h5")

@testset "avalanche against DREAM's Boltzmann mode" begin
    h5open(DREAM_AVALANCHE_KINETIC_DATA) do h5
        setting(key) = only(read(h5["settings/$key"]))
        @test setting("eqsys/n_re/avalanche") == 5            # AVALANCHE_MODE_BOLTZMANN_KNOCK_ON
        @test setting("collisions/collfreq_mode") == 2        # FULL
        @test setting("collisions/collfreq_type") == 1        # COMPLETELY_SCREENED
        @test setting("collisions/lnlambda") == 1             # CONSTANT
        @test setting("eqsys/f_hot/adv_interp/p1") == 1       # CENTRED
        @test setting("eqsys/f_hot/synchrotronmode") == 1     # neglected
        p_cutoff = setting("eqsys/n_re/pCutAvalanche")
        temperature, density = read(h5["temperature"]), read(h5["density"])
        field = read(h5["E_field"])
        # the case script has its own physical constants
        @test field ≈ 20 * critical_field(density, relativistic_coulomb_logarithm(temperature, density)) rtol = 1e-7

        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        @test mom.p_faces ≈ biuniform_p_faces(180, 60, 0.0, 0.6, 200.0) rtol = 1e-13
        rad = RadialGrid(read(h5["grid/r_f"]))
        g = cylindrical_grid(mom, rad)
        plasma = PlasmaState(cold_electron_temperature=[temperature], cold_electron_density=[density],
                             ions=[IonSpecies("D", 1, 1, [density])], electric_field=[field],
                             magnetic_field=[read(h5["B"])])
        settings = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:relativistic)
        operator = fokker_planck_operator(g, plasma, settings; scheme=:central)  # DREAM's centred interpolation
        knockon = KnockOnOperator(g, g, p_cutoff; dream_compatible=true, primary_pitch_points=1, table_size=80)
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e12)])
        system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                  FokkerPlanckTerm(:hot_tail_distribution, operator),
                                                  KnockOnTerm(:hot_tail_distribution, knockon;
                                                              target_density=[density], treatment=:explicit))])
        u0 = zeros(length(layout))
        kinetic_view(u0, layout, 1, 1) .= read(h5["f_hot_initial"])[:, :, 1]
        times = read(h5["t"])
        @test length(times) == 193
        solution = integrate(system, u0, times)

        # the distribution at the end, where it is above 1e-12 of its largest value: largest
        # difference 2.2e-10 of the value of DREAM
        a = read(h5["f_hot_final"])[:, :, 1]
        b = kinetic_view(solution.states[:, end], layout, 1, 1)
        cut = 1e-12 * maximum(abs, a)
        difference = maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
        @test difference < 1e-8

        # the runaways above the test-particle critical momentum plus those beyond p_max, and
        # their growth rate over the windows of 20 ms
        critical = read(h5["p_critical"])
        jacobian = phase_space_jacobian(g, 1)
        volume = rad.spatial_jacobian[1]
        runaways(state) = sum(jacobian[i, j] * mom.p_widths[i] * mom.xi_widths[j] * state[i, j, 1]
                              for i in 1:momentum_cell_count(mom), j in 1:pitch_cell_count(mom)
                              if mom.p_centers[i] > critical) / volume
        states = [reshape(solution.states[:, k], size(g)) for k in eachindex(times)]
        outflux = [momentum_flux(operator, s, momentum_cell_count(mom) + 1) for s in states]
        # DREAM's n_re takes the flux through p_max of the new time level
        lost = cumsum([0.0; [outflux[k+1] * (times[k+1] - times[k]) for k in 1:length(times)-1]])
        on_grid = read(h5["runaways_on_grid"])
        # observed 2e-13 of the count of DREAM
        difference_on_grid = maximum(abs(runaways(states[k]) / on_grid[k] - 1) for k in eachindex(times))
        @test difference_on_grid < 1e-10
        # nothing leaves the grid in either code
        n_re = read(h5["n_re"])
        @test n_re[end] < 1e-12 * on_grid[end]
        @test lost[end] < 1e-12 * on_grid[end]
        counts = [runaways(states[k]) + lost[k] for k in eachindex(times)]
        dream_rates = read(h5["window_rates"])
        rates = [log(counts[1 + 48w] / counts[1 + 48 * (w - 1)]) / (times[1 + 48w] - times[1 + 48 * (w - 1)])
                 for w in 1:4]
        # observed 3e-13 of the rate of DREAM
        @test rates ≈ dream_rates rtol = 1e-10
        # central interpolation on this grid: 25.24 s⁻¹ in the last window, 3.3% below
        # exponential fitting on the same grid (26.10)
        @test 25.0 < rates[end] < 25.4
    end
end
