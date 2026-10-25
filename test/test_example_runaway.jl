# DREAM's Dreicer example, examples/runaway/basic.py (Mathias Hoppe, 2024-12-17), run by
# reference/dream/cases/example_runaway.py with the settings the example writes and two
# overrides that leave f as the only unknown (n_cold prescribed, no particle source). The
# test reads the example's values from the recorded settings and checks them, so the grid
# and the physics are the example's by construction. It compares the upwind interpolation
# with DREAM.
using HDF5

const DREAM_EXAMPLE_RUNAWAY_DATA = reference_file("dream_example_runaway.h5")

@testset "DREAM example runaway/basic.py" begin
    h5open(DREAM_EXAMPLE_RUNAWAY_DATA) do h5
        attr = attributes(h5)
        @test read(attr["example"]) == "examples/runaway/basic.py"
        @test read(attr["example_author"]) == "Mathias Hoppe"
        setting(key) = only(read(h5["settings/$key"]))

        # the example's values (DREAM defaults where the example sets nothing)
        @test (setting("hottailgrid/np"), setting("hottailgrid/pmax"),
            setting("hottailgrid/nxi")) ==
              (300, 1.0, 20)
        @test setting("eqsys/E_field/data/x") == 6.0
        @test setting("eqsys/T_cold/data/x") == 100.0
        @test setting("eqsys/n_cold/data/x") == 5e19          # override: prescribed at n_D
        @test (
            setting("radialgrid/B0"), setting("radialgrid/a"), setting("radialgrid/nr")) ==
              (5.0, 0.22, 1)
        @test setting("collisions/collfreq_mode") == 2        # COLLFREQ_MODE_FULL (default)
        @test setting("collisions/collfreq_type") == 3        # PARTIALLY_SCREENED
        @test setting("collisions/lnlambda") == 3             # LNLAMBDA_THERMAL (default)
        @test setting("eqsys/f_hot/adv_interp/p1") == 2       # AD_INTERP_UPWIND
        @test setting("eqsys/f_hot/adv_interp/p2") == 2
        @test setting("eqsys/f_hot/boundarycondition") == 1   # BC_F_0
        @test setting("eqsys/f_hot/synchrotronmode") == 1     # neglected
        @test setting("eqsys/f_hot/particleSource") == 1      # override: PARTICLE_SOURCE_ZERO
        @test setting("eqsys/n_re/avalanche") == 1            # neglected
        @test setting("eqsys/n_re/dreicer") == 1              # disabled (default)
        @test (setting("timestep/tmax"), setting("timestep/nt")) == (1e-3, 20)

        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        @test (momentum_cell_count(mom), pitch_cell_count(mom)) == (300, 20)
        rad = RadialGrid(read(h5["grid/r_f"]))
        g = cylindrical_grid(mom, rad)
        density = setting("eqsys/n_cold/data/x")
        plasma = PlasmaState(cold_electron_temperature = [read(h5["T_cold"])],
            cold_electron_density = [density],
            ions = [IonSpecies("D", 1, 1, [density])], electric_field = [read(h5["E_field"])],
            magnetic_field = [setting("radialgrid/B0")])
        settings = CollisionSettings(
            mode = :full, screening = :partial, coulomb_logarithm = :thermal,
            ion_size_table = :dream, bremsstrahlung = :neglect)
        op = fokker_planck_operator(
            g, plasma, settings; scheme = :upwind, synchrotron = false)
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale = density)])
        u0 = zeros(length(layout))
        kinetic_view(u0, layout, 1, 1) .= read(h5["f_hot_initial"])[:, :, 1]
        times = read(h5["t"])
        system = EquationSystem(layout,
            [Equation(:hot_tail_distribution, TimeDerivative(),
                FokkerPlanckTerm(:hot_tail_distribution, op))])
        solution = integrate(system, u0, times)

        # observed: 2.0e-11 in f, 1.4e-11 in every n_re increment; over the example's 1 ms
        # the rate still grows from 1.2e18 to 9.7e20 m⁻³s⁻¹, so the run ends before the
        # rate settles. With `scheme = :central` the same comparison is off by a relative
        # 330 in f and a factor 5 in the n_re increments, so the test tells the schemes apart.
        a = read(h5["f_hot_final"])[:, :, 1]
        b = kinetic_view(solution.states[:, end], layout, 1, 1)
        cut = 1e-12 * maximum(abs, a)
        @test maximum(abs(a[k] - b[k]) / abs(a[k])
        for k in eachindex(a) if abs(a[k]) > cut) < 1e-9
        # n_re collects the flux through p_max: each increment is dt times the flux at the
        # end of the step (backward Euler in both codes)
        n_re = read(h5["n_re"])
        fluxes = [momentum_flux(op, reshape(solution.states[:, k], size(g)), length(mom.p_faces))
                  for k in 2:length(times)]
        @test all(isapprox.(diff(n_re), fluxes .* diff(times); rtol = 1e-9))
    end
end
