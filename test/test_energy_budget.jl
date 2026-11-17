# The total energy of the electrons in the fully kinetic mode, an unknown W_tot = W_cold +
# W_hot, with W_hot the energy of the hot region of the hot-tail grid, whose time derivative
# the row of the temperature holds, so that T follows from W_cold = W_tot − W_hot: the
# transfer from a slowing-down tail to the cold electrons against the friction power of
# superthermal electrons, the temperature of a cold plasma under a hot tail a hundred times
# its energy, the chain that determines the temperature, the checks of the row with the
# losses that take no flux through p_max, the refusal of radial transport set after the
# build, by an update or by hand, the energy of the knock-ons of fluid runaways, and DREAM's
# fluid heating by runaways.
import ForwardDiff

@testset "energy budget" begin
    e = Constants.ELEMENTARY_CHARGE
    c, r0, me = Constants.SPEED_OF_LIGHT, Constants.CLASSICAL_ELECTRON_RADIUS,
    Constants.ELECTRON_MASS
    minus(name) = LinearTerm(name; coefficient = -1)
    density_row(n) = Equation(:cold_electron_density, Algebraic(),
        PrescribedValueTerm(n), minus(:cold_electron_density))

    @testset "the slowing down at p = 0 with plasma derivatives" begin
        # the energy-dependent Coulomb logarithm at a momentum of zero that carries derivatives
        # (the plasma columns of the Jacobian): finite, the thermal value
        base = PlasmaState(; cold_electron_temperature = [100.0],
            ions = [IonSpecies("D", 1, 1, [1e19])], electric_field = [0.0],
            magnetic_field = [0.0], cold_electron_density = [1e19])
        dual = Runes2P.dual_plasma(base, true)
        for lnl in (:energy_dependent, :thermal)
            model = Runes2P.CollisionModel(dual,
                CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = lnl))
            nu = Runes2P.p3_slowing_down_frequency_at_zero(model, 1)
            @test all(isfinite, ForwardDiff.partials(nu))
        end
        @test coulomb_logarithm(:electron_electron, :energy_dependent, 0.0, 1e19, 100.0) ==
              thermal_coulomb_logarithm(1e19, 100.0)
    end

    @testset "transfer from a slowing-down tail" begin
        # A bump of electrons at p = 0.3 above a 100 eV bulk, collisions with the free electrons
        # only (complete screening), no field: the energy the tail loses goes to the cold
        # electrons. The friction power of a superthermal electron on cold free electrons is
        # v F_s = 4π n r₀² mₑc³ lnΛ_ee(p) γ/p, and the transfer is its sum over the tail. The
        # collisions are those of the fully kinetic mode, finite at p = 0 (the superthermal
        # friction diverges there); their thermal corrections at the bump are of order T/E ≈ 0.5 %.
        T0, n, n_b = 100.0, 1e19, 1e16
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(300, 0.0, 0.6), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 0.5)))
        m = momentum_grid(grid)
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        bulk = [exp(-p^2 / (1 + sqrt(1 + p^2)) / theta)
                for p in m.p_centers, _ in 1:2, _ in 1:1]
        bulk .*= n / density(grid, bulk)[1]
        bump = [exp(-((p - 0.3) / 0.03)^2) for p in m.p_centers, _ in 1:2, _ in 1:1]
        bump .*= n_b / density(grid, bump)[1]
        f0 = bulk .+ bump
        base = PlasmaState(;
            cold_electron_temperature = [T0], ions = [IonSpecies("D", 1, 1, [n])],
            electric_field = [0.0], magnetic_field = [0.0])
        settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :energy_dependent)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        fp = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base), settings)
        # the row lists the sinks of the hot region and the energy that leaves through p_max;
        # both are below 1e-10 of the total energy over the step here
        equations = [Equation(:f, TimeDerivative(), fp),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                PrescribedValueTerm([0.0]),
                KineticEnergyLossTerm(fp; part = :sinks, region),
                MomentumBoundaryFluxTerm(:f, fp; weight = :energy, coefficient = -1)),
            density_row([n]), thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        u0 = zeros(length(layout))
        kinetic_array(u0, layout, 1) .= f0
        unknown_values(u0, layout, 2) .= T0
        unknown_values(u0, layout, 3) .= n
        system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
        dt = 5e-8
        solution = integrate(system, u0, [0.0, dt]; stepper = FixedSteps())
        W_cold = [unknown_values(solution.states[:, k], layout, 4)[1] for k in 1:2]
        W_tot = [unknown_values(solution.states[:, k], layout, 5)[1] for k in 1:2]
        # the total holds (no source in the row of the temperature), and the cold electrons
        # gain what the tail loses
        @test W_tot[2] ≈ W_tot[1] rtol = 1e-10
        @test W_tot[1] ≈
              W_cold[1] + only(Runes2P.rhs!(
            zeros(length(u0)), EnergyMomentTerm(:f, grid; region), solution.states[:, 1],
            layout, 5, 0.0)[Runes2P.unknown_range(layout, 5, 1)]) rtol = 1e-12
        f1 = kinetic_array(solution.states[:, 2], layout, 1)
        T1 = unknown_values(solution.states[:, 2], layout, 2)[1]
        p_th = 5 * sqrt(2 * T1 / Constants.ELECTRON_REST_ENERGY)
        V = phase_space_jacobian(grid, 1)
        friction = sum(4π * n * r0^2 * me * c^3 *
                       coulomb_logarithm(:electron_electron, :energy_dependent, p, n, T1) *
                       sqrt(1 + p^2) / p * V[i, j] * m.p_widths[i] * m.xi_widths[j] *
                       f1[i, j, 1]
        for (i, p) in enumerate(m.p_centers) if p > p_th for j in 1:2) /
                   grid.radial_grid.spatial_jacobian[1]
        transfer = (W_cold[2] - W_cold[1]) / dt
        @test transfer > 0
        @test transfer ≈ friction rtol = 2e-2
    end

    @testset "the row of the total energy lists every channel of W_hot" begin
        T0, n = 100.0, 1e19
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(60, 0.0, 0.4), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(2, 0.5)))
        base = PlasmaState(;
            cold_electron_temperature = [T0, T0], ions = [IonSpecies("D", 1, 1, [n, n])],
            electric_field = zeros(2), magnetic_field = fill(2.5, 2))
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        plasma = PlasmaFromUnknowns(base)
        fp(boundary; settings = CollisionSettings(mode = :full, screening = :partial)) = FokkerPlanckTerm(
            :f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = boundary);
            plasma, settings)
        term = fp(:f_zero)
        sinks = KineticEnergyLossTerm(term; part = :sinks, region)
        outflow = MomentumBoundaryFluxTerm(:f, term; weight = :energy, coefficient = -1)
        energy = [thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        row(terms...) = Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy), terms...)
        check(f_terms, row_terms...) = Runes2P._check_plasma_pairings([Equation(:f,
                                                                           TimeDerivative(),
                                                                           f_terms...);
                                                                       row(row_terms...);
                                                                       energy])
        @test check((term,), sinks, outflow) === nothing
        @test_throws "part = :sinks" check((term,), outflow)
        @test_throws "through p_max" check((term,), sinks)
        # the boundary term of this Fokker-Planck term, charged as a loss
        @test_throws "through p_max" check((term,), sinks,
            MomentumBoundaryFluxTerm(:f, term; weight = :energy, coefficient = 1))
        # a copy of the term built from the same arguments: the identity check refuses it before the
        # row check
        @test_throws "not a term of the system" check((term,), sinks,
            MomentumBoundaryFluxTerm(:f, fp(:f_zero); weight = :energy, coefficient = -1))
        # complete screening without radiation has no sinks
        bare = fp(:f_zero;
            settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect))
        bare = FokkerPlanckTerm(
            :f, bare.operator; plasma, settings = bare.settings, synchrotron = false)
        @test check((bare,), MomentumBoundaryFluxTerm(:f, bare; weight = :energy, coefficient = -1)) ===
              nothing
        # no screening counts the bound electrons as targets, which the free-electron operator of
        # the sinks term does not: a sink
        none = fp(:f_zero;
            settings = CollisionSettings(mode = :full, screening = :none, bremsstrahlung = :neglect))
        none = FokkerPlanckTerm(
            :f, none.operator; plasma, settings = none.settings, synchrotron = false)
        @test_throws "part = :sinks" check(
            (none,), MomentumBoundaryFluxTerm(:f, none; weight = :energy, coefficient = -1))
        @test check((none,), KineticEnergyLossTerm(none; part = :sinks, region),
            MomentumBoundaryFluxTerm(:f, none; weight = :energy, coefficient = -1)) ===
              nothing
        # the sinks term charged as a gain, or over another region than the energy moment
        @test_throws "negative coefficient" check(
            (term,), KineticEnergyLossTerm(term; part = :sinks, region, coefficient = 1),
            outflow)
        @test_throws "another region" check((term,),
            KineticEnergyLossTerm(term; part = :sinks, region = MomentumRegion(0.1)),
            outflow)
        # the interface to a runaway grid under the total energy: the interface term of this term,
        # charged as a loss
        upper = runaway_grid(grid, uniform_p_faces(20, 0.4, 5.0)[2:end])
        lower = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings = CollisionSettings(mode = :full, screening = :partial))
        re = FokkerPlanckTerm(:runaway,
            AdvectionDiffusion(upper; scheme = :exponential_fitting, p_lower_boundary = :interface);
            plasma, settings = lower.settings)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        two(row_terms...) = Runes2P._check_plasma_pairings([Equation(:f, TimeDerivative(), lower, interface);
                                                            Equation(:runaway, TimeDerivative(), re, interface);
                                                            row(row_terms...); energy])
        crossing = KineticEnergyLossTerm(lower; part = :sinks, region)
        work = InterfaceCollisionWorkTerm(interface)
        @test two(crossing, InterfaceEnergyFluxTerm(interface), work) === nothing
        @test_throws "add InterfaceEnergyFluxTerm" two(crossing, work)
        @test_throws "negative coefficient" two(crossing, InterfaceEnergyFluxTerm(interface; coefficient = 1), work)
        # the work of the collisions on the electrons that cross: required with the energy that
        # crosses, of the same interface, charged to the cold electrons
        @test_throws "add InterfaceCollisionWorkTerm" two(crossing, InterfaceEnergyFluxTerm(interface))
        @test_throws "gives the cold electrons the work that they pay" two(
            crossing, InterfaceEnergyFluxTerm(interface), InterfaceCollisionWorkTerm(interface; coefficient = 1))
        # the work of a copy of the interface: the identity check refuses it before the row check
        copy_of_interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        @test_throws "not a term of the system" two(
            crossing, InterfaceEnergyFluxTerm(interface), InterfaceCollisionWorkTerm(copy_of_interface))
        # the work of the interface of another Fokker-Planck term of f
        other = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings = lower.settings)
        other_interface = MomentumInterfaceTerm(:f, :runaway, other, re)
        @test_throws "its collision work at the interface of f reads the interface of another term" Runes2P._check_plasma_pairings([
            Equation(:f, TimeDerivative(), lower, other, interface, other_interface);
            Equation(:runaway, TimeDerivative(), re, interface, other_interface);
            row(crossing, KineticEnergyLossTerm(other; part = :sinks, region),
                InterfaceEnergyFluxTerm(interface), InterfaceCollisionWorkTerm(other_interface)); energy])
        # radial transport and knock-on secondaries on the grid of f have no row term: the
        # system refuses them
        transport = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
        add_rechester_rosenbluth_diffusion!(transport, 1e-3; major_radius = 6.2)
        @test_throws "Radial transport" check((term, FokkerPlanckTerm(:f, transport)), sinks, outflow)
        # the knock-ons of the fluid runaways above the grid bring their energy into W_tot: the
        # row takes one energy term of this very source, with a positive coefficient and no
        # region
        rp = RosenbluthPutvinskiSourceTerm(:f, grid)
        @test check((term, rp), sinks, outflow, KineticEnergyChangeTerm(rp)) === nothing
        @test_throws "add KineticEnergyChangeTerm(source)" check((term, rp), sinks, outflow)
        for coefficient in (0, -1)
            @test_throws "give it a positive coefficient" check(
                (term, rp), sinks, outflow, KineticEnergyChangeTerm(rp; coefficient))
        end
        @test_throws "give it no region" check((term, rp), sinks, outflow,
            KineticEnergyChangeTerm(rp; region = MomentumRegion(0.1)))
        @test_throws "more than once" check((term, rp), sinks, outflow,
            KineticEnergyChangeTerm(rp), KineticEnergyChangeTerm(rp))
        # the energy term of a copy of the source built from the same arguments: the identity
        # check refuses it before the row check
        @test_throws "not a term of the system" check((term, rp), sinks, outflow,
            KineticEnergyChangeTerm(RosenbluthPutvinskiSourceTerm(:f, grid)))
        # two sources in the equation of f: each needs its own energy term, matched by the
        # identity of the source, not by the distribution
        rp_low = RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff = 0.02)
        @test check((term, rp, rp_low), sinks, outflow, KineticEnergyChangeTerm(rp),
            KineticEnergyChangeTerm(rp_low)) === nothing
        @test_throws "add KineticEnergyChangeTerm(source)" check(
            (term, rp, rp_low), sinks, outflow, KineticEnergyChangeTerm(rp))
        @test_throws "add KineticEnergyChangeTerm(source)" check(
            (term, rp, rp_low), sinks, outflow, KineticEnergyChangeTerm(rp_low))
        # the energy term of the source of a distribution outside W_tot: its knock-ons are not
        # in W_tot, so the row refuses the term
        rp_g = RosenbluthPutvinskiSourceTerm(:g, grid)
        outside(row_terms...) = Runes2P._check_plasma_pairings([Equation(:f, TimeDerivative(), term, rp);
                                                                Equation(:g, TimeDerivative(), rp_g);
                                                                row(row_terms...); energy])
        @test outside(sinks, outflow, KineticEnergyChangeTerm(rp)) === nothing
        @test_throws "whose distribution is not in the total energy" outside(
            sinks, outflow, KineticEnergyChangeTerm(rp), KineticEnergyChangeTerm(rp_g))
        @test_throws "whose distribution is not in the total energy" outside(
            sinks, outflow, KineticEnergyChangeTerm(rp_g))
        # a prescribed loss rate and a Maxwellian source move electrons with their energy out
        # of W_hot and into it, which no row term carries
        @test_throws "prescribed loss rate is refused" check(
            (term, LossRateTerm(:f, [1e3, 1e3])), sinks, outflow)
        @test_throws "Maxwellian source is refused" check(
            (term, MaxwellianSourceTerm(:f, grid, T0; target_density = n, rate = 1e5)),
            sinks, outflow)
        # a LinearTerm of a kinetic unknown moves electrons into f or out of it, the same
        # channel as the loss rate. With a zero coefficient it moves none.
        @test_throws "LinearTerm of f in the equation of f" check(
            (term, LinearTerm(:f; coefficient = -1e3)), sinks, outflow)
        @test_throws "LinearTerm with a nonzero coefficient is refused" check(
            (term, LinearTerm(:g; coefficient = [0.0, 1e3])), sinks, outflow)
        @test check((term, LinearTerm(:f; coefficient = 0.0)), sinks, outflow) === nothing
        @test check((term, LinearTerm(:g; coefficient = [0.0, 0.0])), sinks, outflow) ===
              nothing
        # in an algebraic equation of f a LinearTerm of f is the closure of the row
        @test Runes2P._check_plasma_pairings([Equation(:f, Algebraic(), LinearTerm(:g),
                                                  LinearTerm(:f; coefficient = -1));
                                              row(); energy]) === nothing
        # a system without the total energy needs none of these row terms
        @test Runes2P._check_plasma_pairings([Equation(
            :f, TimeDerivative(), term, FokkerPlanckTerm(:f, transport))]) === nothing
        @test Runes2P._check_plasma_pairings([Equation(
            :f, TimeDerivative(), term, LossRateTerm(:f, [1e3, 1e3]),
            MaxwellianSourceTerm(:f, grid, T0; target_density = n, rate = 1e5),
            LinearTerm(:f; coefficient = -1e3))]) === nothing

        # Neither part of the loss takes a flux through p_max, whatever the boundary of the
        # source: the energy of the electrons that cross it is the boundary term's alone.
        # A tail at the end of the grid, where the flux through p_max is large.
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e16),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19)])
        u = zeros(length(layout))
        m = momentum_grid(grid)
        for ir in 1:2
            kinetic_view(u, layout, 1, ir) .= [1e16 * exp(-((p - 0.38) / 0.05)^2)
                                               for p in m.p_centers, _ in 1:2]
        end
        unknown_values(u, layout, 2) .= T0
        unknown_values(u, layout, 3) .= n
        loss(boundary, part) = Runes2P.rhs!(zeros(length(u)),
            KineticEnergyLossTerm(fp(boundary); part, region = nothing),
            u, layout, 2, 0.0)[Runes2P.unknown_range(layout, 2, 1)]
        for part in (:sinks, :free_electrons)
            reference = loss(:interface, part)
            @test only(reference) != 0
            for boundary in (:f_zero, :constant_flux)
                @test loss(boundary, part) == reference
            end
        end
        # the electron-electron operator of the source carries electrons through p_max here,
        # so the free-electron part of a source at :constant_flux would differ with the flux
        source = fp(:constant_flux)
        ee = electron_electron_operator(
            grid, Runes2P.plasma_values(plasma, u, layout, 0.0), source.settings;
            scheme = source.operator.p_scheme, p_upper_boundary = :constant_flux)
        through_p_max = MomentumBoundaryFluxTerm(:f, ee; weight = :energy)
        @test only(Runes2P.rhs!(zeros(length(u)), through_p_max, u, layout, 2, 0.0)[Runes2P.unknown_range(
            layout, 2, 1)]) != 0
    end

    @testset "radial transport set after the build under the total energy" begin
        # An operator of radial transport built by hand declares radial transport and holds no
        # coefficients when the system is built, so the system accepts it. A
        # RechesterRosenbluthUpdate sets its coefficients in the run, and the run refuses it
        # while the distribution is in the total energy, as the system refuses transport set
        # before it is built. Coefficients set by hand after the build are refused by the run.
        T0, n = 100.0, 1e19
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(20, 0.0, 0.4), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(2, 0.5)))
        base = PlasmaState(;
            cold_electron_temperature = [T0, T0], ions = [IonSpecies("D", 1, 1, [n, n])],
            electric_field = zeros(2), magnetic_field = zeros(2))
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        fp = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base),
            settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect))
        transport = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
        @test transport.radial_transport[] && all(iszero, transport.diffusion_r_r)
        update = RechesterRosenbluthUpdate(transport, t -> 1e-3; major_radius = 6.2)
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        equations = [Equation(:f, TimeDerivative(), fp, FokkerPlanckTerm(:f, transport)),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                PrescribedValueTerm([0.0, 0.0]), KineticEnergyLossTerm(fp; part = :sinks, region),
                MomentumBoundaryFluxTerm(:f, fp; weight = :energy, coefficient = -1)),
            density_row([n, n]), thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        u0 = zeros(length(layout))
        for ir in 1:2
            kinetic_view(u0, layout, 1, ir) .= [n * exp(-p^2 / (2theta))
                                                for p in momentum_grid(grid).p_centers,
            _ in 1:2]
        end
        unknown_values(u0, layout, 2) .= T0
        unknown_values(u0, layout, 3) .= n
        # accepted when built: the operator moves nothing yet
        system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
        @test system isa EquationSystem
        @test_throws "sets the radial transport of f" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps(), updates = (update,))
        # the same coefficients set by hand on the built system: the run refuses them before
        # its first step, also when it resumes, and the system refuses them when it is built
        add_rechester_rosenbluth_diffusion!(transport, 1e-3; major_radius = 6.2)
        @test_throws "Radial transport of f is refused" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps())
        @test_throws "Radial transport of f is refused" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps(),
            resume = joinpath(mktempdir(), "missing.h5"))
        @test_throws "Radial transport of f is refused" EquationSystem(
            layout, equations; time_scale = 1e-7, initial_state = u0)
    end

    @testset "a cold plasma under a hot tail" begin
        # A frozen tail with a hundred times the energy of the cold electrons above a fixed
        # momentum, and a prescribed loss P of the cold electrons: T(t) = T₀ − P t/(3/2 e n).
        # The temperature follows from W_tot − W_hot; its error stays near the tolerance
        # times W_tot/W_cold, since the difference amplifies a relative error of W_tot by
        # that ratio.
        T0, n, n_hot = 100.0, 1e19, 1e19
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(200, 0.0, 1.0), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 0.5)))
        m = momentum_grid(grid)
        hot = [exp(-p^2 / (1 + sqrt(1 + p^2)) / (1e4 / Constants.ELECTRON_REST_ENERGY))
               for p in m.p_centers, _ in 1:2, _ in 1:1]
        hot .*= n_hot / density(grid, hot)[1]
        region = MomentumRegion(0.05)
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e23)])
        P = THERMAL_ENERGY_FACTOR * n * 50 / 1e-3     # half the temperature in 1 ms
        equations = [Equation(:f, TimeDerivative(), LinearTerm(:f; coefficient = 0.0)),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                PrescribedValueTerm([-P])),
            density_row([n]), thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        u0 = zeros(length(layout))
        kinetic_array(u0, layout, 1) .= hot
        unknown_values(u0, layout, 2) .= T0
        unknown_values(u0, layout, 3) .= n
        # the temperature is determined through W_tot ← W_cold ← T although the threshold is fixed
        system = EquationSystem(layout, equations; initial_state = u0)
        times = collect(range(0.0, 1e-3; length = 11))
        solution = integrate(system, u0, times; stepper = FixedSteps())
        W = solution.states[:, 1]
        @test unknown_values(W, layout, 5)[1] / unknown_values(W, layout, 4)[1] > 90
        for (k, t) in enumerate(times)
            @test unknown_values(solution.states[:, k], layout, 2)[1] ≈
                  T0 - P * t / (THERMAL_ENERGY_FACTOR * n) rtol = 1e-7
        end
        # without the chain, nothing would determine the temperature: its row holds the
        # derivative of an energy that does not read it
        unrelated = [equations[1],
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                PrescribedValueTerm([-P])),
            density_row([n]), Equation(:cold_thermal_energy, Algebraic(),
                PrescribedValueTerm([1.0]), minus(:cold_thermal_energy)),
            total_energy_equation(:f, grid; region)]
        @test_throws "nothing determines it" EquationSystem(layout, unrelated; initial_state = u0)
    end

    @testset "the energy of the knock-ons of fluid runaways" begin
        # A frozen tail: the equation of f holds the Rosenbluth-Putvinski source alone, with
        # prescribed n_re and n_tot, and the row of the temperature the energy P of its
        # knock-ons. With the hot region below the cutoff every knock-on joins W_hot, so the
        # cold temperature stays. With the region above p_max, W_hot = 0 and the cold
        # electrons take P: T(t) = T₀ + P t/(3/2 e n).
        T0, n, n_re, n_tot = 100.0, 1e19, 3e16, 1e20
        p_cutoff = 0.04
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(40, 0.0, 0.8), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 0.5)))
        m = momentum_grid(grid)
        source = RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff, electric_field = nothing)
        # P = n_re n_tot Σ_i mₑc²(γ_i − 1) R_i over the cells above the cutoff, a face here, each
        # in the pitch cell ξ > 0 of the two
        P = n_re * n_tot *
            sum(Constants.ELECTRON_REST_ENERGY * e * (sqrt(1 + m.p_centers[i]^2) - 1) *
                rosenbluth_putvinski_rate(m.p_faces[i], m.p_faces[i + 1])
            for i in 1:40 if m.p_faces[i] >= p_cutoff)
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e16),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:runaway_density; scale = 1e16),
                RadialUnknown(:total_electron_density; scale = 1e20)])
        prescribed(name, value) = Equation(name, Algebraic(), PrescribedValueTerm([value]), minus(name))
        function knockon_run(region)
            equations = [Equation(:f, TimeDerivative(), source),
                prescribed(:runaway_density, n_re), prescribed(:total_electron_density, n_tot),
                Equation(
                    :cold_electron_temperature, TimeDerivative(:total_electron_energy),
                    KineticEnergyChangeTerm(source)),
                density_row([n]), thermal_energy_equation(), total_energy_equation(:f, grid; region)]
            u0 = zeros(length(layout))
            unknown_values(u0, layout, 2) .= T0
            unknown_values(u0, layout, 3) .= n
            unknown_values(u0, layout, 6) .= n_re
            unknown_values(u0, layout, 7) .= n_tot
            system = EquationSystem(layout, equations; initial_state = u0)
            times = collect(range(0.0, 1e-3; length = 11))
            solution = integrate(system, u0, times; stepper = FixedSteps())
            temperatures = [unknown_values(solution.states[:, k], layout, 2)[1]
                            for k in 1:11]
            return times, temperatures
        end
        # every knock-on in W_hot: the limit 0.02 lies a cell below the cutoff 0.04, the lower
        # face of the first cell of the source, and gives an envelope of 1 in every cell of
        # the source. Measured: |T − T₀|/T₀ of at most 2.8e-16, against the rise (T − T₀)/T₀
        # of 9.1e-2 of the second run.
        times, temperatures = knockon_run(MomentumRegion(0.02))
        @test maximum(abs.(temperatures .- T0)) / T0 < 1e-12
        # no knock-on in W_hot: the cold electrons take their whole energy. Measured:
        # |T − T_exact|/T_exact of at most 9.1e-16.
        times, temperatures = knockon_run(MomentumRegion(0.9))
        exact = T0 .+ P .* times ./ (THERMAL_ENERGY_FACTOR * n)
        @test exact[end] - T0 > 0.05 * T0
        @test maximum(abs.(temperatures .- exact) ./ exact) < 1e-12
    end

    @testset "a driven fully kinetic plasma" begin
        # One hot-tail grid with the bulk in f, a prescribed field, the cold density from the
        # free electrons minus the hot ones, the particle source that keeps the bulk, DREAM's
        # corrected ohmic current, and the total energy of the electrons. Once f
        # carries its ohmic current, the electrons gain σ E² with Braams and Karney's σ, within
        # the accuracy of DREAM's fit of the conductivity of the kinetic operator.
        T0, n, E0 = 100.0, 1e20, 1.0
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(80, 0.0, 10 * sqrt(2theta)), uniform_xi_faces(6)),
            RadialGrid(uniform_r_faces(1, 0.5)))
        base = PlasmaState(; cold_electron_temperature = [T0],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [0.0],
            magnetic_field = [0.0], cold_electron_density = [n])
        settings = CollisionSettings(mode = :full, screening = :complete)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        plasma = PlasmaFromUnknowns(base)
        fp = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma, settings, electric_field = :electric_field)
        correction = ConductivityTerm(:electric_field,
            ConductivityModel(PlasmaFromUnknowns(base; density = nothing);
                kinetic = :dream_fit))
        current(; kwargs...) = CurrentMomentTerm(:f, grid; pitch_rule = :cell_center, kwargs...)
        layout = UnknownLayout(grid.radial_grid,
            [
                RadialUnknown(:electric_field), RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e20), RadialUnknown(
                    :hot_electron_density; scale = 1e10),
                RadialUnknown(:ohmic_current_density; scale = 1e6), RadialUnknown(
                    :hot_electron_current_density; scale = 1e2),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e22),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e22),
                RadialUnknown(:particle_source; scale = 1e22), KineticUnknown(:f, grid; scale = 1e20)])
        equations = [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm([E0]), minus(:electric_field)),
            Equation(:hot_electron_density, Algebraic(),
                DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
            Equation(:cold_electron_density, Algebraic(),
                PrescribedValueTerm([n]), minus(:hot_electron_density),
                minus(:cold_electron_density)),
            Equation(:particle_source, Constraint(),
                DensityMomentTerm(:f, grid), minus(:cold_electron_density),
                minus(:hot_electron_density); scale = n),
            Equation(:f, TimeDerivative(), fp,
                ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature)),
            Equation(:hot_electron_current_density, Algebraic(),
                current(; region), minus(:hot_electron_current_density)),
            Equation(:ohmic_current_density, Algebraic(), current(),
                minus(:hot_electron_current_density), correction,
                minus(:ohmic_current_density)),
            thermal_energy_equation(), total_energy_equation(:f, grid; region),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                ProductTerm(:electric_field, :ohmic_current_density), ProductTerm(
                    :electric_field, :hot_electron_current_density),
                KineticEnergyLossTerm(fp; part = :sinks, region), MomentumBoundaryFluxTerm(
                    :f, fp; weight = :energy, coefficient = -1))]
        u0 = zeros(length(layout))
        f0 = [exp(-p^2 / (1 + sqrt(1 + p^2)) / theta)
              for p in momentum_grid(grid).p_centers, _ in 1:6, _ in 1:1]
        f0 .*= n / density(grid, f0)[1]
        kinetic_array(u0, layout, unknown_id(layout, :f)) .= f0
        unknown_values(u0, layout, unknown_id(layout, :electric_field)) .= E0
        unknown_values(u0, layout, unknown_id(layout, :cold_electron_temperature)) .= T0
        unknown_values(u0, layout, unknown_id(layout, :cold_electron_density)) .= n
        system = EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0)
        times = collect(range(0.0, 2e-5; length = 21))     # about 4 collision times per step
        solution = integrate(system, u0, times; stepper = FixedSteps())
        value(k, name) = only(unknown_values(solution.states[:, k], layout, unknown_id(layout, name)))
        rate = (value(21, :total_electron_energy) - value(11, :total_electron_energy)) /
               (times[21] - times[11])
        T_mid = value(16, :cold_electron_temperature)
        @test value(21, :cold_electron_temperature) > T0
        @test rate ≈ braams_karney_conductivity(n, T_mid, 1.0) * E0^2 rtol = 5e-2
        # The gap between the work of the field on the hot region, which W_hot gains in the
        # discrete equation of f, and E j_hot, which the row of the temperature charges: with
        # the total energy as the unknown the difference heats the cold electrons spuriously.
        # Here the hot region holds the tail above 5 thermal momenta.
        field = AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        add_electric_field!(field,
            PlasmaState(;
                cold_electron_temperature = [T0], ions = [IonSpecies("D", 1, 1, [n])],
                electric_field = [E0], magnetic_field = [0.0]))
        f_end = kinetic_array(solution.states[:, 21], layout, unknown_id(layout, :f))
        df = zeros(size(f_end))
        Runes2P.apply!(df, field, f_end)
        # the energy moment over the hot region of the change of f by the field, at the
        # temperature of the end state
        changed = copy(solution.states[:, 21])
        kinetic_array(changed, layout, unknown_id(layout, :f)) .= df
        row = unknown_id(layout, :hot_electron_density)
        work = only(unknown_values(
            Runes2P.rhs!(zeros(length(changed)),
                EnergyMomentTerm(:f, grid; region), changed, layout, row,
                times[end]),
            layout,
            row))
        ej_hot = E0 * value(21, :hot_electron_current_density)
        # measured: 1.11 W/m³ of work against 0.023 W/m³ of E j_hot, 7e-7 of the heating rate.
        # Most of the work is the energy of the electrons that the field moves across the
        # threshold, E_k(p₀) Γ, which the total energy correctly takes from the cold electrons.
        @test abs(work - ej_hot) < 1e-2 * abs(rate)
    end

    @testset "the energy budgets of two grids across the interface" begin
        # A hot-tail grid to p = 0.6 and a runaway grid above it on two radii, a prescribed field,
        # argon in partial screening, bremsstrahlung and synchrotron loss, and the knock-ons of a
        # Rosenbluth-Putvinski source on the runaway grid. The schemes are central, so that the
        # energy change of each operator is the sum of the changes of its parts (the field, the
        # electron-electron collisions, the sinks), and the flux through the face is that of its
        # parts at the same interpolated face value. After one step the energy of each grid
        # changes by its channels, written out here with operators of their own.
        T0, n = 100.0, 1e19
        energy_of(p) = Constants.ELECTRON_REST_ENERGY * e * p^2 / (1 + sqrt(1 + p^2))
        radii = RadialGrid(uniform_r_faces(2, 0.5))
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(30, 0.0, 0.6), uniform_xi_faces(4)), radii)
        upper = runaway_grid(hot, uniform_p_faces(20, 0.6, 5.0)[2:end], uniform_xi_faces(6))
        mixed = PlasmaState(; cold_electron_temperature = [T0, 2T0],
            ions = [IonSpecies("D", 1, 1, [n, 2n]), IonSpecies("Ar", 18, 2, [1e18, 3e18])],
            electric_field = zeros(2), magnetic_field = fill(2.5, 2))
        partial = CollisionSettings(mode = :full, screening = :partial)
        plasma = PlasmaFromUnknowns(mixed)
        lower = FokkerPlanckTerm(:f,
            AdvectionDiffusion(hot; scheme = :central, p_upper_boundary = :interface);
            plasma, settings = partial, electric_field = :electric_field)
        re = FokkerPlanckTerm(:runaway,
            AdvectionDiffusion(upper; scheme = :central, p_lower_boundary = :interface,
                p_upper_boundary = :f_zero);
            plasma, settings = partial, electric_field = :electric_field)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        source = RosenbluthPutvinskiSourceTerm(:runaway, upper; p_cutoff = 0.6, electric_field = nothing)
        layout = UnknownLayout(radii,
            [KineticUnknown(:f, hot; scale = 1e16), KineticUnknown(:runaway, upper; scale = 1e14),
                RadialUnknown(:electric_field; scale = 10.0),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:runaway_density; scale = 1e16),
                RadialUnknown(:total_electron_density; scale = 1e20), RadialUnknown(:probe)])
        field = [30.0, 20.0]
        prescribed(name, values) = Equation(name, Algebraic(), PrescribedValueTerm(values), minus(name))
        u0 = zeros(length(layout))
        hot_m, upper_m = momentum_grid(hot), momentum_grid(upper)
        for ir in 1:2
            kinetic_view(u0, layout, 1, ir) .= [1e16 * exp(-p / 0.15) for p in hot_m.p_centers, _ in 1:4]
            kinetic_view(u0, layout, 2, ir) .= [2e14 * exp(-p / 2) for p in upper_m.p_centers, _ in 1:6]
        end
        values = Dict(:electric_field => field, :cold_electron_temperature => [T0, 2T0],
            :cold_electron_density => [n, 2n], :runaway_density => [1e16, 2e16],
            :total_electron_density => [1.4e20, 2.4e20], :probe => [0.0, 0.0])
        for (name, v) in values
            unknown_values(u0, layout, unknown_id(layout, name)) .= v
        end
        system = EquationSystem(layout,
            [Equation(:f, TimeDerivative(), lower, interface),
                Equation(:runaway, TimeDerivative(), re, interface, source),
                [prescribed(name, v) for (name, v) in values]...]; initial_state = u0)
        dt = 1e-7
        solution = integrate(system, u0, [0.0, dt]; stepper = FixedSteps())
        u = solution.states[:, 2]
        F = zeros(length(u))
        rhs!(F, u, system, dt)
        probe = unknown_id(layout, :probe)
        value(term) = collect(unknown_values(
            Runes2P.rhs!(zeros(length(u)), term, u, layout, probe, dt), layout, probe))
        # the energy moment of a change `df` of a distribution on its grid, per radius
        moment(grid, df) = [sum(Runes2P._energy_weights(grid, ir) .* view(df, :, :, ir)) for ir in 1:2]
        # the change of a distribution by the field alone, without a flux through the face or p_max
        function field_work(grid, boundaries)
            op = AdvectionDiffusion(grid; scheme = :central, boundaries...)
            Runes2P.set_unit_field_coefficients!(op)
            id = unknown_id(layout, grid === hot ? :f : :runaway)
            f = kinetic_array(u, layout, id)
            df = zeros(size(f))
            Runes2P.apply!(df, op, f; electric_field = field)
            return moment(grid, df)
        end
        change(name, grid) = moment(grid, kinetic_array(F, layout, unknown_id(layout, name)))
        # over the step, backward Euler: the change of the energy of a grid is the rate at its end
        W(state, name, grid) = moment(grid, kinetic_array(state, layout, unknown_id(layout, name)))
        for (name, grid) in ((:f, hot), (:runaway, upper))
            @test (W(u, name, grid) .- W(u0, name, grid)) ./ dt ≈ change(name, grid) rtol = 1e-8
        end

        # the runaway grid alone: the field, the heating of the cold electrons, the sinks, the
        # outflow through its p_max, the influx through the interface and the knock-ons
        field_re = field_work(upper, (p_lower_boundary = :interface, p_upper_boundary = :interface))
        heating_re = value(free_electron_energy_transfer_term(re))
        sinks_re = value(KineticEnergyLossTerm(re; part = :sinks, coefficient = 1))
        outflow = value(MomentumBoundaryFluxTerm(:runaway, re; weight = :energy))
        influx = change(:runaway, upper) .- moment(upper,
            kinetic_array(Runes2P.rhs!(zeros(length(u)), re, u, layout, 2, dt) .+
                          Runes2P.rhs!(zeros(length(u)), source, u, layout, 2, dt), layout, 2))
        crossing = [interface_flux(interface, u, layout, ir) for ir in 1:2]
        @test maximum(abs, influx .- energy_of(upper_m.p_centers[1]) .* crossing) <=
              1e-12 * maximum(abs, change(:runaway, upper))
        knockons = value(KineticEnergyChangeTerm(source))
        channels_re = field_re .- heating_re .- sinks_re .- outflow .+ influx .+ knockons
        scale_re = maximum(abs, hcat(field_re, heating_re, sinks_re, outflow, influx, knockons))
        @test maximum(abs, change(:runaway, upper) .- channels_re) <= 1e-12 * scale_re

        # the two grids together: the electrons that cross change the energy by the step between
        # the two cells at the face, which the field, the sinks and the electron-electron
        # collisions pay for their parts of the flux, the last from the cold electrons
        field_f = field_work(hot, (p_upper_boundary = :interface,))
        heating_f = value(free_electron_energy_transfer_term(lower))
        sinks_f = value(KineticEnergyLossTerm(lower; part = :sinks, coefficient = 1))
        op = Runes2P._operator_at(lower, u, layout, dt)
        ee = electron_electron_operator(hot, op.coefficient_source[].plasma, partial;
            scheme = :central, p_upper_boundary = :interface)
        # the flux through the face of the coefficients (A, D) of lower pitch cell j, at the face
        # value of the central interpolation
        function part(A_of, D_of, ir)
            distance, weight_left = Runes2P._interface_distance(interface, ir)
            f_l = kinetic_view(u, layout, 1, ir)
            f_u = kinetic_view(u, layout, 2, ir)
            overlap = interface.overlaps[ir]
            total = 0.0
            for j in 1:4, s in overlap.left_ranges[j]

                fl, fr = f_l[30, j], f_u[1, overlap.right_cells[s]]
                face = weight_left * fl + (1 - weight_left) * fr
                gamma = A_of(j) * face - D_of(j) * (fr - fl) / distance
                total += hot_m.xi_widths[j] * overlap.left_fractions[s] *
                         op.jacobian_p_faces[31, j, ir] * gamma
            end
            return total / radii.spatial_jacobian[ir]
        end
        energy_step = energy_of(upper_m.p_centers[1]) - energy_of(hot_m.p_centers[end])
        face_field = [energy_step * part(j -> field[ir] * op.unit_field_advection_p[31, j, ir], j -> 0.0, ir)
                      for ir in 1:2]
        face_sinks = [energy_step * part(j -> op.advection_p[31, j, ir] - ee.advection_p[31, j, ir],
                          j -> op.diffusion_p_p[31, j, ir] - ee.diffusion_p_p[31, j, ir], ir)
                      for ir in 1:2]
        # the term: the energy that the cold electrons gain, the negative of the work that the
        # collisions do on the electrons that cross
        cold_work = value(InterfaceCollisionWorkTerm(interface))
        both = change(:f, hot) .+ change(:runaway, upper)
        without = field_f .+ field_re .+ face_field .- heating_f .- heating_re .- sinks_f .-
                  sinks_re .+ face_sinks .- outflow .+ knockons
        scale = maximum(abs, hcat(field_f, field_re, face_field, heating_f, heating_re, sinks_f,
            sinks_re, face_sinks, outflow, knockons))
        miss = maximum(abs, both .- without)
        closure = maximum(abs, both .- (without .- cold_work))
        @info "the energy of two grids across the interface: misses by $(miss / scale) of the largest channel without the work of the collisions, $(closure / scale) with it; the work is $(cold_work) W/m³, the influx $(influx) W/m³"
        # measured: closed to 6.0e-16 of the largest channel (the field) with the term. Without
        # it the budget misses by the work, 71 and 146 W/m³ at the two radii, 1.9e-5 of the
        # largest channel and 27 % and 20 % of the energy influx through the interface.
        @test closure <= 1e-12 * scale
        @test miss > 1e-6 * scale
        @test all(cold_work .> 0)
    end

    @testset "the total energy of two grids conserved across the interface" begin
        # A tail on the hot-tail grid and a runaway grid above p = 0.6, no field, collisions with
        # the free electrons only (complete screening, no radiation): the electrons exchange
        # energy with the cold electrons alone, so W_tot of the row and the energy of the runaway
        # grid together are conserved. The friction pulls electrons down through the face, and
        # each gives the cold electrons the energy step between the two cells at the face, the
        # work of the collisions.
        T0, n = 100.0, 1e19
        radii = RadialGrid(uniform_r_faces(1, 0.5))
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(60, 0.0, 0.6), uniform_xi_faces(2)), radii)
        upper = runaway_grid(hot, uniform_p_faces(10, 0.6, 3.0)[2:end], uniform_xi_faces(3))
        hot_m, upper_m = momentum_grid(hot), momentum_grid(upper)
        base = PlasmaState(; cold_electron_temperature = [T0], ions = [IonSpecies("D", 1, 1, [n])],
            electric_field = [0.0], magnetic_field = [0.0])
        settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect)
        plasma = PlasmaFromUnknowns(base)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        lower = FokkerPlanckTerm(:f,
            AdvectionDiffusion(hot; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings, synchrotron = false)
        re = FokkerPlanckTerm(:runaway,
            AdvectionDiffusion(upper; scheme = :exponential_fitting, p_lower_boundary = :interface,
                p_upper_boundary = :outflow);
            plasma, settings, synchrotron = false)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        layout = UnknownLayout(radii,
            [KineticUnknown(:f, hot; scale = 1e19), KineticUnknown(:runaway, upper; scale = 1e16),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        u0 = zeros(length(layout))
        bulk = [exp(-p^2 / (1 + sqrt(1 + p^2)) / theta) for p in hot_m.p_centers, _ in 1:2]
        bulk .*= n / density(hot, reshape(bulk, size(bulk)..., 1))[1]
        tail = [1e16 * exp(-p / 0.3) for p in hot_m.p_centers, _ in 1:2]
        kinetic_view(u0, layout, 1, 1) .= bulk .+ tail
        kinetic_view(u0, layout, 2, 1) .= [1e16 * exp(-p / 0.3) for p in upper_m.p_centers, _ in 1:3]
        unknown_values(u0, layout, 3) .= T0
        unknown_values(u0, layout, 4) .= n
        function run(work_terms...)
            equations = [Equation(:f, TimeDerivative(), lower, interface),
                Equation(:runaway, TimeDerivative(), re, interface),
                Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                    InterfaceEnergyFluxTerm(interface), work_terms...,
                    free_electron_energy_transfer_term(re)),
                density_row([n]), thermal_energy_equation(), total_energy_equation(:f, hot; region)]
            system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
            times = collect(range(0.0, 5e-7; length = 6))
            solution = integrate(system, u0, times; stepper = FixedSteps())
            # W_tot and the energy of the runaway grid, relative to W_tot at the start
            totals = map(1:6) do k
                state = solution.states[:, k]
                W_re = sum(Runes2P._energy_weights(upper, 1) .* kinetic_view(state, layout, 2, 1))
                unknown_values(state, layout, 6)[1] + W_re
            end
            flux = interface_flux(interface, solution.states[:, end], layout, 1)
            return (totals .- totals[1]) ./ totals[1], flux
        end
        # without the work the system refuses the row
        @test_throws "add InterfaceCollisionWorkTerm" run()
        drift, flux = run(InterfaceCollisionWorkTerm(interface))
        @test flux < 0
        # a coefficient of −1e-30, which the check accepts, stands for a row without the work
        missed, _ = run(InterfaceCollisionWorkTerm(interface; coefficient = -1e-30))
        @info "the total energy of two grids: drifts by $(maximum(abs, drift)) of W_tot with the work of the collisions at the interface, by $(maximum(abs, missed)) without it"
        # measured: a drift of 1.2e-16 of W_tot with the work, 3.9e-7 without it
        @test maximum(abs, drift) < 1e-12
        @test maximum(abs, missed) > 1e3 * max(maximum(abs, drift), 1e-15)
    end

    @testset "DREAM's fluid heating by runaways" begin
        # e c E_c n_re with E_c = n e³ lnΛ/(4π ε₀² mₑc²): the friction of runaways at the speed of light
        T, n, n_re = 50.0, 2e19, 1e16
        plasma = PlasmaState(; cold_electron_temperature = [T],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [0.0],
            magnetic_field = [0.0])
        layout = UnknownLayout(RadialGrid(uniform_r_faces(1, 0.5)),
            [RadialUnknown(:cold_electron_temperature),
                RadialUnknown(:cold_electron_density),
                RadialUnknown(:runaway_density)])
        u = [T, n, n_re]
        term = RunawayFluidEnergyTransferTerm(plasma)
        Runes2P.validate(term, layout, 1)
        out = Runes2P.rhs!(zeros(3), term, u, layout, 1, 0.0)
        lnL = coulomb_logarithm(:electron_electron, :energy_dependent, 20.0, n, T)
        E_c = n * e^3 * lnL / (4π * Constants.VACUUM_PERMITTIVITY^2 * me * c^2)
        @test out[1] ≈ e * c * E_c * n_re rtol = 1e-6
        @test Runes2P._temperature_source(term) == (:unknown, :cold_electron_temperature)
        @test describe(term, Runes2P.DescriptionContext())["coulomb_logarithm"] ==
              "energy_dependent"
        @test_throws ArgumentError RunawayFluidEnergyTransferTerm(plasma; coulomb_logarithm = :other)
    end
end
