# The total energy of the electrons in the fully kinetic mode, an unknown W_tot = W_cold +
# W_hot, with W_hot the energy of the hot region of the hot-tail grid, whose time derivative
# the row of the temperature holds, so that T follows from W_cold = W_tot − W_hot: the
# transfer from a slowing-down tail to the cold electrons against the friction power of
# superthermal electrons, the temperature of a cold plasma under a hot tail a hundred times
# its energy, the chain that determines the temperature, the checks of the row with the
# losses that take no flux through p_max, the energy of radial transport in the total energy
# with its budget at each radius, and the energy of the knock-ons of fluid runaways. Radial
# transport set after the build, by an update or by hand, needs the energy term as well.
# Last, DREAM's fluid heating by runaways.
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

    @testset "transfer from a slowing-down tail ($scheme)" for scheme in (:exponential_fitting, :tcdf)
        # A bump of electrons at p = 0.3 above a 100 eV bulk, collisions with the free electrons
        # only (complete screening), no field: the energy the tail loses goes to the cold
        # electrons. Under the limiter TCDF the system accepts the sinks of the row
        # (KineticEnergyLossTerm), which are nearly zero here, and the budget closes as under
        # exponential fitting. The friction power of a superthermal electron on cold free electrons is
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
            AdvectionDiffusion(grid; scheme, p_upper_boundary = :f_zero);
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
        @test_throws "negative coefficient" two(
            crossing, InterfaceEnergyFluxTerm(interface; coefficient = 1), work)
        # the work of the collisions on the electrons that cross: required with the energy that
        # crosses, of the same interface, charged to the cold electrons
        @test_throws "add InterfaceCollisionWorkTerm" two(crossing, InterfaceEnergyFluxTerm(interface))
        @test_throws "gives the cold electrons the work that they pay" two(
            crossing, InterfaceEnergyFluxTerm(interface),
            InterfaceCollisionWorkTerm(interface; coefficient = 1))
        # the work of a copy of the interface: the identity check refuses it before the row check
        copy_of_interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        @test_throws "not a term of the system" two(
            crossing, InterfaceEnergyFluxTerm(interface),
            InterfaceCollisionWorkTerm(copy_of_interface))
        # the work of the interface of another Fokker-Planck term of f
        other = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings = lower.settings)
        other_interface = MomentumInterfaceTerm(:f, :runaway, other, re)
        f_both = Equation(:f, TimeDerivative(), lower, other, interface, other_interface)
        re_both = Equation(:runaway, TimeDerivative(), re, interface, other_interface)
        row_other = row(crossing, KineticEnergyLossTerm(other; part = :sinks, region),
            InterfaceEnergyFluxTerm(interface), InterfaceCollisionWorkTerm(other_interface))
        other_equations = [f_both, re_both, row_other, energy...]
        other_message = "collision work at the interface of f reads the interface of another term"
        @test_throws other_message Runes2P._check_plasma_pairings(other_equations)
        # two terms of the work of one interface count it twice
        @test_throws "count the work more than once" two(
            crossing, InterfaceEnergyFluxTerm(interface), work, work)
        # the work at the interface of a distribution outside W_tot: the electrons that cross
        # there are not in W_tot, so the row refuses the term
        g_lower = FokkerPlanckTerm(:g,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings = lower.settings)
        g_interface = MomentumInterfaceTerm(:g, :runaway, g_lower, re)
        g_equations = [Equation(:f, TimeDerivative(), term),
            Equation(:g, TimeDerivative(), g_lower, g_interface),
            Equation(:runaway, TimeDerivative(), re, g_interface)]
        @test Runes2P._check_plasma_pairings([g_equations; row(sinks, outflow); energy]) ===
              nothing
        g_row = row(sinks, outflow, InterfaceCollisionWorkTerm(g_interface))
        g_system = [g_equations; g_row; energy]
        outside_message = "reads the interface of g, whose distribution is not in the total energy"
        @test_throws outside_message Runes2P._check_plasma_pairings(g_system)
        # radial transport moves the energy of the electrons of f: the row takes one energy term
        # of this very Fokker-Planck term, with a positive coefficient and no region
        transport = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
        add_rechester_rosenbluth_diffusion!(transport, 1e-3; major_radius = 6.2)
        moving = FokkerPlanckTerm(:f, transport)
        transport_energy = KineticEnergyChangeTerm(moving)
        @test check((term, moving), sinks, outflow, transport_energy) === nothing
        @test_throws "add KineticEnergyChangeTerm(term)" check((term, moving), sinks, outflow)
        @test_throws "more than once" check(
            (term, moving), sinks, outflow, transport_energy, transport_energy)
        for coefficient in (0, -1)
            @test_throws "give it a positive coefficient" check(
                (term, moving), sinks, outflow,
                KineticEnergyChangeTerm(moving; coefficient))
        end
        # a region: refused when the term is built, and by the row for a term built otherwise
        @test_throws "takes no region" KineticEnergyChangeTerm(moving; region)
        with_region = KineticEnergyChangeTerm(:f, moving, 1.0, region, nothing)
        @test_throws "give it no region" check((term, moving), sinks, outflow, with_region)
        # the energy term of a copy of the transport term: the identity check refuses it
        @test_throws "not a term of the system" check((term, moving), sinks, outflow,
            KineticEnergyChangeTerm(FokkerPlanckTerm(:f, transport)))
        # a transport term that holds no coefficients yet needs no energy term, and takes one
        idle = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme = :central))
        @test check((term, idle), sinks, outflow) === nothing
        @test check((term, idle), sinks, outflow, KineticEnergyChangeTerm(idle)) ===
              nothing
        @test_throws "more than once" check((term, idle), sinks, outflow,
            KineticEnergyChangeTerm(idle), KineticEnergyChangeTerm(idle))
        # the energy term of the transport of a distribution outside W_tot
        g_moving = FokkerPlanckTerm(:g, transport)
        g_transport = [Equation(:f, TimeDerivative(), term, moving),
            Equation(:g, TimeDerivative(), g_moving)]
        g_energy = KineticEnergyChangeTerm(g_moving)
        g_row = row(sinks, outflow, transport_energy, g_energy)
        @test_throws "whose distribution is not in the total energy" Runes2P._check_plasma_pairings([g_transport;
                                                                                                     g_row;
                                                                                                     energy])
        f_row = row(sinks, outflow, transport_energy)
        @test Runes2P._check_plasma_pairings([g_transport; f_row; energy]) === nothing
        # a row of the energy of the cold electrons alone holds no distribution, so the energy
        # term of the transport would give the cold electrons the energy of f
        cold_row = Equation(
            :cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
            transport_energy)
        @test_throws "an energy of the cold electrons alone without a distribution" Runes2P._check_plasma_pairings([
            Equation(:f, TimeDerivative(), term, moving), cold_row, thermal_energy_equation()])
        # the Rechester-Rosenbluth heat transport next to the energy term of radial transport
        # moves the heat of the bulk twice, whatever its level: refused. A prescribed
        # diffusivity is allowed, and so is the Rechester-Rosenbluth heat transport without
        # the energy term, in a row of the total energy or of the cold thermal energy alone
        rr_heat = HeatTransportTerm(; delta_b_over_b = 1e-3, major_radius = 6.2)
        prescribed_heat = HeatTransportTerm(; diffusion = 1.0)
        @test_throws "the heat of the bulk would move twice" check(
            (term, moving), sinks, outflow, transport_energy, rr_heat)
        @test check((term, moving), sinks, outflow, transport_energy, prescribed_heat) ===
              nothing
        @test check((term,), sinks, outflow, rr_heat) === nothing
        cold_heat_row = Equation(
            :cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
            rr_heat)
        @test Runes2P._check_plasma_pairings([
            Equation(:f, TimeDerivative(), term, moving), cold_heat_row,
            thermal_energy_equation()]) === nothing
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

    @testset "the energy of radial transport in the total energy" begin
        # A hot bump on a 100 eV bulk at three radii of a nonuniform grid, collisions with the
        # free electrons alone (no sinks), the outflow at p_max, and radial transport with a
        # closed edge: Rechester-Rosenbluth diffusion and an inward advection. The row of the
        # total energy holds the energy term of the transport, so at each radius W_tot changes
        # by the step times the transport moment P of the end of the step, computed here from
        # apply! of the transport operator over the whole grid, less the outflow. P integrates
        # to zero over the volume, and so does the change of W_tot apart from the outflow.
        # Without the term, a coefficient of 1e-30 that the
        # check accepts, the row misses P at each radius, and the cold electrons pay it.
        T0, n = 100.0, 1e19
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(40, 0.0, 0.6), uniform_xi_faces(2)),
            RadialGrid([0.0, 0.15, 0.3, 0.5]))
        m = momentum_grid(grid)
        base = PlasmaState(;
            cold_electron_temperature = fill(T0, 3), ions = [IonSpecies("D", 1, 1, fill(n, 3))],
            electric_field = zeros(3), magnetic_field = zeros(3))
        region = MomentumRegion(0.1)
        settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect)
        fp() = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base), settings, synchrotron = false)
        function transport_operator(; filled = true, r_boundary = :zero_flux, r_scheme = :central)
            op = AdvectionDiffusion(grid; scheme = :central, r_scheme, r_boundary,
                cross_diffusion = false)
            if filled
                add_rechester_rosenbluth_diffusion!(op, 2e-3; major_radius = 6.2)
                op.advection_r .= -2e3
            end
            return op
        end
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        function equations_of(collisions, transport, moment_region, row_terms...)
            [Equation(:f, TimeDerivative(), collisions, transport),
                Equation(
                    :cold_electron_temperature, TimeDerivative(:total_electron_energy),
                    MomentumBoundaryFluxTerm(:f, collisions; weight = :energy, coefficient = -1),
                    row_terms...),
                density_row(fill(n, 3)), thermal_energy_equation(),
                total_energy_equation(:f, grid; region = moment_region)]
        end
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        u0 = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u0, layout, 1, ir) .= [n * exp(-p^2 / 2theta) +
                                                1e17 * (4 - ir) *
                                                exp(-((p - 0.3) / 0.05)^2)
                                                for p in m.p_centers, _ in 1:2]
        end
        unknown_values(u0, layout, 2) .= T0
        unknown_values(u0, layout, 3) .= n
        volumes = grid.radial_grid.spatial_jacobian .* grid.radial_grid.r_widths
        # the envelope of the hot region at radius ir: a fixed limit, or the threshold of the
        # temperature of the state u
        function envelope_at(moment_region, u, ir)
            moment_region.limit isa ThermalMomentum ||
                return momentum_envelope(moment_region, grid, ir)
            T = unknown_values(u, layout, 2)[ir]
            p0 = moment_region.limit.factor * sqrt(2T / Constants.ELECTRON_REST_ENERGY)
            return momentum_envelope!(zeros(40), moment_region, m, p0)
        end
        # the moment of the transport from apply!, per radius [W/m³], over the whole grid
        # (`nothing`) or over the hot region of `moment_region`
        function transport_moment(op, u, moment_region)
            df = apply!(zeros(size(grid)), op, kinetic_array(u, layout, 1))
            moments = zeros(3)
            for ir in 1:3, j in 1:2, i in 1:40
                envelope = moment_region === nothing ? ones(40) :
                           envelope_at(moment_region, u, ir)
                moments[ir] += envelope[i] * Runes2P._kinetic_energy(m.p_centers[i]) *
                               df[i, j, ir] * phase_space_jacobian(grid, ir)[i, j] *
                               m.p_widths[i] * m.xi_widths[j] /
                               grid.radial_grid.spatial_jacobian[ir]
            end
            return moments
        end
        values_of(u, k) = unknown_values(u, layout, k)
        dt = 1e-7
        function run(coefficient; updates = (), moment_region = region,
                temperatures = fill(T0, 3), heat = (), r_boundary = :zero_flux,
                r_scheme = :central)
            collisions = fp()
            op = transport_operator(; filled = isempty(updates), r_boundary, r_scheme)
            transport = FokkerPlanckTerm(:f, op)
            equations = equations_of(collisions, transport, moment_region,
                KineticEnergyChangeTerm(transport; coefficient), heat...)
            initial = copy(u0)
            unknown_values(initial, layout, 2) .= temperatures
            system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = initial)
            update_list = [RechesterRosenbluthUpdate(op, t -> 2e-3; major_radius = 6.2)
                           for _ in updates]
            if !isempty(updates)
                op.advection_r .= -2e3
            end
            solution = integrate(
                system, initial, [0.0, dt]; stepper = FixedSteps(), updates = Tuple(update_list))
            start, finish = solution.states[:, 1], solution.states[:, 2]
            rows = [first(Runes2P.unknown_range(layout, 2, ir)) for ir in 1:3]
            outflow = Runes2P.rhs!(
                zeros(length(finish)), equations[2].terms[1], finish, layout, 2, 0.0)[rows]
            return (; op,
                start, finish, outflow, moment = transport_moment(op, finish, nothing),
                hot = transport_moment(op, finish, moment_region))
        end
        with = run(1.0)
        # the transport moves energy at each radius, and its moment integrates to zero.
        # Measured: the largest |P| is 9.4e6 W/m³, so a step moves 0.26 % of W_tot at the
        # inner radius
        scale = maximum(abs, with.moment)
        @test scale > 1e-3 * maximum(values_of(with.start, 5)) / dt
        @test abs(sum(volumes .* with.moment)) < 1e-13 * sum(volumes .* abs.(with.moment))
        # each radius: the change of W_tot is that of the transport and the outflow. Measured:
        # at most 2.0e-14 of the largest |P|, and |∫P| of 1.5e-16 of ∫|P|
        change = (values_of(with.finish, 5) .- values_of(with.start, 5)) ./ dt
        @test maximum(abs.(change .- (with.moment .+ with.outflow))) < 1e-12 * scale
        @test abs(sum(volumes .* (change .- with.outflow))) <
              1e-12 * sum(volumes .* abs.(with.moment))
        # without the term the row misses the transport at each radius, which the cold
        # electrons pay: W_cold changes by minus the step times the moment, against the run
        # with it. Measured: 9.9e-7 relative, from the collisions at the two temperatures
        without = run(1e-30)
        missed = (values_of(without.finish, 5) .- values_of(without.start, 5)) ./ dt
        @test maximum(abs.(missed .- without.outflow)) < 1e-12 * scale
        paid = (values_of(without.finish, 4) .- values_of(with.finish, 4)) ./ dt
        @test paid ≈ -with.moment rtol = 1e-5
        # the coefficients set in the run by an update: the same budget
        updated = run(1.0; updates = (:rechester_rosenbluth,))
        change = (values_of(updated.finish, 5) .- values_of(updated.start, 5)) ./ dt
        @test maximum(abs.(updated.moment .- with.moment)) < 1e-6 * scale
        @test maximum(abs.(change .- (updated.moment .+ updated.outflow))) < 1e-12 * scale
        # a threshold that follows a temperature that differs between the radii: the row holds
        # the transport over the whole grid, so the budget closes at each radius and in the
        # volume as with one threshold. The moment over the hot region alone, the rejected
        # alternative, no longer integrates to zero, since the transport carries electrons
        # across the threshold. Measured: the budget per radius to 1.9e-14 of the largest |P|,
        # |∫P| of 7.5e-17 of ∫|P|, and the hot region 3.0e-6 of ∫|P|
        thermal = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 3))
        profiled = run(1.0; moment_region = thermal, temperatures = [100.0, 70.0, 40.0])
        change = (values_of(profiled.finish, 5) .- values_of(profiled.start, 5)) ./ dt
        thermal_scale = maximum(abs, profiled.moment)
        thermal_volume_scale = sum(volumes .* abs.(profiled.moment))
        @test maximum(abs.(change .- (profiled.moment .+ profiled.outflow))) <
              1e-12 * thermal_scale
        @test abs(sum(volumes .* profiled.moment)) < 1e-13 * thermal_volume_scale
        @test abs(sum(volumes .* (change .- profiled.outflow))) <
              1e-12 * thermal_volume_scale
        @test abs(sum(volumes .* profiled.hot)) > 1e-7 * thermal_volume_scale
        # an absorbing edge and a four-point radial scheme: the same budget at each radius.
        # The volume integral of P is minus the energy flux out through the edge, the
        # particle flux of ε f there. Measured: at each radius 1.6e-14 (absorbing) and
        # 2.7e-14 (TCDF) of the largest |P|, in the volume 1.7e-16 and 7.4e-17 of ∫|P|
        energies = [Runes2P._kinetic_energy(p) for p in m.p_centers]
        for (r_boundary, r_scheme) in ((:absorbing, :central), (:zero_flux, :tcdf))
            case = run(1.0; r_boundary, r_scheme)
            change = (values_of(case.finish, 5) .- values_of(case.start, 5)) ./ dt
            case_scale = maximum(abs, case.moment)
            @test maximum(abs.(change .- (case.moment .+ case.outflow))) <
                  1e-12 * case_scale
            edge = radial_particle_flux(
                case.op, energies .* kinetic_array(case.finish, layout, 1), 4)
            if r_boundary === :absorbing
                @test edge > 0
            end
            @test abs(sum(volumes .* case.moment) + edge) <
                  1e-13 * sum(volumes .* abs.(case.moment))
        end
        # the Rechester-Rosenbluth heat transport next to the term is refused when the system
        # is built; a prescribed diffusivity steps, and the heat transport moves energy
        # between the radii alone
        rr_heat = HeatTransportTerm(; delta_b_over_b = 2e-3, major_radius = 6.2)
        @test_throws "the heat of the bulk would move twice" run(1.0; heat = (rr_heat,))
        heated = run(1.0; heat = (HeatTransportTerm(; diffusion = 1.0),),
            temperatures = [100.0, 70.0, 40.0])
        change = (values_of(heated.finish, 5) .- values_of(heated.start, 5)) ./ dt
        @test all(isfinite, heated.finish)
        @test abs(sum(volumes .* (change .- heated.outflow))) <
              1e-12 * sum(volumes .* abs.(heated.moment))
    end

    @testset "radial transport next to the sinks under an upwind side that a field turns" begin
        # Partial screening gives the collisions of f sinks, which the row takes over the hot
        # region under :upwind in p, where every part of the loss takes the side of the whole
        # advection. A field of 2 V/m beats the friction above a few thermal momenta, so the
        # whole advection changes its sign along p. Radial transport of f on three radii with
        # a closed edge, with its energy term. At each radius W_tot changes by the sinks, the
        # outflow at p_max and the transport, and in the volume by the sinks and the outflow
        # alone. Measured: 18 of the 117 inner faces of the pitch cell ξ > 0 with friction and
        # 99 with the field ahead, the budget at each radius to 9.9e-15 of the largest channel,
        # |∫P| of 6.3e-17 of ∫|P|, and the volume change less the losses 2.2e-14 of ∫|P|
        T0, n, E = 100.0, 1e19, 2.0
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(40, 0.0, 0.6), uniform_xi_faces(2)),
            RadialGrid([0.0, 0.15, 0.3, 0.5]))
        m = momentum_grid(grid)
        base = PlasmaState(;
            cold_electron_temperature = fill(T0, 3), ions = [
                IonSpecies("Ar", 18, 2, fill(1e18, 3)),
                IonSpecies("D", 1, 1, fill(n, 3))],
            electric_field = fill(E, 3), magnetic_field = fill(2.5, 3))
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 3))
        collisions = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :upwind, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base),
            settings = CollisionSettings(mode = :full, screening = :partial))
        transport_op = AdvectionDiffusion(grid; scheme = :central, cross_diffusion = false)
        add_rechester_rosenbluth_diffusion!(transport_op, 2e-3; major_radius = 6.2)
        transport = FokkerPlanckTerm(:f, transport_op)
        sinks = KineticEnergyLossTerm(collisions; part = :sinks, region)
        outflow = MomentumBoundaryFluxTerm(:f, collisions; weight = :energy, coefficient = -1)
        energy = KineticEnergyChangeTerm(transport)
        layout = UnknownLayout(grid.radial_grid,
            [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        equations = [Equation(:f, TimeDerivative(), collisions, transport),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                sinks, outflow, energy),
            density_row(fill(n, 3)), thermal_energy_equation(),
            total_energy_equation(:f, grid; region)]
        theta = T0 / Constants.ELECTRON_REST_ENERGY
        u0 = zeros(length(layout))
        for ir in 1:3
            kinetic_view(u0, layout, 1, ir) .= [n * exp(-p^2 / 2theta) +
                                                1e17 * (4 - ir) *
                                                exp(-((p - 0.3) / 0.05)^2)
                                                for p in m.p_centers, _ in 1:2]
        end
        unknown_values(u0, layout, 2) .= [100.0, 70.0, 40.0]
        unknown_values(u0, layout, 3) .= n
        system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
        dt = 1e-7
        solution = integrate(system, u0, [0.0, dt]; stepper = FixedSteps())
        start, finish = solution.states[:, 1], solution.states[:, 2]
        # the whole advection takes both signs along p in each pitch cell of each radius
        advection = Runes2P._operator_at(collisions, finish, layout, dt).advection_p
        @test any(<(0), advection[2:40, 2, :]) && any(>(0), advection[2:40, 2, :])
        rows = [first(Runes2P.unknown_range(layout, 2, ir)) for ir in 1:3]
        channel(term) = Runes2P.rhs!(zeros(length(finish)), term, finish, layout, 2, dt)[rows]
        losses = channel(sinks) .+ channel(outflow)
        transported = channel(energy)
        change = (unknown_values(finish, layout, 5) .- unknown_values(start, layout, 5)) ./
                 dt
        volumes = grid.radial_grid.spatial_jacobian .* grid.radial_grid.r_widths
        scale = maximum(abs, transported)
        @test maximum(abs, losses) > 0 && scale > 0
        @test maximum(abs.(change .- (losses .+ transported))) <
              1e-12 * max(scale, maximum(abs, losses))
        @test abs(sum(volumes .* transported)) < 1e-13 * sum(volumes .* abs.(transported))
        @test abs(sum(volumes .* (change .- losses))) <
              1e-12 * sum(volumes .* abs.(transported))
    end

    @testset "radial transport set after the build under the total energy" begin
        # An operator of radial transport built by hand declares radial transport and holds no
        # coefficients when the system is built, so the system accepts it without the energy
        # term of the transport. A RechesterRosenbluthUpdate sets its coefficients in the run,
        # and the run then requires the term, as the system does for transport set before it
        # is built. Coefficients set by hand after the build need it as well.
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
        moving = FokkerPlanckTerm(:f, transport)
        equations(row_terms...) = [Equation(:f, TimeDerivative(), fp, moving),
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy),
                PrescribedValueTerm([0.0, 0.0]), KineticEnergyLossTerm(fp; part = :sinks, region),
                MomentumBoundaryFluxTerm(:f, fp; weight = :energy, coefficient = -1), row_terms...),
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
        system = EquationSystem(layout, equations(); time_scale = 1e-7, initial_state = u0)
        @test system isa EquationSystem
        @test_throws "add KineticEnergyChangeTerm(term)" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps(), updates = (update,))
        # with the energy term of the transport the run takes the update; a second term is
        # refused when the system is built
        with_term = EquationSystem(
            layout, equations(KineticEnergyChangeTerm(moving));
            time_scale = 1e-7, initial_state = u0)
        solution = integrate(
            with_term, u0, [0.0, 1e-8]; stepper = FixedSteps(), updates = (update,))
        @test all(isfinite, solution.states)
        @test any(!iszero, transport.diffusion_r_r)
        fill!(transport.diffusion_r_r, 0.0)
        twice = equations(KineticEnergyChangeTerm(moving), KineticEnergyChangeTerm(moving))
        @test_throws "more than once" EquationSystem(
            layout, twice; time_scale = 1e-7, initial_state = u0)
        # the same coefficients set by hand on the built system: the run refuses them before
        # its first step, also when it resumes, and the system refuses them when it is built
        add_rechester_rosenbluth_diffusion!(transport, 1e-3; major_radius = 6.2)
        @test_throws "add KineticEnergyChangeTerm(term)" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps())
        @test_throws "add KineticEnergyChangeTerm(term)" integrate(
            system, u0, [0.0, 1e-8]; stepper = FixedSteps(),
            resume = joinpath(mktempdir(), "missing.h5"))
        @test_throws "add KineticEnergyChangeTerm(term)" EquationSystem(
            layout, equations(); time_scale = 1e-7, initial_state = u0)
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
            [KineticUnknown(:f, hot; scale = 1e16),
                KineticUnknown(:runaway, upper; scale = 1e14),
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
            kinetic_view(u0, layout, 1, ir) .= [1e16 * exp(-p / 0.15)
                                                for p in hot_m.p_centers, _ in 1:4]
            kinetic_view(u0, layout, 2, ir) .= [2e14 * exp(-p / 2)
                                                for p in upper_m.p_centers, _ in 1:6]
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
        moment(grid, df) = [sum(Runes2P._energy_weights(grid, ir) .* view(df, :, :, ir))
                            for ir in 1:2]
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
        field_re = field_work(upper, (
            p_lower_boundary = :interface, p_upper_boundary = :interface))
        heating_re = value(free_electron_energy_transfer_term(re))
        sinks_re = value(KineticEnergyLossTerm(re; part = :sinks, coefficient = 1))
        outflow = value(MomentumBoundaryFluxTerm(:runaway, re; weight = :energy))
        influx = change(:runaway, upper) .- moment(upper,
            kinetic_array(
                Runes2P.rhs!(zeros(length(u)), re, u, layout, 2, dt) .+
                Runes2P.rhs!(zeros(length(u)), source, u, layout, 2, dt),
                layout,
                2))
        crossing = [interface_flux(interface, u, layout, ir) for ir in 1:2]
        @test maximum(abs, influx .- energy_of(upper_m.p_centers[1]) .* crossing) <=
              1e-12 * maximum(abs, change(:runaway, upper))
        knockons = value(KineticEnergyChangeTerm(source))
        channels_re = field_re .- heating_re .- sinks_re .- outflow .+ influx .+ knockons
        scale_re = maximum(abs, hcat(
            field_re, heating_re, sinks_re, outflow, influx, knockons))
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
        face_field = [energy_step *
                      part(j -> field[ir] * op.unit_field_advection_p[31, j, ir], j -> 0.0, ir)
                      for ir in 1:2]
        face_sinks = [energy_step *
                      part(j -> op.advection_p[31, j, ir] - ee.advection_p[31, j, ir],
                          j -> op.diffusion_p_p[31, j, ir] - ee.diffusion_p_p[31, j, ir], ir)
                      for ir in 1:2]
        # the term: the energy that the cold electrons gain, the negative of the work that the
        # collisions do on the electrons that cross
        cold_work = value(InterfaceCollisionWorkTerm(interface))
        both = change(:f, hot) .+ change(:runaway, upper)
        without = field_f .+ field_re .+ face_field .- heating_f .- heating_re .- sinks_f .-
                  sinks_re .+ face_sinks .- outflow .+ knockons
        scale = maximum(abs,
            hcat(field_f, field_re, face_field, heating_f, heating_re, sinks_f,
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
        base = PlasmaState(;
            cold_electron_temperature = [T0], ions = [IonSpecies("D", 1, 1, [n])],
            electric_field = [0.0], magnetic_field = [0.0])
        settings = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect)
        plasma = PlasmaFromUnknowns(base)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        lower = FokkerPlanckTerm(:f,
            AdvectionDiffusion(hot; scheme = :exponential_fitting, p_upper_boundary = :interface);
            plasma, settings, synchrotron = false)
        re = FokkerPlanckTerm(:runaway,
            AdvectionDiffusion(
                upper; scheme = :exponential_fitting, p_lower_boundary = :interface,
                p_upper_boundary = :outflow);
            plasma, settings, synchrotron = false)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, re)
        layout = UnknownLayout(radii,
            [KineticUnknown(:f, hot; scale = 1e19),
                KineticUnknown(:runaway, upper; scale = 1e16),
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
        kinetic_view(u0, layout, 2, 1) .= [1e16 * exp(-p / 0.3)
                                           for p in upper_m.p_centers, _ in 1:3]
        unknown_values(u0, layout, 3) .= T0
        unknown_values(u0, layout, 4) .= n
        function run(work_terms...)
            equations = [Equation(:f, TimeDerivative(), lower, interface),
                Equation(:runaway, TimeDerivative(), re, interface),
                Equation(
                    :cold_electron_temperature, TimeDerivative(:total_electron_energy),
                    InterfaceEnergyFluxTerm(interface), work_terms...,
                    free_electron_energy_transfer_term(re)),
                density_row([n]), thermal_energy_equation(), total_energy_equation(:f, hot; region)]
            system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
            times = collect(range(0.0, 5e-7; length = 6))
            solution = integrate(system, u0, times; stepper = FixedSteps())
            # W_tot and the energy of the runaway grid, relative to W_tot at the start
            totals = map(1:6) do k
                state = solution.states[:, k]
                W_re = sum(Runes2P._energy_weights(upper, 1) .*
                           kinetic_view(state, layout, 2, 1))
                unknown_values(state, layout, 6)[1] + W_re
            end
            flux = interface_flux(interface, solution.states[:, end], layout, 1)
            return (totals .- totals[1]) ./ totals[1], flux, solution, totals[1]
        end
        # without the work the system refuses the row
        @test_throws "add InterfaceCollisionWorkTerm" run()
        drift, flux, _, _ = run(InterfaceCollisionWorkTerm(interface))
        @test flux < 0
        # a coefficient of −1e-30, which the check accepts, stands for a row without the work
        missed, _, without, start = run(InterfaceCollisionWorkTerm(interface; coefficient = -1e-30))
        # without the term the sum drifts by minus the time integral of the term, the energy that
        # the cold electrons would gain: backward Euler takes the term at the end of each step
        cold_work = InterfaceCollisionWorkTerm(interface)
        work_at(k) = only(unknown_values(
            Runes2P.rhs!(zeros(length(u0)), cold_work, without.states[:, k], layout, 3,
                without.times[k]),
            layout, 3))
        dt = without.times[2] - without.times[1]
        integral = [0.0; cumsum([-dt * work_at(k) for k in 2:6])] ./ start
        @info "the total energy of two grids: drifts by $(maximum(abs, drift)) of W_tot with the work of the collisions at the interface, by $(maximum(abs, missed)) without it, minus the integral of the work $(maximum(abs, integral)); the drift without it less that integral: $(maximum(abs, missed .- integral))"
        # measured: a drift of 1.2e-16 of W_tot with the work, 3.9e-7 without it, which equals minus
        # the time integral of the term to 2.0e-16 of W_tot
        @test maximum(abs, drift) < 1e-12
        @test maximum(abs, missed) > 1e3 * max(maximum(abs, drift), 1e-15)
        @test maximum(abs, missed .- integral) <= 1e-3 * maximum(abs, missed)

        # a row of the cold thermal energy alone takes the term without the conditions of the
        # total energy: the system builds and steps, and refuses a second term of one interface
        cold_layout = UnknownLayout(radii,
            [KineticUnknown(:f, hot; scale = 1e19),
                KineticUnknown(:runaway, upper; scale = 1e16),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        cold_u0 = u0[1:length(cold_layout)]
        function cold_system(row_terms...)
            equations = [Equation(:f, TimeDerivative(), lower, interface),
                Equation(:runaway, TimeDerivative(), re, interface),
                Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                    row_terms...),
                density_row([n]), thermal_energy_equation()]
            return EquationSystem(
                cold_layout, equations; time_scale = 1e-7, initial_state = cold_u0)
        end
        row_work = InterfaceCollisionWorkTerm(interface)
        cold_solution = integrate(cold_system(row_work), cold_u0, [0.0, 1e-7];
            stepper = FixedSteps())
        @test all(cold_solution.statistics.accepted)
        @test unknown_values(cold_solution.states[:, 2], cold_layout, 3)[1] != T0
        @test_throws "count the work more than once" cold_system(row_work, row_work)
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
