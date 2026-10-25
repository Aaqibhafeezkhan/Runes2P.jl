# The total energy of the electrons in the fully kinetic mode, an unknown W_tot = W_cold +
# W_hot, with W_hot the energy of the hot region of the hot-tail grid, whose time derivative
# the row of the temperature holds, so that T follows from W_cold = W_tot − W_hot: the
# transfer from a slowing-down tail to the cold electrons against the friction power of
# superthermal electrons, the temperature of a cold plasma under a hot tail a hundred times
# its energy, the chain that determines the temperature, and DREAM's fluid heating by
# runaways.
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
        @test coulomb_logarithm(:electron_electron, :energy_dependent, 0.0, 100.0, 1e19) ==
              thermal_coulomb_logarithm(100.0, 1e19)
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
        radial_values(u0, layout, 2) .= T0
        radial_values(u0, layout, 3) .= n
        system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
        dt = 5e-8
        solution = integrate(system, u0, [0.0, dt]; stepper = FixedSteps())
        W_cold = [radial_values(solution.states[:, k], layout, 4)[1] for k in 1:2]
        W_tot = [radial_values(solution.states[:, k], layout, 5)[1] for k in 1:2]
        # the total holds (no source in the row of the temperature), and the cold electrons
        # gain what the tail loses
        @test W_tot[2] ≈ W_tot[1] rtol = 1e-10
        @test W_tot[1] ≈
              W_cold[1] + only(Runes2P.rhs!(
            zeros(length(u0)), EnergyMomentTerm(:f, grid; region), solution.states[:, 1],
            layout, 5, 0.0)[Runes2P.unknown_range(layout, 5, 1)]) rtol = 1e-12
        f1 = kinetic_array(solution.states[:, 2], layout, 1)
        T1 = radial_values(solution.states[:, 2], layout, 2)[1]
        p_th = 5 * sqrt(2 * T1 / Constants.ELECTRON_REST_ENERGY)
        V = phase_space_jacobian(grid, 1)
        friction = sum(4π * n * r0^2 * me * c^3 *
                       coulomb_logarithm(:electron_electron, :energy_dependent, p, T1, n) *
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
        @test two(crossing, InterfaceEnergyFluxTerm(interface)) === nothing
        @test_throws "add InterfaceEnergyFluxTerm" two(crossing)
        @test_throws "negative coefficient" two(crossing, InterfaceEnergyFluxTerm(interface; coefficient = 1))
        # radial transport and knock-on secondaries on the grid of f have no row term: the
        # system refuses them
        transport = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
        add_rechester_rosenbluth_diffusion!(transport, 1e-3; major_radius = 6.2)
        @test_throws "Radial transport" check((term, FokkerPlanckTerm(:f, transport)), sinks, outflow)
        @test_throws "fluid runaways above the grid" check(
            (term, RosenbluthPutvinskiSourceTerm(:f, grid)), sinks, outflow)
        # a system without the total energy needs none of these row terms
        @test Runes2P._check_plasma_pairings([Equation(
            :f, TimeDerivative(), term, FokkerPlanckTerm(:f, transport))]) === nothing

        # The sinks take no flux through p_max, whatever the boundary of the source: the energy
        # of the electrons that leave is the boundary term's alone.
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
        radial_values(u, layout, 2) .= T0
        radial_values(u, layout, 3) .= n
        loss(boundary) = Runes2P.rhs!(zeros(length(u)),
            KineticEnergyLossTerm(fp(boundary); part = :sinks, region = nothing),
            u, layout, 2, 0.0)[Runes2P.unknown_range(layout, 2, 1)]
        reference = loss(:interface)
        @test only(reference) != 0
        for boundary in (:f_zero, :constant_flux)
            @test loss(boundary) == reference
        end
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
        radial_values(u0, layout, 2) .= T0
        radial_values(u0, layout, 3) .= n
        # the temperature is determined through W_tot ← W_cold ← T although the threshold is fixed
        system = EquationSystem(layout, equations; initial_state = u0)
        times = collect(range(0.0, 1e-3; length = 11))
        solution = integrate(system, u0, times; stepper = FixedSteps())
        W = solution.states[:, 1]
        @test radial_values(W, layout, 5)[1] / radial_values(W, layout, 4)[1] > 90
        for (k, t) in enumerate(times)
            @test radial_values(solution.states[:, k], layout, 2)[1] ≈
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
        radial_values(u0, layout, unknown_id(layout, :electric_field)) .= E0
        radial_values(u0, layout, unknown_id(layout, :cold_electron_temperature)) .= T0
        radial_values(u0, layout, unknown_id(layout, :cold_electron_density)) .= n
        system = EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0)
        times = collect(range(0.0, 2e-5; length = 21))     # about 4 collision times per step
        solution = integrate(system, u0, times; stepper = FixedSteps())
        value(k, name) = only(radial_values(solution.states[:, k], layout, unknown_id(layout, name)))
        rate = (value(21, :total_electron_energy) - value(11, :total_electron_energy)) /
               (times[21] - times[11])
        T_mid = value(16, :cold_electron_temperature)
        @test value(21, :cold_electron_temperature) > T0
        @test rate ≈ braams_karney_conductivity(T_mid, 1.0, n) * E0^2 rtol = 5e-2
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
        work = only(radial_values(
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
        lnL = coulomb_logarithm(:electron_electron, :energy_dependent, 20.0, T, n)
        E_c = n * e^3 * lnL / (4π * Constants.VACUUM_PERMITTIVITY^2 * me * c^2)
        @test out[1] ≈ e * c * E_c * n_re rtol = 1e-6
        @test Runes2P._temperature_source(term) == (:unknown, :cold_electron_temperature)
        @test describe(term, Runes2P.DescriptionContext())["coulomb_logarithm"] ==
              "energy_dependent"
        @test_throws ArgumentError RunawayFluidEnergyTransferTerm(plasma; coulomb_logarithm = :other)
    end
end
