# The energy that the runaway grid and the electrons leaving the hot-tail grid exchange with
# the total energy of the electrons: the collisional transfer of a
# runaway population against the friction power of superthermal electrons, its Jacobian
# with the plasma columns, and the energy carried through p_max and through the interface.
using SparseArrays: findnz

@testset "energy transfer" begin
    e, c, r0, me = Constants.ELEMENTARY_CHARGE,
    Constants.SPEED_OF_LIGHT, Constants.CLASSICAL_ELECTRON_RADIUS,
    Constants.ELECTRON_MASS
    energy_of(p) = Constants.ELECTRON_REST_ENERGY * e * p^2 / (1 + sqrt(1 + p^2))
    T0, n = 50.0, 1e19
    base = PlasmaState(; cold_electron_temperature = [T0],
        ions = [IonSpecies("D", 1, 1, [n])], electric_field = [0.0],
        magnetic_field = [0.0], cold_electron_density = [n])
    settings = CollisionSettings(
        mode = :superthermal, screening = :complete, coulomb_logarithm = :energy_dependent)
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(300, 2.0, 30.0), uniform_xi_faces(4)), rad)
    m = momentum_grid(grid)
    f = [exp(-((p - 10) / 1.5)^2) for p in m.p_centers, _ in 1:4, _ in 1:1]
    f .*= 1e16 / density(grid, f)[1]
    layout = UnknownLayout(rad,
        [KineticUnknown(:runaway, grid; scale = 1e16),
            RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:cold_electron_density; scale = 1e19)])
    u = zeros(length(layout))
    kinetic_array(u, layout, 1) .= f
    unknown_values(u, layout, 2) .= T0
    unknown_values(u, layout, 3) .= n
    fp(lagged) = FokkerPlanckTerm(:runaway,
        AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
        plasma = PlasmaFromUnknowns(base), settings, lagged)

    @testset "the friction of runaways on the cold electrons" begin
        # v F_s = 4π n r₀² mₑc³ lnΛ_ee(p) γ/p per superthermal electron on cold free electrons
        term = free_electron_energy_transfer_term(fp(:ions))
        Runes2P.validate(term, layout, 2)
        power = Runes2P.rhs!(zeros(length(u)), term, u, layout, 2, 0.0)[first(Runes2P.unknown_range(
            layout, 2, 1))]
        V = phase_space_jacobian(grid, 1)
        friction = sum(4π * n * r0^2 * me * c^3 *
                       coulomb_logarithm(:electron_electron, :energy_dependent, p, T0, n) *
                       sqrt(1 + p^2) / p * V[i, j] * m.p_widths[i] * m.xi_widths[j] *
                       f[i, j, 1]
        for (i, p) in enumerate(m.p_centers), j in 1:4) / rad.spatial_jacobian[1]
        @test power > 0
        @test power ≈ friction rtol = 1e-2
        # p = 20: the friction of DREAM's fluid term per runaway
        @test 4π * n * r0^2 * me * c^3 * sqrt(1 + 20^2) / 20 ≈
              Runes2P.RUNAWAY_FLUID_ENERGY_PREFACTOR * n rtol = 2e-3
        @test Runes2P._temperature_source(term) == (:unknown, :cold_electron_temperature)
    end

    @testset "Jacobian and treatment" begin
        # the transfer reads the Fokker-Planck term of the system itself
        system(lagged, collisions = fp(lagged)) = EquationSystem(layout,
            [Equation(:runaway, TimeDerivative(), collisions),
                Equation(:cold_electron_temperature, Algebraic(),
                    free_electron_energy_transfer_term(collisions),
                    LinearTerm(:cold_electron_temperature; coefficient = -1e3)),
                Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n]),
                    LinearTerm(:cold_electron_density; coefficient = -1))];
            initial_state = u)
        implicit = system(:ions)
        J = jacobian_prototype(implicit)
        jacobian!(J, implicit, u, 0.0)
        @test all(isfinite, J.nzval)
        # the temperature, the density and two cells of the distribution
        for column in (first(Runes2P.unknown_range(layout, 2, 1)),
            first(Runes2P.unknown_range(layout, 3, 1)),
            Runes2P._kinetic_index(layout, 1, 1, 120, 2), Runes2P._kinetic_index(
            layout, 1, 1, 150, 4))
            # a step of the scale of the unknown: a cell far on the flank of the bump holds a
            # value too small for a relative step against the round-off of the total power
            scale = column == first(Runes2P.unknown_range(layout, 2, 1)) ? 100.0 :
                    column == first(Runes2P.unknown_range(layout, 3, 1)) ? 1e19 : 1e16
            h = 1e-6 * scale
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(u)), zeros(length(u))
            rhs!(F_up, up, implicit, 0.0)
            rhs!(F_down, down, implicit, 0.0)
            @test J[first(Runes2P.unknown_range(layout, 2, 1)), column] ≈
                  ((F_up .- F_down) ./ (2h))[first(Runes2P.unknown_range(layout, 2, 1))] rtol = 1e-6
        end
        # lagged: the plasma of the start of the step, so a changed temperature changes nothing
        term = free_electron_energy_transfer_term(fp(:all))
        @test Runes2P.dependencies(term) == [:runaway]
        Runes2P.prepare_step!(term, u, layout, 2, 0.0)
        hotter = copy(u)
        unknown_values(hotter, layout, 2) .= 2T0
        power(state) = Runes2P.rhs!(zeros(length(u)), term, state, layout, 2, 0.0)[first(Runes2P.unknown_range(
            layout, 2, 1))]
        @test power(hotter) == power(u)
    end

    @testset "energy through p_max and through the interface" begin
        # a field that drives electrons out at p_max: the energy is the particle flux times
        # mₑc²(γ − 1) at the edge
        driven = PlasmaState(; cold_electron_temperature = [T0],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [50.0],
            magnetic_field = [0.0])
        op = AdvectionDiffusion(grid; p_upper_boundary = :outflow)
        op.advection_p[end, :, :] .= 0.5      # outwards at p_max
        flat = UnknownLayout(rad, [KineticUnknown(:runaway, grid), RadialUnknown(:loss)])
        state = zeros(length(flat))
        kinetic_array(state, flat, 1) .= 1e14
        # the radial unknowns come before the kinetic ones in the state: read them by name
        flux(weight) = only(unknown_values(
            Runes2P.rhs!(
                zeros(length(state)), MomentumBoundaryFluxTerm(:runaway, op; weight),
                state, flat, 2, 0.0),
            flat,
            2))
        particles, energy = flux(:density), flux(:energy)
        @test particles != 0
        # the energy of the last cell, which the outflow empties
        @test energy ≈ energy_of(m.p_centers[end]) * particles rtol = 1e-13

        # two grids that meet at p = 2: the energy that crosses with the electrons
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(60, 0.0, 2.0), uniform_xi_faces(4)), rad)
        upper = runaway_grid(hot, uniform_p_faces(40, 2.0, 30.0)[2:end])
        lower_op = fokker_planck_operator(hot, driven, settings; p_upper_boundary = :interface)
        upper_op = fokker_planck_operator(upper, driven, settings; p_lower_boundary = :interface)
        interface = MomentumInterfaceTerm(:f, :runaway, lower_op, upper_op)
        two = UnknownLayout(rad,
            [KineticUnknown(:f, hot; scale = 1e16),
                KineticUnknown(:runaway, upper; scale = 1e16),
                RadialUnknown(:cold_electron_temperature)])
        w = zeros(length(two))
        kinetic_array(w, two, 1) .= [exp(-p)
                                     for p in momentum_grid(hot).p_centers, _ in 1:4,
        _ in 1:1] .* 1e16
        kinetic_array(w, two, 2) .= 1e13
        term = InterfaceEnergyFluxTerm(interface)
        Runes2P.validate(term, two, 3)
        out = only(unknown_values(Runes2P.rhs!(zeros(length(w)), term, w, two, 3, 0.0), two, 3))
        crossing = interface_flux(interface, w, two, 1)
        @test crossing != 0
        @test out ≈ -energy_of(momentum_grid(hot).p_centers[end]) * crossing rtol = 1e-13
        # closure: the energy moment of what the interface takes from the lower grid is the
        # energy the term charges, to round-off
        taken = Runes2P.rhs!(zeros(length(w)), interface, w, two, 1, 0.0)
        weights = Runes2P._energy_weights(hot, 1)
        moment = sum(weights[i, j] * taken[Runes2P._kinetic_index(two, 1, 1, i, j)]
        for i in 1:60, j in 1:4)
        @test moment ≈ out rtol = 1e-12
        record = describe(term, Runes2P.DescriptionContext())
        @test record["coefficient"] == -1.0
        # the Jacobian of the term, by the colored AD over its pattern
        equations = [Equation(:f, TimeDerivative(), interface),
            Equation(:runaway, TimeDerivative(), interface),
            Equation(:cold_electron_temperature, Algebraic(), term,
                LinearTerm(:cold_electron_temperature; coefficient = -1e3))]
        system = EquationSystem(two, equations; initial_state = w)
        J = jacobian_prototype(system)
        jacobian!(J, system, w, 0.0)
        T_row = first(Runes2P.unknown_range(two, 3, 1))
        for column in (Runes2P._kinetic_index(two, 1, 1, 60, 1),
            Runes2P._kinetic_index(two, 1, 1, 60, 3),
            Runes2P._kinetic_index(two, 2, 1, 1, 2), Runes2P._kinetic_index(
            two, 2, 1, 1, 4))
            h = 1e-6 * 1e16
            up, down = copy(w), copy(w)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(w)), zeros(length(w))
            rhs!(F_up, up, system, 0.0)
            rhs!(F_down, down, system, 0.0)
            @test J[T_row, column] ≈ (F_up[T_row] - F_down[T_row]) / (2h) rtol = 1e-6
        end
        # a copy of the interface that is not a term of the system
        stray = InterfaceEnergyFluxTerm(MomentumInterfaceTerm(:f, :runaway, lower_op, upper_op))
        @test_throws "not a term of the system" EquationSystem(two,
            [equations[1:2]...,
                Equation(:cold_electron_temperature, Algebraic(), stray,
                    LinearTerm(:cold_electron_temperature; coefficient = -1e3))];
            initial_state = w)
    end
end
