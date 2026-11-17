# The energy that the runaway grid and the electrons leaving the hot-tail grid exchange with
# the total energy of the electrons: the collisional transfer of a
# runaway population against the friction power of superthermal electrons, its Jacobian
# with the plasma columns and with the plasma lagged (no plasma columns, in a term and in a
# step of the system), the Jacobian of the energy loss of f (KineticEnergyLossTerm) with a
# lagged source, its error when never prepared, its values in the prototype of a system and
# the plasma it keeps there when prepared before, and the energy carried through p_max and
# through the interface, with its column of the field when the interface reads the field,
# and the particles through the interface in a radial row (InterfaceParticleFluxTerm), with
# the rows of both on pitch grids that differ between radii. The loss of f under the
# schemes whose face value depends on the side of the advection: the face values of the
# flux functions with a side, the face form that a flux limiter gives and the moment of the
# applied operators that a linear source keeps, the parts that sum to the moment of the
# whole, the pitch fluxes that leave it, the face form of the linear schemes with an
# envelope of dual numbers, the Jacobian, the sign of the field, the jump of the parts where
# a face changes its side, two small systems that step and the allocations. The work of the
# collisions on the electrons that cross the interface (InterfaceCollisionWorkTerm): its parts
# of the flux, its Jacobian, its checks and the rebuilds of the coefficients that it keeps.
# The energy of radial transport (KineticEnergyChangeTerm of a Fokker-Planck term): its
# moment against that of the applied operator, its volume integral, its Jacobian, its
# allocations and its source.
using ForwardDiff: ForwardDiff
using Random: Xoshiro
using SparseArrays: findnz, sparse

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
                       coulomb_logarithm(:electron_electron, :energy_dependent, p, n, T0) *
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
        # lagged at a hotter start of the step: the columns of f hold the operators of that
        # plasma, against central differences at u. The term is linear in f, so a step of 1e-6
        # of the scale of f leaves round-off alone, at most 1e-10 of the column. The first cell
        # has a zero column, so the cells start at the second.
        T_row = first(Runes2P.unknown_range(layout, 2, 1))
        T_column, n_column = first(Runes2P.unknown_range(layout, 2, 1)),
        first(Runes2P.unknown_range(layout, 3, 1))
        for part in (:free_electrons, :sinks)
            part_loss = KineticEnergyLossTerm(fp(:all); part)
            implicit_loss = KineticEnergyLossTerm(fp(:ions); part)
            Runes2P.prepare_step!(part_loss, hotter, layout, 2, 0.0)
            loss_rhs(x) = Runes2P.rhs!(zeros(length(x)), part_loss, x, layout, 2, 0.0)[T_row]
            @test loss_rhs(u) ≈ Runes2P.rhs!(zeros(length(u)), implicit_loss, hotter,
                layout, 2, 0.0)[T_row] rtol = 1e-14
            # the sinks hardly depend on the temperature (7e-15 between T0 and 2T0), so only
            # the free-electron part tells the lagged plasma from that of the state
            if part === :free_electrons
                @test loss_rhs(u) != Runes2P.rhs!(zeros(length(u)), implicit_loss, u,
                    layout, 2, 0.0)[T_row]
            end
            # the pattern holds the columns of f alone: no column of the temperature or the
            # density, which the implicit loss has
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], part_loss, layout, 2)
            @test all(in(Runes2P.unknown_range(layout, 1, 1)), last.(entries))
            JL = sparse(first.(entries), last.(entries), zeros(length(entries)),
                length(u), length(u))
            Runes2P.add_jacobian!(JL, part_loss, u, layout, 2, 0.0)
            for (i, j) in ((2, 1), (60, 2), (120, 2), (150, 4), (300, 3))
                column = Runes2P._kinetic_index(layout, 1, 1, i, j)
                h = 1e-6 * 1e16
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (loss_rhs(up) - loss_rhs(down)) / 2h
                @test difference != 0
                @test abs(JL[T_row, column] - difference) <= 1e-8 * abs(difference)
            end
        end
        # the plasma lagged: the system builds a loss of its own, and `loss` on the same
        # collisions, outside the system, is never prepared by it
        collisions = fp(:all)
        lagged = system(:all, collisions)
        loss = free_electron_energy_transfer_term(collisions)
        # never prepared, the lagged loss has no plasma of the start of the step: its residual
        # and its Jacobian throw an ArgumentError that names prepare_step!
        @test_throws ArgumentError Runes2P.rhs!(zeros(length(u)), loss, u, layout, 2, 0.0)
        @test_throws "prepare_step!" Runes2P.rhs!(zeros(length(u)), loss, u, layout, 2, 0.0)
        loss_entries = Runes2P.sparsity!(Tuple{Int, Int}[], loss, layout, 2)
        loss_J = sparse(first.(loss_entries), last.(loss_entries),
            zeros(length(loss_entries)), length(u), length(u))
        @test_throws ArgumentError Runes2P.add_jacobian!(loss_J, loss, u, layout, 2, 0.0)
        @test_throws "prepare_step!" Runes2P.add_jacobian!(loss_J, loss, u, layout, 2, 0.0)
        # the system takes its initial state u as the start of the step of its prototype: the
        # columns of f in the loss row, the whole block of the one radius, are those of a loss
        # prepared at u, bit for bit
        prototype = jacobian_prototype(lagged)
        Runes2P.prepare_step!(loss, u, layout, 2, 0.0)
        Runes2P.add_jacobian!(loss_J, loss, u, layout, 2, 0.0)
        f_columns = Runes2P.unknown_range(layout, 1, 1)
        @test count(!iszero, loss_J[T_row, f_columns]) > 1000
        @test prototype[T_row, f_columns] == loss_J[T_row, f_columns]
        # a loss prepared before keeps its plasma: prepared at the hotter state and then put
        # into a system built at u, it evaluates as before, where a loss prepared at u differs
        kept_collisions = fp(:all)
        kept = free_electron_energy_transfer_term(kept_collisions)
        Runes2P.prepare_step!(kept, hotter, layout, 2, 0.0)
        kept_power() = Runes2P.rhs!(zeros(length(u)), kept, u, layout, 2, 0.0)[T_row]
        before = kept_power()
        EquationSystem(layout,
            [Equation(:runaway, TimeDerivative(), kept_collisions),
                Equation(:cold_electron_temperature, Algebraic(), kept,
                    LinearTerm(:cold_electron_temperature; coefficient = -1e3)),
                Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm([n]),
                    LinearTerm(:cold_electron_density; coefficient = -1))];
            initial_state = u)
        @test kept_power() == before
        @test Runes2P.rhs!(zeros(length(u)), loss, u, layout, 2, 0.0)[T_row] != before
        # lagged in a step: the Newton residual prepares the terms at the start of the step
        # u_old, and the loss row holds the columns of f alone. u_old is hotter than the
        # iterate, so that the columns of f tell the plasma of u_old from that of the iterate.
        u_old, dt = copy(hotter), 1e-6
        residual!(zeros(length(u)), u, u_old, lagged, 0.0, dt)
        J = jacobian_prototype(lagged)
        jacobian!(J, lagged, u, 0.0)
        # the row in the system: the temperature column is that of the linear term alone
        @test J[T_row, T_column] == -1e3 && J[T_row, n_column] == 0
        # The loss is linear in f with the plasma held, so the central difference of the
        # residual is exact up to round-off: a step of 1e-6 of the scale of f, and the
        # tolerance of the implicit columns above (the difference is within 1e-11).
        # The row is algebraic, so its residual is minus the right-hand side.
        function residual_row(x)
            F = zeros(length(x))
            residual!(F, x, u_old, lagged, 0.0, dt)
            return F[T_row]
        end
        for column in (Runes2P._kinetic_index(layout, 1, 1, 120, 2),
            Runes2P._kinetic_index(layout, 1, 1, 150, 4))
            h = 1e-6 * 1e16
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            @test J[T_row, column] ≈ -(residual_row(up) - residual_row(down)) / 2h rtol = 1e-6
        end
        # the loss reads the plasma of u_old, held as numbers, so a step in the temperature
        # or the density of the iterate leaves it unchanged exactly, not to round-off
        Runes2P.prepare_step!(loss, u_old, layout, 2, 0.0)
        loss_row(x) = Runes2P.rhs!(zeros(length(x)), loss, x, layout, 2, 0.0)[T_row]
        for (column, h) in ((T_column, 1e-6 * 100.0), (n_column, 1e-6 * 1e19))
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            @test loss_row(up) - loss_row(down) == 0
        end
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

    @testset "the field column of the flux through the interface ($mapping)" for mapping in (
        :overlap, :dream)
        # the lower Fokker-Planck term takes the field as an unknown, so the interface flux,
        # the energy it carries and the particles it brings to a radial row depend on it
        driven = PlasmaState(; cold_electron_temperature = [T0],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [0.0],
            magnetic_field = [0.0])
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(60, 0.0, 2.0), uniform_xi_faces(4)), rad)
        # 30 upper pitch cells: the row has 4 + 30 + 1 columns, more than one evaluation of
        # its entries seeds (Runes2P.INTERFACE_COLUMN_CHUNK), so the field is seeded in the
        # second evaluation
        upper = runaway_grid(hot, uniform_p_faces(40, 2.0, 30.0)[2:end], uniform_xi_faces(30))
        @test 4 + 30 + 1 > Runes2P.INTERFACE_COLUMN_CHUNK
        lower_op = fokker_planck_operator(hot, driven, settings;
            p_upper_boundary = :interface, field_in_coefficients = false)
        upper_op = fokker_planck_operator(upper, driven, settings;
            p_lower_boundary = :interface, field_in_coefficients = false)
        lower_term = FokkerPlanckTerm(:f, lower_op; electric_field = :electric_field)
        upper_term = FokkerPlanckTerm(:runaway, upper_op; electric_field = :electric_field)
        interface = MomentumInterfaceTerm(:f, :runaway, lower_term, upper_op;
            pitch_mapping = mapping)
        term = InterfaceEnergyFluxTerm(interface)
        particles = InterfaceParticleFluxTerm(interface)
        @test :electric_field in Runes2P.dependencies(term) && !is_linear(term)
        @test :electric_field in Runes2P.dependencies(particles)
        two = UnknownLayout(rad,
            [KineticUnknown(:f, hot; scale = 1e16),
                KineticUnknown(:runaway, upper; scale = 1e16),
                RadialUnknown(:electric_field; scale = 10.0),
                RadialUnknown(:cold_electron_temperature),
                RadialUnknown(:runaway_density; scale = 1e10)])
        w = zeros(length(two))
        kinetic_array(w, two, 1) .= [exp(-p) * (1 + 0.2xi)
                                     for p in momentum_grid(hot).p_centers,
        xi in momentum_grid(hot).xi_centers, _ in 1:1] .* 1e16
        kinetic_array(w, two, 2) .= 1e13
        unknown_values(w, two, 3) .= 50.0
        system = EquationSystem(two,
            [Equation(:f, TimeDerivative(), lower_term, interface),
                Equation(:runaway, TimeDerivative(), upper_term, interface),
                Equation(:electric_field, Algebraic(), PrescribedValueTerm([50.0]),
                    LinearTerm(:electric_field; coefficient = -1)),
                Equation(:cold_electron_temperature, Algebraic(), term,
                    LinearTerm(:cold_electron_temperature; coefficient = -1e3)),
                Equation(:runaway_density, TimeDerivative(), particles)];
            initial_state = w)
        J = jacobian_prototype(system)
        jacobian!(J, system, w, 0.0)
        T_row = first(Runes2P.unknown_range(two, 4, 1))
        n_row = first(Runes2P.unknown_range(two, 5, 1))
        column = first(Runes2P.unknown_range(two, 3, 1))
        @test (T_row, column) in zip(findnz(J)[1:2]...)
        # a relative step of 1e-6 of the field: the difference is at most 1e-10 of the entry
        # in both rows with both mappings
        h = 1e-6 * w[column]
        up, down = copy(w), copy(w)
        up[column] += h
        down[column] -= h
        F_up, F_down = zeros(length(w)), zeros(length(w))
        rhs!(F_up, up, system, 0.0)
        rhs!(F_down, down, system, 0.0)
        for row in (T_row, n_row)
            difference = (F_up[row] - F_down[row]) / (2h)
            @test difference != 0
            @test J[row, column] ≈ difference rtol = 1e-8
        end
        # the two rows assemble their entries themselves: the cells at the face and the
        # field against the AD of the residual
        @test jacobian_assembly(term) === jacobian_assembly(particles) === :direct
        dense = ForwardDiff.jacobian(v -> rhs!(similar(v), v, system, 0.0), w)
        for row in (T_row, n_row)
            @test any(!iszero, dense[row, Runes2P.unknown_range(two, 1, 1)])
            @test any(!iszero, dense[row, Runes2P.unknown_range(two, 2, 1)])
            @test Vector(J[row, :]) ≈ dense[row, :] rtol = 1e-12
        end
    end
    @testset "the particle flux into a radial row ($mapping)" for mapping in (:overlap, :dream)
        # DREAM's n_re takes the flux through the interface (PXiExternalKineticKinetic,
        # TYPE_DENSITY): the lower grid loses what the upper grid and the radial row gain
        rng = Xoshiro(54)
        radii = RadialGrid(uniform_r_faces(2, 0.5))
        lower_grid = cylindrical_grid(
            MomentumGrid(biuniform_p_faces(9, 4, 0.0, 0.4, 2.5), uniform_xi_faces(9)), radii)
        # 30 upper pitch cells: the row has 39 columns, more than one evaluation of its
        # entries seeds (Runes2P.INTERFACE_COLUMN_CHUNK)
        upper_grid = runaway_grid(
            lower_grid, uniform_p_faces(5, 2.5, 12.0)[2:end], uniform_xi_faces(30))
        @test 9 + 30 > Runes2P.INTERFACE_COLUMN_CHUNK
        lower_op = AdvectionDiffusion(lower_grid; scheme = :central, p_upper_boundary = :interface)
        for A in (lower_op.advection_p, lower_op.advection_xi, lower_op.advection_r)
            A .= randn(rng, size(A))
        end
        for D in (lower_op.diffusion_p_p, lower_op.diffusion_xi_xi, lower_op.diffusion_r_r)
            D .= rand(rng, size(D))
        end
        upper_op = AdvectionDiffusion(upper_grid; p_lower_boundary = :interface)
        interface = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution,
            lower_op, upper_op; pitch_mapping = mapping)
        particles = InterfaceParticleFluxTerm(interface)
        @test particles.coefficient == 1
        layout = UnknownLayout(radii,
            [KineticUnknown(:hot_tail_distribution, lower_grid),
                KineticUnknown(:runaway_distribution, upper_grid),
                RadialUnknown(:runaway_density)])
        system = EquationSystem(layout,
            [Equation(:hot_tail_distribution, TimeDerivative(), interface),
                Equation(:runaway_distribution, TimeDerivative(), interface),
                Equation(:runaway_density, TimeDerivative(), particles)])
        u = rand(rng, length(layout))
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        rates = unknown_values(du, layout, unknown_id(layout, :runaway_density))
        @test rates == [interface_flux(interface, u, layout, ir) for ir in 1:2]
        @test all(!=(0), rates)
        # Σ 𝒱' Δp Δξ Δr of the rate of a kinetic unknown
        function content(name)
            id = unknown_id(layout, name)
            grid = kinetic_grid(layout, id)
            return sum(1:2) do ir
                m = momentum_grid(grid, ir)
                V = phase_space_jacobian(grid, ir)
                f = kinetic_view(du, layout, id, ir)
                sum(V[i, j] * m.p_widths[i] * m.xi_widths[j] * radii.r_widths[ir] * f[i, j]
                for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m))
            end
        end
        lower_change = content(:hot_tail_distribution)
        upper_change = content(:runaway_distribution)
        radial_change = sum(rates[ir] * radii.spatial_jacobian[ir] * radii.r_widths[ir]
        for ir in 1:2)
        @test upper_change ≈ -lower_change rtol = 1e-13
        @test radial_change ≈ upper_change rtol = 1e-13
        # linear in the state: the stored Jacobian is the term
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        @test J * u ≈ du rtol = 1e-12
        # the row assembles its entries itself, against the AD of the residual, and leaves
        # the coloring: in it, each of the 39 cells of the row would take a color of its own
        @test jacobian_assembly(particles) === :direct
        @test Matrix(J) ≈ ForwardDiff.jacobian(v -> rhs!(similar(v), v, system, 0.0), u) rtol = 1e-12
        @test jacobian_colors(system) < 9
        # the cells at the face hold one momentum cell, and reading another one throws
        cells = Runes2P.FaceCells([1.0, 2.0, 3.0], 4, 4)
        @test size(cells) == (4, 3)
        @test cells[4, 2] == 2.0
        @test_throws ArgumentError cells[3, 1]
        context = Runes2P.DescriptionContext()
        record = describe(particles, context)
        interface_record = describe(interface, context)
        @test record["type"] == "InterfaceParticleFluxTerm" && record["coefficient"] == 1.0
        @test record["interface"]["type"] == "MomentumInterfaceTerm"
        @test record["interface"] == interface_record
        # the record of the energy through the interface: the record of a term with its
        # coefficient and the record of its interface, as the particle flux
        energy = InterfaceEnergyFluxTerm(interface)
        energy_record = describe(energy, context)
        expected = Runes2P._term_record(energy)
        expected["coefficient"] = -1.0
        expected["interface"] = interface_record
        @test energy_record == expected
        @test energy_record["type"] == "InterfaceEnergyFluxTerm"
        @test Set(keys(energy_record)) == Set(keys(record))
    end

    @testset "the rows of the interface on pitch grids that differ between radii ($mapping)" for mapping in (
        :overlap, :dream)
        # the particle and energy rows assemble their entries directly, on a lower grid whose
        # pitch faces differ between the two radii (uniform and cosine), against the AD of the
        # residual; the counts are the same, since an AdvectionDiffusion takes the same cell
        # counts at all radii
        rng = Xoshiro(54)
        radii = RadialGrid(uniform_r_faces(2, 0.5))
        p_faces = biuniform_p_faces(9, 4, 0.0, 0.4, 2.5)
        uniform_pitch = MomentumGrid(p_faces, uniform_xi_faces(9))
        cosine_pitch = MomentumGrid(p_faces, -cos.(range(0, π; length = 10)))
        lower_grid = cylindrical_grid([uniform_pitch, cosine_pitch], radii)
        upper_grid = runaway_grid(
            lower_grid, uniform_p_faces(5, 2.5, 12.0)[2:end], uniform_xi_faces(30))
        lower_op = AdvectionDiffusion(lower_grid; scheme = :central,
            p_upper_boundary = :interface, radial_transport = false, cross_diffusion = false)
        for A in (lower_op.advection_p, lower_op.advection_xi)
            A .= randn(rng, size(A))
        end
        for D in (lower_op.diffusion_p_p, lower_op.diffusion_xi_xi)
            D .= rand(rng, size(D))
        end
        upper_op = AdvectionDiffusion(upper_grid; p_lower_boundary = :interface)
        interface = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution,
            lower_op, upper_op; pitch_mapping = mapping)
        particles = InterfaceParticleFluxTerm(interface)
        energy = InterfaceEnergyFluxTerm(interface)
        layout = UnknownLayout(radii,
            [KineticUnknown(:hot_tail_distribution, lower_grid),
                KineticUnknown(:runaway_distribution, upper_grid),
                RadialUnknown(:runaway_density), RadialUnknown(:energy)])
        system = EquationSystem(layout,
            [Equation(:hot_tail_distribution, TimeDerivative(), interface),
                Equation(:runaway_distribution, TimeDerivative(), interface),
                Equation(:runaway_density, TimeDerivative(), particles),
                Equation(:energy, TimeDerivative(), energy)])
        @test jacobian_assembly(particles) === :direct
        @test jacobian_assembly(energy) === :direct
        u = rand(rng, length(layout))
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        reference = ForwardDiff.jacobian(
            (du, x) -> rhs!(du, x, system, 0.0), zeros(length(layout)), u)
        largest = maximum(abs, reference)
        @test maximum(abs, Matrix(J) - reference) <= 1e-12 * largest
    end

    @testset "the work of the collisions at the interface ($(case))" for case in (
        (mapping = :overlap, scheme = :upwind), (mapping = :overlap, scheme = :central),
        (mapping = :overlap, scheme = :exponential_fitting),
        (mapping = :overlap, scheme = :tcdf), (mapping = :dream, scheme = :upwind),
        (mapping = :overlap, scheme = :exponential_fitting, peclet = :small),
        (mapping = :overlap, scheme = :exponential_fitting, mode = :superthermal,
        pitch = :per_radius),
        (mapping = :overlap, scheme = :upwind, lagged = :all))
        # Two grids that meet at p = 0.6 on two radii, the field an unknown, partial screening
        # of argon, bremsstrahlung and synchrotron loss: the flux through the face is the part of
        # the field, of the electron-electron collisions and of the sinks, each with the face
        # value of the whole flux. The part of the electron-electron collisions times the energy
        # step is the term. `:tcdf` is a limited scheme on the hot grid, whose interface flux is
        # upwind; the other schemes are those of the interface on an operator of the same scheme.
        # `peclet = :small` takes keV temperatures, cells of 0.02 on both sides of the face and a
        # field near the friction, so that the Péclet number of exponential fitting spans 0.05 to
        # above 1 over the pitch cells. `mode = :superthermal` has no energy diffusion, so
        # exponential fitting is upwind there; `pitch = :per_radius` takes other pitch faces at
        # the second radius. `lagged = :all` takes the plasma of the last preparation of the
        # Fokker-Planck term, and the system is built before any.
        (; mapping, scheme) = case
        peclet, mode = get(case, :peclet, :large), get(case, :mode, :full)
        pitch, lagged = get(case, :pitch, :uniform), get(case, :lagged, ())
        temperatures = peclet === :small ? [5e3, 1e4] : [T0, 2T0]
        radii = RadialGrid(uniform_r_faces(2, 0.5))
        hot_p = uniform_p_faces(30, 0.0, 0.6)
        hot = if pitch === :per_radius
            cylindrical_grid(
                [MomentumGrid(hot_p, uniform_xi_faces(4)),
                    MomentumGrid(hot_p, -cos.(range(0, π; length = 5)))],
                radii)
        else
            cylindrical_grid(MomentumGrid(hot_p, uniform_xi_faces(4)), radii)
        end
        upper_p = peclet === :small ? uniform_p_faces(100, 0.6, 2.6) :
                  uniform_p_faces(20, 0.6, 5.0)
        upper = runaway_grid(hot, upper_p[2:end], uniform_xi_faces(6))
        mixed = PlasmaState(; cold_electron_temperature = temperatures,
            ions = [IonSpecies("D", 1, 1, [n, 2n]), IonSpecies("Ar", 18, 2, [1e18, 3e18])],
            electric_field = zeros(2), magnetic_field = fill(2.5, 2))
        partial = CollisionSettings(; mode, screening = :partial)
        plasma = PlasmaFromUnknowns(mixed)
        lower = FokkerPlanckTerm(:f,
            AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface);
            plasma, settings = partial, electric_field = :electric_field, lagged)
        re = FokkerPlanckTerm(:runaway,
            AdvectionDiffusion(upper; scheme = :exponential_fitting, p_lower_boundary = :interface);
            plasma, settings = partial, electric_field = :electric_field)
        interface_scheme = scheme === :tcdf ? (;) : (; scheme)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, re; pitch_mapping = mapping,
            interface_scheme...)
        @test interface.scheme === (scheme === :tcdf ? :upwind : scheme)
        work = InterfaceCollisionWorkTerm(interface)
        @test work.coefficient == -1
        layout = UnknownLayout(radii,
            [KineticUnknown(:f, hot; scale = 1e16),
                KineticUnknown(:runaway, upper; scale = 1e14),
                RadialUnknown(:electric_field; scale = 10.0),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19), RadialUnknown(:work)])
        prescribed(name, values) = Equation(name, Algebraic(), PrescribedValueTerm(values),
            LinearTerm(name; coefficient = -1))
        rng = Xoshiro(17)
        hot_m, upper_m = momentum_grid(hot, 1), momentum_grid(upper, 1)
        state(field) = begin
            u = zeros(length(layout))
            for ir in 1:2
                kinetic_view(u, layout, 1, ir) .= [1e16 * exp(-p / 0.15) *
                                                   (1 + 0.3 * rand(rng))
                                                   for p in hot_m.p_centers, _ in 1:4]
                kinetic_view(u, layout, 2, ir) .= [2e14 * exp(-p / 2) *
                                                   (1 + 0.3 * rand(rng))
                                                   for p in upper_m.p_centers, _ in 1:6]
            end
            unknown_values(u, layout, 3) .= field
            unknown_values(u, layout, 4) .= temperatures
            unknown_values(u, layout, 5) .= [n, 2n]
            return u
        end
        step = energy_of(upper_m.p_centers[1]) - energy_of(hot_m.p_centers[end])
        @test step > 0
        # the Bernoulli function and the weight of the lower cell in the face value of
        # exponential fitting, written out here
        bernoulli(x) = x == 0 ? 1.0 : x / expm1(x)
        fitting(w) = abs(w) < 1e-6 ? 0.5 + w / 12 : (bernoulli(-w) - 1) / w
        # the field of each radius at which the Péclet number of the last pitch cell is w
        function field_at_peclet(w)
            op = Runes2P._operator_at(lower, state([0.0, 0.0]), layout, 0.0)
            return map(1:2) do ir
                distance, _ = Runes2P._interface_distance(interface, ir)
                A = w * op.diffusion_p_p[31, 4, ir] / distance - op.advection_p[31, 4, ir]
                A / op.unit_field_advection_p[31, 4, ir]
            end
        end
        if lagged === :all
            Runes2P.prepare_step!(lower, state([0.0, 0.0]), layout, 1, 0.0)
        end
        fields = if peclet === :small
            (field_at_peclet(0.05), field_at_peclet(-0.05))
        else
            ([30.0, -20.0], [-30.0, 20.0])
        end
        for field in fields
            u = state(field)
            op = Runes2P._operator_at(lower, u, layout, 0.0)
            if peclet === :small
                for ir in 1:2
                    distance, _ = Runes2P._interface_distance(interface, ir)
                    w = [(op.advection_p[31, j, ir] +
                          field[ir] * op.unit_field_advection_p[31, j, ir]) * distance /
                         op.diffusion_p_p[31, j, ir] for j in 1:4]
                    @test minimum(abs, w) < 0.1 && maximum(abs, w) > 1
                end
            end
            ee = electron_electron_operator(hot, op.coefficient_source[].plasma, partial;
                p_upper_boundary = :interface)
            values = Runes2P.rhs!(zeros(length(u)), work, u, layout, 6, 0.0)
            # the flux of the coefficients (A, D) of lower pitch cell j at radius ir, with the face
            # value of the whole coefficients
            function part(A_of, D_of, ir)
                distance, weight_left = Runes2P._interface_distance(interface, ir)
                f_l, f_u = kinetic_view(u, layout, 1, ir), kinetic_view(u, layout, 2, ir)
                overlap = interface.overlaps[ir]
                xi_widths = momentum_grid(hot, ir).xi_widths
                total = 0.0
                for j in 1:4
                    A = op.advection_p[31, j, ir] +
                        field[ir] * op.unit_field_advection_p[31, j, ir]
                    D = op.diffusion_p_p[31, j, ir]
                    V = op.jacobian_p_faces[31, j, ir]
                    fl = f_l[30, j]
                    if mapping === :dream
                        fr = Runes2P._dream_upper_value(interface, f_u, j, ir)
                        face = A > 0 ? fl : fr
                        total += xi_widths[j] * V *
                                 (A_of(j) * face - D_of(j) * (fr - fl) / distance)
                        continue
                    end
                    for s in overlap.left_ranges[j]
                        fr = f_u[1, overlap.right_cells[s]]
                        weight = if interface.scheme === :upwind || D == 0
                            A >= 0 ? 1.0 : 0.0
                        elseif interface.scheme === :central
                            weight_left
                        else
                            fitting(A * distance / D)
                        end
                        face = weight * fl + (1 - weight) * fr
                        gamma = A_of(j) * face - D_of(j) * (fr - fl) / distance
                        total += xi_widths[j] * overlap.left_fractions[s] * V * gamma
                    end
                end
                return total / radii.spatial_jacobian[ir]
            end
            for ir in 1:2
                flux = interface_flux(interface, u, layout, ir)
                by_field = part(j -> field[ir] * op.unit_field_advection_p[31, j, ir], j -> 0.0, ir)
                by_ee = part(j -> ee.advection_p[31, j, ir], j -> ee.diffusion_p_p[31, j, ir], ir)
                by_sinks = part(j -> op.advection_p[31, j, ir] - ee.advection_p[31, j, ir],
                    j -> op.diffusion_p_p[31, j, ir] - ee.diffusion_p_p[31, j, ir], ir)
                @test by_field + by_ee + by_sinks ≈ flux rtol = 1e-12
                @test by_ee != 0 && by_sinks != 0 && by_field != 0
                # the term: −Γ_ee times the energy step
                row_value = only(unknown_values(values, layout, 6)[ir:ir])
                @test row_value ≈ -step * by_ee rtol = 1e-12
            end
        end

        # the Jacobian of the row against central differences in every column, and against
        # the AD of the residual in the columns of the distributions and the field, which the
        # coefficients of the plasma do not hide
        u = state(first(fields))
        system = EquationSystem(layout,
            [Equation(:f, TimeDerivative(), lower, interface),
                Equation(:runaway, TimeDerivative(), re, interface),
                prescribed(:electric_field, first(fields)),
                prescribed(:cold_electron_temperature, temperatures),
                prescribed(:cold_electron_density, [n, 2n]),
                Equation(:work, Algebraic(), work, LinearTerm(:work; coefficient = -1))];
            initial_state = u)
        if lagged === :all
            # the system is built and steps with a term never prepared before: its operator
            # holds no collisions until the first preparation
            fresh = FokkerPlanckTerm(:f,
                AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface);
                plasma, settings = partial, electric_field = :electric_field, lagged)
            fresh_interface = MomentumInterfaceTerm(:f, :runaway, fresh, re;
                pitch_mapping = mapping, interface_scheme...)
            @test fresh.operator.coefficient_source[] === nothing
            # the work is zero until the first preparation, and nonzero after the first step
            fresh_work = InterfaceCollisionWorkTerm(fresh_interface)
            before = Runes2P.rhs!(zeros(length(u)), fresh_work, u, layout, 6, 0.0)
            @test all(iszero, unknown_values(before, layout, 6))
            fresh_system = EquationSystem(layout,
                [Equation(:f, TimeDerivative(), fresh, fresh_interface),
                    Equation(:runaway, TimeDerivative(), re, fresh_interface),
                    system.equations[3:5]...,
                    Equation(
                        :work, Algebraic(), fresh_work, LinearTerm(:work; coefficient = -1))];
                initial_state = u)
            solution = integrate(fresh_system, u, [0.0, 1e-9]; stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            @test all(!=(0), unknown_values(solution.states[:, 2], layout, 6))
            # a lagged plasma has no columns: the stored Jacobian treats it as fixed
            @test !Runes2P.has_plasma_columns(work)
        end
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        rows = [first(Runes2P.unknown_range(layout, 6, ir)) for ir in 1:2]
        # the rows of the term alone: with a lagged plasma that of its last preparation, as the
        # stored Jacobian takes it
        row_values(x) = Runes2P.rhs!(zeros(eltype(x), length(x)), work, x, layout, 6, 0.0)[rows]
        differences = zeros(2, length(u))
        for column in eachindex(u)
            h = 1e-6 * max(abs(u[column]), 1.0)
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            differences[:, column] .= (row_values(up) .- row_values(down)) ./ (2h)
        end
        work_columns = setdiff(eachindex(u), rows)
        for (k, r) in enumerate(rows)
            largest = maximum(abs, differences[k, work_columns])
            @test maximum(abs, Vector(J[r, work_columns]) .-
                               differences[k, work_columns]) <=
                  1e-6 * largest
            # the plasma columns hold the derivative of the collision frequencies, and a lagged
            # plasma has none
            lagged === :all && continue
            for name in (:cold_electron_temperature, :cold_electron_density)
                column = first(Runes2P.unknown_range(layout, unknown_id(layout, name), k))
                @test J[r, column] != 0
                @test J[r, column] ≈ differences[k, column] rtol = 1e-5
            end
        end
        field_columns = [first(Runes2P.unknown_range(layout, 3, ir)) for ir in 1:2]
        stored = Set(zip(findnz(J)[1:2]...))
        # the field enters through the upwind side or a fixed weight alone, except with exponential
        # fitting, whose weight follows the Péclet number. Without energy diffusion
        # (`mode = :superthermal`) the face is upwind, and the row holds no column of the field.
        with_field = interface.scheme === :exponential_fitting && mapping === :overlap &&
                     mode === :full
        @test ((rows[1], field_columns[1]) in stored) == with_field
        # the column of the field is about 1/w² of the others: at a large Péclet number below
        # the resolution of the differences, near w = 1 against them
        if with_field && peclet === :small
            for (k, r) in enumerate(rows)
                @test J[r, field_columns[k]] != 0
                @test J[r, field_columns[k]] ≈ differences[k, field_columns[k]] rtol = 1e-5
            end
        end
        dense = ForwardDiff.jacobian(row_values, u)
        kinetic = vcat(collect(Runes2P.unknown_range(layout, 1, 1)),
            collect(Runes2P.unknown_range(layout, 1, 2)),
            collect(Runes2P.unknown_range(layout, 2, 1)), collect(Runes2P.unknown_range(layout, 2, 2)),
            field_columns)
        for (k, r) in enumerate(rows)
            @test Vector(J[r, kinetic]) ≈ dense[k, kinetic] rtol = 1e-12
        end

        # the record: the record of a term with its coefficient and its interface
        context = Runes2P.DescriptionContext()
        record = describe(work, context)
        expected = Runes2P._term_record(work)
        expected["coefficient"] = -1.0
        expected["interface"] = describe(interface, context)
        @test record == expected
        @test record["type"] == "InterfaceCollisionWorkTerm"

        # rhs! allocates nothing of its own once its coefficients are kept: only the bytes of the
        # check of the plasma of the Fokker-Planck term, which the interface refreshes (32 bytes
        # at -O2, as in the flux of InterfaceEnergyFluxTerm)
        out = zeros(length(u))
        refresh = call_allocation(Runes2P.refresh_coefficients!, lower, u, layout, 0.0)
        @test call_allocation(Runes2P.rhs!, out, work, u, layout, 6, 0.0) <= refresh skip = !OPTIMIZED
    end

    @testset "the work of the collisions at the interface: its checks" begin
        # the collisions of a lower operator built by hand: no plasma, refused
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(10, 0.0, 0.6), uniform_xi_faces(2)), rad)
        upper = runaway_grid(hot, uniform_p_faces(5, 0.6, 5.0)[2:end])
        bare = MomentumInterfaceTerm(:f, :runaway,
            AdvectionDiffusion(hot; p_upper_boundary = :interface),
            AdvectionDiffusion(upper; p_lower_boundary = :interface))
        two = UnknownLayout(rad,
            [
                KineticUnknown(:f, hot), KineticUnknown(:runaway, upper), RadialUnknown(:work)])
        @test_throws "holds no collisions" Runes2P.validate(InterfaceCollisionWorkTerm(bare), two, 3)
        relativistic = fokker_planck_operator(hot, base,
            CollisionSettings(mode = :ultra_relativistic, screening = :complete);
            p_upper_boundary = :interface)
        fast = MomentumInterfaceTerm(:f, :runaway, relativistic,
            AdvectionDiffusion(upper; p_lower_boundary = :interface))
        @test_throws "mode :full or :superthermal" Runes2P.validate(
            InterfaceCollisionWorkTerm(fast), two, 3)
        # an operator built from a plasma is accepted, and the work follows its collisions
        built = fokker_planck_operator(hot, base, settings; p_upper_boundary = :interface)
        accepted = MomentumInterfaceTerm(:f, :runaway, built,
            AdvectionDiffusion(upper; p_lower_boundary = :interface))
        @test Runes2P.validate(InterfaceCollisionWorkTerm(accepted), two, 3) === nothing
        # its rhs! reads an operator that no Fokker-Planck term refreshes, and allocates nothing
        w = zeros(length(two))
        kinetic_array(w, two, 1) .= 1e16
        kinetic_array(w, two, 2) .= 1e14
        prescribed_work = InterfaceCollisionWorkTerm(accepted)
        out = zeros(length(w))
        @test only(Runes2P.rhs!(out, prescribed_work, w, two, 3, 0.0)[Runes2P.unknown_range(two, 3, 1)]) !=
              0
        @test call_allocation(Runes2P.rhs!, out, prescribed_work, w, two, 3, 0.0) == 0 skip = !OPTIMIZED
        @test_throws "radial unknown with one component" Runes2P.validate(
            InterfaceCollisionWorkTerm(accepted), two, 1)
    end

    @testset "the work of the collisions at the interface: its cache" begin
        # The coefficients of the face that the term keeps are rebuilt once per change of one
        # input of the electron-electron collisions, and not at an unchanged state. A plasma
        # state changed in place, and an operator rebuilt by hand from it, give the values of a
        # fresh term.
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(10, 0.0, 0.6), uniform_xi_faces(2)), rad)
        upper = runaway_grid(hot, uniform_p_faces(5, 0.6, 5.0)[2:end])
        two = UnknownLayout(rad,
            [
                KineticUnknown(:f, hot), KineticUnknown(:runaway, upper), RadialUnknown(:work)])
        w = zeros(length(two))
        kinetic_array(w, two, 1) .= [1e16 * exp(-p / 0.1)
                                     for p in momentum_grid(hot).p_centers, _ in 1:2,
        _ in 1:1]
        kinetic_array(w, two, 2) .= 1e13
        row_value(term, t) = only(unknown_values(
            Runes2P.rhs!(zeros(length(w)), term, w, two, 3, t), two, 3))
        full = CollisionSettings(mode = :full, screening = :partial)
        # one state, which the prescribed plasma returns at every time and the test changes in place
        state = PlasmaState(; cold_electron_temperature = [100.0],
            ions = [IonSpecies("D", 1, 1, [n]), IonSpecies("Ar", 18, 2, [1e17])],
            electric_field = [5.0], magnetic_field = [2.0])
        lower = FokkerPlanckTerm(
            :f, AdvectionDiffusion(hot; p_upper_boundary = :interface);
            plasma = PrescribedPlasma(t -> state), settings = full)
        upper_op = AdvectionDiffusion(upper; p_lower_boundary = :interface)
        interface = MomentumInterfaceTerm(:f, :runaway, lower, upper_op)
        work = InterfaceCollisionWorkTerm(interface)
        fresh(t) = row_value(InterfaceCollisionWorkTerm(interface), t)
        start = row_value(work, 0.0)
        @test start != 0
        @test work.rebuilds[] == 1
        # an unchanged state, at the same time and at another one: no rebuild
        @test row_value(work, 0.0) == start
        @test row_value(work, 1e-9) == start
        @test work.rebuilds[] == 1
        # one input alone, changed in place: one rebuild each, and the values of a fresh term
        for (k, change!) in enumerate((
            s -> s.cold_electron_temperature[1] = 150.0,
            s -> s.cold_electron_density[1] *= 2,
            s -> s.magnetic_field[1] = 3.0,
            # the field alone: the Fokker-Planck term rebuilds its coefficients, which the
            # count of its rebuilds shows, while the key of the term leaves the field out
            s -> s.electric_field[1] = 8.0))
            change!(state)
            t = k * 1e-8
            value = row_value(work, t)
            @test work.rebuilds[] == 1 + k
            @test value == fresh(t)
            @test row_value(work, t) == value
            @test work.rebuilds[] == 1 + k
        end
        @test row_value(work, 1.0) != start

        # an operator rebuilt by hand from the state changed in place
        op = fokker_planck_operator(hot, state, full; p_upper_boundary = :interface)
        bare = MomentumInterfaceTerm(:f, :runaway, op, upper_op)
        by_hand = InterfaceCollisionWorkTerm(bare)
        fresh_bare() = row_value(InterfaceCollisionWorkTerm(bare), 0.0)
        built = row_value(by_hand, 0.0)
        @test built == fresh_bare()
        @test row_value(by_hand, 0.0) == built
        @test by_hand.rebuilds[] == 1
        state.cold_electron_temperature[1] = 200.0
        set_fokker_planck_coefficients!(op, state, full)
        @test row_value(by_hand, 0.0) == fresh_bare()
        @test row_value(by_hand, 0.0) != built
        @test by_hand.rebuilds[] == 2
        # cleared, the term is zero, and built again from an equal plasma it is rebuilt
        zero_coefficients!(op)
        @test row_value(by_hand, 0.0) == 0
        set_fokker_planck_coefficients!(op, state, full)
        @test row_value(by_hand, 0.0) == fresh_bare()
        @test row_value(by_hand, 0.0) != 0
        @test by_hand.rebuilds[] == 3
        # equal settings of another object: no rebuild. Other settings: one rebuild.
        set_fokker_planck_coefficients!(op, state, CollisionSettings(mode = :full, screening = :partial))
        @test row_value(by_hand, 0.0) == fresh_bare()
        @test by_hand.rebuilds[] == 3
        set_fokker_planck_coefficients!(
            op, state, CollisionSettings(mode = :superthermal, screening = :partial))
        @test row_value(by_hand, 0.0) == fresh_bare()
        @test by_hand.rebuilds[] == 4
    end

    @testset "the loss under schemes that depend on the side of the advection" begin
        # A bulk of 100 eV with a monotone tail on a grid from p = 0, two radii with different
        # pitch grids, collisions with a synchrotron loss (B = 5 T) and bremsstrahlung as the
        # sinks, and a field of 2 V/m, which beats the friction above p ≈ 0.06 at ξ of its
        # sign, so that the side of the whole advection differs from that of the collisions
        T1, n1 = 100.0, 1e19
        p_faces = uniform_p_faces(60, 0.0, 0.6)
        radii = RadialGrid(uniform_r_faces(2, 0.5))
        hot = cylindrical_grid(
            [MomentumGrid(p_faces, uniform_xi_faces(6)),
                MomentumGrid(p_faces, -cos.(range(0, π; length = 7)))],
            radii)
        # the same pitch grids above p_min = 0.1
        above = cylindrical_grid(
            [MomentumGrid(uniform_p_faces(30, 0.1, 0.6), uniform_xi_faces(6)),
                MomentumGrid(uniform_p_faces(30, 0.1, 0.6), -cos.(range(0, π; length = 7)))],
            radii)
        collision_settings = CollisionSettings(mode = :full, screening = :complete,
            coulomb_logarithm = :energy_dependent, bremsstrahlung = :stopping_power)
        hot_region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        theta = T1 / Constants.ELECTRON_REST_ENERGY
        function monotone_tail(grid)
            g = zeros(size(grid))
            for ir in 1:2
                mr = momentum_grid(grid, ir)
                for (j, xi) in enumerate(mr.xi_centers), (i, p) in enumerate(mr.p_centers)

                    bulk = n1 * exp(-p^2 / (1 + sqrt(1 + p^2)) / theta) / (2π * theta)^1.5
                    g[i, j, ir] = (bulk + 1e20 * exp(-p / 0.1)) * (1 + 0.3xi)
                end
            end
            return g
        end
        tail = monotone_tail(hot)
        function loss_case(scheme, field; xi_scheme = scheme, field_unknown = false,
                region = hot_region, grid = hot, boundaries = (;), lagged = :ions)
            plasma = PlasmaState(; cold_electron_temperature = fill(T1, 2),
                ions = [IonSpecies("D", 1, 1, fill(n1, 2))], electric_field = fill(field, 2),
                magnetic_field = fill(5.0, 2), cold_electron_density = fill(n1, 2))
            unknowns = [KineticUnknown(:f, grid; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19)]
            if field_unknown
                push!(unknowns, RadialUnknown(:electric_field))
            end
            case_layout = UnknownLayout(radii, unknowns)
            state = zeros(length(case_layout))
            f0 = grid === hot ? tail : monotone_tail(grid)
            kinetic_array(state, case_layout, 1) .= f0
            unknown_values(state, case_layout, 2) .= T1
            unknown_values(state, case_layout, 3) .= n1
            op = AdvectionDiffusion(
                grid; scheme, xi_scheme, p_upper_boundary = :f_zero, boundaries...)
            source = if field_unknown
                unknown_values(state, case_layout, 4) .= field
                set_unit_field_coefficients!(op)
                FokkerPlanckTerm(:f, op; plasma = PlasmaFromUnknowns(plasma),
                    settings = collision_settings, electric_field = :electric_field, lagged)
            else
                FokkerPlanckTerm(:f, op; plasma = PlasmaFromUnknowns(plasma),
                    settings = collision_settings, lagged)
            end
            sinks = KineticEnergyLossTerm(source; part = :sinks, region, coefficient = 1)
            free = KineticEnergyLossTerm(source; part = :free_electrons, region)
            if lagged === :all
                Runes2P.prepare_step!(source, state, case_layout, 1, 0.0)
                Runes2P.prepare_step!(sinks, state, case_layout, 2, 0.0)
                Runes2P.prepare_step!(free, state, case_layout, 2, 0.0)
            end
            whole = Runes2P._operator_at(source, state, case_layout, 0.0)
            side = Runes2P._loss_side(sinks, state, case_layout, 0.0)
            collisions, electron_electron = Runes2P._loss_operators(sinks, grid,
                Runes2P._transfer_plasma(sinks, state, case_layout, 0.0))
            # the field part: the advection of the whole without the collisions
            field_part = AdvectionDiffusion(grid; scheme, xi_scheme, p_upper_boundary = :interface)
            field_part.advection_p .= _p_advection(whole, state, case_layout, source) .-
                                      collisions.advection_p
            return (; layout = case_layout, state, f0, source, sinks, free, whole, side,
                collisions, electron_electron, field_part)
        end
        # the advection of the whole on the p faces, with the field of an unknown
        function _p_advection(op, state, case_layout, source)
            source.electric_field === nothing && return op.advection_p
            E = unknown_values(state, case_layout, unknown_id(case_layout, source.electric_field))
            return op.advection_p .+ reshape(E, 1, 1, :) .* op.unit_field_advection_p
        end
        # the rows of the radii in the row 2 of the temperature
        rows_of(layout) = [first(Runes2P.unknown_range(layout, 2, ir)) for ir in 1:2]
        limiters = (:tcdf, :muscl, :smart, :ospre, :van_albada)
        linear_schemes = (
            :upwind, :quick, :upwind_second_order, :central, :exponential_fitting)

        @testset "the face values of the flux functions with a side ($scheme, E = $field)" for scheme in (
                :upwind, :quick, :upwind_second_order, limiters...),
            field in (2.0, -2.0)

            # Face by face, the field part, the free-electron part and the sinks with the side of
            # the whole sum to the momentum flux of the whole to round-off (2e-16 to 6e-16 of the
            # largest flux), while the parts on their own sides differ from it where the field
            # beats the friction: by 0.070 to 0.18 of the largest flux under the limiters and
            # by 0.065 to 0.44 under the linear schemes, measured
            (; layout, state, side, whole, collisions,
                electron_electron, field_part, source) = loss_case(
                scheme, field)
            field_values = Runes2P._field_values(source.electric_field, state, layout)
            flux(op, k, j, ir, s = nothing) = Runes2P._p_flux(
                op, tail, k, j, ir, nothing, nothing, s)
            shared, own, largest = 0.0, 0.0, 0.0
            for ir in 1:2, j in 1:6, k in 1:60
                reference = Runes2P._p_flux(whole, tail, k, j, ir, nothing, field_values)
                sinks_flux = flux(collisions, k, j, ir, side) -
                             flux(electron_electron, k, j, ir, side)
                parts = flux(field_part, k, j, ir, side) +
                        flux(electron_electron, k, j, ir, side) +
                        sinks_flux
                own_sides = flux(field_part, k, j, ir) + flux(electron_electron, k, j, ir) +
                            (flux(collisions, k, j, ir) - flux(electron_electron, k, j, ir))
                shared = max(shared, abs(parts - reference))
                own = max(own, abs(own_sides - reference))
                largest = max(largest, abs(reference))
            end
            @test shared <= 2e-15 * largest
            @test own > 1e-2 * largest
        end

        @testset "the parts sum to the moment of the whole ($scheme, $xi_scheme, E = $field)" for (scheme, xi_scheme) in (
                ((limiter, limiter) for limiter in limiters)..., (:upwind, :tcdf)),
            field in (2.0, -2.0)

            # the energy moment of the whole over the region, in the operator's own form, is the
            # moment of the field part plus the rows of the free-electron part and of the sinks
            # (at most 6.4e-15 of it measured), also with upwind in p and a limiter in ξ alone.
            # The whole takes no flux through p_max here, as the parts.
            (; layout, state, side, whole, field_part, sinks, free, source) = loss_case(
                scheme, field; xi_scheme)
            interfaced = AdvectionDiffusion(
                hot; scheme, xi_scheme, p_upper_boundary = :interface)
            for name in (:advection_p, :diffusion_p_p, :advection_xi, :diffusion_xi_xi,
                :advection_p_times_p2_at_zero)
                getfield(interfaced, name) .= getfield(whole, name)
            end
            interfaced.advection_p .= _p_advection(whole, state, layout, source)
            df = apply!(zeros(size(tail)), interfaced, tail)
            sinks_rhs = Runes2P.rhs!(zeros(length(state)), sinks, state, layout, 2, 0.0)
            free_rhs = Runes2P.rhs!(zeros(length(state)), free, state, layout, 2, 0.0)
            for ir in 1:2
                envelope = Runes2P._envelope_at(
                    sinks.envelopes, hot, ir, state, layout, 0.0)
                weights = Runes2P._energy_weights(hot, ir) .* envelope
                moment = sum(weights .* view(df, :, :, ir))
                field_moment = Runes2P._face_energy_moment(
                    (field_part, nothing), tail, side, envelope, ir)
                # the coefficients 1 of both parts: the rows hold −(the moment)
                r = first(Runes2P.unknown_range(layout, 2, ir))
                parts = field_moment - sinks_rhs[r] - free_rhs[r]
                @test parts ≈ moment rtol = 1e-13
            end
        end

        @testset "the form of the term follows the linearity of the source" begin
            # a flux limiter in p, in ξ or in r makes the source nonlinear and gives the face
            # form; linear schemes in every direction keep the moment of the applied operators
            one_pitch_grid = cylindrical_grid(MomentumGrid(p_faces, uniform_xi_faces(6)), radii)
            for (p_scheme, xi_scheme, r_scheme, face) in (
                (:tcdf, :tcdf, :central, true),
                (:upwind, :tcdf, :upwind, true),
                (:exponential_fitting, :ospre, :upwind, true),
                (:upwind, :upwind, :tcdf, true),
                (:quick, :quick, :central, false),
                (:upwind, :central, :upwind, false))
                op = AdvectionDiffusion(one_pitch_grid; p_scheme, xi_scheme, r_scheme)
                source = FokkerPlanckTerm(:f, op; settings = collision_settings,
                    plasma = PlasmaFromUnknowns(PlasmaState(;
                        cold_electron_temperature = fill(T1, 2),
                        ions = [IonSpecies("D", 1, 1, fill(n1, 2))], electric_field = zeros(2),
                        magnetic_field = fill(5.0, 2))))
                @test Runes2P._face_form(KineticEnergyLossTerm(source; part = :sinks)) ==
                      face
            end
        end

        @testset "a limiter in ξ alone under exponential fitting in p ($part)" for part in (
            :sinks, :free_electrons)
            # exponential fitting takes no side, so the face form is the moment of the applied
            # operators to round-off of the moments of the cells
            (; layout, state, source) = loss_case(:exponential_fitting, 2.0; xi_scheme = :tcdf)
            term = KineticEnergyLossTerm(source; part, region = hot_region)
            @test Runes2P._face_form(term)
            out = Runes2P.rhs!(zeros(length(state)), term, state, layout, 2, 0.0)
            plasma = Runes2P._transfer_plasma(term, state, layout, 0.0)
            df = Runes2P._loss_change(Runes2P._loss_operators(term, hot, plasma), tail)
            for ir in 1:2
                cells = Runes2P._loss_weights(term, hot, ir, state, layout, 0.0) .*
                        view(df, :, :, ir)
                @test abs(out[rows_of(layout)[ir]] + term.coefficient * sum(cells)) <=
                      1e-14 * sum(abs, cells)
            end
        end

        @testset "a linear source keeps the moment of the applied operators ($scheme, $part)" for scheme in linear_schemes,
            part in (:sinks, :free_electrons)

            # With a field unknown that beats the friction, the row is the energy moment of
            # the operators applied to f, each on the side of its own advection, in that order
            # of the sums: the term reads no side and lists no field unknown
            (; layout, state, source) = loss_case(scheme, 2.0; field_unknown = true)
            term = KineticEnergyLossTerm(source; part, region = hot_region)
            @test !Runes2P._face_form(term)
            @test :electric_field ∉ Runes2P.dependencies(term)
            @test Runes2P.dependencies(term) ==
                  [:f, :cold_electron_temperature, :cold_electron_density]
            out = Runes2P.rhs!(zeros(length(state)), term, state, layout, 2, 0.0)
            plasma = Runes2P._transfer_plasma(term, state, layout, 0.0)
            df = Runes2P._loss_change(Runes2P._loss_operators(term, hot, plasma), tail)
            for ir in 1:2
                weights = Runes2P._loss_weights(term, hot, ir, state, layout, 0.0)
                @test out[rows_of(layout)[ir]] ==
                      -term.coefficient * sum(weights .* view(df, :, :, ir))
            end
        end

        @testset "the pitch fluxes leave the energy moment ($scheme)" for scheme in (:tcdf, :upwind)
            # the energy moment of the pitch divergence alone, with advection, diffusion and
            # cross diffusion in pitch, is zero to round-off of the moments of its cells (at
            # most 2.7e-17 of their absolute sum measured)
            (; layout, state, whole, sinks) = loss_case(scheme, 2.0)
            pitch = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
            pitch.advection_xi .= whole.advection_xi
            pitch.diffusion_xi_xi .= whole.diffusion_xi_xi
            pitch.diffusion_xi_p .= 0.1 .* whole.diffusion_xi_xi
            df = apply!(zeros(size(tail)), pitch, tail)
            for ir in 1:2
                envelope = Runes2P._envelope_at(
                    sinks.envelopes, hot, ir, state, layout, 0.0)
                cells = Runes2P._energy_weights(hot, ir) .* envelope .* view(df, :, :, ir)
                @test any(!iszero, cells)
                @test abs(sum(cells)) <= 1e-14 * sum(abs, cells)
            end
        end

        @testset "the face form of a linear scheme is the moment of apply! ($scheme)" for scheme in linear_schemes
            # With the face values on their own sides, the sum over the momentum faces is the
            # energy moment of the operators applied to f, at p = 0 with both of its
            # boundaries and at p_min > 0 with a prescribed flux. The random coefficients
            # cancel in the moment, so the bound is 1e-14 of the sum of the absolute moments
            # of the cells (at most 4.6e-17 measured). The same holds for the derivative by the
            # temperature through an envelope of dual numbers with a cell of value 0 and a
            # nonzero partial, the edge of a region at a face, which the sum keeps.
            (; layout, state, sinks) = loss_case(:upwind, 2.0)
            rng = Xoshiro(17)
            D = ForwardDiff.Dual{Runes2P.PlasmaColumnTag, Float64, 1}
            for (grid_d, boundaries) in ((hot, (; p_zero_boundary = :outflow)),
                (hot, (; p_zero_boundary = :zero_flux)), (
                above, (; p_lower_boundary = :prescribed)))
                op = AdvectionDiffusion(grid_d; scheme, p_upper_boundary = :interface, boundaries...)
                op.advection_p .= randn(rng, size(op.advection_p))
                op.diffusion_p_p .= rand(rng, size(op.diffusion_p_p))
                op.diffusion_p_xi .= 0.1 .* randn(rng, size(op.diffusion_p_xi))
                op.advection_xi .= randn(rng, size(op.advection_xi))
                op.advection_p_times_p2_at_zero .= randn(rng, size(op.advection_p_times_p2_at_zero))
                op.p_flux_at_lower_boundary .= randn(rng, size(op.p_flux_at_lower_boundary))
                g = rand(rng, size(grid_d)...)
                df = apply!(zeros(size(g)), op, g)
                np = size(grid_d)[1]
                for ir in 1:2
                    envelope = Runes2P._envelope_at(
                        sinks.envelopes, hot, ir, state, layout, 0.0)
                    envelope = grid_d === hot ? envelope : ones(30)
                    energies = Runes2P._energy_weights(grid_d, ir)
                    cells = energies .* envelope .* view(df, :, :, ir)
                    face = Runes2P._face_energy_moment(
                        (op, nothing), g, nothing, envelope, ir)
                    @test abs(face - sum(cells)) <= 1e-14 * sum(abs, cells)
                    # zero below cell 10, the value 0 and the partial −1 in it, and the
                    # cells above 1 with random partials
                    dual_envelope = zeros(D, np)
                    dual_envelope[10] = D(0.0, ForwardDiff.Partials((-1.0,)))
                    for i in 11:np
                        dual_envelope[i] = D(1.0, ForwardDiff.Partials((randn(rng),)))
                    end
                    dual_face = Runes2P._face_energy_moment(
                        (op, nothing), g, nothing, dual_envelope, ir)
                    dual_cells = [ForwardDiff.partials(dual_envelope[i], 1) *
                                  energies[i, j] * df[i, j, ir]
                                  for i in 1:np, j in axes(df, 2)]
                    @test abs(ForwardDiff.partials(dual_face, 1) - sum(dual_cells)) <=
                          1e-14 * sum(abs, dual_cells)
                end
            end
        end

        # the cases of the Jacobian: (p scheme, part, options of loss_case)
        jacobian_cases = (
            (:tcdf, :sinks, (;)), (:tcdf, :free_electrons, (;)), (:ospre, :sinks, (;)),
            (:ospre, :free_electrons, (;)), (:van_albada, :sinks, (;)),
            (:tcdf, :sinks, (; region = nothing, field = 200.0)),
            (:tcdf, :sinks,
                (; region = nothing, field = 200.0,
                    boundaries = (; p_zero_boundary = :zero_flux))),
            (:tcdf, :sinks,
                (; grid = above, boundaries = (; p_lower_boundary = :prescribed))),
            (:tcdf, :sinks, (; lagged = :all)), (:upwind, :sinks, (; xi_scheme = :tcdf)),
            (:exponential_fitting, :free_electrons, (; xi_scheme = :tcdf)))

        @testset "the Jacobian of the face form ($scheme, $part, $options)" for (scheme, part, options) in jacobian_cases
            # At a monotone f, where the limiters are smooth, the row in the columns of f
            # agrees with the automatic differentiation of the residual to 1e-15 of its
            # largest entry in them. Against central differences of 1e-6 of each value, the
            # columns of f agree to 1e-6 of their largest entry, and the columns of the
            # temperature and the density each to 1e-6 of the larger of the difference and
            # the row over the unknown (at most 1.9e-7 in the columns of f and 1.3e-9 in the others
            # measured). With the plasma lagged, the row has the column of the temperature
            # through the region alone.
            # Without a region and at 200 V/m the faces near p = 0 carry weight and read the
            # ghost cells.
            field = get(options, :field, 2.0)
            case_options = Base.structdiff(options, NamedTuple{(:field,)})
            (; layout, state, source, f0) = loss_case(scheme, field; case_options...)
            term = KineticEnergyLossTerm(source; part, region = get(options, :region, hot_region))
            if get(options, :lagged, :ions) === :all
                Runes2P.prepare_step!(term, state, layout, 2, 0.0)
            end
            @test Runes2P._face_form(term) && !is_linear(term)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 2)
            JL = sparse(
                first.(entries), last.(entries), zeros(length(entries)), length(state),
                length(state))
            Runes2P.add_jacobian!(JL, term, state, layout, 2, 0.0)
            loss_rhs(x) = Runes2P.rhs!(zeros(eltype(x), length(x)), term, x, layout, 2, 0.0)
            dense = ForwardDiff.jacobian(loss_rhs, state)
            function difference(column, r)
                h = 1e-6 * abs(state[column])
                up, down = copy(state), copy(state)
                up[column] += h
                down[column] -= h
                return (loss_rhs(up)[r] - loss_rhs(down)[r]) / 2h
            end
            for ir in 1:2
                r = first(Runes2P.unknown_range(layout, 2, ir))
                f_columns = Runes2P.unknown_range(layout, 1, ir)
                @test maximum(abs, Vector(JL[r, f_columns]) - dense[r, f_columns]) <=
                      1e-15 * maximum(abs, dense[r, f_columns])
                differences = [difference(column, r) for column in f_columns]
                f_error = maximum(abs, Vector(JL[r, f_columns]) - differences) /
                          maximum(abs, differences)
                @test f_error <= 1e-6
                row_value = abs(loss_rhs(state)[r])
                for id in (2, 3)
                    column = first(Runes2P.unknown_range(layout, id, ir))
                    d = difference(column, r)
                    scale = max(abs(d), row_value / abs(state[column]))
                    @test abs(JL[r, column] - d) <= 1e-6 * scale
                end
            end
        end

        @testset "the ghost cell at p = 0 enters the row without a region" begin
            # The friction points to p = 0 at the faces near it, which then read no ghost cell.
            # A field of 200 V/m, above the Dreicer field, turns the advection there at ξ of
            # its sign, and the mirrored ghost cell of :outflow and the zero one of :zero_flux
            # then give those faces different values, which the row holds without a region.
            row(case) = Runes2P.rhs!(zeros(length(case.state)), case.sinks, case.state,
                case.layout, 2, 0.0)[rows_of(case.layout)]
            outflow = loss_case(:tcdf, 200.0; region = nothing)
            zero_flux = loss_case(:tcdf, 200.0; region = nothing,
                boundaries = (; p_zero_boundary = :zero_flux))
            @test row(outflow) != row(zero_flux)
        end

        @testset "the sign of the field unknown" begin
            # the face form reads the field unknown of the source only through the sign of the
            # whole advection and has no column of it: a change of the field that keeps the
            # side of every face leaves the row unchanged exactly
            (; layout, state, sinks, whole) = loss_case(:tcdf, 2.0; field_unknown = true)
            @test Runes2P.dependencies(sinks) ==
                  [:f, :cold_electron_temperature, :cold_electron_density, :electric_field]
            @test !is_linear(sinks)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], sinks, layout, 2)
            @test !any(entry -> last(entry) in Runes2P.unknown_range(layout, 4, 1), entries)
            rows = rows_of(layout)
            loss_rhs(x) = Runes2P.rhs!(zeros(length(x)), sinks, x, layout, 2, 0.0)[rows]
            changed, reversed = copy(state), copy(state)
            unknown_values(changed, layout, 4) .*= 1 + 1e-6
            unknown_values(reversed, layout, 4) .*= -1
            sides(x) = sign.(whole.advection_p .+
                             reshape(unknown_values(x, layout, 4), 1, 1, :) .*
                             whole.unit_field_advection_p)
            @test sides(changed) == sides(state)
            @test loss_rhs(changed) == loss_rhs(state)
            @test loss_rhs(reversed) != loss_rhs(state)
            # the field of the plasma of the source gives the same side and the same row
            in_plasma = loss_case(:tcdf, 2.0)
            plasma_rhs = Runes2P.rhs!(zeros(length(in_plasma.state)), in_plasma.sinks,
                in_plasma.state, in_plasma.layout, 2, 0.0)
            @test plasma_rhs[rows_of(in_plasma.layout)] ≈ loss_rhs(state) rtol = 1e-13
        end

        @testset "the jump of the parts where a face changes its side" begin
            # Under TCDF a face changes its side where the field unknown cancels the advection
            # of the collisions. Just below and just above the field of a face inside the
            # region, each part jumps, while the field part, the free-electron part and the
            # sinks together stay continuous to round-off of the row.
            (; layout, state, whole, sinks) = loss_case(:tcdf, 2.0; field_unknown = true)
            region_weights = [Runes2P._envelope_at(
                                  sinks.envelopes, hot, ir, state, layout, 0.0)[k]
                              for k in 1:60, j in 1:6, ir in 1:2]
            flips = -whole.advection_p[1:60, :, :] ./
                    whole.unit_field_advection_p[1:60, :, :]
            candidates = [(abs(flips[k, j, ir] - 3.56), k, j, ir)
                          for k in 2:60, j in 1:6, ir in 1:2
                          if region_weights[k, j, ir] > 0 && 1 < flips[k, j, ir] < 10]
            @test !isempty(candidates)
            _, k, j, ir = minimum(candidates)
            E_flip = flips[k, j, ir]
            function rows_at(E)
                case = loss_case(:tcdf, E; field_unknown = true)
                free_rhs = Runes2P.rhs!(zeros(length(case.state)), case.free, case.state,
                    case.layout, 2, 0.0)
                sinks_rhs = Runes2P.rhs!(zeros(length(case.state)), case.sinks, case.state,
                    case.layout, 2, 0.0)
                envelope = Runes2P._envelope_at(
                    case.sinks.envelopes, hot, ir, case.state, case.layout, 0.0)
                field_moment = Runes2P._face_energy_moment(
                    (case.field_part, nothing), tail, case.side, envelope, ir)
                r = rows_of(case.layout)[ir]
                return (; free = free_rhs[r], sinks = sinks_rhs[r], field = field_moment)
            end
            below = rows_at(E_flip * (1 - 1e-12))
            above_flip = rows_at(E_flip * (1 + 1e-12))
            jump_free = above_flip.free - below.free
            jump_sinks = above_flip.sinks - below.sinks
            # the rows hold −(the moment) with the coefficients 1 of both parts
            whole_below = below.field - below.free - below.sinks
            whole_above = above_flip.field - above_flip.free - above_flip.sinks
            @info "a face changes its side at $(E_flip) V/m: the free-electron row jumps by " *
                  "$(jump_free) W/m³ ($(abs(jump_free / below.free)) of the row), the sinks row by " *
                  "$(jump_sinks) W/m³, the sum of the parts by $(whole_above - whole_below) W/m³"
            @test abs(jump_free) > 1e-12 * abs(below.free)
            @test abs(jump_sinks) > 1e-12 * abs(below.sinks)
            @test abs(whole_above - whole_below) <= 1e-4 * abs(jump_free)
        end

        # A small system: f with a Fokker-Planck term in a field unknown that the system holds
        # at a prescribed field, and the row of the energy that f loses to the sinks
        function sinks_system(scheme, xi_scheme, field)
            plasma = PlasmaState(; cold_electron_temperature = fill(T1, 2),
                ions = [IonSpecies("D", 1, 1, fill(n1, 2))], electric_field = zeros(2),
                magnetic_field = fill(5.0, 2), cold_electron_density = fill(n1, 2))
            layout = UnknownLayout(radii,
                [KineticUnknown(:f, hot; scale = 1e19), RadialUnknown(:electric_field),
                    RadialUnknown(:sink_energy; scale = 1e3)])
            op = AdvectionDiffusion(hot; scheme, xi_scheme, p_upper_boundary = :f_zero)
            set_unit_field_coefficients!(op)
            source = FokkerPlanckTerm(:f, op; plasma, settings = collision_settings,
                electric_field = :electric_field)
            sinks = KineticEnergyLossTerm(source; part = :sinks, coefficient = 1)
            equations = [Equation(:f, TimeDerivative(), source),
                Equation(:electric_field, Algebraic(), PrescribedValueTerm(field),
                    LinearTerm(:electric_field; coefficient = -1)),
                Equation(:sink_energy, TimeDerivative(), sinks)]
            u0 = zeros(length(layout))
            kinetic_array(u0, layout, 1) .= tail
            unknown_values(u0, layout, 2) .= field(0.0)
            system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
            return (; layout, u0, system, op, source)
        end

        @testset "a system with a limiter in pitch alone steps" begin
            # upwind in p and TCDF in ξ: the term takes the face form, and two steps of 1e-8 s
            # at 2 V/m converge
            (; layout, u0, system) = sinks_system(:upwind, :tcdf, t -> 2.0)
            solution = integrate(system, u0, [0.0, 1e-8, 2e-8]; stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            @test unknown_values(solution.states[:, end], layout, 3) != zeros(2)
            @info "upwind in p and TCDF in ξ: Newton iterations per step $(solution.statistics.newton_iterations)"
        end

        @testset "a system whose faces change their side between steps" begin
            # Under TCDF the field rises from 3.0 to 4.2 V/m over four steps, so faces inside the
            # hot grid change their side between the steps and within the Newton iterations of
            # each step, where the field moves from its old value to the new one. The steps
            # converge within the default tolerances.
            ramp(t) = 3.0 + 1.2 * t / 4e-8
            (; layout, u0, system, op, source) = sinks_system(:tcdf, :tcdf, ramp)
            times = collect(range(0.0, 4e-8; length = 5))
            solution = integrate(system, u0, times; stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            whole = Runes2P._operator_at(source, u0, layout, 0.0)
            sides(E) = sign.(whole.advection_p[2:60, :, :] .+
                             reshape(E, 1, 1, :) .*
                             whole.unit_field_advection_p[2:60, :, :])
            changed = [count(sides(unknown_values(solution.states[:, s], layout, 2)) .!=
                             sides(unknown_values(solution.states[:, s + 1], layout, 2)))
                       for s in 1:4]
            @test sum(changed) > 0
            @info "TCDF with the field from 3.0 to 4.2 V/m: faces that change their side per step " *
                  "$changed, Newton iterations per step $(solution.statistics.newton_iterations)"
        end

        @testset "allocations of the face form" begin
            # the sum over the faces allocates nothing, nor do the columns of f of the row. The
            # residual allocates its operators (_loss_operators) and at most 4096 bytes more
            # (2048 measured), and the Jacobian less than that of the applied operators under
            # exponential fitting
            (; layout, state, source, side, collisions, electron_electron) = loss_case(:tcdf, 2.0)
            term = KineticEnergyLossTerm(source; part = :sinks)
            ops = (collisions, electron_electron)
            @test call_allocation(
                Runes2P._face_energy_moment, ops, tail, side, nothing, 1) == 0 skip = !OPTIMIZED
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 2)
            JL = sparse(
                first.(entries), last.(entries), zeros(length(entries)), length(state),
                length(state))
            colored = Runes2P.ColoredCells(tail)
            @test call_allocation(Runes2P._add_face_loss_columns!, JL, term, ops, colored,
                side, hot, state, layout, 2, 0.0) == 0 skip = !OPTIMIZED
            fitted = KineticEnergyLossTerm(loss_case(:exponential_fitting, 2.0).source; part = :sinks)
            out = zeros(length(state))
            plasma = Runes2P._transfer_plasma(term, state, layout, 0.0)
            operators = call_allocation(Runes2P._loss_operators, term, hot, plasma)
            residual = call_allocation(Runes2P.rhs!, out, term, state, layout, 2, 0.0)
            jacobian = call_allocation(
                Runes2P.add_jacobian!, JL, term, state, layout, 2, 0.0)
            fitted_jacobian = call_allocation(
                Runes2P.add_jacobian!, JL, fitted, state, layout, 2, 0.0)
            @test residual <= operators + 4096 skip = !OPTIMIZED
            @test jacobian <= fitted_jacobian skip = !OPTIMIZED
        end
    end

    @testset "the energy of radial transport" begin
        # KineticEnergyChangeTerm of a Fokker-Planck term of radial transport: the energy moment
        # of the radial part of its operator, against the moment of the operator applied to f
        # by apply!, for the radial schemes, diffusion and advection with and without a
        # dependence on the momentum, uniform and nonuniform radii, one pitch grid and pitch
        # grids that differ between radii. Its volume integral is the energy flux through the
        # edge with one envelope at all radii, and with envelopes that differ it also holds
        # the energy that the transport carries across the threshold. The Jacobian against
        # ForwardDiff, with the column of the temperature of a thermal threshold.
        rng = Xoshiro(17)
        momentum = MomentumGrid(uniform_p_faces(12, 0.0, 0.6), uniform_xi_faces(4))
        uniform_radii = uniform_r_faces(5, 0.8)
        nonuniform_radii = [0.0, 0.05, 0.2, 0.3, 0.55, 0.8]
        one_pitch(r_faces) = cylindrical_grid(momentum, RadialGrid(r_faces))
        pitch_faces = [uniform_xi_faces(4), biuniform_theta_xi_faces(4, 2, 0.9),
            uniform_xi_faces(4), biuniform_theta_xi_faces(4, 2, 0.7), uniform_xi_faces(4)]
        per_radius(r_faces) = cylindrical_grid(
            [MomentumGrid(momentum.p_faces, xi) for xi in pitch_faces], RadialGrid(r_faces))
        energies = [energy_of(p) for p in momentum.p_centers]
        # the transport alone, with random coefficients that depend on p (or a profile in r
        # alone) and the edge `r_boundary`
        function transport_operator(grid, r_scheme, parts, p_dependent, r_boundary)
            op = AdvectionDiffusion(grid; scheme = :central, r_scheme, r_boundary,
                cross_diffusion = false)
            shape = size(op.diffusion_r_r)
            profile(k) = 0.5 + 0.1 * k
            if :diffusion in parts
                op.diffusion_r_r .= p_dependent ? rand(rng, shape...) :
                                    [profile(k)
                                     for _ in 1:shape[1], _ in 1:shape[2], k in 1:shape[3]]
            end
            if :advection in parts
                op.advection_r .= p_dependent ? randn(rng, shape...) :
                                  [-profile(k)
                                   for _ in 1:shape[1], _ in 1:shape[2], k in 1:shape[3]]
            end
            return op
        end
        function transport_case(grid, op; region = nothing)
            layout = UnknownLayout(grid.radial_grid,
                [KineticUnknown(:f, grid; scale = 1e16),
                    RadialUnknown(:cold_electron_temperature; scale = 100.0),
                    RadialUnknown(:total_electron_energy; scale = 1e3)])
            u = zeros(length(layout))
            kinetic_array(u, layout, 1) .= 1e16 .* (1 .+ rand(rng, size(grid)...))
            unknown_values(u, layout, 2) .= 100.0 .*
                                            (1 .+ 0.5 .* rand(rng, radial_cell_count(grid)))
            source = FokkerPlanckTerm(:f, op)
            term = KineticEnergyChangeTerm(source; region)
            Runes2P.validate(term, layout, 3)
            return (; layout, u, source, term)
        end
        radial_rows(layout, k) = [first(Runes2P.unknown_range(layout, k, ir))
                                  for ir in 1:radial_cell_count(layout)]
        row_values(term, layout, u) = Runes2P.rhs!(
            zeros(eltype(u), length(u)), term, u, layout, 3, 0.0)[radial_rows(layout, 3)]
        # the energy moment of apply! of the transport alone over the envelopes, and the sum
        # of the absolute values of its cells, which sets the scale of the round-off
        function applied_moment(grid, op, f, envelopes)
            df = apply!(zeros(size(f)), op, f)
            moments, scales = zeros(radial_cell_count(grid)), zeros(radial_cell_count(grid))
            for ir in 1:radial_cell_count(grid)
                m = momentum_grid(grid, ir)
                V = phase_space_jacobian(grid, ir)
                for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

                    cell = envelopes[ir][i] * energies[i] * df[i, j, ir] * V[i, j] *
                           m.p_widths[i] * m.xi_widths[j] /
                           grid.radial_grid.spatial_jacobian[ir]
                    moments[ir] += cell
                    scales[ir] += abs(cell)
                end
            end
            return moments, scales
        end
        volume(grid) = grid.radial_grid.spatial_jacobian .* grid.radial_grid.r_widths
        no_envelope(grid) = [ones(12) for _ in 1:radial_cell_count(grid)]

        cases = [(r_scheme, parts, p_dependent, radii, pitch)
                 for r_scheme in (:central, :upwind, :exponential_fitting, :tcdf, :quick),
                     parts in ((:diffusion,), (:advection,), (:diffusion, :advection)),
                     p_dependent in (true, false), radii in (:uniform, :nonuniform),
                     pitch in (:one, :per_radius)
                 if !(pitch === :per_radius && Runes2P.is_four_point_scheme(r_scheme))]
        @testset "the moment of the operator ($r_scheme, $parts, p-dependent $p_dependent, $radii radii, $pitch pitch grid)" for (
                r_scheme, parts, p_dependent, radii, pitch) in cases
            r_faces = radii === :uniform ? uniform_radii : nonuniform_radii
            grid = pitch === :one ? one_pitch(r_faces) : per_radius(r_faces)
            for r_boundary in (:zero_flux, :absorbing, :absorbing_dream)
                op = transport_operator(grid, r_scheme, parts, p_dependent, r_boundary)
                (; layout, u, term) = transport_case(grid, op)
                f = kinetic_array(u, layout, 1)
                values = row_values(term, layout, u)
                moments, scales = applied_moment(grid, op, f, no_envelope(grid))
                @test maximum(abs.(values .- moments) ./ scales) < 1e-13
                # the energy only moves between the radii: the integral is the energy flux
                # through the edge, zero for a closed edge. The flux through the edge is
                # linear in the cell values of the last radius, so the particle flux of ε f is
                # the energy flux.
                edge = radial_particle_flux(op, energies .* f, radial_cell_count(grid) + 1)
                integral = sum(volume(grid) .* values)
                bound = 1e-13 * sum(volume(grid) .* scales)
                if r_boundary === :zero_flux
                    @test edge == 0
                    @test abs(integral) < bound
                else
                    @test edge >= 0
                    @test abs(integral + edge) < bound
                end
            end
        end

        @testset "envelopes that differ between the radii" begin
            # a fixed threshold per radius: the moment of apply! over the envelopes, and the
            # integral, which now holds the energy that the transport carries across the
            # threshold at the inner faces, Σ_k Σ_ij (w_{i,k} − w_{i,k−1}) ε_i Δp_i Δξ_j Φ_{ij,k}
            grid = one_pitch(uniform_radii)
            limits = [0.1, 0.25, 0.15, 0.3, 0.2]
            region = MomentumRegion(limits)
            op = transport_operator(
                grid, :central, (:diffusion, :advection), true, :zero_flux)
            (; layout, u, term) = transport_case(grid, op; region)
            f = kinetic_array(u, layout, 1)
            envelopes = [momentum_envelope(region, grid, ir) for ir in 1:5]
            values = row_values(term, layout, u)
            moments, scales = applied_moment(grid, op, f, envelopes)
            @test maximum(abs.(values .- moments) ./ scales) < 1e-13
            mode = Runes2P._radial_flux_mode(op)
            crossing = 0.0
            for k in 2:5, j in 1:4, i in 1:12
                flux = Runes2P._r_flux(op, f, i, j, k, k, nothing, mode)
                crossing += (envelopes[k][i] - envelopes[k - 1][i]) * energies[i] *
                            momentum.p_widths[i] * momentum.xi_widths[j] * flux
            end
            integral = sum(volume(grid) .* values)
            @test abs(crossing) > 1e-3 * sum(volume(grid) .* scales)
            @test integral ≈ crossing rtol = 1e-12
        end

        @testset "the Jacobian ($label)" for (label, r_scheme, pitch, region) in (
            ("central", :central, :one, nothing),
            ("TCDF", :tcdf, :one, nothing),
            ("QUICK, a fixed threshold", :quick, :one, MomentumRegion(0.2)),
            ("upwind, pitch grids per radius", :upwind, :per_radius, nothing),
            ("exponential fitting, a thermal threshold", :exponential_fitting, :one,
            MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 3);
                envelope = :cell_fraction)))
            grid = pitch === :one ? one_pitch(nonuniform_radii) :
                   per_radius(nonuniform_radii)
            op = transport_operator(
                grid, r_scheme, (:diffusion, :advection), true, :absorbing)
            (; layout, u, term) = transport_case(grid, op; region)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 3)
            J = sparse(first.(entries), last.(entries),
                zeros(length(entries)), length(u), length(u))
            Runes2P.add_jacobian!(J, term, u, layout, 3, 0.0)
            rows = radial_rows(layout, 3)
            reference = ForwardDiff.jacobian(x -> row_values(term, layout, x), u)
            @test count(!iszero, reference) > 100
            # every nonzero derivative is in the pattern, and the entries match it
            pattern = sparse(first.(entries), last.(entries),
                trues(length(entries)), length(u), length(u))
            @test all(pattern[rows, :][findall(!iszero, reference)])
            @test maximum(abs.(Matrix(J[rows, :]) .- reference)) <=
                  1e-12 * maximum(abs, reference)
            # the temperature column of the thermal threshold is in the Jacobian
            if region !== nothing && !isempty(Runes2P.region_temperature(region))
                T_columns = radial_rows(layout, 2)
                @test any(!iszero, reference[:, T_columns])
                @test Runes2P.dependencies(term) == [:f, :cold_electron_temperature]
            end
            thermal = region !== nothing && !isempty(Runes2P.region_temperature(region))
            @test Runes2P.is_linear(term) == (!thermal && r_scheme !== :tcdf)
        end

        @testset "allocations" begin
            # the residual and the columns of f of the Jacobian allocate nothing at a fixed
            # region, nor does the sum of one radius
            grid = one_pitch(uniform_radii)
            op = transport_operator(grid, :tcdf, (:diffusion, :advection), true, :absorbing)
            (; layout, u, term) = transport_case(grid, op; region = MomentumRegion(0.2))
            out = zeros(length(u))
            f = kinetic_array(u, layout, 1)
            envelope = momentum_envelope(MomentumRegion(0.2), grid, 1)
            mode = Runes2P._radial_flux_mode(op)
            @test call_allocation(Runes2P._transport_energy, op, f, envelope, 2, mode) == 0 skip = !OPTIMIZED
            @test call_allocation(Runes2P.rhs!, out, term, u, layout, 3, 0.0) == 0 skip = !OPTIMIZED
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 3)
            J = sparse(first.(entries), last.(entries),
                zeros(length(entries)), length(u), length(u))
            @test call_allocation(Runes2P.add_jacobian!, J, term, u, layout, 3, 0.0) == 0 skip = !OPTIMIZED
        end

        @testset "the source" begin
            grid = one_pitch(uniform_radii)
            # a term whose operator declares no radial transport, or a term with a plasma source,
            # which clears the declaration of its operator
            @test_throws "declares none" KineticEnergyChangeTerm(FokkerPlanckTerm(
                :f, AdvectionDiffusion(grid; radial_transport = false)))
            collisions = FokkerPlanckTerm(:f, AdvectionDiffusion(grid);
                plasma = PlasmaState(; cold_electron_temperature = fill(T0, 5),
                    ions = [IonSpecies("D", 1, 1, fill(n, 5))], electric_field = zeros(5),
                    magnetic_field = zeros(5)))
            @test_throws "declares none" KineticEnergyChangeTerm(collisions)
            @test_throws "or a FokkerPlanckTerm" KineticEnergyChangeTerm(LossRateTerm(:f, fill(1e3, 5)))
            # the record names the source and the region
            op = transport_operator(grid, :central, (:diffusion,), false, :absorbing)
            term = KineticEnergyChangeTerm(FokkerPlanckTerm(:f, op); region = MomentumRegion(0.2))
            record = Runes2P.describe(term, DescriptionContext())
            @test record["kinetic"] == "f" && record["coefficient"] == 1.0
            @test haskey(record, "source") && haskey(record, "region")
            # and a region that follows the temperature, that of a run of the total energy
            thermal = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
            thermal_record = Runes2P.describe(
                KineticEnergyChangeTerm(FokkerPlanckTerm(:f, op); region = thermal),
                DescriptionContext())
            @test thermal_record["region"] ==
                  Runes2P.describe(thermal, DescriptionContext())
        end
    end
end
