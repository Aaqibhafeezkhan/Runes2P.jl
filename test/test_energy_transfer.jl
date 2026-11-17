# The energy that the runaway grid and the electrons leaving the hot-tail grid exchange with
# the total energy of the electrons: the collisional transfer of a
# runaway population against the friction power of superthermal electrons, its Jacobian
# with the plasma columns and with the plasma lagged (no plasma columns, in a term and in a
# step of the system), the Jacobian of the energy loss of f (KineticEnergyLossTerm) with a
# lagged source, its error when never prepared, its values in the prototype of a system and
# the plasma it keeps there when prepared before, and the energy carried through p_max and
# through the interface, with its column of the field when the interface reads the field,
# and the particles through the interface in a radial row (InterfaceParticleFluxTerm), with
# the rows of both on pitch grids that differ between radii. Under the schemes whose face
# value depends on the side of the advection, the loss of f in the form of the momentum
# faces: the parts that sum to the flux of the whole, the energy moment, the pitch fluxes
# that leave it, the form of the linear schemes, the Jacobian, the sign of the field and the
# allocations.
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
        collision_settings = CollisionSettings(mode = :full, screening = :complete,
            coulomb_logarithm = :energy_dependent, bremsstrahlung = :stopping_power)
        hot_region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        theta = T1 / Constants.ELECTRON_REST_ENERGY
        tail = zeros(60, 6, 2)
        for ir in 1:2
            mr = momentum_grid(hot, ir)
            for (j, xi) in enumerate(mr.xi_centers), (i, p) in enumerate(mr.p_centers)

                bulk = n1 * exp(-p^2 / (1 + sqrt(1 + p^2)) / theta) / (2π * theta)^1.5
                tail[i, j, ir] = (bulk + 1e20 * exp(-p / 0.1)) * (1 + 0.3xi)
            end
        end
        function loss_case(scheme, field; field_unknown = false, region = hot_region)
            plasma = PlasmaState(; cold_electron_temperature = fill(T1, 2),
                ions = [IonSpecies("D", 1, 1, fill(n1, 2))], electric_field = fill(field, 2),
                magnetic_field = fill(5.0, 2), cold_electron_density = fill(n1, 2))
            unknowns = [KineticUnknown(:f, hot; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:cold_electron_density; scale = 1e19)]
            if field_unknown
                push!(unknowns, RadialUnknown(:electric_field))
            end
            case_layout = UnknownLayout(radii, unknowns)
            state = zeros(length(case_layout))
            kinetic_array(state, case_layout, 1) .= tail
            unknown_values(state, case_layout, 2) .= T1
            unknown_values(state, case_layout, 3) .= n1
            op = AdvectionDiffusion(hot; scheme, p_upper_boundary = :f_zero)
            source = if field_unknown
                unknown_values(state, case_layout, 4) .= field
                set_unit_field_coefficients!(op)
                FokkerPlanckTerm(:f, op; plasma = PlasmaFromUnknowns(plasma),
                    settings = collision_settings, electric_field = :electric_field)
            else
                FokkerPlanckTerm(:f, op; plasma = PlasmaFromUnknowns(plasma),
                    settings = collision_settings)
            end
            sinks = KineticEnergyLossTerm(source; part = :sinks, region, coefficient = 1)
            free = KineticEnergyLossTerm(source; part = :free_electrons, region)
            whole = Runes2P._operator_at(source, state, case_layout, 0.0)
            side = Runes2P._loss_side(sinks, state, case_layout, 0.0)
            collisions, electron_electron = Runes2P._loss_operators(sinks, hot,
                Runes2P._transfer_plasma(sinks, state, case_layout, 0.0))
            # the field part: the advection of the whole without the collisions
            field_part = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
            field_part.advection_p .= _p_advection(whole, state, case_layout, source) .-
                                      collisions.advection_p
            return (; layout = case_layout, state, source, sinks, free, whole, side,
                collisions, electron_electron, field_part)
        end
        # the advection of the whole on the p faces, with the field of an unknown
        function _p_advection(op, state, case_layout, source)
            source.electric_field === nothing && return op.advection_p
            E = unknown_values(state, case_layout, unknown_id(case_layout, source.electric_field))
            return op.advection_p .+ reshape(E, 1, 1, :) .* op.unit_field_advection_p
        end
        side_schemes = (
            :upwind, :quick, :upwind_second_order, :tcdf, :muscl, :smart, :ospre,
            :van_albada)

        @testset "the parts sum to the flux of the whole ($scheme, E = $field)" for scheme in side_schemes,
            field in (2.0, -2.0)

            case = loss_case(scheme, field)
            (; layout, state, side, whole, collisions, electron_electron, field_part) = case
            (; sinks, free, source) = case
            field_values = Runes2P._field_values(source.electric_field, state, layout)
            # a: face by face, the field part, the free-electron part and the sinks with the
            # side of the whole sum to the momentum flux of the whole to round-off (2e-16 to
            # 6e-16 of the largest flux), while the parts on their own sides differ from it
            # (0.07 to 0.4 of the largest flux) where the field beats the friction
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
            # b: the energy moment of the whole over the region, in the operator's own form,
            # is the moment of the field part plus the free-electron part plus the sinks (at
            # most 6.4e-15 of it measured). The whole takes no flux through p_max here, as the parts.
            interfaced = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
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
                field_moment = Runes2P._face_loss(
                    (field_part, nothing), tail, side, envelope, ir)
                # the coefficients 1 of both parts: the rows hold −(the moment)
                r = first(Runes2P.unknown_range(layout, 2, ir))
                parts = field_moment - sinks_rhs[r] - free_rhs[r]
                @test parts ≈ moment rtol = 1e-13
            end
        end

        @testset "the pitch fluxes leave the energy moment ($scheme)" for scheme in (:tcdf, :upwind)
            # c: the energy moment of the pitch divergence alone, with advection, diffusion and
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

        @testset "the face form of a linear scheme is the moment of apply! ($scheme)" for scheme in (
            :upwind, :central, :exponential_fitting, :quick, :upwind_second_order)
            # d: with the face values on their own sides, the sum over the momentum faces is
            # the energy moment of the operators applied to f, at p = 0 with both of its
            # boundaries and at p_min > 0 with a prescribed flux. The random coefficients
            # cancel in the moment, so the bound is 1e-14 of the sum of the absolute moments
            # of the cells (at most 4.6e-17 measured).
            (; layout, state, sinks) = loss_case(:upwind, 2.0)
            rng = Xoshiro(17)
            above = cylindrical_grid(
                [MomentumGrid(uniform_p_faces(30, 0.1, 0.6), uniform_xi_faces(6)),
                    MomentumGrid(uniform_p_faces(30, 0.1, 0.6), -cos.(range(0, π; length = 7)))],
                radii)
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
                for ir in 1:2
                    envelope = Runes2P._envelope_at(
                        sinks.envelopes, hot, ir, state, layout, 0.0)
                    envelope = grid_d === hot ? envelope : ones(30)
                    cells = Runes2P._energy_weights(grid_d, ir) .* envelope .*
                            view(df, :, :, ir)
                    face = Runes2P._face_loss((op, nothing), g, nothing, envelope, ir)
                    @test abs(face - sum(cells)) <= 1e-14 * sum(abs, cells)
                end
            end
        end

        @testset "the Jacobian of the face form ($scheme, $part)" for scheme in (:tcdf, :ospre),
            part in (:sinks, :free_electrons)

            # e: at a monotone f, where the limiters are smooth, the row in the columns of f
            # agrees with the automatic differentiation of the residual to 1e-15 of its
            # largest entry in them. Against central differences of 1e-6 of each value, the
            # columns of f agree to 1e-6 of their largest entry, and the columns of the
            # temperature and the density each to 1e-6 of the larger of the difference and
            # the row over the unknown (at most 1.9e-7 in the columns of f and 1.3e-9 in the others
            # measured).
            (; layout, state, source) = loss_case(scheme, 2.0)
            term = KineticEnergyLossTerm(source; part, region = hot_region)
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

        @testset "the sign of the field unknown" begin
            # the face form reads the field unknown of the source only through the sign of the
            # whole advection and has no column of it: a change of the field that keeps the
            # side of every face leaves the row unchanged exactly
            (; layout, state, sinks, whole) = loss_case(:tcdf, 2.0; field_unknown = true)
            @test :electric_field in Runes2P.dependencies(sinks) && !is_linear(sinks)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], sinks, layout, 2)
            @test !any(entry -> last(entry) in Runes2P.unknown_range(layout, 4, 1), entries)
            rows = [first(Runes2P.unknown_range(layout, 2, ir)) for ir in 1:2]
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
            @test [plasma_rhs[first(Runes2P.unknown_range(in_plasma.layout, 2, ir))]
                   for ir in 1:2] ≈ loss_rhs(state) rtol = 1e-13
        end

        @testset "allocations of the face form" begin
            # the sum over the faces allocates nothing, nor do the columns of f of the row. The
            # residual allocates its operators (_loss_operators) and at most 4096 bytes more
            # (2048 measured), and the Jacobian less than that of the applied operators under
            # exponential fitting
            (; layout, state, source, side, collisions, electron_electron) = loss_case(:tcdf, 2.0)
            term = KineticEnergyLossTerm(source; part = :sinks)
            ops = (collisions, electron_electron)
            @test call_allocation(Runes2P._face_loss, ops, tail, side, nothing, 1) == 0 skip = !OPTIMIZED
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
end
