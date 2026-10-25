# The radiative and bound-electron energy losses of the kinetic electrons: the
# bremsstrahlung cross section of Koch and Motz against its limits of Bethe-Heitler and
# Heitler, the synchrotron loss against the Larmor power, the loss of the
# hot electrons charged to their total energy, and the transfer to the cold electrons that
# remains exactly the free-electron part.
using QuadGK: quadgk

@testset "radiation losses" begin
    e, c, r0, me = Constants.ELEMENTARY_CHARGE,
    Constants.SPEED_OF_LIGHT, Constants.CLASSICAL_ELECTRON_RADIUS,
    Constants.ELECTRON_MASS
    alpha = Constants.FINE_STRUCTURE_CONSTANT

    @testset "bremsstrahlung cross section" begin
        # ∫₀^X ln(1 + x)/x dx in the three ranges of its evaluation
        for X in (0.3, 0.5, 1.0, 1.9, 2.0, 7.0, 1e4)
            @test Runes2P._bremsstrahlung_integral(X) ≈
                  quadgk(x -> log1p(x) / x, 0, X; rtol = 1e-13)[1] rtol = 1e-11
        end
        # Bethe and Heitler without screening, p → 0: 16/3 α r₀² Z²
        @test bremsstrahlung_cross_section(1e-4) ≈ 16 / 3 rtol = 1e-3
        # Heitler without screening, γ ≫ 1: 4 α r₀² Z² (ln 2γ − 1/3); the corrections fall as ln²γ/γ
        p = 2000.0
        gamma = sqrt(1 + p^2)
        @test bremsstrahlung_cross_section(p) ≈ 4 * (log(2gamma) - 1 / 3) rtol = 1e-2
        # the slowing down it adds: n Z² α r₀² c φ/β per charge state
        plasma = PlasmaState(; cold_electron_temperature = [10.0],
            ions = [IonSpecies("Ar", 18, 2, [1e18]), IonSpecies("D", 1, 1, [1e20])],
            electric_field = [0.0], magnetic_field = [0.0])
        with = CollisionModel(plasma, CollisionSettings(bremsstrahlung = :stopping_power))
        without = CollisionModel(plasma, CollisionSettings(bremsstrahlung = :neglect))
        for p in (0.5, 5.0, 50.0)
            beta = p / sqrt(1 + p^2)
            expected = (18^2 * 1e18 + 1e20) * alpha * r0^2 * c *
                       bremsstrahlung_cross_section(p) / beta
            @test slowing_down_frequency(with, 1, p) -
                  slowing_down_frequency(without, 1, p) ≈ expected rtol = 1e-12
        end
        @test_throws ArgumentError CollisionSettings(bremsstrahlung = :other)
        # DREAM's integer division writes the constant 4/3 of 4BN as 1
        # (`src/Equations/SlowingDownFrequency.cpp:234`); `constant = :dream` gives it, for the
        # comparisons with the stopping power on
        for p in (0.1, 1.0, 10.0)
            @test bremsstrahlung_cross_section(p; constant = :dream) -
                  bremsstrahlung_cross_section(p) ≈ 1 / 3 rtol = 1e-12
        end
        dream = CollisionModel(plasma,
            CollisionSettings(bremsstrahlung = :stopping_power, bremsstrahlung_constant = :dream))
        @test slowing_down_frequency(dream, 1, 5.0) -
              slowing_down_frequency(without, 1, 5.0) ≈
              (18^2 * 1e18 + 1e20) * alpha * r0^2 * c *
              bremsstrahlung_cross_section(5.0; constant = :dream) / (5 / sqrt(26)) rtol = 1e-12
        @test_throws ArgumentError CollisionSettings(bremsstrahlung_constant = :other)
    end

    @testset "a Maxwellian radiates Gould's bremsstrahlung" begin
        # The radiative stopping as a pure drag: the full operator on a Maxwellian at the
        # temperature of the bulk loses the bremsstrahlung power of the electrons on the
        # nuclei, whose nonrelativistic Born limit (φ → 16/3) averaged over the Maxwellian is
        # the formula of R. J. Gould, Astrophys. J. 238, 1026 (1980), with its relativistic
        # correction 11/24 T/mₑc² (GOULD_BREMSSTRAHLUNG_PREFACTOR, GOULD_ION_CORRECTION; at this
        # tolerance 19/24 would pass as well, test_independent_references.jl tells them apart). The
        # collisions with the free electrons conserve the energy of a Maxwellian at their
        # temperature, up to the discretization, so the loss is measured against the same
        # operator without bremsstrahlung (the e-e part alone changes the energy of this
        # Maxwellian by 1.6e-1 and 1.9e-1 of Gould's power on these grids). With DREAM's form
        # the drag has the thermal diffusion of ν_∥, and the Maxwellian radiates nothing.
        n_D = 1e20
        for T in (100.0, 1000.0)
            plasma = PlasmaState(;
                cold_electron_temperature = [T], ions = [IonSpecies("D", 1, 1, [n_D])],
                electric_field = [0.0], magnetic_field = [0.0])
            theta = T / Constants.ELECTRON_REST_ENERGY
            p_max = 12 * sqrt(2theta)
            maxwell_grid = cylindrical_grid(
                MomentumGrid(uniform_p_faces(400, 0.0, p_max), uniform_xi_faces(2)),
                RadialGrid(uniform_r_faces(1, 0.5)))
            mm = momentum_grid(maxwell_grid)
            fm = [exp(-(sqrt(1 + p^2) - 1) / theta)
                  for p in mm.p_centers, _ in 1:2, _ in 1:1]
            fm .*= n_D / density(maxwell_grid, fm)[1]
            power(diffusion; bremsstrahlung = :stopping_power) = begin
                settings = CollisionSettings(;
                    mode = :full, screening = :complete, bremsstrahlung,
                    bremsstrahlung_diffusion = diffusion)
                op = fokker_planck_operator(maxwell_grid, plasma, settings;
                    synchrotron = false, scheme = :exponential_fitting,
                    p_upper_boundary = :f_zero)
                df = zeros(size(fm))
                Runes2P.apply!(df, op, fm)
                -sum(Runes2P._energy_weights(maxwell_grid, 1) .* df[:, :, 1])
            end
            gould = Runes2P.GOULD_BREMSSTRAHLUNG_PREFACTOR * sqrt(T) * n_D * n_D *
                    (1 + Runes2P.GOULD_ION_CORRECTION * theta)
            collisions = power(:none; bremsstrahlung = :neglect)
            @test power(:none) - collisions ≈ gould rtol = 2e-3
            @test abs(power(:dream) - collisions) < 1e-3 * gould
        end
        @test CollisionSettings().bremsstrahlung_diffusion === :none
        @test_throws ArgumentError CollisionSettings(bremsstrahlung_diffusion = :other)
        # ν_∥ of the two forms: Θγ times the collisional part, or times the whole ν_s
        plasma = PlasmaState(;
            cold_electron_temperature = [500.0], ions = [IonSpecies("Ar", 18, 2, [1e18])],
            electric_field = [0.0], magnetic_field = [0.0])
        drag = CollisionModel(plasma, CollisionSettings(bremsstrahlung_diffusion = :none))
        dream = CollisionModel(plasma, CollisionSettings(bremsstrahlung_diffusion = :dream))
        neglect = CollisionModel(plasma, CollisionSettings(bremsstrahlung = :neglect))
        theta = 500.0 / Constants.ELECTRON_REST_ENERGY
        for p in (0.1, 1.0)
            @test parallel_diffusion_frequency(drag, 1, p) ≈
                  parallel_diffusion_frequency(neglect, 1, p) rtol = 1e-14
            @test parallel_diffusion_frequency(dream, 1, p) ≈
                  theta * sqrt(1 + p^2) * slowing_down_frequency(dream, 1, p) rtol = 1e-14
            @test slowing_down_frequency(drag, 1, p) == slowing_down_frequency(dream, 1, p)
        end
    end

    # electrons at p ≈ 5 on a grid with all pitches
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(400, 0.0, 10.0), uniform_xi_faces(8)), rad)
    m = momentum_grid(grid)
    V = phase_space_jacobian(grid, 1)
    volume(i, j) = V[i, j] * m.p_widths[i] * m.xi_widths[j] / rad.spatial_jacobian[1]
    f = [exp(-((p - 5) / 0.3)^2) for p in m.p_centers, _ in 1:8, _ in 1:1]
    f .*= 1e15 / density(grid, f)[1]
    energy_moment(df) = sum(Runes2P._energy_weights(grid, 1) .* df[:, :, 1])

    @testset "synchrotron against the Larmor power" begin
        # P = 2 σ_T c U_B γ²β² (1 − ξ²) per electron, σ_T = 8π r₀²/3, U_B = B²/(2μ₀)
        B = 5.0
        op = AdvectionDiffusion(grid; scheme = :upwind)
        add_synchrotron!(op,
            PlasmaState(;
                cold_electron_temperature = [10.0], ions = [IonSpecies("D", 1, 1, [1e20])],
                electric_field = [0.0], magnetic_field = [B]))
        df = zeros(size(f))
        Runes2P.apply!(df, op, f)
        sigma_T, U_B = 8π * r0^2 / 3, B^2 / (2 * Constants.VACUUM_PERMEABILITY)
        larmor = sum(2 * sigma_T * c * U_B * p^2 * (1 - m.xi_centers[j]^2) * f[i, j, 1] *
                     volume(i, j)
        for (i, p) in enumerate(m.p_centers), j in 1:8)
        @test -energy_moment(df) ≈ larmor rtol = 1e-2
    end

    # the hot electrons of a 20 eV argon plasma
    T0, n_D, n_Ar = 20.0, 1e20, 1e18
    argon = zeros(19, 1)
    argon[3, 1] = n_Ar
    base = PlasmaState(; cold_electron_temperature = [T0],
        ions = [IonSpecies("D", 1, 1, [n_D]), IonSpecies("Ar", 18, argon)],
        electric_field = [0.0], magnetic_field = [5.0], cold_electron_density = [n_D +
                                                                                 2n_Ar])
    layout = UnknownLayout(rad,
        [KineticUnknown(:f, grid; scale = 1e15),
            RadialUnknown(:cold_electron_temperature; scale = 10.0),
            RadialUnknown(:cold_electron_density; scale = 1e20)])
    u = zeros(length(layout))
    kinetic_array(u, layout, 1) .= f
    radial_values(u, layout, 2) .= T0
    radial_values(u, layout, 3) .= n_D + 2n_Ar
    fp(settings; synchrotron = false, lagged = :ions) = FokkerPlanckTerm(:f,
        AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
        plasma = PlasmaFromUnknowns(base), settings, synchrotron, lagged)
    loss(term) = only(radial_values(
        Runes2P.rhs!(
            zeros(length(u)), term, u, layout, 2, 0.0), layout, 2))

    @testset "both losses are on by default" begin
        @test CollisionSettings().bremsstrahlung === :stopping_power
        settings = CollisionSettings(mode = :full, screening = :complete)
        default = fokker_planck_operator(grid, base, settings)
        with = fokker_planck_operator(grid, base, settings; synchrotron = true)
        without = fokker_planck_operator(grid, base, settings; synchrotron = false)
        @test default.coefficient_source[].synchrotron
        @test default.advection_p == with.advection_p &&
              default.advection_xi == with.advection_xi
        @test default.advection_p != without.advection_p
        rebuilt = AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        set_fokker_planck_coefficients!(rebuilt, base, settings)
        @test rebuilt.advection_p == with.advection_p
        # a term that rebuilds a bare operator takes the same defaults, and one on a built
        # operator the options that built it
        bare = FokkerPlanckTerm(:f,
            AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base))
        @test bare.synchrotron && bare.settings.bremsstrahlung === :stopping_power
        @test !FokkerPlanckTerm(:f, without; plasma = PlasmaFromUnknowns(base)).synchrotron
        # the default slowing down holds the bremsstrahlung loss
        plasma = PlasmaState(;
            cold_electron_temperature = [10.0], ions = [IonSpecies("Ar", 18, 2, [1e18])],
            electric_field = [0.0], magnetic_field = [0.0])
        @test slowing_down_frequency(CollisionModel(plasma, CollisionSettings()), 1, 5.0) ==
              slowing_down_frequency(
            CollisionModel(plasma, CollisionSettings(bremsstrahlung = :stopping_power)),
            1, 5.0)
    end

    @testset "the sinks of the hot electrons" begin
        complete = CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :neglect)
        # bremsstrahlung alone: m c² β p ν_s,brems per electron
        brems = KineticEnergyLossTerm(
            fp(CollisionSettings(mode = :full, screening = :complete, bremsstrahlung = :stopping_power));
            part = :sinks, coefficient = 1)
        expected = sum(me * c^2 * p^2 / sqrt(1 + p^2) * (n_D + 18^2 * n_Ar) * alpha * r0^2 *
                       c * bremsstrahlung_cross_section(p) *
                       sqrt(1 + p^2) / p * f[i, j, 1] * volume(i, j)
        for (i, p) in enumerate(m.p_centers), j in 1:8)
        @test loss(brems) ≈ expected rtol = 1e-2
        # synchrotron alone: the Larmor power
        sync = KineticEnergyLossTerm(fp(complete; synchrotron = true); part = :sinks, coefficient = 1)
        sigma_T, U_B = 8π * r0^2 / 3, 5.0^2 / (2 * Constants.VACUUM_PERMEABILITY)
        larmor = sum(2 * sigma_T * c * U_B * p^2 * (1 - m.xi_centers[j]^2) * f[i, j, 1] *
                     volume(i, j)
        for (i, p) in enumerate(m.p_centers), j in 1:8)
        @test loss(sync) ≈ larmor rtol = 1e-2
        # no sink with complete screening, no radiation
        @test abs(loss(KineticEnergyLossTerm(fp(complete); part = :sinks, coefficient = 1))) <
              1e-10 * loss(sync)
        # the friction on bound electrons is a sink with partial screening
        @test loss(KineticEnergyLossTerm(
            fp(CollisionSettings(mode = :full, screening = :partial, bremsstrahlung = :neglect));
            part = :sinks,
            coefficient = 1)) > 0
        @test_throws ArgumentError KineticEnergyLossTerm(fp(complete); part = :other)
    end

    @testset "the cold electrons receive the free-electron part" begin
        # The hot electrons above the threshold lose energy to every channel; with the sinks
        # charged to the total energy, the cold electrons gain exactly the free-electron part
        # of what the hot electrons lose (the total energy holds the hot-tail grid only).
        settings = CollisionSettings(mode = :full, screening = :partial, bremsstrahlung = :stopping_power)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        full = UnknownLayout(rad,
            [KineticUnknown(:f, grid; scale = 1e15),
                RadialUnknown(:cold_electron_temperature; scale = 10.0),
                RadialUnknown(:cold_electron_density; scale = 1e20),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21),
                RadialUnknown(:total_electron_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        collisions = fp(settings; synchrotron = true)
        sinks = KineticEnergyLossTerm(collisions; part = :sinks, region)
        free = KineticEnergyLossTerm(collisions; part = :free_electrons, region)
        equations = [Equation(:f, TimeDerivative(), collisions),
            Equation(
                :cold_electron_temperature, TimeDerivative(:total_electron_energy), sinks,
                MomentumBoundaryFluxTerm(:f, collisions; weight = :energy, coefficient = -1)),
            Equation(
                :cold_electron_density, Algebraic(), PrescribedValueTerm([n_D + 2n_Ar]),
                LinearTerm(:cold_electron_density; coefficient = -1)),
            thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        w = zeros(length(full))
        kinetic_array(w, full, 1) .= f
        radial_values(w, full, 2) .= T0
        radial_values(w, full, 3) .= n_D + 2n_Ar
        system = EquationSystem(full, equations; time_scale = 1e-7, initial_state = w)
        dt = 1e-8
        solution = integrate(system, w, [0.0, dt]; stepper = FixedSteps())
        W_cold = [only(radial_values(solution.states[:, k], full, 4)) for k in 1:2]
        state = solution.states[:, 2]
        transfer = only(radial_values(
            Runes2P.rhs!(
                zeros(length(state)), free, state, full, 2, dt), full, 2))
        @test transfer > 0
        @test (W_cold[2] - W_cold[1]) / dt ≈ transfer rtol = 1e-6
        # the Jacobian of the sinks: the temperature (plasma and region), the density, two cells
        J = jacobian_prototype(system)
        jacobian!(J, system, state, dt)
        T_row = first(Runes2P.unknown_range(full, 2, 1))
        for (column, scale) in ((first(Runes2P.unknown_range(full, 2, 1)), 10.0),
            (first(Runes2P.unknown_range(full, 3, 1)), 1e20),
            (Runes2P._kinetic_index(full, 1, 1, 200, 3), 1e15), (
            Runes2P._kinetic_index(full, 1, 1, 210, 7), 1e15))
            h = 1e-6 * scale
            up, down = copy(state), copy(state)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(state)), zeros(length(state))
            Runes2P.rhs!(F_up, sinks, up, full, 2, dt)
            Runes2P.rhs!(F_down, sinks, down, full, 2, dt)
            single = copy(J)
            single.nzval .= 0
            Runes2P.add_jacobian!(single, sinks, state, full, 2, dt)
            # the sinks hardly depend on the temperature and the density (bound electrons,
            # bremsstrahlung and synchrotron do not read them): an absolute tolerance from the loss
            scale_of_loss = abs(only(radial_values(
                Runes2P.rhs!(zeros(length(state)), sinks, state, full, 2, dt), full, 2))) /
                            scale
            @test single[T_row, column]≈(F_up[T_row]-F_down[T_row])/(2h) rtol=1e-5 atol=1e-6*scale_of_loss
        end
        record = describe(sinks, Runes2P.DescriptionContext())
        @test record["part"] == "sinks" && record["coefficient"] == -1.0
    end
end
