# The ion charge states in a fully kinetic system: every piece together
# in one run. A hot-tail grid that covers the bulk; the Fokker-Planck term reads the
# temperature, the cold density and the charge states of the iterate; the ionization is
# split between the fluid rate of the cold electrons and the kinetic rate of f above 5
# thermal momenta; the cold density is the closure n_free − n_hot of the ions; the particle
# source keeps the density of f at n_cold + n_hot; the temperature follows the total electron
# energy with the radiated power, the binding energy at the fluid
# coefficient and the losses of f to the bound electrons (default); the charged states diffuse with a zero flux at the
# edge. Neutral deuterium and argon deposited at the start, the step controller over 20 µs.
# Checked: the species are conserved, the closure and the constraint hold, the total energy
# is the cold energy and that of the tail, and the argon ionizes. Needs the ADF11 cache.

@testset "ions in a fully kinetic system" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the integrated run is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        composition = IonComposition(["D" => 1, "Ar" => 18])
        nr, T0 = 2, [30.0, 20.0]
        radial = RadialGrid(uniform_r_faces(nr, 0.5))
        p_max = 10 * sqrt(2 * maximum(T0) / Constants.ELECTRON_REST_ENERGY)
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(61, 0.0, p_max), uniform_xi_faces(3)), radial)
        background = [IonSpecies("D", 1, 1, fill(2e19, nr)), IonSpecies("Ar", 18, zeros(19, nr))]
        prescribed = PlasmaState(; cold_electron_temperature=T0, ions=background, electric_field=zeros(nr),
                                 magnetic_field=fill(2.5, nr))
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=5))
        layout = UnknownLayout(radial, [ion_densities_unknown(composition),
                                        RadialUnknown(:cold_electron_density; scale=1e19),
                                        RadialUnknown(:hot_electron_density; scale=1e13),
                                        RadialUnknown(:particle_source; scale=1e24),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0),
                                        RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e20),
                                        RadialUnknown(:total_electron_energy; scale=THERMAL_ENERGY_FACTOR * 1e20),
                                        KineticUnknown(:f, grid; scale=1e19)])
        minus(name) = LinearTerm(name; coefficient=-1)
        source = PlasmaFromUnknowns(prescribed; ions=:ion_densities)
        fp = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero);
                              plasma=source, settings=CollisionSettings(mode=:full, screening=:partial))
        kinetic = KineticIonizationTerm(composition, :f, grid; region)
        radiation = RadiatedPowerTerm(prescribed; ions=:ion_densities, kinetic)
        equations = [
            Equation(:ion_densities, TimeDerivative(), IonRateTerm(composition; kinetic), kinetic,
                     IonDiffusionTerm(composition; diffusion=1.0)),
            population_equations(composition; hot_density=:hot_electron_density)...,
            Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
            Equation(:particle_source, Constraint(), DensityMomentTerm(:f), minus(:cold_electron_density),
                     minus(:hot_electron_density); scale=1e19),
            Equation(:f, TimeDerivative(), fp, ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature)),
            # the losses of f to the bound electrons and the radiation leave the electrons
            # (part = :sinks); the energy of the kinetic ionization is among them
            Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy), radiation,
                     KineticEnergyLossTerm(fp; part=:sinks, region),
                     MomentumBoundaryFluxTerm(:f, fp; weight=:energy, coefficient=-1)),
            thermal_energy_equation(), total_energy_equation(:f, grid; region)]
        u0 = zeros(length(layout))
        set_ion_densities!(u0, layout, background)
        deposit_neutrals!(u0, layout, composition, "D" => 2e18, "Ar" => 2e17)
        n_free = [free_electron_density(ion_densities(u0, layout, composition), ir) for ir in 1:nr]
        radial_values(u0, layout, 5) .= T0
        for ir in 1:nr
            m = momentum_grid(grid, ir)
            theta = T0[ir] / Constants.ELECTRON_REST_ENERGY
            shape = [exp(-p^2 / (sqrt(1 + p^2) + 1) / theta) for p in m.p_centers, _ in 1:3]
            kinetic_view(u0, layout, 8, ir) .= n_free[ir] .* shape ./ sum(kinetic.weights[ir] .* shape)
        end
        hot = Runes2P.rhs!(zeros(length(u0)), DensityMomentTerm(:f, grid; region), u0, layout, 3, 0.0)
        radial_values(u0, layout, 3) .= radial_values(hot, layout, 3)
        radial_values(u0, layout, 2) .= n_free .- radial_values(u0, layout, 3)
        system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u0)
        solution = integrate(system, u0, [0.0, 20e-6]; stepper=StepController(; relative_tolerance=1e-4,
                                                                                 initial_step=1e-9))
        @test run_status(solution) === :finished
        statistics = solution.statistics
        accepted, rejected = count(statistics.accepted), count(!, statistics.accepted)
        @info "ions in a fully kinetic system: $(jacobian_colors(system)) colors of the automatic Jacobian, " *
              "$accepted accepted steps, $rejected rejected, " *
              "$(sum(statistics.newton_iterations)) Newton iterations, statuses of the rejected " *
              "$(unique(statistics.status[.!statistics.accepted]))"
        @test rejected <= accepted
        final = solution.states[:, end]
        rg = radial
        volume(ir) = rg.spatial_jacobian[ir] * rg.r_widths[ir]
        # each species conserved over the plasma (the rates and the zero-flux diffusion)
        for s in 1:2
            total(x) = sum(volume(ir) * sum(radial_view(x, layout, 1, ir)[species_components(composition, s)]) for ir in 1:nr)
            @test total(final) ≈ total(solution.states[:, 1]) rtol = 1e-9
        end
        # the closure and the constraint at the end
        ions = ion_densities(final, layout, composition)
        moment(term, row) = radial_values(Runes2P.rhs!(zeros(length(final)), term, final, layout, row, 20e-6), layout, row)
        n_hot = moment(DensityMomentTerm(:f, grid; region), 3)
        n_f = moment(DensityMomentTerm(:f), 2)
        n_cold = radial_values(final, layout, 2)
        for ir in 1:nr
            @test n_cold[ir] ≈ free_electron_density(ions, ir) - n_hot[ir] rtol = 1e-9
            @test n_f[ir] ≈ n_cold[ir] + n_hot[ir] rtol = 1e-8
            @test radial_values(final, layout, 3)[ir] ≈ n_hot[ir] rtol = 1e-8
        end
        # the electrons follow the ionization: the free electrons grew with the argon charge
        @test all(ir -> free_electron_density(ions, ir) > n_free[ir], 1:nr)
        mean_charge = [sum((0:18) .* ions[2].charge_state_densities[:, ir]) / sum(ions[2].charge_state_densities[:, ir])
                       for ir in 1:nr]
        @test all(>(0.5), mean_charge)
        # the total electron energy is the cold energy and that of f above the threshold, and it
        # fell by the radiated power and the binding energy
        energy_above = moment(EnergyMomentTerm(:f, grid; region), 7)
        W_cold, W_tot = radial_values(final, layout, 6), radial_values(final, layout, 7)
        @test W_tot ≈ W_cold .+ energy_above rtol = 1e-9
        @test all(W_tot .< radial_values(solution.states[:, 1], layout, 7))
        @test all(radial_values(final, layout, 5) .< T0)
    end
end

# The sign test of the heating: a cold neon plasma (n_bound ≫ n_free) with a
# superthermal tail of 1e-3 at 300 eV, the same fully kinetic system as above, run for 2 µs
# with and without the heating by the secondaries of the kinetic ionization. The cold
# electrons end warmer with it, by the energy of the secondaries that the friction took from
# f. Neon has a sourced width of the secondary spectrum (24.2 eV).
function _neon_run(heating::Bool)
    composition = IonComposition(["D" => 1, "Ne" => 10])
    nr, T0 = 1, [10.0]
    radial = RadialGrid(uniform_r_faces(nr, 0.5))
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(80, 0.0, 0.08), uniform_xi_faces(3)), radial)
    background = [IonSpecies("D", 1, 1, fill(2e19, nr)), IonSpecies("Ne", 10, zeros(11, nr))]
    prescribed = PlasmaState(; cold_electron_temperature=T0, ions=background, electric_field=zeros(nr),
                             magnetic_field=fill(2.5, nr))
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=5))
    layout = UnknownLayout(radial, [ion_densities_unknown(composition),
                                    RadialUnknown(:cold_electron_density; scale=1e19),
                                    RadialUnknown(:hot_electron_density; scale=1e16),
                                    RadialUnknown(:particle_source; scale=1e24),
                                    RadialUnknown(:cold_electron_temperature; scale=10.0),
                                    RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e20),
                                    RadialUnknown(:total_electron_energy; scale=THERMAL_ENERGY_FACTOR * 1e20),
                                    KineticUnknown(:f, grid; scale=1e19)])
    minus(name) = LinearTerm(name; coefficient=-1)
    fp = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero);
                          plasma=PlasmaFromUnknowns(prescribed; ions=:ion_densities),
                          settings=CollisionSettings(mode=:full, screening=:partial))
    kinetic = KineticIonizationTerm(composition, :f, grid; region)
    energy_terms = EquationTerm[RadiatedPowerTerm(prescribed; ions=:ion_densities, kinetic),
                                KineticEnergyLossTerm(fp; part=:sinks, region),
                                MomentumBoundaryFluxTerm(:f, fp; weight=:energy, coefficient=-1)]
    heating && push!(energy_terms, SecondaryHeatingTerm(kinetic))
    equations = [
        Equation(:ion_densities, TimeDerivative(), IonRateTerm(composition; kinetic), kinetic),
        population_equations(composition; hot_density=:hot_electron_density)...,
        Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
        Equation(:particle_source, Constraint(), DensityMomentTerm(:f), minus(:cold_electron_density),
                 minus(:hot_electron_density); scale=1e19),
        Equation(:f, TimeDerivative(), fp, ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature)),
        Equation(:cold_electron_temperature, TimeDerivative(:total_electron_energy), Tuple(energy_terms)),
        thermal_energy_equation(), total_energy_equation(:f, grid; region)]
    u0 = zeros(length(layout))
    set_ion_densities!(u0, layout, background)
    deposit_neutrals!(u0, layout, composition, "Ne" => 2e19)
    n_free = free_electron_density(ion_densities(u0, layout, composition), 1)
    radial_values(u0, layout, 5) .= T0
    m = momentum_grid(grid, 1)
    shape(T) = [exp(-p^2 / (sqrt(1 + p^2) + 1) / (T / Constants.ELECTRON_REST_ENERGY)) for p in m.p_centers, _ in 1:3]
    normalized(T) = shape(T) ./ sum(kinetic.weights[1] .* shape(T))
    kinetic_view(u0, layout, 8, 1) .= n_free .* ((1 - 1e-3) .* normalized(T0[1]) .+ 1e-3 .* normalized(300.0))
    hot = Runes2P.rhs!(zeros(length(u0)), DensityMomentTerm(:f, grid; region), u0, layout, 3, 0.0)
    radial_values(u0, layout, 3) .= radial_values(hot, layout, 3)
    radial_values(u0, layout, 2) .= n_free .- radial_values(u0, layout, 3)
    system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u0)
    solution = integrate(system, u0, collect(range(0.0, 2e-6; length=21)); stepper=FixedSteps())
    return solution, layout
end

@testset "heating by the secondaries of the kinetic ionization (issue #177)" begin
    if isdir(adas_cache_directory())
        with, layout = _neon_run(true)
        without, _ = _neon_run(false)
        @test run_status(with) === :finished && run_status(without) === :finished
        T_with = radial_values(with.states[:, end], layout, 5)[1]
        T_without = radial_values(without.states[:, end], layout, 5)[1]
        @info "neon plasma with a 300 eV tail after 2 µs: T_cold $T_with eV with the heating by secondaries, " *
              "$T_without eV without"
        @test T_with > T_without
    end
end
