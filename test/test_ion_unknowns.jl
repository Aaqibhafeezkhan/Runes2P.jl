# The ion charge states as unknowns: the composition of the species, the unknown with a group
# of components per species, its lower limit and Newton tolerance relative to the density of
# each species, the free-electron density as a weighted sum, the closure of the cold density,
# the deposition of neutrals, a plasma whose ions are read from the unknown, and the guards
# on the target of the e-e collisions.
using HDF5
using ForwardDiff

const ION_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

# deuterium and argon on `nr` radii, with argon at 1e-4 of the deuterium density
function _ion_layout(; nr = 2, extra = ())
    radial = RadialGrid(uniform_r_faces(nr, 0.5))
    unknowns = [ion_densities_unknown(ION_COMPOSITION),
        RadialUnknown(:cold_electron_density; scale = 1e19), extra...]
    return UnknownLayout(radial, unknowns)
end

function _ion_species(nr; deuterium = 1e19, argon = 1e15)
    argon_states = zeros(19, nr)
    argon_states[2, :] .= 0.5argon
    argon_states[3, :] .= 0.5argon
    return [IonSpecies("D", 1, 1, fill(deuterium, nr)), IonSpecies("Ar", 18, argon_states)]
end

@testset "ion unknowns" begin
    @testset "composition" begin
        c = ION_COMPOSITION
        @test length(c) == 2 && component_count(c) == 21
        @test species_components(c, 1) == 1:2 && species_components(c, "Ar") == 3:21
        @test species_index(c, "Ar") == 2
        @test_throws ArgumentError species_index(c, "Ne")
        @test IonComposition(_ion_species(1)) == c
        @test IonComposition(PlasmaState(;
            cold_electron_temperature = [1.0], ions = _ion_species(1),
            electric_field = [0.0], magnetic_field = [0.0])) == c
        @test_throws ArgumentError IonComposition(["D" => 1, "D" => 1])
        @test_throws ArgumentError IonComposition(["X" => 0])
        @test free_electron_weights(c) == [0.0; 1.0; 0.0:18.0]
    end

    @testset "unknown with groups and its record" begin
        u = ion_densities_unknown(ION_COMPOSITION)
        @test u.components == 21 && component_groups(u) == [1:2, 3:21]
        @test u.lower_limit.reference === :group && u.lower_limit.tolerance == 1e-10
        @test component_groups(RadialUnknown(:x; components = 3)) == [1:3]
        @test_throws ArgumentError RadialUnknown(:x; components = 3, groups = [1:1, 3:3])
        @test_throws ArgumentError RadialUnknown(:x; components = 3, groups = [1:2])
        @test_throws ArgumentError RadialUnknown(:x; components = 3, groups = [1:0, 1:3])
        # a group reference needs a radial unknown
        radial = RadialGrid(uniform_r_faces(1, 0.5))
        @test_throws ArgumentError UnknownLayout(
            radial, [ScalarUnknown(:s; nonnegative = LowerLimit(reference = :group))])
        layout = _ion_layout()
        record = Runes2P.describe(layout, Runes2P.DescriptionContext())
        @test record["unknowns"]["ion_densities"]["group_starts"] == [1, 3]
        @test record["unknowns"]["ion_densities"]["lower_limit_reference"] == "group"
        @test !haskey(record["unknowns"]["cold_electron_density"], "group_starts")
    end

    @testset "initial state, deposition and read back" begin
        nr = 3
        layout = _ion_layout(; nr)
        u = zeros(length(layout))
        ions = _ion_species(nr)
        set_ion_densities!(u, layout, ions)
        back = ion_densities(u, layout, ION_COMPOSITION)
        @test [s.charge_state_densities for s in back] ==
              [s.charge_state_densities for s in ions]
        deposit_neutrals!(
            u, layout, ION_COMPOSITION, "D" => 1e20, "Ar" => [1.0, 2.0, 3.0] .* 1e19)
        back = ion_densities(u, layout, ION_COMPOSITION)
        @test back[1].charge_state_densities[1, :] == fill(1e20, nr)
        @test back[1].charge_state_densities[2, :] == fill(1e19, nr)
        @test back[2].charge_state_densities[1, :] == [1.0, 2.0, 3.0] .* 1e19
        @test_throws ArgumentError deposit_neutrals!(u, layout, ION_COMPOSITION, "Ne" =>
            1.0)
        @test_throws ArgumentError deposit_neutrals!(u, layout, ION_COMPOSITION, "D" =>
            -1.0)
        @test_throws DimensionMismatch deposit_neutrals!(u, layout, ION_COMPOSITION, "D" =>
            [1.0, 2.0])
        @test_throws ArgumentError set_ion_densities!(u, layout, ions[1:1])
    end

    @testset "free-electron density and the closure" begin
        nr = 2
        layout = _ion_layout(; nr, extra = [RadialUnknown(:hot_electron_density; scale = 1e17)])
        term = WeightedSumTerm(:ion_densities, free_electron_weights(ION_COMPOSITION))
        @test_throws DimensionMismatch Runes2P.validate(
            WeightedSumTerm(:ion_densities, [
                1.0, 2.0]), layout, 2)
        @test_throws ArgumentError Runes2P.validate(term, layout, 1)
        u = zeros(length(layout))
        ions = _ion_species(nr)
        set_ion_densities!(u, layout, ions)
        out = Runes2P.rhs!(zeros(length(layout)), term, u, layout, 2, 0.0)
        n_free = [free_electron_density(ions, ir) for ir in 1:nr]
        @test unknown_values(out, layout, 2) ≈ n_free rtol = 1e-14
        @test n_free ≈ fill(1e19 + 1.5e15, nr) rtol = 1e-14
        equations = [Equation(:ion_densities, TimeDerivative()),
            population_equations(ION_COMPOSITION; hot_density = :hot_electron_density)...,
            Equation(:hot_electron_density, Algebraic(), PrescribedValueTerm(1e17),
                LinearTerm(:hot_electron_density; coefficient = -1))]
        system = EquationSystem(layout, equations)
        unknown_values(u, layout, 3) .= 1e17
        unknown_values(u, layout, 2) .= n_free .- 1e17
        @test maximum(abs, Runes2P.rhs!(zeros(length(layout)), u, system, 0.0)) <
              1e-12 * 1e19
        # the direct Jacobian against the automatic one of its rhs
        J = Runes2P.jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        reference = ForwardDiff.jacobian(x -> Runes2P.rhs!(zeros(eltype(x), length(x)), x, system, 0.0), u)
        @test Matrix(J) ≈ reference rtol = 1e-14
    end

    @testset "plasma with the ions from the unknown" begin
        nr = 2
        layout = _ion_layout(; nr, extra = [RadialUnknown(:cold_electron_temperature; scale = 10.0)])
        prescribed = PlasmaState(; cold_electron_temperature = fill(5.0, nr),
            ions = _ion_species(nr; argon = 0.0),
            electric_field = zeros(nr), magnetic_field = zeros(nr))
        @test_throws ArgumentError PlasmaFromUnknowns(prescribed; ions = :ion_densities, density = nothing)
        source = PlasmaFromUnknowns(prescribed; ions = :ion_densities)
        @test Runes2P.source_dependencies(source) ==
              (:cold_electron_temperature, :cold_electron_density, :ion_densities)
        @test Runes2P.ion_dependency(source) === :ion_densities
        u = zeros(length(layout))
        ions = _ion_species(nr)
        set_ion_densities!(u, layout, ions)
        unknown_values(u, layout, 3) .= [10.0, 20.0]
        unknown_values(u, layout, 2) .= 7e18
        plasma = plasma_values(source, u, layout, 0.0)
        @test plasma.ions[2].charge_state_densities == ions[2].charge_state_densities
        @test plasma.cold_electron_temperature == [10.0, 20.0] &&
              plasma.cold_electron_density == fill(7e18, nr)
        @test plasma.cold_density_given
        # the key of a cache sees a change of a charge state
        key = Runes2P.PlasmaKey()
        @test Runes2P.changed_plasma!(key, source, u, layout, 0.0) !== nothing
        @test Runes2P.changed_plasma!(key, source, u, layout, 0.0) === nothing
        radial_view(u, layout, 1, 2)[21] = 1.0
        @test Runes2P.changed_plasma!(key, source, u, layout, 0.0).ions[2].charge_state_densities[19, 2] ==
              1.0
        # an unknown of the wrong size
        wrong = UnknownLayout(layout.radial_grid,
            [RadialUnknown(:ion_densities; components = 3),
                RadialUnknown(:cold_electron_density),
                RadialUnknown(:cold_electron_temperature)])
        wrong_state = zeros(length(wrong))
        unknown_values(wrong_state, wrong, 3) .= 10.0
        @test_throws ArgumentError plasma_values(source, wrong_state, wrong, 0.0)
        # a Fokker-Planck term takes the ions and lags only the group of its ions
        g = cylindrical_grid(
            MomentumGrid(uniform_p_faces(8, 0.0, 0.1), uniform_xi_faces(2)),
            layout.radial_grid)
        kinetic = UnknownLayout(
            layout.radial_grid, [layout.unknowns...;
                                 KineticUnknown(:f, g)])
        fp(; lagged = ()) = FokkerPlanckTerm(
            :f, AdvectionDiffusion(g; p_upper_boundary = :f_zero); plasma = source, lagged)
        @test Runes2P.validate(fp(), kinetic, 4, 0.0) === nothing
        @test Runes2P.validate(fp(; lagged = (:ion_densities,)), kinetic, 4, 0.0) ===
              nothing
        @test_throws ArgumentError Runes2P.validate(
            fp(; lagged = (:cold_electron_temperature,)), kinetic, 4, 0.0)
    end

    @testset "lower limit and tolerance per species" begin
        nr = 1
        layout = _ion_layout(; nr)
        equations = [
            Equation(:ion_densities, TimeDerivative(), ExchangeTerm(:ion_densities, 1e3)),
            population_equations(ION_COMPOSITION)...]
        system = EquationSystem(layout, equations)
        tol = tolerance(system, :ion_densities)
        @test tol.group == 1e-10 && tol.absolute == 1e-20 * 1e19
        @test tolerance(system, :cold_electron_density).group == 0
        @test Runes2P.describe(Tolerance(1.0, 2.0)) ==
              Runes2P.SettingsRecord("absolute" => 1.0, "relative" => 2.0)
        @test Runes2P.describe(Tolerance(1.0, 2.0; group = 3.0))["group"] == 3.0
        @test_throws ArgumentError Tolerance(1.0, 2.0; group = -1.0)
        ws = NewtonWorkspace(system)
        u = zeros(length(layout))
        set_ion_densities!(u, layout, _ion_species(nr))
        ions = unknown_range(layout, 1, 1)
        # the limit of each charge state: −1e-10 times the density of its species
        for (k, i) in enumerate(ws.limited)
            i in ions || continue
            expected = i in ions[1:2] ? -1e-10 * 1e19 : -1e-10 * 1e15
            @test Runes2P._limit_value(ws, u, k) ≈ expected rtol = 1e-14
        end
        @test ws.group_tolerance[ions] == fill(1e-10, 21) &&
              iszero(ws.group_tolerance[last(ions) + 1])
        @test ws.group_ranges[ions[5]] == ions[3:21]
        # an argon state below −1e-10 of the argon density is below its limit, a deuterium
        # state at the same density is not
        trial = copy(u)
        trial[ions[4]] = -1e6
        @test Runes2P._below_lower_limit(ws, trial)
        trial = copy(u)
        trial[ions[1]] = -1e6
        @test !Runes2P._below_lower_limit(ws, trial)
        # a step of the exchange between the neutral and the ion of deuterium (backward
        # Euler, rate 1e3/s over 1e-3 s): the difference falls by 1 + 2 k dt = 3, the sum
        # and the trace argon stay
        unknown_values(u, layout, 2) .= free_electron_density(_ion_species(nr), 1)
        solution = integrate(system, u, [0.0, 1e-3])
        final = solution.states[:, end]
        @test final[ions[1]] - final[ions[2]] ≈ (u[ions[1]] - u[ions[2]]) / 3 rtol = 1e-8
        @test final[ions[1]] + final[ions[2]] ≈ u[ions[1]] + u[ions[2]] rtol = 1e-12
        @test final[ions[3:21]] == u[ions[3:21]]
    end

    @testset "groups in the output file" begin
        layout = _ion_layout(; nr = 1)
        system = EquationSystem(layout,
            [Equation(:ion_densities, TimeDerivative()),
                population_equations(ION_COMPOSITION)...])
        u = zeros(length(layout))
        set_ion_densities!(u, layout, _ion_species(1))
        unknown_values(u, layout, 2) .= free_electron_density(_ion_species(1), 1)
        solution = integrate(system, u, [0.0, 1e-6])
        file = joinpath(mktempdir(), "ions.h5")
        write_output(file, solution)
        back = read_output(file)
        unknown = Runes2P.unknown(back.layout, :ion_densities)
        @test component_groups(unknown) == [1:2, 3:21]
        @test unknown.lower_limit.reference === :group
    end

    @testset "the target of the e-e collisions" begin
        nr = 2
        radial = RadialGrid(uniform_r_faces(nr, 0.5))
        g = cylindrical_grid(MomentumGrid(uniform_p_faces(8, 0.0, 0.1), uniform_xi_faces(2)), radial)
        layout = UnknownLayout(radial,
            [RadialUnknown(:cold_electron_temperature; scale = 10.0),
                RadialUnknown(:cold_electron_density; scale = 1e19), KineticUnknown(:f, g)])
        default = PlasmaState(; cold_electron_temperature = fill(10.0, nr),
            ions = _ion_species(nr; argon = 0.0),
            electric_field = zeros(nr), magnetic_field = zeros(nr))
        @test !default.cold_density_given
        given = PlasmaState(; cold_electron_temperature = fill(10.0, nr),
            ions = _ion_species(nr; argon = 0.0),
            electric_field = zeros(nr), magnetic_field = zeros(nr), cold_electron_density = fill(1e19, nr))
        @test given.cold_density_given &&
              !with_electric_field(default, ones(nr)).cold_density_given
        term(plasma; mode = :full) = FokkerPlanckTerm(
            :f, AdvectionDiffusion(g; p_upper_boundary = :f_zero); plasma,
            settings = CollisionSettings(; mode))
        @test_throws ArgumentError Runes2P.validate(
            term(PlasmaFromUnknowns(default; density = nothing)), layout, 3, 0.0)
        @test_throws ArgumentError Runes2P.validate(term(default), layout, 3, 0.0)
        @test Runes2P.validate(term(default; mode = :superthermal), layout, 3, 0.0) ===
              nothing
        @test Runes2P.validate(term(given), layout, 3, 0.0) === nothing
        @test Runes2P.validate(term(PlasmaFromUnknowns(default)), layout, 3, 0.0) ===
              nothing
        # without the unknown of the cold density the default stays allowed
        alone = UnknownLayout(radial, [KineticUnknown(:f, g)])
        @test Runes2P.validate(term(default), alone, 1, 0.0) === nothing
    end
end

@testset "a labelled species takes the data of its element" begin
    # DREAM's disruption example injects deuterium as a species of its own (Dinj), with an
    # ion energy of its own: Runes2P's "D_inj" takes the atomic data of D
    @test Runes2P.atomic_element("D_inj") == "D" && Runes2P.atomic_element("Ar") == "Ar"
    @test ionization_energy("D_inj", 0) == ionization_energy("D", 0)
    @test ion_mass(IonSpecies("D_inj", 1, 0, [1e19])) ==
          ion_mass(IonSpecies("D", 1, 0, [1e19]))
    @test ionization_cross_section("D_inj", 0, 0.1) == ionization_cross_section("D", 0, 0.1)
    @test Runes2P.adf11_file_name("D_inj", :scd) == Runes2P.adf11_file_name("D", :scd)
    composition = IonComposition(["D" => 1, "D_inj" => 1, "Ar" => 18])
    layout = UnknownLayout(RadialGrid(uniform_r_faces(1, 0.5)), [ion_densities_unknown(composition)])
    u = zeros(length(layout))
    deposit_neutrals!(u, layout, composition, "D_inj" => 2e19)
    @test radial_view(u, layout, 1, 1)[species_components(composition, "D_inj")] ==
          [2e19, 0.0]
    @test all(iszero, radial_view(u, layout, 1, 1)[species_components(composition, "D")])
    if isdir(adas_cache_directory())
        rates = IonRateTerm(composition)
        @test rates.tables[2].scd.source == rates.tables[1].scd.source
    end
end
