# The ionization of the ions by the electrons of a distribution: the cross sections of DREAM's
# fits against the ADAS coefficients at low density, the conservation of each species, the
# Jacobian of the term against central differences, and the split of the ionization at the hot
# threshold: a Maxwellian f ionizes at the ADAS rate for every threshold, and DREAM's pairing
# counts the tail above the threshold twice. Needs the ADF11 cache (scripts/fetch_adas.py).
using SparseArrays: sparse
using ForwardDiff
using QuadGK: quadgk
using Runes2P: add_local_jacobian!

const KINETIC_IONIZATION_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

# a grid that covers the bulk at 20 and 10 eV to 10 thermal momenta, with 81 cells so that
# no threshold of 5 or 7 thermal momenta lies on a face, where the interpolated envelope
# has a kink (the split refuses factors below 5 with DREAM's cross sections and below 7 with
# CHIANTI's): below 7 the case takes DREAM's form
function _ionization_case(; factor = 5.0, temperatures = [20.0, 10.0], n_M = 5e19,
        form = factor < Runes2P.MINIMUM_SPLIT_THRESHOLD_FACTORS[:chianti] ? :dream :
               :chianti)
    radial = RadialGrid(uniform_r_faces(2, 0.5))
    p_max = 10 * sqrt(2 * maximum(temperatures) / Constants.ELECTRON_REST_ENERGY)
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(81, 0.0, p_max), uniform_xi_faces(3)), radial)
    layout = UnknownLayout(radial,
        [ion_densities_unknown(KINETIC_IONIZATION_COMPOSITION),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            RadialUnknown(:cold_electron_temperature; scale = 10.0),
            KineticUnknown(:f, grid; scale = 1e19)])
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor))
    kinetic = KineticIonizationTerm(
        KINETIC_IONIZATION_COMPOSITION, :f, grid; region, ionization_cross_section_form = form)
    u = zeros(length(layout))
    argon = zeros(19, 2)
    argon[1:6, :] .= [1e17, 3e17, 5e17, 2e17, 1e17, 5e16]
    set_ion_densities!(u, layout, [
        IonSpecies("D", 1, [1e18 2e18; 4e19 3e19]), IonSpecies("Ar", 18, argon)])
    unknown_values(u, layout, 3) .= temperatures
    # a relativistic Maxwellian of density n_M on the grid, with the weights of the term
    for ir in 1:2
        m = momentum_grid(grid, ir)
        theta = temperatures[ir] / Constants.ELECTRON_REST_ENERGY
        shape = [exp(-p^2 / (sqrt(1 + p^2) + 1) / theta) for p in m.p_centers, _ in 1:3]
        kinetic_view(u, layout, 4, ir) .= n_M .* shape ./ sum(kinetic.weights[ir] .* shape)
    end
    return layout, kinetic, u
end

@testset "kinetic ionization" begin
    @testset "cross sections" begin
        @test ionization_cross_section("Ar", 0, 0.0) == 0
        @test ionization_cross_section("Ar", 18, 1.0) == 0
        # above the threshold of 15.76 eV of neutral argon, at 100 eV: of the order of πa₀²
        p = sqrt((1 + 100 / Constants.ELECTRON_REST_ENERGY)^2 - 1)
        @test 1e-21 < ionization_cross_section("Ar", 0, p) < 1e-19
        @test_throws ArgumentError ionization_cross_section("Xe", 0, 1.0)
        @test_throws ArgumentError ionization_cross_section("Ar", 19, 1.0)
        @test size(KINETIC_IONIZATION_FITS["Ar"]) == (18, 3) &&
              size(KINETIC_IONIZATION_FITS["D"]) == (1, 3)
        @test all(size(KINETIC_IONIZATION_BETHE_FITS[e]) ==
                  size(KINETIC_IONIZATION_FITS[e])
        for e in keys(KINETIC_IONIZATION_FITS))
        @test keys(KINETIC_IONIZATION_BETHE_FITS) == keys(KINETIC_IONIZATION_FITS)
        @test_throws ArgumentError ionization_cross_section("Ar", 0, 1.0; form = :other)
    end

    @testset "the relativistic term joins the Burgess-Chidichimo limit" begin
        # At U ≫ 1 and E ≪ mₑc² the Burgess-Chidichimo term tends to C π a₀² (R_y/I)² ln U/U,
        # which is C π a₀² α² (R_y/I) ln U/β² with E = β²mₑc²/2 and R_y = α²mₑc²/2. The form
        # with 1/β² therefore equals it there, and the blend of the two with any weight S is
        # the Burgess-Chidichimo term; DREAM's form is β² times smaller and leaves (1 − S + Sβ²)
        # of it. The Burgess-Chidichimo term is evaluated here with the parameters of each table.
        mc2 = Constants.ELECTRON_REST_ENERGY
        a0, Ry = Constants.BOHR_RADIUS, Constants.RYDBERG_ENERGY
        for (element, Z0) in (("Ar", 0), ("Ar", 1), ("D", 0)), E in (2e3, 5e3, 1e4)

            gamma = 1 + E / mc2
            p = sqrt(gamma^2 - 1)
            S = 1 / (1 + exp(1 - E * 1e-5))
            burgess(fits) = (row = fits[element][Z0 + 1, :]; U = E / row[2];
                row[1] * π * a0^2 * (Ry / row[2])^2 * log(U)^(1 + row[3] / U) / U)
            @test ionization_cross_section(element, Z0, p; form = :bethe) /
                  burgess(KINETIC_IONIZATION_BETHE_FITS) ≈ 1 rtol = 0.03
            @test ionization_cross_section(element, Z0, p; form = :dream) /
                  burgess(KINETIC_IONIZATION_FITS) ≈
                  1 - S + S * (p / gamma)^2 rtol = 2e-3
        end
        # the blend is floored at zero, where the Bethe logarithm lies below β² just above the
        # threshold of a state of several keV
        I = KINETIC_IONIZATION_BETHE_FITS["W"][65, 2]
        near = sqrt((1 + 1.001I / mc2)^2 - 1)
        @test ionization_cross_section("W", 64, near; form = :bethe) == 0
        @test ionization_cross_section("W", 64, sqrt((1 + 3I / mc2)^2 - 1); form = :bethe) >
              0
    end

    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparisons with ADAS are skipped"
        @test_skip isdir(adas_cache_directory())
    else
        @testset "Maxwellian average against ADAS at low density" begin
            # DREAM's fits were made to the SCD coefficients at zero density: the Maxwellian
            # average of v σ at the ionization energy of each state, against SCD at the lowest
            # density node, per state of argon and neon. DREAM states typical errors of its fits
            # (DREAM's tools/ADAS/fit.py) of at most tens of per cent.
            # CHIANTI's cross sections are independent of ADAS and differ from it by up to 2.6
            # (Ne⁺: 0.40, from Dere's fit to three measurements agreeing within 3 %).
            p = collect(range(0.0, 1.0; length = 200_001))[2:end]
            for element in ("Ar", "Ne"), form in (:dream, :bethe, :chianti)

                scd = read_adas(element).scd
                Z = size(KINETIC_IONIZATION_FITS[element], 1)
                ratios = Float64[]
                for Z0 in 0:(Z - 1)
                    T = ionization_energy(element, Z0)
                    theta = T / Constants.ELECTRON_REST_ENERGY
                    weight = @. p^2 * exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)
                    rate = sum(weight .* [Constants.SPEED_OF_LIGHT * x / sqrt(1 + x^2) *
                                ionization_cross_section(element, Z0, x; form) for x in p]) /
                           sum(weight)
                    push!(ratios, rate /
                                  adas_coefficient(scd, Z0, exp10(first(scd.log_density)), T))
                end
                @info "Maxwellian average over ADAS SCD at T = I per state of $element, form $form: " *
                      join(round.(ratios; digits = 3), ", ")
                @test all(r -> 0.3 < r < 3.0, ratios)
            end
        end

        @testset "conservation and Jacobian of the term" begin
            layout, kinetic, u = _ionization_case()
            n = length(u)
            Runes2P.validate(kinetic, layout, 1)
            du = Runes2P.rhs!(zeros(n), kinetic, u, layout, 1, 0.0)
            for ir in 1:2, s in 1:2

                rates = du[unknown_range(layout, 1, ir)[species_components(KINETIC_IONIZATION_COMPOSITION, s)]]
                @test abs(sum(rates)) <= 1e-12 * maximum(abs, rates)
            end
            @test all(du[unknown_range(layout, 1, 1)[3:5]] .!= 0)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], kinetic, layout, 1)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            Runes2P.add_jacobian!(J, kinetic, u, layout, 1, 0.0)
            term_rhs(x) = Runes2P.rhs!(zeros(n), kinetic, x, layout, 1, 0.0)
            size_of_rhs = maximum(abs, du)
            scales = unknown_scales(layout)
            # the columns of radius 2: ion densities, the temperature, a few cells of f
            cells = unknown_range(layout, 4, 2)
            for column in [collect(unknown_range(layout, 1, 2));
                           unknown_range(layout, 3, 2); cells[[1, 20, 40, 41, 120]]]
                h = column in unknown_range(layout, 3, 2) ? 1e-6 * u[column] :
                    1e-3 * max(abs(u[column]), 1e-3 * scales[column])
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (term_rhs(up) - term_rhs(down)) / 2h
                @test maximum(abs, Vector(J[:, column]) - difference) <=
                      1e-6 * maximum(abs, difference) + 1e-13 * size_of_rhs / h
            end
        end

        @testset "split at the threshold" begin
            # below 5 thermal momenta the split is refused with every form: the fluid coefficient
            # is negative there for populated argon states
            for factor in (2.0, 3.0, 4.9)
                _, low, _ = _ionization_case(; factor)
                @test_throws "is refused" IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = low)
                prescribed = PlasmaState(; cold_electron_temperature = [20.0, 10.0],
                    ions = [IonSpecies("D", 1, 1, [1e19, 1e19]),
                        IonSpecies("Ar", 18, 1, [1e18, 1e18])],
                    electric_field = zeros(2), magnetic_field = zeros(2))
                @test_throws "is refused" RadiatedPowerTerm(
                    prescribed; ions = :ion_densities, kinetic_ionization = low)
            end
            # a threshold at a fixed momentum does not follow the temperature
            _, fixed, _ = _ionization_case()
            fixed_region = KineticIonizationTerm(KINETIC_IONIZATION_COMPOSITION, :f,
                fixed.grid; region = MomentumRegion(0.05))
            @test_throws "ThermalMomentum" IonRateTerm(
                KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = fixed_region)
            # the minimum per form: 7 with CHIANTI's cross sections, 5 with DREAM's
            @test Runes2P.MINIMUM_SPLIT_THRESHOLD_FACTORS ==
                  Dict(:chianti => 7.0, :dream => 5.0, :bethe => 5.0)
            for factor in (5.0, 6.9)
                _, chianti, _ = _ionization_case(; factor, form = :chianti)
                @test_throws "is refused" IonRateTerm(
                    KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = chianti)
            end
            _, chianti, _ = _ionization_case(; factor = 7.0, form = :chianti)
            @test IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = chianti) isa
                  IonRateTerm
            _, dream, _ = _ionization_case(; factor = 5.0, form = :dream)
            @test IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = dream) isa
                  IonRateTerm
            for factor in (5.0, 7.0)
                layout, kinetic, u = _ionization_case(; factor)
                n = length(u)
                splits = [Runes2P.threshold_split(kinetic, ir, u, layout, 0.0)
                          for ir in 1:2]
                @test all(s -> 0 < s.fraction < 1e-6, splits)
                # the closure: the cold electrons are those of the Maxwellian below the threshold
                n_cold = [5e19 * (1 - s.fraction) for s in splits]
                unknown_values(u, layout, 2) .= n_cold
                paired = IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = kinetic)
                dream = IonRateTerm(KINETIC_IONIZATION_COMPOSITION)
                kinetic_part = Runes2P.rhs!(zeros(n), kinetic, u, layout, 1, 0.0)
                for (fluid, exact) in ((paired, true), (dream, false))
                    du = Runes2P.rhs!(zeros(n), fluid, u, layout, 1, 0.0) .+ kinetic_part
                    for ir in 1:2
                        T = unknown_values(u, layout, 3)[ir]
                        rows = unknown_range(layout, 1, ir)
                        for s in 1:2
                            name = KINETIC_IONIZATION_COMPOSITION.names[s]
                            tables = read_adas(name)
                            states = species_components(KINETIC_IONIZATION_COMPOSITION, s)
                            # the ionization flow of each state: the net flow plus recombination
                            for (j, k) in enumerate(states[1:(end - 1)])
                                Z0 = j - 1
                                net = -sum(du[rows[states[1:j]]])
                                recombination = n_cold[ir] *
                                                adas_coefficient(tables.acd, Z0 + 1, n_cold[ir], T) *
                                                u[rows[k + 1]]
                                ionization = net + recombination
                                expected = 5e19 *
                                           adas_coefficient(tables.scd, Z0, n_cold[ir], T) *
                                           u[rows[k]]
                                u[rows[k]] > 0 || continue
                                if exact
                                    # the flows are differences of the rates of the states: round-off
                                    @test ionization ≈ expected rtol = 1e-9
                                else
                                    # DREAM's pairing: the cold electrons at the whole ADAS rate and
                                    # the tail above the threshold once more, n_M ((1 − f_>) SCD + A_>)
                                    split = splits[ir]
                                    scd = adas_coefficient(tables.scd, Z0, n_cold[ir], T)
                                    @test ionization ≈
                                          5e19 *
                                          ((1 - split.fraction) * scd +
                                           split.coefficients[k]) *
                                          u[rows[k]] rtol = 1e-9
                                end
                            end
                        end
                    end
                end
            end
        end

        @testset "the fluid rate's Jacobian with the split" begin
            layout, kinetic, u = _ionization_case()
            unknown_values(u, layout, 2) .= 4.5e19
            equations = [
                Equation(:ion_densities, TimeDerivative(),
                    IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic_ionization = kinetic),
                    kinetic),
                # the closure of the cold density: a kinetic ionization needs it
                population_equations(KINETIC_IONIZATION_COMPOSITION)...,
                Equation(:cold_electron_temperature, Algebraic(),
                    PrescribedValueTerm([20.0, 10.0]),
                    LinearTerm(:cold_electron_temperature; coefficient = -1)),
                Equation(:f, TimeDerivative())]
            system = EquationSystem(layout, equations; initial_state = u)
            J = jacobian_prototype(system)
            rhs!(zeros(length(u)), u, system, 0.0)
            jacobian!(J, system, u, 0.0)
            rhs_all(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out)
            ion_rows = unknown_range(layout, 1, 1)
            for column in (first(unknown_range(layout, 3, 1)), first(unknown_range(layout, 2, 1)))
                h = 1e-6 * u[column]
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (rhs_all(up) - rhs_all(down))[ion_rows] / 2h
                @test Vector(J[ion_rows, column])≈difference rtol=1e-5 atol=1e-6*maximum(abs, difference)
            end
            # the binding energy at the fluid coefficient of the split
            prescribed = PlasmaState(; cold_electron_temperature = [20.0, 10.0],
                ions = ion_densities(u, layout,
                    KINETIC_IONIZATION_COMPOSITION),
                electric_field = zeros(2), magnetic_field = zeros(2))
            full = radiated_power(RadiatedPowerTerm(prescribed; ions = :ion_densities), u, layout, 0.0).binding
            split = radiated_power(
                RadiatedPowerTerm(prescribed; ions = :ion_densities, kinetic_ionization = kinetic),
                u, layout, 0.0).binding
            @test all(split .< full)
            # the direct block of the radiated power with the split, the temperature column
            # through the moving threshold included, against the forward-mode Jacobian
            n = length(u)
            term = RadiatedPowerTerm(prescribed; ions = :ion_densities, kinetic_ionization = kinetic)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 3)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            add_local_jacobian!(J, term, u, layout, 3, 0.0)
            reference = ForwardDiff.jacobian(
                x -> Runes2P.rhs!(
                    zeros(eltype(x), n), term, x, layout, 3, 0.0), u)
            @test Matrix(J)≈reference rtol=1e-10 atol=1e-12*maximum(abs, reference)
            # and the temperature column against a central difference
            column = first(unknown_range(layout, 3, 1))
            h = 1e-6 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (Runes2P.rhs!(zeros(n), term, up, layout, 3, 0.0) -
                          Runes2P.rhs!(zeros(n), term, down, layout, 3, 0.0)) / 2h
            @test Vector(J[:, column])≈difference rtol=1e-5 atol=1e-8*maximum(abs, difference)
        end
        @testset "secondaries and the energy of the kinetic ionization" begin
            @test mean_secondary_energy(24.2, 10.0, 15.8, 1e3) == 0
            # without a cut: the mean of the Opal spectrum up to (E − I)/2
            top = (1000.0 - 15.8) / 2
            @test mean_secondary_energy(24.2, 1000.0, 15.8, Inf) ≈
                  12.1 * log(1 + (top / 24.2)^2) / atan(top / 24.2)
            @test mean_secondary_energy(24.2, 1000.0, 15.8, 50.0) <
                  mean_secondary_energy(24.2, 1000.0, 15.8, 100.0)
            # independent of the closed form: the mean energy of the spectrum of Opal et al.,
            # σ(E_s) ∝ 1/(1 + (E_s/w)²), by quadrature up to the smaller of (E − I)/2 and the cut
            for (w, E, I, cut) in ((24.2, 1000.0, 15.8, Inf),
                (10.0, 300.0, 15.76, 40.0), (8.3, 60.0, 13.6, Inf),
                (15.8, 5.0e4, 24.6, 120.0))
                top = min((E - I) / 2, cut)
                spectrum(x) = 1 / (1 + (x / w)^2)
                mean = quadgk(x -> x * spectrum(x), 0, top; rtol = 1e-12)[1] /
                       quadgk(spectrum, 0, top; rtol = 1e-12)[1]
                @test mean_secondary_energy(w, E, I, cut) ≈ mean rtol = 1e-10
            end
            layout, kinetic, u = _ionization_case()
            n = length(u)
            # the widths of Opal et al. (1971), Table II: H₂ for deuterium, argon 10.0 eV
            # the spectrum of Opal et al. (the default is that of Kim and Rudd)
            heating = SecondaryHeatingTerm(kinetic; spectrum = :opal)
            @test heating.widths == [8.3, 10.0]
            # one number type of the weights and of the heating with a dual temperature, under both
            # cuts: the cut :none is Inf in the type of the temperature
            for cut in (:none, :threshold)
                h = SecondaryHeatingTerm(kinetic; spectrum = :opal, cut)
                Td = ForwardDiff.Dual{Nothing, Float64, 1}(20.0, ForwardDiff.Partials((1.0,)))
                np, components = size(kinetic.rates[1])
                @test (@inferred Runes2P._secondary_weights(h, 1, Td)) isa
                      Matrix{typeof(Td)}
                @test (@inferred Runes2P._secondary_heating(
                    h, 1, ones(np), ones(components), Td)) isa typeof(Td)
                @test (@inferred Runes2P._secondary_weights(h, 1, 20.0)) isa Matrix{Float64}
            end
            @test SecondaryHeatingTerm(kinetic; widths = Dict("Ar" => 12.0), spectrum = :opal).widths ==
                  [8.3, 12.0]
            # an element without a width is refused: iron
            fe = KineticIonizationTerm(IonComposition(["Fe" => 26]), :f, kinetic.grid;
                region = kinetic.envelopes.region)
            @test_throws ArgumentError SecondaryHeatingTerm(fe; spectrum = :opal)
            widths = Dict{String, Float64}()
            row = 3
            Runes2P.validate(heating, layout, row)
            Q = unknown_values(Runes2P.rhs!(zeros(n), heating, u, layout, row, 0.0), layout, row)
            # the heating against a direct sum over the cells, the states and the species
            for ir in 1:2
                T = unknown_values(u, layout, 3)[ir]
                m = momentum_grid(kinetic.grid, ir)
                p0 = 5.0 * sqrt(2T / Constants.ELECTRON_REST_ENERGY)
                cut = (sqrt(1 + p0^2) - 1) * Constants.ELECTRON_REST_ENERGY
                envelope = momentum_envelope!(zeros(length(m.p_centers)), kinetic.envelopes.region, m, p0)
                f = kinetic_view(u, layout, 4, ir)
                states = radial_view(u, layout, 1, ir)
                expected = 0.0
                for (s, name, Z, width) in ((1, "D", 1, 8.3), (2, "Ar", 18, 10.0)),
                    Z0 in 0:(Z - 1)

                    k = species_components(KINETIC_IONIZATION_COMPOSITION, s)[Z0 + 1]
                    for (i, p) in enumerate(m.p_centers), j in 1:3

                        E = (sqrt(1 + p^2) - 1) * Constants.ELECTRON_REST_ENERGY
                        v = Constants.SPEED_OF_LIGHT * p / sqrt(1 + p^2)
                        expected += states[k] * envelope[i] * v *
                                    ionization_cross_section(name, Z0, p;
                                        form = kinetic.ionization_cross_section_form) *
                                    Constants.ELEMENTARY_CHARGE *
                                    mean_secondary_energy(width, E, ionization_energy(name, Z0), Inf) *
                                    kinetic.weights[ir][i, j] * f[i, j]
                    end
                end
                @test Q[ir] ≈ expected rtol = 1e-12
                @test Q[ir] > 0
            end
            # the direct block against central differences: a cell of f, the charge states, the temperature
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], heating, layout, row)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            Runes2P.add_jacobian!(J, heating, u, layout, row, 0.0)
            term_rhs(x) = Runes2P.rhs!(zeros(n), heating, x, layout, row, 0.0)
            size_of_rhs = maximum(abs, term_rhs(u))
            scales = unknown_scales(layout)
            for column in [collect(unknown_range(layout, 1, 2));
                           unknown_range(layout, 3, 2);
                           unknown_range(layout, 4, 2)[[30, 41]]]
                h = column in unknown_range(layout, 3, 2) ? 1e-6 * u[column] :
                    1e-3 * max(abs(u[column]), 1e-3 * scales[column])
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (term_rhs(up) - term_rhs(down)) / 2h
                @test maximum(abs, Vector(J[:, column]) - difference) <=
                      1e-5 * maximum(abs, difference) + 1e-13 * size_of_rhs / h
            end
            # the potential energy is e I_j times the kinetic flows, the secondaries the heating
            flows = ionization_energy_flows(
                kinetic, u, layout, 0.0; widths, spectrum = :opal)
            @test flows.secondaries ≈ Q rtol = 1e-14
            du = Runes2P.rhs!(zeros(n), kinetic, u, layout, 1, 0.0)
            for ir in 1:2
                rows = unknown_range(layout, 1, ir)
                expected = 0.0
                for (s, name, Z) in ((1, "D", 1), (2, "Ar", 18))
                    states = species_components(KINETIC_IONIZATION_COMPOSITION, s)
                    for Z0 in 0:(Z - 1)
                        flow = -sum(du[rows[states[1:(Z0 + 1)]]])
                        expected += Constants.ELEMENTARY_CHARGE *
                                    ionization_energy(name, Z0) * flow
                    end
                end
                @test flows.potential[ir] ≈ expected rtol = 1e-10
            end
            @test Runes2P.describe(heating, Runes2P.DescriptionContext())["widths"]["Ar"] ==
                  10.0
            # the two cuts: every secondary (cut = :none, the default) against the mean
            # below the hot threshold (cut = :threshold). With f in one cell at about 90 keV
            # and argon in its neutral state alone, the ratio of the two heatings is that of
            # the two means of the spectrum, 2.56 at E_c = 250 eV (10 eV, 5 thermal momenta)
            fast_grid = cylindrical_grid(
                MomentumGrid(uniform_p_faces(50, 0.0, 1.0), uniform_xi_faces(2)),
                RadialGrid(uniform_r_faces(1, 0.5)))
            argon_only = IonComposition(["Ar" => 18])
            fast_layout = UnknownLayout(fast_grid.radial_grid,
                [ion_densities_unknown(argon_only),
                    RadialUnknown(:cold_electron_temperature; scale = 10.0),
                    KineticUnknown(:f, fast_grid; scale = 1e10)])
            fast = zeros(length(fast_layout))
            radial_view(fast, fast_layout, 1, 1)[1] = 1e18
            unknown_values(fast, fast_layout, 2) .= 10.0
            cell = 32
            kinetic_view(fast, fast_layout, 3, 1)[cell, :] .= 1e10
            fast_kinetic = KineticIonizationTerm(argon_only, :f, fast_grid;
                region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5)))
            heat(cut) = only(unknown_values(
                Runes2P.rhs!(zeros(length(fast)),
                    SecondaryHeatingTerm(fast_kinetic; cut, spectrum = :opal), fast,
                    fast_layout, 2, 0.0),
                fast_layout,
                2))
            p = momentum_grid(fast_grid).p_centers[cell]
            E = (sqrt(1 + p^2) - 1) * Constants.ELECTRON_REST_ENERGY
            E_c = (sqrt(1 + 50 * 10.0 / Constants.ELECTRON_REST_ENERGY) - 1) *
                  Constants.ELECTRON_REST_ENERGY
            I0 = ionization_energy("Ar", 0)
            ratio = mean_secondary_energy(10.0, E, I0, Inf) /
                    mean_secondary_energy(10.0, E, I0, E_c)
            @test heat(:none) / heat(:threshold) ≈ ratio rtol = 1e-12
            @test 2.5 < ratio < 2.6
            # without a width the diagnostic still gives the potential energy: iron
            fe_layout = UnknownLayout(layout.radial_grid,
                [ion_densities_unknown(IonComposition(["Fe" => 26])),
                    layout.unknowns[2:end]...])
            fe_state = zeros(length(fe_layout))
            unknown_values(fe_state, fe_layout, 3) .= unknown_values(u, layout, 3)
            for ir in 1:2
                radial_view(fe_state, fe_layout, 1, ir)[1:3] .= 1e17
                kinetic_view(fe_state, fe_layout, 4, ir) .= kinetic_view(u, layout, 4, ir)
            end
            no_width = ionization_energy_flows(
                fe, fe_state, fe_layout, 0.0; spectrum = :opal)
            @test ionization_energy_flows(fe, fe_state, fe_layout, 0.0).secondaries !==
                  nothing
            @test no_width.secondaries === nothing && all(>(0), no_width.potential)
        end
    end
end

# The friction on bound electrons tied to the ionization: CollisionSettings(bound_friction =
# :ionization_split) charges f the ionization potential and the mean energy of every secondary
# for each ionization that the kinetic ionization counts, and keeps the rest of the Bethe term
# floored at zero.
@testset "the friction tied to the ionization" begin
    mc2 = Constants.ELECTRON_REST_ENERGY
    momentum(E) = sqrt((1 + E / mc2)^2 - 1)
    @testset "settings" begin
        @test CollisionSettings().bound_friction === :bethe
        @test_throws ArgumentError CollisionSettings(bound_friction = :other)
        @test_throws "partial screening" CollisionSettings(
            bound_friction = :ionization_split, screening = :complete)
        @test_throws ArgumentError CollisionSettings(ionization_threshold = -1.0)
        @test_throws ArgumentError CollisionSettings(ionization_threshold_width = 1.0)
        @test_throws ArgumentError CollisionSettings(secondary_widths = Dict("Fe" => 0.0))
        split = CollisionSettings(bound_friction = :ionization_split, secondary_spectrum = :opal)
        @test CollisionSettings().secondary_spectrum === :beb
        plasma(name, Z) = PlasmaState(;
            cold_electron_temperature = [10.0], ions = [IonSpecies(name, Z, 0, [1e18])],
            electric_field = [0.0], magnetic_field = [0.0], cold_electron_density = [1e19])
        # iron has a fit of the cross sections and no width of the secondary spectrum
        @test_throws "secondary_widths" Runes2P.CollisionModel(plasma("Fe", 26), split)
        @test Runes2P.CollisionModel(plasma("Fe", 26),
            CollisionSettings(bound_friction = :ionization_split,
                secondary_widths = Dict("Fe" => 9.0))) isa Runes2P.CollisionModel
        # krypton has a width and no fit of the cross sections
        @test_throws "DREAM's tables lack" Runes2P.CollisionModel(plasma("Kr", 36), split)
        # the Bethe model needs neither
        @test Runes2P.CollisionModel(plasma("Kr", 36), CollisionSettings()) isa
              Runes2P.CollisionModel
    end

    @testset "the slowing down per charge state" begin
        # one charge state at a time: ν_split − ν_Bethe = max(e ν_ion − w ν_Bethe, 0), with the
        # ionization part γ n σ v (I + ⟨E_s⟩)/(mₑc² p²) from the cross section of the kinetic
        # ionization and the mean energy of every secondary
        T, n = 10.0, 1e18
        for Z0 in (0, 1, 5, 16), window in (:excitation_energy, :dream),
            factor in (0.0, 5.0), spectrum in (:opal, :beb)
            densities = zeros(19, 1)
            densities[Z0 + 1] = n
            plasma = PlasmaState(;
                cold_electron_temperature = [T], ions = [IonSpecies("Ar", 18, densities)],
                electric_field = [0.0], magnetic_field = [0.0], cold_electron_density = [1e19])
            common = (
                mode = :superthermal, screening_window = window, bremsstrahlung = :neglect)
            bethe = Runes2P.CollisionModel(plasma, CollisionSettings(; common...))
            split = Runes2P.CollisionModel(plasma,
                CollisionSettings(; common..., bound_friction = :ionization_split,
                    ionization_threshold = factor, secondary_spectrum = spectrum))
            # the free electrons of the same plasma alone
            free = Runes2P.CollisionModel(plasma, CollisionSettings(; common..., screening = :complete))
            I = ionization_energy("Ar", Z0)
            p0 = factor * sqrt(2T / mc2)
            for E in (0.5I, 1.5I, 3I, 30I, 3e3I, 2e6)
                p = momentum(E)
                gamma = sqrt(1 + p^2)
                v = Constants.SPEED_OF_LIGHT * p / gamma
                ion = gamma * n * ionization_cross_section("Ar", Z0, p) * v *
                      (I + Runes2P.secondary_spectrum_mean(spectrum, 10.0, E, I, Inf)) /
                      (mc2 * p^2)
                envelope = factor == 0 ? 1.0 : p < 0.9p0 ? 0.0 : p > 1.1p0 ? 1.0 : NaN
                isnan(envelope) && continue
                nu_free = slowing_down_frequency(free, 1, p)
                nu_bethe = slowing_down_frequency(bethe, 1, p) - nu_free
                nu_split = slowing_down_frequency(split, 1, p) - nu_free
                @test nu_split≈nu_bethe+max(envelope*ion-nu_bethe, 0.0) rtol=1e-10 atol=1e-12*nu_free
                @test nu_split >= nu_bethe - 1e-12 * nu_free
                # below the ionization energy nothing changes
                E < I && @test nu_split≈nu_bethe rtol=1e-12 atol=1e-12*nu_free
            end
        end
    end

    @testset "the ionization part tabulated per momentum grid" begin
        m = MomentumGrid(biuniform_p_faces(40, 10, 0.0, 0.07, 1.0), uniform_xi_faces(2))
        densities = zeros(19, 1)
        densities[1:4] .= [1e17, 3e17, 2e17, 1e16]
        plasma = PlasmaState(; cold_electron_temperature = [10.0],
            ions = [IonSpecies("D", 1, 1, [1e19]), IonSpecies("Ar", 18, densities)],
            electric_field = [0.0], magnetic_field = [0.0], cold_electron_density = [1e19])
        for spectrum in (:beb, :opal)
            settings = CollisionSettings(;
                mode = :full, bound_friction = :ionization_split,
                ionization_threshold = 5.0,
                secondary_spectrum = spectrum)
            direct = Runes2P.CollisionModel(plasma, settings)
            tabulated = Runes2P.CollisionModel(plasma, settings; faces = [m.p_faces])
            @test direct.ionization_tables[1] === nothing &&
                  tabulated.ionization_tables[1] !== nothing
            # the same numbers at the faces, from the table, and between them, computed
            @test all(slowing_down_frequency(tabulated, 1, p) ==
                      slowing_down_frequency(direct, 1, p) for p in m.p_faces)
            @test all(slowing_down_frequency(tabulated, 1, p) ==
                      slowing_down_frequency(direct, 1, p) for p in m.p_centers)
            # one table per grid and spectrum, shared by the models
            again = Runes2P.CollisionModel(plasma, settings; faces = [m.p_faces])
            @test last(again.ionization_tables[1]) === last(tabulated.ionization_tables[1])
            # a temperature with partials, through the envelope of the threshold
            dual = Runes2P.dual_plasma(plasma, true)
            dual_direct = Runes2P.CollisionModel(dual, settings)
            dual_tabulated = Runes2P.CollisionModel(dual, settings; faces = [m.p_faces])
            @test all(slowing_down_frequency(dual_tabulated, 1, p) ==
                      slowing_down_frequency(dual_direct, 1, p) for p in m.p_faces)
        end
        # the Bethe friction takes no table
        @test Runes2P.CollisionModel(plasma, CollisionSettings(); faces = [m.p_faces]).ionization_tables[1] ===
              nothing
        # the operator builds its model with the faces of its grid
        grid = cylindrical_grid(m, RadialGrid(uniform_r_faces(1, 0.5)))
        settings = CollisionSettings(;
            mode = :full, bound_friction = :ionization_split, ionization_threshold = 5.0)
        op = fokker_planck_operator(
            grid, plasma, settings; field_in_coefficients = false, synchrotron = false)
        reference = Runes2P.CollisionModel(plasma, settings)
        @test all(op.advection_p[k, 1, 1] ==
                  -m.p_faces[k] * slowing_down_frequency(reference, 1, m.p_faces[k])
        for k in eachindex(m.p_faces))
    end

    @testset "neutral argon at minimum ionization against the PDG" begin
        # Independent of DREAM's fits: argon at 20 °C and 1 atm (2.504e25 atoms per m³) has
        # N_P = 25 primary ionizations per cm for minimum ionizing particles (βγ ≈ 3 to 4),
        # Table 31.5 of the Review of Particle Physics, J. Beringer et al. (Particle Data
        # Group), Phys. Rev. D 86, 010001 (2012), section 31 "Particle detectors at
        # accelerators" (the table notes that such values differ between sources). The default,
        # CHIANTI's fit of the cross section of Ar⁰, gives 24.1 per cm at βγ = 3.5, DREAM's 21.6.
        n_ntp = 2.504e25
        @test ionization_cross_section("Ar", 0, 3.5) * n_ntp / 100 ≈ 25 rtol = 0.2
        # at p = 3 to 4 the ionization takes 0.95 to 0.97 of the Bethe stopping with the neutral
        # width of Opal et al. (0.83 to 0.89 with DREAM's form), and about 0.57 with the default
        # spectrum of Kim and Rudd, so the split leaves the friction of MeV electrons unchanged
        gas = PlasmaState(;
            cold_electron_temperature = [1.0], cold_electron_density = [0.0],
            ions = [IonSpecies("Ar", 18, 0, [n_ntp])], electric_field = [0.0], magnetic_field = [0.0])
        common = (
            mode = :superthermal, screening_window = :dream, bremsstrahlung = :neglect)
        bethe = Runes2P.CollisionModel(gas, CollisionSettings(; common...))
        split = Runes2P.CollisionModel(gas, CollisionSettings(; common..., bound_friction = :ionization_split))
        for p in (3.0, 3.5, 4.0)
            @test slowing_down_frequency(split, 1, p) ≈ slowing_down_frequency(bethe, 1, p) rtol = 1e-14
            gamma = sqrt(1 + p^2)
            stopping = mc2 * p^2 * slowing_down_frequency(bethe, 1, p) / gamma     # eV/s
            I = ionization_energy("Ar", 0)
            ionization = n_ntp * ionization_cross_section("Ar", 0, p) *
                         Constants.SPEED_OF_LIGHT * p / gamma *
                         (I + mean_secondary_energy(10.0, (gamma - 1) * mc2, I, Inf))
            @test 0.7 < ionization / stopping < 1
        end
    end

    @testset "f pays the energy of the kinetic ionization" begin
        # A bump of electrons at 60 eV in neutral argon and a deuterium plasma: between the
        # ionization energy of Ar⁰ (15.8 eV) and its mean excitation energy (188 eV), where the
        # default screening window gives no Bethe friction. The energy the bump loses to the
        # bound electrons (KineticEnergyLossTerm with part = :sinks) is zero with :bethe, while
        # the kinetic ionization takes the potential and the energy of the secondaries
        # (ionization_energy_flows); with :ionization_split the two agree to the discretization.
        composition = IonComposition(["D" => 1, "Ar" => 18])
        radial = RadialGrid(uniform_r_faces(1, 0.5))
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(240, 0.0, momentum(170.0)), uniform_xi_faces(2)),
            radial)
        m = momentum_grid(grid)
        layout = UnknownLayout(radial,
            [ion_densities_unknown(composition),
                RadialUnknown(:cold_electron_temperature; scale = 10.0),
                KineticUnknown(:f, grid; scale = 1e16)])
        argon = zeros(19, 1)
        argon[1] = 1e19
        ions = [IonSpecies("D", 1, reshape([0.0, 1e19], :, 1)), IonSpecies("Ar", 18, argon)]
        base = PlasmaState(; cold_electron_temperature = [5.0], ions,
            electric_field = [0.0], magnetic_field = [0.0],
            cold_electron_density = [1e19])
        u = zeros(length(layout))
        set_ion_densities!(u, layout, ions)
        unknown_values(u, layout, 2) .= 5.0
        center, width = momentum(60.0), 0.15 * momentum(60.0)
        kinetic_view(u, layout, 3, 1) .= [1e16 * exp(-((p - center) / width)^2)
                                          for p in m.p_centers, _ in 1:2]
        kinetic = KineticIonizationTerm(composition, :f, grid)
        flows = ionization_energy_flows(kinetic, u, layout, 0.0)
        paid = flows.potential[1] + flows.secondaries[1]
        @test paid > 0
        operator() = AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        function sinks(bound_friction)
            settings = CollisionSettings(; mode = :superthermal, bremsstrahlung = :neglect, bound_friction)
            fp = FokkerPlanckTerm(:f, operator(); plasma = base, settings)
            loss = KineticEnergyLossTerm(fp; part = :sinks, coefficient = 1)
            return only(unknown_values(
                Runes2P.rhs!(
                    zeros(length(u)), loss, u, layout, 2, 0.0), layout, 2))
        end
        @test abs(sinks(:bethe)) <= 1e-12 * paid
        @test sinks(:ionization_split) ≈ paid rtol = 2e-2
        # with the threshold of 5 thermal momenta at 0.5 eV (12.5 eV, 15 eV at the top of the
        # step of the faces) the bump lies above it: the same agreement
        cold = 0.5
        unknown_values(u, layout, 2) .= cold
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        above = KineticIonizationTerm(composition, :f, grid; region)
        paid_above = sum(ionization_energy_flows(above, u, layout, 0.0)[k][1]
        for k in (:potential, :secondaries))
        @test paid_above ≈ paid rtol = 1e-3
        cold_base = PlasmaState(; cold_electron_temperature = [cold], ions,
            electric_field = [0.0], magnetic_field = [0.0],
            cold_electron_density = [1e19])
        settings = CollisionSettings(; mode = :superthermal, bremsstrahlung = :neglect,
            bound_friction = :ionization_split,
            ionization_threshold = 5.0)
        fp = FokkerPlanckTerm(:f, operator(); plasma = cold_base, settings)
        loss = KineticEnergyLossTerm(fp; part = :sinks, coefficient = 1)
        @test only(unknown_values(
            Runes2P.rhs!(
                zeros(length(u)), loss, u, layout, 2, 0.0), layout, 2)) ≈
              paid_above rtol = 2e-2
    end

    @testset "the spectrum of Kim and Rudd per charge state" begin
        # eq. (52) of Kim and Rudd, Phys. Rev. A 50, 3954 (1994), with Q = 1 and the common
        # factor left out, in w = W/B up to (t − 1)/2
        shape(w, t) = -1 / (t + 1) * (1 / (w + 1) + 1 / (t - w)) +
                      (1 / (w + 1)^2 + 1 / (t - w)^2) +
                      log(t) * (1 / (w + 1)^3 + 1 / (t - w)^3)
        for (E, B, cut) in ((1000.0, 15.76, Inf), (500.0, 15.76, Inf),
            (1e5, 27.63, 250.0), (5e3, 918.0, Inf),
            (60.0, 13.6, Inf), (1e6, 15.76, Inf), (2e3, 422.0, 300.0))
            t = E / B
            top = min((t - 1) / 2, cut / B)
            count = quadgk(w -> shape(w, t), 0, top; rtol = 1e-12)[1]
            moment = quadgk(w -> w * shape(w, t), 0, top; rtol = 1e-12)[1]
            @test beb_mean_secondary_energy(E, B, cut) ≈ B * moment / count rtol = 1e-9
            # the transcription of the shape: over the whole range it integrates to the total
            # cross section of the BEB model, their eq. (57) at Q = 1
            cut == Inf &&
                @test count ≈ log(t) / 2 * (1 - 1 / t^2) + 1 - 1 / t - log(t) / (t + 1) rtol = 1e-10
        end
        @test beb_mean_secondary_energy(10.0, 15.76, Inf) == 0
        # within 1e-6 B of the threshold the flat spectrum, continuous with the closed form
        for top in (0.99e-6, 1.01e-6)
            @test beb_mean_secondary_energy(15.76 * (1 + 2top), 15.76, Inf) ≈
                  15.76 * top / 2 rtol = 1e-3
        end
        @test beb_mean_secondary_energy(15.76 * (1 + 2e-16), 15.76, Inf) >= 0
        # neutral argon with primaries of 500 eV, where Opal et al. measured Ē = 10.0 eV: 19.2
        # against 20.8 eV
        @test beb_mean_secondary_energy(500.0, 15.76, Inf) ≈
              mean_secondary_energy(10.0, 500.0, 15.76, Inf) rtol = 0.15
        # the spectrum grows with the binding energy: Ar¹⁵⁺ (918 eV) at 5 keV, 576 eV against 34 eV
        # with the width of the neutral
        I15 = ionization_energy("Ar", 15)
        @test beb_mean_secondary_energy(5e3, I15, Inf) >
              10 * mean_secondary_energy(10.0, 5e3, I15, Inf)
        # iron has no width and needs none with :beb, in the friction and in the heating
        iron = PlasmaState(;
            cold_electron_temperature = [10.0], ions = [IonSpecies("Fe", 26, 0, [1e18])],
            electric_field = [0.0], magnetic_field = [0.0], cold_electron_density = [1e19])
        @test Runes2P.CollisionModel(iron,
            CollisionSettings(bound_friction = :ionization_split, secondary_spectrum = :beb)) isa
              Runes2P.CollisionModel
        @test_throws ArgumentError CollisionSettings(secondary_spectrum = :other)
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(20, 0.0, 0.2), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 0.5)))
        fe = KineticIonizationTerm(IonComposition(["Fe" => 26]), :f, grid)
        @test SecondaryHeatingTerm(fe).spectrum === :beb
        @test_throws ArgumentError SecondaryHeatingTerm(fe; spectrum = :opal)
        # the heating and the friction take the same spectrum
        composition = IonComposition(["D" => 1, "Ar" => 18])
        two = RadialGrid(uniform_r_faces(2, 0.5))
        g = cylindrical_grid(MomentumGrid(uniform_p_faces(20, 0.0, 0.2), uniform_xi_faces(2)), two)
        base = PlasmaState(; cold_electron_temperature = [10.0, 8.0],
            ions = [IonSpecies("D", 1, 1, [1e19, 1e19])],
            electric_field = zeros(2), magnetic_field = zeros(2))
        k = KineticIonizationTerm(composition,
            :f,
            g;
            region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5)))
        fp(spectrum) = FokkerPlanckTerm(:f,
            AdvectionDiffusion(g; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(base; ions = :ion_densities),
            settings = CollisionSettings(; bound_friction = :ionization_split,
                ionization_threshold = 5.0,
                secondary_spectrum = spectrum))
        system(spectrum, heating) = [Equation(:f, TimeDerivative(), fp(spectrum)),
            Equation(:ion_densities, TimeDerivative(), k),
            Equation(:cold_electron_temperature, TimeDerivative(), heating),
            population_equations(composition)...]
        @test Runes2P._check_plasma_pairings(system(:beb, SecondaryHeatingTerm(k; spectrum = :beb))) ===
              nothing
        @test_throws "the same" Runes2P._check_plasma_pairings(system(
            :beb, SecondaryHeatingTerm(k; spectrum = :opal)))
        @test_throws "the same" Runes2P._check_plasma_pairings(system(
            :opal, SecondaryHeatingTerm(k; spectrum = :beb)))
    end

    @testset "pairings" begin
        composition = IonComposition(["D" => 1, "Ar" => 18])
        radial = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(20, 0.0, 0.2), uniform_xi_faces(2)), radial)
        base = PlasmaState(; cold_electron_temperature = [10.0, 8.0],
            ions = [IonSpecies("D", 1, 1, [1e19, 1e19])],
            electric_field = zeros(2), magnetic_field = zeros(2))
        operator() = AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
        from_unknowns = PlasmaFromUnknowns(base; ions = :ion_densities)
        closure = population_equations(composition)
        fp(factor) = FokkerPlanckTerm(:f, operator(); plasma = from_unknowns,
            settings = CollisionSettings(
                bound_friction = :ionization_split, ionization_threshold = factor,
                secondary_spectrum = :opal))
        region(factor) = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor))
        ionization(factor) = KineticIonizationTerm(composition, :f, grid; region = region(factor))
        check(fp_term, ion_terms...) = Runes2P._check_plasma_pairings([
            Equation(:f, TimeDerivative(), fp_term),
            Equation(:ion_densities, TimeDerivative(), ion_terms...),
            closure...])
        k5 = ionization(5)
        @test check(fp(5.0), k5) === nothing
        @test check(fp(0.0), KineticIonizationTerm(composition, :f, grid)) === nothing
        @test_throws "the same factor" check(fp(7.0), k5)
        @test_throws "the same form" check(fp(5.0),
            KineticIonizationTerm(composition, :f, grid; region = region(5),
                ionization_cross_section_form = :bethe))
        @test_throws ArgumentError KineticIonizationTerm(
            composition, :f, grid; ionization_cross_section_form = :other)
        @test_throws "has none" Runes2P._check_plasma_pairings([Equation(:f, TimeDerivative(), fp(5.0))])
        @test_throws "ThermalMomentum" check(fp(5.0),
            KineticIonizationTerm(composition, :f, grid; region = MomentumRegion(0.1)))
        heating = Equation(:cold_electron_temperature, TimeDerivative(),
            SecondaryHeatingTerm(k5; spectrum = :opal))
        threshold = Equation(:cold_electron_temperature, TimeDerivative(),
            SecondaryHeatingTerm(k5; cut = :threshold, spectrum = :opal))
        other = Equation(:cold_electron_temperature, TimeDerivative(),
            SecondaryHeatingTerm(k5; widths = Dict("Ar" => 12.0), spectrum = :opal))
        system(heat) = [Equation(:f, TimeDerivative(), fp(5.0)),
            Equation(:ion_densities, TimeDerivative(), k5), heat,
            closure...]
        @test Runes2P._check_plasma_pairings(system(heating)) === nothing
        # where the sinks of f leave the electrons, the secondaries need the heating
        source = fp(5.0)
        sinks = Equation(:total_electron_energy, TimeDerivative(), KineticEnergyLossTerm(source; part = :sinks))
        booked(heat...) = [Equation(:f, TimeDerivative(), source),
            Equation(:ion_densities, TimeDerivative(), k5), sinks, heat...,
            closure...]
        @test_throws "SecondaryHeatingTerm(kinetic_ionization)" Runes2P._check_plasma_pairings(booked())
        @test Runes2P._check_plasma_pairings(booked(heating)) === nothing
        @test_throws "cut = :none" Runes2P._check_plasma_pairings(system(threshold))
        @test_throws "different widths" Runes2P._check_plasma_pairings(system(other))
        # DREAM's Bethe friction pairs with any kinetic ionization
        @test Runes2P._check_plasma_pairings([
            Equation(:f,
                TimeDerivative(),
                FokkerPlanckTerm(:f, operator(); plasma = from_unknowns,
                    settings = CollisionSettings())),
            Equation(:ion_densities, TimeDerivative(), ionization(7)), closure...]) ===
              nothing
        @test SecondaryHeatingTerm(k5).cut === :none
        @test_throws ArgumentError SecondaryHeatingTerm(k5; cut = :other)
    end
end
