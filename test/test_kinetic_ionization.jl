# The ionization of the ions by the electrons of a distribution: the cross
# sections of DREAM's fits against the ADAS coefficients at low density, the conservation of
# each species, the Jacobian of the term against central differences, and the split of the
# ionization at the hot threshold: a Maxwellian f ionizes at the ADAS rate for every
# threshold, and DREAM's pairing counts the tail above the threshold twice. Needs the ADF11
# cache (scripts/fetch_adas.py).
using SparseArrays: sparse
using ForwardDiff
using QuadGK: quadgk

const KINETIC_IONIZATION_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

# a grid that covers the bulk at 20 and 10 eV to 10 thermal momenta, with 81 cells so that
# no threshold of 2 or 3 thermal momenta lies on a face, where the interpolated envelope
# has a kink
function _ionization_case(; factor=3.0, temperatures=[20.0, 10.0], n_M=5e19)
    radial = RadialGrid(uniform_r_faces(2, 0.5))
    p_max = 10 * sqrt(2 * maximum(temperatures) / Constants.ELECTRON_REST_ENERGY)
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(81, 0.0, p_max), uniform_xi_faces(3)), radial)
    layout = UnknownLayout(radial, [ion_densities_unknown(KINETIC_IONIZATION_COMPOSITION),
                                    RadialUnknown(:cold_electron_density; scale=1e19),
                                    RadialUnknown(:cold_electron_temperature; scale=10.0),
                                    KineticUnknown(:f, grid; scale=1e19)])
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor))
    kinetic = KineticIonizationTerm(KINETIC_IONIZATION_COMPOSITION, :f, grid; region)
    u = zeros(length(layout))
    argon = zeros(19, 2)
    argon[1:6, :] .= [1e17, 3e17, 5e17, 2e17, 1e17, 5e16]
    set_ion_densities!(u, layout, [IonSpecies("D", 1, [1e18 2e18; 4e19 3e19]), IonSpecies("Ar", 18, argon)])
    radial_values(u, layout, 3) .= temperatures
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
        @test size(KINETIC_IONIZATION_FITS["Ar"]) == (18, 3) && size(KINETIC_IONIZATION_FITS["D"]) == (1, 3)
    end

    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparisons with ADAS are skipped"
    else
        @testset "Maxwellian average against ADAS at low density" begin
            # the fit was made to the SCD coefficients at zero density: the Maxwellian average
            # of v σ at the ionization energy of each state, against SCD at the lowest density
            # node, per state of argon and neon (DREAM #46: errors of tens of per cent)
            p = collect(range(0.0, 1.0; length=200_001))[2:end]
            for element in ("Ar", "Ne")
                scd = read_adas(element).scd
                Z = size(KINETIC_IONIZATION_FITS[element], 1)
                ratios = Float64[]
                for Z0 in 0:Z-1
                    T = ionization_energy(element, Z0)
                    theta = T / Constants.ELECTRON_REST_ENERGY
                    weight = @. p^2 * exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)
                    rate = sum(weight .* [Constants.SPEED_OF_LIGHT * x / sqrt(1 + x^2) *
                                          ionization_cross_section(element, Z0, x) for x in p]) / sum(weight)
                    push!(ratios, rate / adas_coefficient(scd, Z0, exp10(first(scd.log_density)), T))
                end
                @info "Maxwellian average over ADAS SCD at T = I per state of $element: " *
                      join(round.(ratios; digits=3), ", ")
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
            entries = Runes2P.sparsity!(Tuple{Int,Int}[], kinetic, layout, 1)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            Runes2P.add_jacobian!(J, kinetic, u, layout, 1, 0.0)
            term_rhs(x) = Runes2P.rhs!(zeros(n), kinetic, x, layout, 1, 0.0)
            size_of_rhs = maximum(abs, du)
            scales = unknown_scales(layout)
            # the columns of radius 2: ion densities, the temperature, a few cells of f
            cells = unknown_range(layout, 4, 2)
            for column in [collect(unknown_range(layout, 1, 2)); unknown_range(layout, 3, 2); cells[[1, 20, 40, 41, 120]]]
                h = column in unknown_range(layout, 3, 2) ? 1e-6 * u[column] : 1e-3 * max(abs(u[column]), 1e-3 * scales[column])
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (term_rhs(up) - term_rhs(down)) / 2h
                @test maximum(abs, Vector(J[:, column]) - difference) <=
                      1e-6 * maximum(abs, difference) + 1e-13 * size_of_rhs / h
            end
        end

        @testset "split at the threshold" begin
            for factor in (2.0, 3.0)
                layout, kinetic, u = _ionization_case(; factor)
                n = length(u)
                splits = [threshold_split(kinetic, ir, u, layout, 0.0) for ir in 1:2]
                @test all(s -> 0 < s.fraction < 0.1, splits)
                # the closure: the cold electrons are those of the Maxwellian below the threshold
                n_cold = [5e19 * (1 - s.fraction) for s in splits]
                radial_values(u, layout, 2) .= n_cold
                paired = IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic)
                dream = IonRateTerm(KINETIC_IONIZATION_COMPOSITION)
                kinetic_part = Runes2P.rhs!(zeros(n), kinetic, u, layout, 1, 0.0)
                for (fluid, exact) in ((paired, true), (dream, false))
                    du = Runes2P.rhs!(zeros(n), fluid, u, layout, 1, 0.0) .+ kinetic_part
                    for ir in 1:2
                        T = radial_values(u, layout, 3)[ir]
                        rows = unknown_range(layout, 1, ir)
                        for s in 1:2
                            name = KINETIC_IONIZATION_COMPOSITION.names[s]
                            tables = read_adas(name)
                            states = species_components(KINETIC_IONIZATION_COMPOSITION, s)
                            # the ionization flow of each state: the net flow plus recombination
                            for (j, k) in enumerate(states[1:end-1])
                                Z0 = j - 1
                                net = -sum(du[rows[states[1:j]]])
                                recombination = n_cold[ir] * adas_coefficient(tables.acd, Z0 + 1, n_cold[ir], T) * u[rows[k+1]]
                                ionization = net + recombination
                                expected = 5e19 * adas_coefficient(tables.scd, Z0, n_cold[ir], T) * u[rows[k]]
                                u[rows[k]] > 0 || continue
                                if exact
                                    # the flows are differences of the rates of the states: round-off
                                    @test ionization ≈ expected rtol = 1e-9
                                else
                                    # DREAM's pairing: the tail above the threshold twice
                                    @test ionization > expected * (1 + 1e-9) || splits[ir].coefficients[k] == 0
                                end
                            end
                        end
                    end
                end
            end
        end

        @testset "the fluid rate's Jacobian with the split" begin
            layout, kinetic, u = _ionization_case()
            radial_values(u, layout, 2) .= 4.5e19
            equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(KINETIC_IONIZATION_COMPOSITION; kinetic),
                                  kinetic),
                         Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(4.5e19),
                                  LinearTerm(:cold_electron_density; coefficient=-1)),
                         Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm([20.0, 10.0]),
                                  LinearTerm(:cold_electron_temperature; coefficient=-1)),
                         Equation(:f, TimeDerivative())]
            system = EquationSystem(layout, equations; initial_state=u)
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
                @test Vector(J[ion_rows, column]) ≈ difference rtol = 1e-5 atol = 1e-6 * maximum(abs, difference)
            end
            # the binding energy at the fluid coefficient of the split
            prescribed = PlasmaState(; cold_electron_temperature=[20.0, 10.0], ions=ion_densities(u, layout,
                                                                                                   KINETIC_IONIZATION_COMPOSITION),
                                     electric_field=zeros(2), magnetic_field=zeros(2))
            full = radiated_power(RadiatedPowerTerm(prescribed; ions=:ion_densities), u, layout, 0.0).binding
            split = radiated_power(RadiatedPowerTerm(prescribed; ions=:ion_densities, kinetic), u, layout, 0.0).binding
            @test all(split .< full)
            # the direct block of the radiated power with the split, the temperature column
            # through the moving threshold included, against the forward-mode Jacobian
            n = length(u)
            term = RadiatedPowerTerm(prescribed; ions=:ion_densities, kinetic)
            entries = Runes2P.sparsity!(Tuple{Int,Int}[], term, layout, 3)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            add_local_jacobian!(J, term, u, layout, 3, 0.0)
            reference = ForwardDiff.jacobian(x -> Runes2P.rhs!(zeros(eltype(x), n), term, x, layout, 3, 0.0), u)
            @test Matrix(J) ≈ reference rtol = 1e-10 atol = 1e-12 * maximum(abs, reference)
            # and the temperature column against a central difference
            column = first(unknown_range(layout, 3, 1))
            h = 1e-6 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (Runes2P.rhs!(zeros(n), term, up, layout, 3, 0.0) - Runes2P.rhs!(zeros(n), term, down, layout, 3, 0.0)) / 2h
            @test Vector(J[:, column]) ≈ difference rtol = 1e-5 atol = 1e-8 * maximum(abs, difference)
        end
        @testset "secondaries and the energy of the kinetic ionization (issue #177)" begin
            @test mean_secondary_energy(24.2, 10.0, 15.8, 1e3) == 0
            # without a cut: the mean of the Opal spectrum up to (E − I)/2
            top = (1000.0 - 15.8) / 2
            @test mean_secondary_energy(24.2, 1000.0, 15.8, Inf) ≈ 12.1 * log(1 + (top / 24.2)^2) / atan(top / 24.2)
            @test mean_secondary_energy(24.2, 1000.0, 15.8, 50.0) < mean_secondary_energy(24.2, 1000.0, 15.8, 100.0)
            # independent of the closed form: the mean energy of the spectrum of Opal et al.,
            # σ(E_s) ∝ 1/(1 + (E_s/w)²), by quadrature up to the smaller of (E − I)/2 and the cut
            for (w, E, I, cut) in ((24.2, 1000.0, 15.8, Inf), (10.0, 300.0, 15.76, 40.0), (8.3, 60.0, 13.6, Inf),
                                   (15.8, 5.0e4, 24.6, 120.0))
                top = min((E - I) / 2, cut)
                spectrum(x) = 1 / (1 + (x / w)^2)
                mean = quadgk(x -> x * spectrum(x), 0, top; rtol=1e-12)[1] / quadgk(spectrum, 0, top; rtol=1e-12)[1]
                @test mean_secondary_energy(w, E, I, cut) ≈ mean rtol = 1e-10
            end
            layout, kinetic, u = _ionization_case()
            n = length(u)
            # the widths of Opal et al. (1971), Table II: H₂ for deuterium, argon 10.0 eV
            heating = SecondaryHeatingTerm(kinetic)
            @test heating.widths == [8.3, 10.0]
            @test SecondaryHeatingTerm(kinetic; widths=Dict("Ar" => 12.0)).widths == [8.3, 12.0]
            # an element without a width is refused: iron
            fe = KineticIonizationTerm(IonComposition(["Fe" => 26]), :f, kinetic.grid; region=kinetic.envelopes.region)
            @test_throws ArgumentError SecondaryHeatingTerm(fe)
            widths = Dict{String,Float64}()
            row = 3
            Runes2P.validate(heating, layout, row)
            Q = radial_values(Runes2P.rhs!(zeros(n), heating, u, layout, row, 0.0), layout, row)
            # the heating against a direct sum over the cells, the states and the species
            for ir in 1:2
                T = radial_values(u, layout, 3)[ir]
                m = momentum_grid(kinetic.grid, ir)
                p0 = 3.0 * sqrt(2T / Constants.ELECTRON_REST_ENERGY)
                cut = (sqrt(1 + p0^2) - 1) * Constants.ELECTRON_REST_ENERGY
                envelope = momentum_envelope!(zeros(length(m.p_centers)), kinetic.envelopes.region, m, p0)
                f = kinetic_view(u, layout, 4, ir)
                states = radial_view(u, layout, 1, ir)
                expected = 0.0
                for (s, name, Z, width) in ((1, "D", 1, 8.3), (2, "Ar", 18, 10.0)), Z0 in 0:Z-1
                    k = species_components(KINETIC_IONIZATION_COMPOSITION, s)[Z0+1]
                    for (i, p) in enumerate(m.p_centers), j in 1:3
                        E = (sqrt(1 + p^2) - 1) * Constants.ELECTRON_REST_ENERGY
                        v = Constants.SPEED_OF_LIGHT * p / sqrt(1 + p^2)
                        expected += states[k] * envelope[i] * v * ionization_cross_section(name, Z0, p) *
                                    Constants.ELEMENTARY_CHARGE *
                                    mean_secondary_energy(width, E, ionization_energy(name, Z0), cut) *
                                    kinetic.weights[ir][i, j] * f[i, j]
                    end
                end
                @test Q[ir] ≈ expected rtol = 1e-12
                @test Q[ir] > 0
            end
            # the direct block against central differences: a cell of f, the charge states, the temperature
            entries = Runes2P.sparsity!(Tuple{Int,Int}[], heating, layout, row)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            Runes2P.add_jacobian!(J, heating, u, layout, row, 0.0)
            term_rhs(x) = Runes2P.rhs!(zeros(n), heating, x, layout, row, 0.0)
            size_of_rhs = maximum(abs, term_rhs(u))
            scales = unknown_scales(layout)
            for column in [collect(unknown_range(layout, 1, 2)); unknown_range(layout, 3, 2); unknown_range(layout, 4, 2)[[30, 41]]]
                h = column in unknown_range(layout, 3, 2) ? 1e-6 * u[column] : 1e-3 * max(abs(u[column]), 1e-3 * scales[column])
                up, down = copy(u), copy(u)
                up[column] += h
                down[column] -= h
                difference = (term_rhs(up) - term_rhs(down)) / 2h
                @test maximum(abs, Vector(J[:, column]) - difference) <= 1e-5 * maximum(abs, difference) + 1e-13 * size_of_rhs / h
            end
            # the potential energy is e I_j times the kinetic flows, the secondaries the heating
            flows = ionization_energy_flows(kinetic, u, layout, 0.0; widths)
            @test flows.secondaries ≈ Q rtol = 1e-14
            du = Runes2P.rhs!(zeros(n), kinetic, u, layout, 1, 0.0)
            for ir in 1:2
                rows = unknown_range(layout, 1, ir)
                expected = 0.0
                for (s, name, Z) in ((1, "D", 1), (2, "Ar", 18))
                    states = species_components(KINETIC_IONIZATION_COMPOSITION, s)
                    for Z0 in 0:Z-1
                        flow = -sum(du[rows[states[1:Z0+1]]])
                        expected += Constants.ELEMENTARY_CHARGE * ionization_energy(name, Z0) * flow
                    end
                end
                @test flows.potential[ir] ≈ expected rtol = 1e-10
            end
            @test Runes2P.describe(heating, Runes2P.DescriptionContext())["widths"]["Ar"] == 10.0
            # without a width the diagnostic still gives the potential energy: iron
            fe_layout = UnknownLayout(layout.radial_grid, [ion_densities_unknown(IonComposition(["Fe" => 26])),
                                                           layout.unknowns[2:end]...])
            fe_state = zeros(length(fe_layout))
            radial_values(fe_state, fe_layout, 3) .= radial_values(u, layout, 3)
            for ir in 1:2
                radial_view(fe_state, fe_layout, 1, ir)[1:3] .= 1e17
                kinetic_view(fe_state, fe_layout, 4, ir) .= kinetic_view(u, layout, 4, ir)
            end
            no_width = ionization_energy_flows(fe, fe_state, fe_layout, 0.0)
            @test no_width.secondaries === nothing && all(>(0), no_width.potential)
        end

    end
end
