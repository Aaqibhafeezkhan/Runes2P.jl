# The heating and the radiative losses of the cold electrons, on synthetic
# ADF11 tables of helium: the radiated power against a sum of the tabulated coefficients
# in both modes, the recombination gain of each charge state at every radius,
# Gould's bremsstrahlung against the NRL formulary, the Jacobian, and the ohmic heating
# against σ E² with Spitzer's conductivity.

@testset "radiated power and ohmic heating" begin
    directory = mktempdir()
    log_density = [12.0, 13.0, 14.0, 15.0]                 # cm⁻³
    log_temperature = [-0.3, 0.3, 1.0, 1.7, 2.4, 3.0]
    plane(offset, b) = [offset - 0.4b + 0.05x - 0.3y
                        for x in log_density, y in log_temperature]
    offsets = (acd = -12.0, scd = -8.5, ccd = -9.0, plt = -25.5, prb = -26.5)
    for class in Runes2P.ADF11_CLASSES
        write_adf11(joinpath(directory, adf11_file_name("He", class)),
            2, log_density, log_temperature,
            [plane(offsets[class], b) for b in 1:2])
    end
    radii = 3
    # a different mix of charge states at every radius
    densities = [1e18 2e17 5e17; 3e18 1e18 2e18; 5e17 4e18 1e18]   # rows Z0 = 0, 1, 2
    plasma = PlasmaState(; cold_electron_temperature = fill(20.0, radii),
        ions = [IonSpecies("He", 2, densities)],
        electric_field = zeros(radii), magnetic_field = zeros(radii))
    layout = UnknownLayout(RadialGrid(uniform_r_faces(radii, 0.5)),
        [RadialUnknown(:cold_electron_temperature; scale = 10.0),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e20)])
    u = zeros(length(layout))
    radial_values(u, layout, 1) .= [20.0, 35.0, 8.0]
    radial_values(u, layout, 2) .= [8e18, 1.5e19, 6e18]
    e = Constants.ELEMENTARY_CHARGE
    tables = read_adas("He"; directory)

    @testset "against the tabulated coefficients" begin
        bremsstrahlung = Dict{Symbol, Vector{Float64}}()   # Gould's part per radius, by the corrections
        for (radiation, corrections) in ((:recombination, :gould), (:dream, :gould), (
            :dream, :dream))
            term = RadiatedPowerTerm(plasma; radiation, bremsstrahlung_corrections = corrections, directory)
            parts = radiated_power(term, u, layout, 0.0)
            out = zeros(length(u))
            Runes2P.rhs!(out, term, u, layout, 1, 0.0)
            @test radial_values(out, layout, 1) ≈ .-(parts.radiation .+ parts.binding) rtol = 1e-14
            for ir in 1:radii
                T, n = radial_values(u, layout, 1)[ir], radial_values(u, layout, 2)[ir]
                k(table, Z0) = adas_coefficient(table, Z0, n, T)
                line = sum(densities[Z0 + 1, ir] * (k(tables.plt, Z0) +
                            (radiation === :recombination ? k(tables.prb, Z0) : 0))
                for Z0 in 0:2)
                ionization = sum(densities[Z0 + 1, ir] * e * ionization_energy("He", Z0) *
                                 k(tables.scd, Z0) for Z0 in 0:1)
                # the recombination of Z0 returns the energy that ionized Z0 − 1, at every radius
                gain = sum(densities[Z0 + 1, ir] * e * ionization_energy("He", Z0 - 1) *
                           k(tables.acd, Z0) for Z0 in 1:2)
                if radiation === :recombination
                    @test parts.radiation[ir] ≈ n * line rtol = 1e-13
                    @test parts.binding[ir] ≈ n * (ionization - gain) rtol = 1e-13
                else
                    @test parts.binding[ir] ≈ n * ionization rtol = 1e-13
                    # Gould's bremsstrahlung: 1.69e-38 √T Z² n_e n_i W/m³ (NRL formulary), and
                    # its relativistic corrections at T/mc²: 11/24 and 5(44 − 3π²)/(24√2) of Gould
                    # after his erratum and by his eqs. (18) and (22), or DREAM's 19/24 and
                    # 5(44 − 3π²)/(8√2)
                    ion_sum = sum(densities[Z0 + 1, ir] * Z0^2 for Z0 in 0:2)
                    theta = T / Constants.ELECTRON_REST_ENERGY
                    a, κ = corrections === :gould ?
                           (11 / 24, 5 / (24 * sqrt(2)) * (44 - 3π^2)) :
                           (19 / 24, 5 / (8 * sqrt(2)) * (44 - 3π^2))
                    nrl = 1.69e-38 * sqrt(T) * n *
                          (ion_sum * (1 + a * theta) + κ * n * theta)
                    @test parts.radiation[ir] - n * line ≈ nrl rtol = 3e-3
                    push!(get!(bremsstrahlung, corrections, Float64[]), parts.radiation[ir] -
                                                                        n * line)
                end
            end
        end
        # the two sets of corrections against each other, where the NRL constant (1.69e-38 against
        # the computed 1.6933e-38) hides their difference of 3e-4 at these temperatures;
        # the subtraction of the line part leaves a round-off of 1e-11 in the ratio
        for ir in 1:radii
            T, n = radial_values(u, layout, 1)[ir], radial_values(u, layout, 2)[ir]
            theta = T / Constants.ELECTRON_REST_ENERGY
            ion_sum = sum(densities[Z0 + 1, ir] * Z0^2 for Z0 in 0:2)
            ratio = (ion_sum * (1 + 19 / 24 * theta) +
                     5 / (8 * sqrt(2)) * (44 - 3π^2) * n * theta) /
                    (ion_sum * (1 + 11 / 24 * theta) +
                     5 / (24 * sqrt(2)) * (44 - 3π^2) * n * theta)
            @test bremsstrahlung[:dream][ir] / bremsstrahlung[:gould][ir] ≈ ratio rtol = 1e-9
            @test ratio > 1 + 1e-5
        end
        @test Runes2P.GOULD_BREMSSTRAHLUNG_PREFACTOR ≈ 1.69e-38 rtol = 3e-3
    end

    @testset "Jacobian" begin
        term = RadiatedPowerTerm(plasma; directory)
        system = EquationSystem(layout,
            [
                Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), term),
                Equation(:cold_electron_density, Algebraic(),
                    PrescribedValueTerm([8e18, 1.5e19, 6e18]),
                    LinearTerm(:cold_electron_density; coefficient = -1)),
                thermal_energy_equation()];
            initial_state = u)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        for id in 1:2, ir in 1:radii

            column = first(Runes2P.unknown_range(layout, id, ir))
            h = 1e-6 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(u)), zeros(length(u))
            rhs!(F_up, up, system, 0.0)
            rhs!(F_down, down, system, 0.0)
            @test Vector(J[:, column]) ≈ (F_up .- F_down) ./ (2h) rtol = 1e-6
        end
    end

    @testset "refusals and record" begin
        @test_throws ArgumentError RadiatedPowerTerm(plasma; radiation = :other, directory)
        @test_throws ArgumentError RadiatedPowerTerm(plasma; extrapolation = :other, directory)
        unknown = PlasmaState(;
            cold_electron_temperature = [10.0], ions = [IonSpecies("Xx", 1, 1, [1e19])],
            electric_field = [0.0], magnetic_field = [0.0])
        @test_throws "ADF11 tables" RadiatedPowerTerm(unknown; directory)
        @test_throws "fetch_adas.py" RadiatedPowerTerm(
            PlasmaState(; cold_electron_temperature = [10.0],
                ions = [IonSpecies("Ne", 10, 1, [1e19])],
                electric_field = [0.0], magnetic_field = [0.0]);
            directory)
        record = describe(RadiatedPowerTerm(plasma; radiation = :dream, directory),
            Runes2P.DescriptionContext())
        @test record["radiation"] == "dream" &&
              record["temperature"] == "cold_electron_temperature"
        @test record["bremsstrahlung_corrections"] == "gould"
        @test_throws ArgumentError RadiatedPowerTerm(plasma; bremsstrahlung_corrections = :other, directory)
        @test record["adf11_files"]["He"]["plt"] == "plt96_he.dat"
        # a term of the cold bulk for the guard of one temperature
        @test Runes2P._temperature_source(RadiatedPowerTerm(plasma; directory)) ==
              (:unknown, :cold_electron_temperature)
    end

    @testset "ohmic heating" begin
        # j_ohm from the conductivity of Braams and Karney, which is Spitzer's at low temperature
        T, n = 10.0, 1e20
        one_radius = PlasmaState(;
            cold_electron_temperature = [T], ions = [IonSpecies("D", 1, 1, [n])],
            electric_field = [0.0], magnetic_field = [0.0])
        heated = UnknownLayout(RadialGrid(uniform_r_faces(1, 0.5)),
            [RadialUnknown(:electric_field),
                RadialUnknown(:ohmic_current_density; scale = 1e6),
                RadialUnknown(:heating; scale = 1e6)])
        E = 0.3
        equations = [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm([E]),
                LinearTerm(:electric_field; coefficient = -1)),
            Equation(:ohmic_current_density, Algebraic(),
                ConductivityTerm(:electric_field, ConductivityModel(one_radius)),
                LinearTerm(:ohmic_current_density; coefficient = -1)),
            Equation(:heating, Algebraic(), ohmic_heating_term(), LinearTerm(:heating; coefficient = -1))]
        system = EquationSystem(heated, equations)
        state = [E, braams_karney_conductivity(one_radius, 1) * E, 0.0]
        F = zeros(3)
        rhs!(F, state, system, 0.0)
        sigma = braams_karney_conductivity(one_radius, 1)
        @test F[3] ≈ sigma * E^2 rtol = 1e-14
        # Spitzer and Härm, Table III: σ = 0.5816 σ_L at Z = 1, with the Lorentz conductivity
        # σ_L = 64√(2π) ε₀² T^{3/2} / (√m e² Z lnΛ) (as in test_conductivity.jl)
        lorentz = 64 * sqrt(2π) * Constants.VACUUM_PERMITTIVITY^2 * (T * e)^1.5 /
                  (sqrt(Constants.ELECTRON_MASS) * e^2 * thermal_coulomb_logarithm(T, n))
        @test F[3] / E^2 ≈ 0.5816 * lorentz rtol = 4e-3
    end
end
