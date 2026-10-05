# The OpenADAS files of the cache, where it exists: every file the table
# of Runes2P names is there and reads, and the reader agrees with the tables DREAM compiles
# from the same files (`src/Atomics/adasdata.cpp` of the reference build, 5 decimals in
# log₁₀). No ADAS data enters the repository: without the cache the tests are skipped.

# the arrays `<name> = {...}` of DREAM's adasdata.cpp
function _dream_adas_arrays(path)
    arrays = Dict{String,Vector{Float64}}()
    for m in eachmatch(r"const real_t (\w+)\[\d+\] = \{([^}]*)\};", read(path, String))
        arrays[m.captures[1]] = parse.(Float64, split(m.captures[2], ','))
    end
    return arrays
end

@testset "OpenADAS cache" begin
    directory = adas_cache_directory()
    if !isdir(directory)
        @info "no ADF11 cache at $directory (scripts/fetch_adas.py): the tests on OpenADAS data are skipped"
    else
        elements = sort!(collect(keys(Runes2P.ADAS_FILES)))
        @test all(element -> all(class -> isfile(joinpath(directory, adf11_file_name(element, class))),
                                 Runes2P.ADF11_CLASSES), elements)
        dream = joinpath(@__DIR__, "..", "reference", "dream", "DREAM", "src", "Atomics", "adasdata.cpp")
        arrays = isfile(dream) ? _dream_adas_arrays(dream) : nothing
        arrays === nothing && @info "no adasdata.cpp of the DREAM build: the comparison with DREAM is skipped"
        for element in elements
            tables = read_adas(element)
            charges = unique(table.nuclear_charge for table in tables)
            @test length(charges) == 1
            @test length(IONIZATION_ENERGIES[element]) == only(charges)
            arrays === nothing && continue
            for class in Runes2P.ADF11_CLASSES
                table = tables[class]
                prefix = "$(element)_$(class)"
                @test table.log_density ≈ arrays[prefix*"_n"] atol = 1e-9
                @test table.log_temperature ≈ arrays[prefix*"_T"] atol = 1e-9
                @test vec(table.log_coefficient) ≈ arrays[prefix*"_coeff"] atol = 1e-9
            end
        end
        # argon at a disruption density: the line radiation of the neutral, inside the table
        # and below its lowest temperature of 1 eV, where DREAM extrapolates
        argon = read_adas("Ar")
        @test 10.0^argon.plt.log_temperature[1] ≈ 1.0 rtol = 1e-3
        @test adas_coefficient(argon.plt, 0, 1e20, 5.0) > 0
        @test adas_coefficient(argon.scd, 0, 1e20, 0.5) < adas_coefficient(argon.scd, 0, 1e20, 1.0)

        # An independent reference: the radiation of argon in coronal equilibrium against the
        # cooling factor of Pütterich et al., Nucl. Fusion 59 (2019) 056013, Fig. 1, at
        # n_e = 5e19 m⁻³, read from the vector graphic to a few percent. Its atomic data are
        # newer than the ADAS year 89 of DREAM's argon and good to 20-30 % above 100 eV, worse
        # below; the ratios were 1.37, 1.17, 0.84, 0.61 and 0.69 in a first estimate.
        n_e = 5e19
        cooling = Dict(10.0 => 7.7e-32, 30.0 => 4.6e-32, 100.0 => 2.3e-32, 300.0 => 4.5e-32, 1000.0 => 4.2e-33)
        layout = UnknownLayout(RadialGrid(uniform_r_faces(1, 0.5)), [RadialUnknown(:cold_electron_temperature),
                                                                     RadialUnknown(:cold_electron_density)])
        for (T, reference) in cooling
            # the charge states in coronal equilibrium: n(Z0 + 1) / n(Z0) = SCD(Z0) / ACD(Z0 + 1)
            fractions = [1.0]
            for Z0 in 0:17
                push!(fractions, fractions[end] * adas_coefficient(argon.scd, Z0, n_e, T) /
                                 adas_coefficient(argon.acd, Z0 + 1, n_e, T))
            end
            fractions ./= sum(fractions)
            plasma = PlasmaState(; cold_electron_temperature=[T], ions=[IonSpecies("Ar", 18, reshape(1e17 .* fractions, :, 1))],
                                 electric_field=[0.0], magnetic_field=[0.0])
            term = RadiatedPowerTerm(plasma; radiation=:recombination)
            power = radiated_power(term, [T, n_e], layout, 0.0)
            radiation = only(power.radiation)
            ratio = radiation / (n_e * 1e17) / reference
            @test 0.5 < ratio < 2
            # in coronal equilibrium the ionization of each state balances the recombination
            # onto the next, so the binding energy vanishes; a charge-state slip of SCD, ACD or
            # the ionization energy (DREAM #140) leaves a remainder of the size of the
            # ionization power
            ionization_power = sum(Constants.ELEMENTARY_CHARGE * ionization_energy("Ar", Z0) * n_e * 1e17 *
                                   fractions[Z0+1] * adas_coefficient(argon.scd, Z0, n_e, T) for Z0 in 0:17)
            @test abs(only(power.binding)) < 1e-10 * ionization_power
        end
    end
end
