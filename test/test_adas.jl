# OpenADAS rate coefficients on synthetic tables, with no ADAS data: the
# reader of the ADF11 format, the interpolation against GSL's values for DREAM's rule
# (test/data/gsl_bicubic_reference.txt, from reference/gsl/bicubic_reference.c), exact
# reproduction of bilinear functions, the charge states of each class, the extrapolations,
# the derivatives, and the ionization energies of NIST.
import ForwardDiff

@testset "OpenADAS rate coefficients" begin
    directory = mktempdir()
    log_density = [7.5, 8.0, 9.0, 10.0, 11.5]        # cm⁻³
    log_temperature = [-0.5, 0.0, 0.7, 1.4, 2.0, 3.0]
    Z = 2
    # log₁₀ of the coefficient, bilinear in the logarithms: reproduced exactly
    plane(b, x, y) = -12.0 - b + 0.25 * x - 0.8 * y + 0.03 * x * y
    blocks = [[plane(b, x, y) for x in log_density, y in log_temperature] for b in 1:Z]

    @testset "reader" begin
        table = read_adf11(write_adf11(joinpath(directory, "scd99_xx.dat"), Z, log_density, log_temperature, blocks))
        @test table.class === :scd && table.nuclear_charge == Z
        @test table.log_density ≈ log_density .+ 6 atol = 1e-12
        @test table.log_temperature == log_temperature
        @test size(table.log_coefficient) == (5, 6, 2)
        @test table.log_coefficient[:, :, 2] ≈ round.(blocks[2]; digits=5) .- 6 atol = 1e-12
        @test read_adf11(table.source; class=:plt).class === :plt
        @test_throws ArgumentError read_adf11(write_adf11(joinpath(directory, "acd99_xx.dat"), 3, log_density,
                                                          log_temperature, blocks))
        @test_throws "metastable" read_adf11(write_adf11(joinpath(directory, "acd98_xx.dat"), Z, log_density,
                                                         log_temperature, blocks; iprt=2))
        @test_throws ArgumentError read_adf11(joinpath(directory, "xyz99_xx.dat"))
        # a dashed line after the last block, as in the files of hydrogen
        @test size(read_adf11(write_adf11(joinpath(directory, "scd96_xx.dat"), Z, log_density, log_temperature, blocks;
                                          closing_line=true)).log_coefficient) == (5, 6, 2)
        # IPRT = 0, no parent, as in the line-radiation files of argon (year 89)
        @test read_adf11(write_adf11(joinpath(directory, "plt97_xx.dat"), Z, log_density, log_temperature, blocks;
                                     iprt=0)).class === :plt
    end

    @testset "interpolation against GSL" begin
        lines = filter(line -> !startswith(line, "#"), readlines(joinpath(@__DIR__, "data", "gsl_bicubic_reference.txt")))
        header = readlines(joinpath(@__DIR__, "data", "gsl_bicubic_reference.txt"))
        nodes(prefix) = parse.(Float64, split(only(filter(l -> startswith(l, prefix), header))[length(prefix)+1:end]))
        xs, ys = nodes("# x nodes:"), nodes("# y nodes:")
        f(x, y) = sin(1.3x) * cos(0.7y) + 0.3x * y^2 - 0.1x^2
        z = reshape([f(x, y) for x in xs, y in ys], length(xs), length(ys), 1)
        table = Runes2P.ADF11Table(:scd, 1, xs, ys, z, Runes2P._bicubic_derivatives(xs, ys, z), "synthetic")
        for line in lines
            x, y, expected = parse.(Float64, split(line))
            @test Runes2P._adf11_log_value(table, 1, x, y, :dream) ≈ expected atol = 1e-13
        end
    end

    @testset "charge states, extrapolation, derivatives" begin
        scd = read_adf11(joinpath(directory, "scd99_xx.dat"))
        acd = read_adf11(write_adf11(joinpath(directory, "acd99_xx.dat"), Z, log_density, log_temperature, blocks))
        coefficient(table, Z0, n, T; extrapolation=:dream) = adas_coefficient(table, Z0, n, T; extrapolation)
        exact(b, n, T) = exp10(round(plane(b, log10(n) - 6, log10(T)); digits=5) - 6)
        # inside and outside the table, a bilinear function is reproduced
        for (n, T) in ((3e14, 2.0), (1e17, 40.0), (1e12, 2.0), (1e20, 0.1), (3e14, 3000.0))
            @test coefficient(scd, 0, n, T) ≈ exp10(plane(1, log10(n) - 6, log10(T)) - 6) rtol = 1e-4
        end
        @test coefficient(scd, 0, 1e15, 10.0^0.7) ≈ exact(1, 1e15, 10.0^0.7) rtol = 1e-12
        # ionization of the bare nucleus and recombination of the neutral do not exist
        @test coefficient(scd, Z, 1e18, 10.0) == 0
        @test coefficient(acd, 0, 1e18, 10.0) == 0
        @test coefficient(acd, 1, 1e15, 10.0^0.7) ≈ exact(1, 1e15, 10.0^0.7) rtol = 1e-12
        @test coefficient(scd, 1, 1e15, 10.0^0.7) ≈ exact(2, 1e15, 10.0^0.7) rtol = 1e-12
        @test coefficient(scd, 0, 0.0, 10.0) == 0 && coefficient(scd, 0, 1e18, -1.0) == 0
        # the density clamped at the edge of the table, the temperature extrapolated
        # (at the edge itself, since the value jumps across it between the nodes)
        edge = scd.log_density[1]
        @test log10(coefficient(scd, 0, 1e10, 10.0; extrapolation=:clamp_density)) ≈
              Runes2P._adf11_log_value(scd, 1, edge, 1.0, :dream) rtol = 1e-14
        @test log10(coefficient(scd, 0, 1e10, 0.1; extrapolation=:clamp_density)) ≈
              Runes2P._adf11_log_value(scd, 1, edge, -1.0, :dream) rtol = 1e-14
        @test_throws ArgumentError coefficient(scd, 0, 1e18, 10.0; extrapolation=:other)
        # exact derivatives through the interpolation, inside and outside the table
        for (n, T) in ((3e14, 2.3), (1e20, 0.1))
            dT = ForwardDiff.derivative(T -> coefficient(scd, 0, n, T), T)
            dn = ForwardDiff.derivative(n -> coefficient(scd, 0, n, T), n)
            h = 1e-6
            @test dT ≈ (coefficient(scd, 0, n, T * (1 + h)) - coefficient(scd, 0, n, T * (1 - h))) / (2h * T) rtol = 1e-6
            @test dn ≈ (coefficient(scd, 0, n * (1 + h), T) - coefficient(scd, 0, n * (1 - h), T)) / (2h * n) rtol = 1e-6
        end
    end

    @testset "files and cache" begin
        @test adf11_file_name("Ar", :plt) == "plt89_ar.dat"
        @test adf11_file_name("D", :scd) == "scd12_h.high_dens.dat"
        @test adf11_file_name("D", :ccd) == "ccd96_d.dat"
        @test adf11_file_name("W", :plt) == "plt42_w.dat"
        @test_throws ArgumentError adf11_file_name("Xx", :acd)
        @test_throws "fetch_adas.py" read_adas("Ne"; directory)
        withenv("RUNES2P_ADAS_CACHE" => directory) do
            @test adas_cache_directory() == directory
        end
    end

    @testset "ionization energies (NIST)" begin
        @test ionization_energy("H", 0) == 13.5984345997
        @test ionization_energy("D", 0) == ionization_energy("H", 0)
        @test ionization_energy("Ar", 0) == 15.7596119
        @test length(IONIZATION_ENERGIES["W"]) == 74
        @test issorted(IONIZATION_ENERGIES["Ne"])
        @test_throws ArgumentError ionization_energy("Ar", 18)
        @test_throws ArgumentError ionization_energy("Xx", 0)
    end
end
