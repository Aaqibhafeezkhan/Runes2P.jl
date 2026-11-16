# The reader of the data files of data/: the check of the header, the conversion of the
# arrays, and the header of every file of the directory. The table of mean excitation
# energies against the printed tables of its source.

@testset "data files" begin
    header = """
    [metadata]
    source = "a source"
    version_or_date = "1.0"
    licence = "MIT"
    generator = "none"
    """
    parse_text(text) = mktempdir() do tmp
        path = joinpath(tmp, "table.toml")
        write(path, text)
        Runes2P.parse_data_file(path)
    end

    @testset "the header" begin
        content = parse_text(header * "\n[data]\nvalues = [1.0, 2.5, nan]\n")
        @test content["metadata"]["source"] == "a source"
        @test isequal(Runes2P.float_vector(content["data"]["values"]), [1.0, 2.5, NaN])
        @test_throws r"no table \[metadata\]" parse_text("[data]\nvalues = [1.0]\n")
        @test_throws r"non-empty string `licence`" parse_text(replace(header,
            "licence = \"MIT\"\n" => ""))
        @test_throws r"non-empty string `source`" parse_text(replace(header,
            "\"a source\"" => "\"\""))
        @test_throws r"non-empty string `version_or_date`" parse_text(replace(header,
            "\"1.0\"" => "1.0"))
        # the numbers after the header, without a table of their own, land in [metadata]
        @test_throws r"holds `values`, which is not a string" parse_text(header *
                                                                         "values = [1.0]\n")
        names = [splitext(f)[1]
                 for f in readdir(Runes2P.DATA_DIRECTORY)
                 if endswith(f, ".toml")]
        @test !isempty(names)
        for name in names
            @test Runes2P.read_data_file(name)["metadata"]["generator"] isa String
        end
    end

    @testset "the arrays" begin
        @test Runes2P.float_vector([1, 2.5]) == [1.0, 2.5]
        @test Runes2P.float_vector([1, 2.5]) isa Vector{Float64}
        @test Runes2P.float_vector(Any[]) == Float64[]
        @test_throws r"no exact Float64" Runes2P.float_vector([2^53 + 1])
        @test_throws r"is not a number" Runes2P.float_vector(Any[1.0, "2"])
        @test_throws r"is not a number" Runes2P.float_vector(Any[true])
        @test Runes2P.float_matrix([[1.0, 2.0, 3.0]]) == [1.0 2.0 3.0]
        @test Runes2P.float_matrix([[1, 2], [3, 4]]) == [1.0 2.0; 3.0 4.0]
        @test_throws r"rows of different lengths" Runes2P.float_matrix([[1.0, 2.0], [3.0]])
        @test_throws r"without rows" Runes2P.float_matrix(Any[])
        @test_throws r"every row must be an array" Runes2P.float_matrix(Any[1.0, 2.0])
        @test_throws r"is not a number" Runes2P.float_matrix([Any[1.0, "x"]])
        @test Runes2P.int_vector([1, 2]) == [1, 2]
        @test Runes2P.int_vector(Any[]) == Int[]
        @test_throws r"every element must be an integer" Runes2P.int_vector([1.0])
        # the TOML parser gives the types that the converters accept
        content = parse_text(header * "\n[data]\nm = [[1, 2.0], [3.5, inf]]\nz = [2, 18]\n")
        @test Runes2P.float_matrix(content["data"]["m"]) == [1.0 2.0; 3.5 Inf]
        @test Runes2P.int_vector(content["data"]["z"]) == [2, 18]
    end

    # S. P. A. Sauer, J. R. Sabin and J. Oddershede, J. Chem. Phys. 148, 174307 (2018),
    # doi:10.1063/1.5027708. Table N lists the ions with N bound electrons, Z = N to 18, in
    # I₀ [eV]. The values are typed here as printed, read from the page images.
    # - Tables I and II: p. 174307-2.
    # - Tables III to VI: p. 174307-3.
    # - Tables VII to XII: p. 174307-4.
    # - Tables XIII to XVIII: p. 174307-5.
    # The paper quotes three cells from earlier RPA work and prints them with two decimals:
    # H from its Ref. 8, He and He⁺ from its Ref. 20 (Sauer, Oddershede and Sabin, Adv.
    # Quantum Chem. 71, 29 (2015)). Table III prints the Cl ion with 3 bound electrons as
    # "Cl12+". Its place in the table and its Z_eff = 17 - S make it Cl¹⁴⁺. Each value holds
    # to half a unit of its last printed digit.
    @testset "mean excitation energies against Sauer, Sabin and Oddershede (2018)" begin
        printed = [
            # I: 1 bound electron, H to Ar¹⁷⁺
            [14.99, 59.88, 134.5, 240.2, 374.6, 539.5, 734.3, 959.0, 1213.7, 1498.4,
                1813.9, 2158.8, 2533.5, 2938.3, 3373.1, 3837.8, 4332.5, 4857.2],
            # II: 2, He to Ar¹⁶⁺
            [42.68, 108.3, 205.0, 330.4, 486.2, 672.0, 887.8, 1133.5, 1409.2, 1715.6,
                2051.5, 2417.2, 2813.0, 3238.8, 3694.5, 4180.2, 4695.9],
            # III: 3, Li to Ar¹⁵⁺
            [33.1, 76.9, 136.9, 214.2, 308.7, 420.7, 550.0, 696.8, 861.2, 1043.2, 1242.7,
                1459.8, 1694.6, 1947.0, 2217.2, 2505.0],
            # IV: 4, Be to Ar¹⁴⁺
            [42.2, 82.3, 134.8, 200.2, 278.6, 370.2, 475.0, 593.3, 724.8, 869.6, 1027.9,
                1199.7, 1384.9, 1583.7, 1796.0],
            # V: 5, B to Ar¹³⁺
            [52.6, 92.6, 142.4, 202.2, 272.3, 352.6, 443.5, 544.8, 656.4, 778.6, 911.2,
                1054.5, 1208.2, 1372.6],
            # VI: 6, C to Ar¹²⁺
            [65.9, 107.4, 157.2, 215.6, 282.8, 358.7, 443.8, 537.4, 640.1, 751.7, 872.2,
                1001.8, 1140.3],
            # VII: 7, N to Ar¹¹⁺
            [81.6, 125.2, 176.4, 235.2, 301.9, 376.7, 459.2, 549.7, 648.2, 754.6, 869.1,
                991.6],
            # VIII: 8, O to Ar¹⁰⁺
            [97.9, 144.0, 196.9, 256.8, 324.3, 398.8, 480.6, 569.7, 666.2, 769.9, 881.1],
            # IX: 9, F to Ar⁹⁺
            [116.5, 165.2, 220.4, 282.5, 351.0, 426.5, 508.8, 598.0, 694.0, 797.0],
            # X: 10, Ne to Ar⁸⁺
            [137.2, 189.2, 246.8, 310.8, 381.4, 458.5, 542.1, 632.1, 728.8],
            # XI: 11, Na to Ar⁷⁺
            [125.7, 173.7, 225.8, 283.1, 345.9, 414.4, 488.6, 568.6],
            # XII: 12, Mg to Ar⁶⁺
            [128.0, 172.7, 221.2, 274.3, 332.4, 395.5, 463.9],
            # XIII: 13, Al to Ar⁵⁺
            [132.2, 177.2, 225.2, 277.3, 333.8, 394.9],
            # XIV: 14, Si to Ar⁴⁺
            [140.8, 185.3, 232.8, 284.1, 339.4],
            # XV: 15, P to Ar³⁺
            [151.6, 195.7, 242.9, 293.7],
            # XVI: 16, S to Ar²⁺
            [162.4, 206.8, 254.0],
            # XVII: 17, Cl to Ar⁺
            [174.9, 219.5],
            # XVIII: 18, Ar
            [188.7]
        ]
        two_decimals = ((1, 0), (2, 0), (2, 1))  # (Z, Z0) of H, He and He⁺
        for (bound, row) in enumerate(printed)
            @test length(row) == 19 - bound
            for (k, I0) in enumerate(row)
                Z = bound + k - 1
                Z0 = Z - bound
                tolerance = (Z, Z0) in two_decimals ? 0.005 : 0.05
                @test abs(mean_excitation_energy(Z, Z0) - I0) <= tolerance
            end
        end
        # every charge state of Z = 1 to 18 is pinned above
        @test count(!isnan, Runes2P.MEAN_EXCITATION_ENERGY) == sum(length, printed) == 171

        # Table XIX (p. 174307-6): D_N and S_N0 of eq. 8, N = 1 to 14
        D_N = [0, 0.00, 0.24, 0.34, 0.41, 0.45, 0.48, 0.50, 0.51, 0.52, 0.55, 0.57, 0.58,
            0.59]
        S_N0 = [0, 0.30, 1.51, 2.32, 3.13, 3.90, 4.67, 5.44, 6.21, 6.97, 8.10, 9.08, 10.03,
            10.94]
        @test maximum(abs, Runes2P.MEAN_EXCITATION_ENERGY_D - D_N) <= 0.005
        @test maximum(abs, Runes2P.MEAN_EXCITATION_ENERGY_S0 - S_N0) <= 0.005
    end
end
