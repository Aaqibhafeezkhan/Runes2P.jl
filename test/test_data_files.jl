# The reader of the data files of data/: the check of the header, the conversion of the
# arrays, and the header of every file of the directory. A test of its own checks the
# table of Braams and Karney against the values printed in their paper.

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

    @testset "the conductivity of Braams and Karney against its source" begin
        # B. J. Braams and C. F. F. Karney, Phys. Fluids B 1, 1355 (1989),
        # doi:10.1063/1.858966, Table I on p. 1362: the normalized conductivity σ̄(Θ, Z),
        # as printed with 5 decimals. Rows: Θ = T/mₑc². Columns: Z = 0, 1, 2, 5, 10, ∞.
        # The data file keeps DREAM's port, the printed values rounded to 4 decimals, so
        # every value is pinned at half a unit of its 4th decimal: at most 5 units of the
        # printed 5th decimal. A count in integer units keeps the 12 printed values that end
        # in 5 at exactly 5 units, such as 1.02875 at Z = 1 and Θ = 10, kept as 1.0288. All
        # 84 cells were read with certainty.
        printed = [3.75994 7.42898 8.75460 10.39122 11.33006 12.76615
                   3.75490 7.27359 8.53281 10.07781 10.95869 12.29716
                   3.74920 7.12772 8.32655 9.78962 10.61952 11.87371
                   3.72852 6.73805 7.78445 9.04621 9.75405 10.81201
                   3.68420 6.20946 7.06892 8.09361 8.66306 9.50746
                   3.57129 5.43667 6.06243 6.80431 7.21564 7.82693
                   3.18206 4.13733 4.47244 4.88050 5.11377 5.47602
                   2.65006 3.13472 3.32611 3.57303 3.72206 3.96944
                   2.03127 2.27862 2.39205 2.54842 2.64827 2.82473
                   1.33009 1.45375 1.51805 1.61157 1.67382 1.78870
                   0.94648 1.02875 1.07308 1.13856 1.18263 1.26490
                   0.67042 0.72743 0.75853 0.80472 0.83593 0.89443
                   0.42422 0.46003 0.47965 0.50885 0.52861 0.56569
                   0.29999 0.32528 0.33915 0.35979 0.37377 0.40000]
        printed_temperatures = [
            0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        printed_charges = [0, 1, 2, 5, 10, Inf]
        charges = 1 ./ (1 .+ reverse(printed_charges))
        data = Runes2P.read_data_file("braams_karney_conductivity")["data"]
        table = Runes2P.float_matrix(data["table"])
        # the rows of the file run in X = 1/(1 + Z) from Z = ∞ to Z = 0
        @test Runes2P.float_vector(data["temperatures"]) == printed_temperatures
        @test Runes2P.float_vector(data["charges"]) ≈ charges atol = 1e-14
        @test size(table) == (6, 14)
        for i in 1:14
            for j in 1:6
                kept = round(Int, 1e5 * table[7 - j, i])
                @test abs(kept - round(Int, 1e5 * printed[i, j])) <= 5
            end
        end
    end
end
