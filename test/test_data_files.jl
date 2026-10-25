# The reader of the data files of data/: the check of the header, the conversion of the
# arrays, and the header of every file of the directory.

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
end
