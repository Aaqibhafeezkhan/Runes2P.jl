import TOML

"""
    DATA_DIRECTORY

The directory `data/` at the root of the package, which holds the tables of numbers as TOML
files (`data/README.md`).
"""
const DATA_DIRECTORY = normpath(joinpath(@__DIR__, "..", "..", "data"))

"""
    DATA_FILE_METADATA_KEYS

The keys that the table `[metadata]` of every data file holds: `source` (the full citation
or DREAM's file), `version_or_date` (the version of the source or the date of retrieval),
`licence` (its licence or terms of use) and `generator` (the script that writes the file).
"""
const DATA_FILE_METADATA_KEYS = ("source", "version_or_date", "licence", "generator")

"""
    read_data_file(name) -> Dict{String, Any}

The parsed content of the data file `data/<name>.toml`, its table `metadata` included, as
[`parse_data_file`](@ref) gives it.

It is called in the body of the module, so the file is read once when the package is
precompiled and the constants built from it cost nothing at run time. The file is registered
with `include_dependency` by its content, so a change of the content recompiles the package
and a copy of the depot elsewhere does not.

The arrays of the result are converted with [`float_vector`](@ref), [`float_matrix`](@ref)
and [`int_vector`](@ref).
"""
function read_data_file(name::AbstractString)
    path = joinpath(DATA_DIRECTORY, name * ".toml")
    include_dependency(path; track_content = true)
    return parse_data_file(path)
end

"""
    parse_data_file(path) -> Dict{String, Any}

The parsed content of the TOML data file at `path`, after a check of its header. It throws
an error if the file has no table `metadata`, if one of [`DATA_FILE_METADATA_KEYS`](@ref) is
missing there or is not a non-empty string, or if the table holds a value that is not a
string: TOML puts every key after the header `[metadata]` into it, so the numbers go into a
table of their own.
"""
function parse_data_file(path::AbstractString)
    content = TOML.parsefile(path)
    metadata = get(content, "metadata", nothing)
    metadata isa AbstractDict ||
        error("data file $path: no table [metadata]")
    for key in DATA_FILE_METADATA_KEYS
        value = get(metadata, key, nothing)
        if !(value isa AbstractString && !isempty(value))
            error("data file $path: [metadata] needs a non-empty string `$key`")
        end
    end
    for (key, value) in metadata
        if !(value isa AbstractString)
            error("data file $path: [metadata] holds `$key`, which is not a string; " *
                  "the numbers go into a table of their own after the header")
        end
    end
    return content
end

"""
    float_vector(values) -> Vector{Float64}

The TOML array `values` as a `Vector{Float64}`. Floats are kept bit for bit (`nan` gives
`NaN`); integers become the `Float64` of the same value and throw an error if that value has
no exact `Float64`. Any other element (a boolean, a string, an array) throws an error.
"""
function float_vector(values::AbstractVector)
    result = Vector{Float64}(undef, length(values))
    for (i, value) in enumerate(values)
        result[i] = data_float(value)
    end
    return result
end

"""
    float_matrix(rows) -> Matrix{Float64}

The TOML array of rows `rows` as a `Matrix{Float64}` with one row per element, so that a
file reads like the Julia literal: `[[1.0, 2.0, 3.0]]` is the 1×3 matrix `[1.0 2.0 3.0]`.
The elements are converted as by [`float_vector`](@ref). It throws an error if the rows
differ in length or if `rows` is empty.
"""
function float_matrix(rows::AbstractVector)
    isempty(rows) && error("data file matrix without rows")
    all(row -> row isa AbstractVector, rows) ||
        error("data file matrix: every row must be an array")
    ncols = length(first(rows))
    all(row -> length(row) == ncols, rows) ||
        error("data file matrix: rows of different lengths")
    result = Matrix{Float64}(undef, length(rows), ncols)
    for (i, row) in enumerate(rows)
        for (j, value) in enumerate(row)
            result[i, j] = data_float(value)
        end
    end
    return result
end

"""
    int_vector(values) -> Vector{Int}

The TOML array of integers `values` as a `Vector{Int}`, also when it is empty. Any other
element throws an error.
"""
function int_vector(values::AbstractVector)
    all(value -> value isa Int64, values) ||
        error("data file array: every element must be an integer")
    return Vector{Int}(values)
end

# A float of a TOML file as it is; an integer as the Float64 of the same value
data_float(value::Float64) = value
function data_float(value::Int64)
    result = Float64(value)
    result == value ||
        error("data file integer $value has no exact Float64")
    return result
end
data_float(value) = error("data file array: $(repr(value)) is not a number")
