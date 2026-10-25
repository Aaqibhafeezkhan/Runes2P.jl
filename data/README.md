# Data files

The tables of the package are TOML files in this directory, one file per table: numbers, and where a table names files, strings.
They hold atomic data, fits of cross sections and tabulated coefficients that the source code uses as constants.
The constants keep their names, types and docstrings in `src/`, and each docstring cites the source of its values.

## Header

Every file starts with a table `[metadata]` with four string keys:

- `source`: the full citation of the source, or DREAM's file for a ported table;
- `version_or_date`: the version of the source, or the date when it was retrieved;
- `licence`: the licence or terms of use of the source;
- `generator`: the script that writes the file.

More keys are allowed.
A file written by a generator is not edited by hand.
Its generator writes it again, byte for byte.

TOML assigns every key after a table header to that table.
The numbers therefore follow the header in tables of their own, `[data]` when the file holds one table:

```toml
[metadata]
source = "DREAM, src/Equations/MeanExcitationEnergy_Extended.cpp"
version_or_date = "DREAM v26.5"
licence = "MIT, see THIRD_PARTY_NOTICES.md"
generator = "scripts/port_atomic_data.py"

[data]
values = [1.0, 2.5, nan]
matrix = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]
```

## Numbers

- Floats are written in the shortest form that parses back to the same `Float64`: Julia's `repr` or Python's `repr`, which spell some values differently (`1.0e-15` and `1e-15`).
- `nan`, `inf` and `-inf` are the TOML literals for `NaN`, `Inf` and `-Inf`; Julia's `repr` spelling `NaN` is no TOML.
- An integer in a float array comes back as the `Float64` of the same value.
- A matrix is an array of rows, so the file reads like the Julia literal: `[[1.0, 2.0, 3.0]]` is the 1×3 matrix `[1.0 2.0 3.0]`.

## Reading a file

`src/physics/data_files.jl` reads the files.
A source file calls it in the body of the module, so the file is read once at precompile time:

```julia
const EXAMPLE_DATA = read_data_file("example_table")    # reads data/example_table.toml
const EXAMPLE_VALUES = float_vector(EXAMPLE_DATA["data"]["values"])
const EXAMPLE_MATRIX = float_matrix(EXAMPLE_DATA["data"]["matrix"])
```

- `read_data_file(name)` returns the parsed `Dict{String, Any}` of `data/<name>.toml`, the table `metadata` included.
  It registers the file with `include_dependency` by its content, so a change of the content recompiles the package.
- `parse_data_file(path)` parses and checks a file at any path; `read_data_file` calls it.
  It throws an error if the file has no table `[metadata]`, if a key of the header is missing or is not a non-empty string, or if the header holds a value that is not a string.
- `float_vector(values)` gives a `Vector{Float64}`.
- `float_matrix(rows)` gives a `Matrix{Float64}` with one row per element of `rows`.
- `int_vector(values)` gives a `Vector{Int}`.

`float_vector` and `float_matrix` accept only TOML integers and floats, and `int_vector` only integers.
Any other element, or an integer with no exact `Float64`, throws an error.
A tuple constant is built as `Tuple(float_vector(...))`, and a dictionary from the keys of a TOML table.
