# OpenADAS rate coefficients of the ADF11 class: the reader of the files,
# the interpolation of DREAM (GSL's bicubic interpolation in the logarithms inside the
# table, bilinear outside), the files of each element, and the cache they are read from.
# The OpenADAS terms do not allow redistributing the data with a modelling code: no file
# enters the repository, and `scripts/fetch_adas.py` fills the cache on a machine with
# internet access.

using LinearAlgebra: Tridiagonal

"""
    ADF11Table

Rate coefficients of one ADF11 file of OpenADAS for the element with nuclear charge
`nuclear_charge`, read by [`read_adf11`](@ref):

- `class`: `:acd` (recombination), `:scd` (ionization), `:ccd` (charge exchange), all in
  m³/s, `:plt` (line radiation) and `:prb` (recombination and bremsstrahlung radiation),
  in W m³;
- `log_density` [log₁₀ m⁻³] and `log_temperature` [log₁₀ eV]: the nodes;
- `log_coefficient[i, j, b]`: log₁₀ of the coefficient at the density node `i`, the
  temperature node `j`, in block `b` of the file (one block per charge state);
- `derivatives`: the derivatives at the nodes that DREAM's bicubic interpolation uses, with respect
  to the log density, the log temperature, and both;
- `source`: the file.

[`adas_coefficient`](@ref) evaluates a coefficient.
"""
struct ADF11Table
    class::Symbol
    nuclear_charge::Int
    log_density::Vector{Float64}
    log_temperature::Vector{Float64}
    log_coefficient::Array{Float64, 3}
    derivatives::NTuple{3, Array{Float64, 3}}
    source::String
end

const ADF11_CLASSES = (:acd, :scd, :ccd, :plt, :prb)

"""
    read_adf11(path; class=nothing)

Read an ADF11 file of OpenADAS (unresolved, one block per charge state) into an
[`ADF11Table`](@ref), in SI units: the densities from cm⁻³ to m⁻³ and the coefficients from
cm³ to m³ (log₁₀ + 6 and − 6, as DREAM's `tools/ADAS/data.py`). `class` is taken from the
file name (`acd96_ne.dat` is `:acd`) unless given.

Throws an `ArgumentError` for a file whose blocks do not match its header, and for a
metastable-resolved file (`IGRD` or `IPRT` above 1), which DREAM does not read either.
"""
function read_adf11(path::AbstractString; class::Union{Nothing, Symbol} = nothing)
    class = class === nothing ? Symbol(lowercase(basename(path)[1:3])) : class
    class in ADF11_CLASSES || throw(ArgumentError("unknown ADF11 class $class of $path"))
    lines = readlines(path)
    header = split(lines[1])
    Z, nn, nT = parse.(Int, header[1:3])
    numbers(line) = parse.(Float64, split(line))
    k = 3
    log_density = Float64[]
    while length(log_density) < nn
        append!(log_density, numbers(lines[k]) .+ 6)
        k += 1
    end
    log_temperature = Float64[]
    while !startswith(strip(lines[k], [' ', 'C']), "--")
        append!(log_temperature, numbers(lines[k]))
        k += 1
    end
    if !(length(log_density) == nn && length(log_temperature) == nT)
        throw(ArgumentError("the nodes of $path do not match its header ($nn densities, $nT temperatures)"))
    end
    blocks = Vector{Float64}[]
    while k <= length(lines) && !startswith(lines[k], "C")
        # a blank line may follow the last block (argon of year 89)
        if isempty(strip(lines[k]))
            k += 1
            continue
        end
        separator = lines[k]
        for field in ("IGRD", "IPRT")
            m = match(Regex(field * "=\\s*(\\d+)"), separator)
            if !(m === nothing || parse(Int, m.captures[1]) <= 1)
                throw(ArgumentError("$path is metastable resolved ($field > 1), which the reader does not take"))
            end
        end
        k += 1
        # a dashed line that closes the last block (hydrogen, charge exchange)
        while k <= length(lines) && isempty(strip(lines[k]))
            k += 1
        end
        if k > length(lines) || startswith(lines[k], "C")
            break
        end
        values = Float64[]
        while k <= length(lines) && !startswith(strip(lines[k], [' ', 'C']), "--") &&
              !startswith(lines[k], "C") &&
              !isempty(strip(lines[k]))
            append!(values, numbers(lines[k]) .- 6)
            k += 1
        end
        if length(values) != nn * nT
            throw(ArgumentError("block $(length(blocks) + 1) of $path has $(length(values)) values, not $(nn * nT)"))
        end
        push!(blocks, values)
    end
    if length(blocks) != Z
        throw(ArgumentError("$path has $(length(blocks)) blocks for nuclear charge $Z"))
    end
    # the density varies fastest within a block
    log_coefficient = reshape(reduce(vcat, blocks), nn, nT, Z)
    return ADF11Table(class, Z, log_density, log_temperature, log_coefficient,
        _bicubic_derivatives(log_density, log_temperature, log_coefficient), String(path))
end

"""
    _natural_spline_slopes(x, y) -> Vector{Float64}

First derivative at each node `x[i]` of the natural cubic spline through the points
`(x[i], y[i])`, with `x` strictly increasing and `length(x) ≥ 2`.
"""
function _natural_spline_slopes(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y differ in length"))
    n >= 2 || throw(ArgumentError("a spline needs at least two nodes"))
    # The unknowns are the node slopes s; each piece is the cubic Hermite polynomial
    # with the end values and slopes of its interval. Continuity of the second
    # derivative at the interior nodes and a zero second derivative at both ends give
    # a tridiagonal system for s (the natural cubic spline in its slope form,
    # de Boor, A Practical Guide to Splines, rev. ed. 2001, ch. IV).
    h = [Float64(x[k + 1] - x[k]) for k in 1:(n - 1)]           # interval widths
    d = [Float64(y[k + 1] - y[k]) / h[k] for k in 1:(n - 1)]    # chord slopes
    lower = zeros(n - 1)
    main = zeros(n)
    upper = zeros(n - 1)
    rhs = zeros(n)
    # Left end: zero second derivative of the first piece at x[1].
    main[1] = 2.0
    upper[1] = 1.0
    rhs[1] = 3.0 * d[1]
    # Interior node i joins the pieces of widths h[i-1] and h[i].
    for i in 2:(n - 1)
        lower[i - 1] = h[i]
        main[i] = 2.0 * (h[i - 1] + h[i])
        upper[i] = h[i - 1]
        rhs[i] = 3.0 * (h[i] * d[i - 1] + h[i - 1] * d[i])
    end
    # Right end: zero second derivative of the last piece at x[n].
    lower[n - 1] = 1.0
    main[n] = 2.0
    rhs[n] = 3.0 * d[n - 1]
    return Tridiagonal(lower, main, upper) \ rhs
end

"""
    _bicubic_derivatives(x, y, z) -> (zx, zy, zxy)

Node derivatives of tabulated values `z[i, j, b]` at `(x[i], y[j])` for independent
blocks `b`: `zx` and `zy` are natural-spline slopes along `x` and `y`, and `zxy` is the
natural-spline slope along `x` of `zy`.
"""
function _bicubic_derivatives(x, y, z::Array{Float64, 3})
    nx, ny, nb = size(z)
    (length(x) == nx && length(y) == ny) ||
        throw(DimensionMismatch("grid lengths do not match the size of z"))
    zx = similar(z)
    zy = similar(z)
    zxy = similar(z)
    for b in 1:nb
        # Slopes along x for every column, along y for every row.
        for j in 1:ny
            zx[:, j, b] = _natural_spline_slopes(x, view(z, :, j, b))
        end
        for i in 1:nx
            zy[i, :, b] = _natural_spline_slopes(y, view(z, i, :, b))
        end
        # The cross derivative is the x-slope of the y-slopes.
        for j in 1:ny
            zxy[:, j, b] = _natural_spline_slopes(x, view(zy, :, j, b))
        end
    end
    return zx, zy, zxy
end

# the cell of v: the last node at or below v, at most the last but one
_cell(nodes, v) = clamp(searchsortedlast(nodes, v), 1, length(nodes) - 1)

# cubic Hermite basis on [0, 1]: the weights of the two values h₀, h₁ and of the two slopes g₀, g₁
_hermite_basis(s) = ((1 + 2s) * (1 - s)^2, s^2 * (3 - 2s), s * (1 - s)^2, s^2 * (s - 1))

# the bicubic Hermite interpolation in cell (i, j) of block b, at the cell coordinates t and u,
# from the values and the derivatives at the four corners (Press et al., Numerical Recipes,
# 3rd ed., 2007, section 3.6)
function _bicubic(table::ADF11Table, b, i, j, t, u)
    z = table.log_coefficient
    zx, zy, zxy = table.derivatives
    #! format: off
    dx = table.log_density[i+1] - table.log_density[i]
    dy = table.log_temperature[j+1] - table.log_temperature[j]
    z00, z01, z10, z11 = z[i, j, b], z[i, j+1, b], z[i+1, j, b], z[i+1, j+1, b]
    x00, x01, x10, x11 = zx[i, j, b] * dx, zx[i, j+1, b] * dx, zx[i+1, j, b] * dx, zx[i+1, j+1, b] * dx
    y00, y01, y10, y11 = zy[i, j, b] * dy, zy[i, j+1, b] * dy, zy[i+1, j, b] * dy, zy[i+1, j+1, b] * dy
    s = dx * dy
    c00, c01, c10, c11 = zxy[i, j, b] * s, zxy[i, j+1, b] * s, zxy[i+1, j, b] * s, zxy[i+1, j+1, b] * s
    #! format: on
    ht0, ht1, gt0, gt1 = _hermite_basis(t)
    hu0, hu1, gu0, gu1 = _hermite_basis(u)
    return ht0 * (hu0 * z00 + hu1 * z01 + gu0 * y00 + gu1 * y01) +
           ht1 * (hu0 * z10 + hu1 * z11 + gu0 * y10 + gu1 * y11) +
           gt0 * (hu0 * x00 + hu1 * x01 + gu0 * c00 + gu1 * c01) +
           gt1 * (hu0 * x10 + hu1 * x11 + gu0 * c10 + gu1 * c11)
end

function _bilinear(table::ADF11Table, b, i, j, t, u)
    z = table.log_coefficient
    return (1 - t) * (1 - u) * z[i, j, b] + t * (1 - u) * z[i + 1, j, b] +
           (1 - t) * u * z[i, j + 1, b] +
           t * u * z[i + 1, j + 1, b]
end

# the block of charge state Z0, or 0 where the class has none: recombination (acd) and its
# radiation (prb) belong to the recombining ion Z0 ≥ 1, the others to Z0 ≤ Z − 1 (DREAM's
# shiftZ0, ADAS.cpp)
function _adf11_block(table::ADF11Table, Z0)
    if table.class in (:acd, :prb)
        block = Z0 == 0 ? 0 : Z0
    else
        block = Z0 == table.nuclear_charge ? 0 : Z0 + 1
    end
    return block
end

"""
    adas_coefficient(table, Z0, free_electron_density, temperature; extrapolation=:dream)

The coefficient of `table` ([`ADF11Table`](@ref)) for the ion of charge `Z0` at the
free-electron density `free_electron_density` (`n`) [m⁻³] and the temperature `temperature`
(`T`) [eV], in m³/s (W m³ for `:plt` and `:prb`). Zero where the class has no such charge
state (recombination of the neutral, ionization of the bare nucleus) and at `n ≤ 0` or
`T ≤ 0`, as in DREAM.

The coefficients are those of ADAS (H. P. Summers, The ADAS User Manual, version 2.6 (2004), http://www.adas.ac.uk, with the data of OpenADAS, https://open.adas.ac.uk).
The interpolation is DREAM's (`src/ADASRateInterpolator.cpp`): GSL's bicubic interpolation of
log₁₀ of the coefficient in log₁₀ n and log₁₀ T inside the table, and outside it the
bilinear interpolation of the edge cell, continued linearly (`extrapolation = :dream`).
`extrapolation = :clamp_density` takes the density at the nearest edge of the table and
extrapolates in the temperature alone: at low densities the coefficients tend to their
coronal limit, which does not depend on the density. The default is `:dream` (a choice of
this package; see the docs page *Choices of this package*).

Derivatives with respect to `n` and `T` are exact through ForwardDiff, in place of DREAM's
one-sided differences.
"""
function adas_coefficient(table::ADF11Table, Z0::Integer, free_electron_density,
        temperature; extrapolation::Symbol = :dream)
    extrapolation in (:dream, :clamp_density) ||
        throw(ArgumentError("extrapolation must be :dream or :clamp_density, not $extrapolation"))
    b = _adf11_block(table, Z0)
    (b == 0 || free_electron_density <= 0 || temperature <= 0) &&
        return zero(float(free_electron_density * temperature))
    return exp10(_adf11_log_value(
        table, b, log10(free_electron_density), log10(temperature), extrapolation))
end

# log₁₀ of the coefficient of block b at x = log₁₀ n, y = log₁₀ T: bicubic inside the table,
# the bilinear interpolation of the edge cell outside it, as DREAM. The value jumps at the
# edge of the table between the nodes, where the two interpolations differ.
function _adf11_log_value(table::ADF11Table, b, x, y, extrapolation::Symbol)
    xs, ys = table.log_density, table.log_temperature
    if extrapolation === :clamp_density
        x = clamp(x, xs[1], xs[end])
    end
    i, j = _cell(xs, _value(x)), _cell(ys, _value(y))
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    u = (y - ys[j]) / (ys[j + 1] - ys[j])
    inside = xs[1] <= x <= xs[end] && ys[1] <= y <= ys[end]
    return inside ? _bicubic(table, b, i, j, t, u) : _bilinear(table, b, i, j, t, u)
end

"""
    ADAS_FILES

The ADF11 file of each element and class (DREAM's `tools/elements.json`): a year, which
names the file `<class><yy>_<element>.dat` of OpenADAS, or a file name. The data are those
of ADAS: H. P. Summers, The ADAS User Manual, version 2.6 (2004), http://www.adas.ac.uk,
with the data of OpenADAS, https://open.adas.ac.uk. Deuterium and tritium take the files of
hydrogen, except charge exchange. The high-density files of hydrogen are not served by
OpenADAS; DREAM's cache carries them. The table is read from `data/adas_files.toml`.
"""
const ADAS_FILES = let data = read_data_file("adas_files")["data"]
    function entry(files)
        file_names = map(class -> files[String(class)]::String, ADF11_CLASSES)
        return NamedTuple{ADF11_CLASSES}(file_names)
    end
    Dict(element => entry(files) for (element, files) in data)
end

"""
    adf11_file_name(element, class)

The name of the ADF11 file of `element` and `class` in the cache ([`ADAS_FILES`](@ref)).
"""
function adf11_file_name(element::AbstractString, class::Symbol)
    element = atomic_element(element)
    haskey(ADAS_FILES, element) ||
        throw(ArgumentError("no ADF11 files listed for the element $element"))
    entry = getproperty(ADAS_FILES[element], class)
    endswith(entry, ".dat") && return entry
    file_element = class !== :ccd && element in ("D", "T") ? "h" : lowercase(element)
    return "$(class)$(entry)_$(file_element).dat"
end

"""
    adas_cache_directory()

The directory of the ADF11 files: `ENV["RUNES2P_ADAS_CACHE"]`, or `~/.cache/Runes2P/adas`. It is
filled by `scripts/fetch_adas.py` on a machine with internet access (the compute nodes of
clusters often have none) and is never part of the repository.
"""
function adas_cache_directory()
    get(ENV, "RUNES2P_ADAS_CACHE", joinpath(homedir(), ".cache", "Runes2P", "adas"))
end

"""
    read_adas(element; directory=adas_cache_directory())

The five [`ADF11Table`](@ref)s of `element` as a named tuple `(acd, scd, ccd, plt, prb)`,
read from the cache. Throws an `ArgumentError` naming the missing file and the script that
fetches it.
"""
function read_adas(element::AbstractString; directory::AbstractString = adas_cache_directory())
    tables = map(class -> _read_adas_class(element, class, directory), ADF11_CLASSES)
    return NamedTuple{ADF11_CLASSES}(tables)
end

# the ADF11 table of one class of `element` from the cache `directory`
function _read_adas_class(element, class, directory)
    path = joinpath(directory, adf11_file_name(element, class))
    if !isfile(path)
        throw(ArgumentError("the ADF11 file $path is missing: run scripts/fetch_adas.py on a " *
                            "machine with internet access (see its header)"))
    end
    return read_adf11(path; class)
end
