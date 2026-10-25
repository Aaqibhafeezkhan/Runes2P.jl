# Writes data/chianti_ionization.toml, the table of the CHIANTI parameters of the ionization of
# the ions by electron impact, from the files of the CHIANTI database (K. P. Dere, Astron.
# Astrophys. 466, 771 (2007)):
#
#   julia scripts/generate_chianti_ionization.jl <chianti directory> [output file]
#
# The directory holds, flat, the files <ion>.diparams (direct ionization) and <ion>.easplom
# (excitation-autoionization cross sections) of every ion of the elements of ELEMENTS, and the
# VERSION file of the database. Their source is the SolarSoft mirror of the CHIANTI database,
# https://soho.nascom.nasa.gov/solarsoft/packages/chianti/dbase/<el>/<ion>/<ion>.<ext>, and
# https://soho.nascom.nasa.gov/solarsoft/packages/chianti/dbase/VERSION. The files are not in
# the repository. The .easplups files (Maxwellian-averaged collision strengths) are not read.
# The output is the default path of the repository. src/physics/ionization_chianti_fits.jl
# reads it into CHIANTI_IONIZATION_FITS. test/test_chianti_ionization.jl runs this script on
# the folder of the environment variable RUNES2P_CHIANTI_DIR (default: ~/chianti) and compares
# the output with the committed data file; without the folder that comparison is skipped.

const ELEMENTS = (("H", 1), ("Ne", 10), ("Ar", 18))

# the date on which the files of the mirror were read; it changes when they are read again
const READ_ON = "2026-10-13"

# the shells of a .diparams file and the scale of its excitation-autoionization (the format of
# ChiantiPy's io.diRead: Z, ion, nodes, shells, autoionization flag; per shell a line of the
# Burgess-Tully parameter and the scaled energies, a line of the threshold [eV] and the scaled
# cross sections [1e-14 cm² eV²]; with the flag, a line of the scale; then -1 and the comments)
function read_diparams(path)
    lines = readlines(path)
    header = parse.(Int, split(lines[1]))
    nodes, shells, flag = header[3], header[4], header[5]
    direct = map(1:shells) do k
        a = parse.(Float64, split(lines[2k]))
        b = parse.(Float64, split(lines[2k + 1]))
        length(a) == length(b) == nodes + 1 || error("$path: shell $k has not $nodes nodes")
        (scale = a[1], threshold = b[1], knots = a[2:end], values = b[2:end])
    end
    scale = flag > 0 ? only(parse.(Float64, split(lines[2shells + 2]))) : 1.0
    strip(lines[2shells + 2 + (flag > 0)]) == "-1" ||
        error("$path: no -1 after the parameters")
    return (Z = header[1], stage = header[2], direct, flag, scale)
end

# the transitions of a .easplom file: Z, ion, two levels, type, gf, ΔE [Ry], C, five nodes
function read_easplom(path)
    rows = []
    for line in readlines(path)
        fields = split(line)
        length(fields) == 1 && break
        v = parse.(Float64, fields)
        push!(rows,
            (kind = Int(v[5]), oscillator_strength = v[6],
                energy = v[7], scale = v[8], values = v[9:end]))
    end
    return rows
end

# a Float64 in TOML: the shortest form that parses back to the same value, and the TOML
# literals of NaN and the infinities
function toml(x::Float64)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    return repr(x)
end
toml(x::Int) = repr(x)
toml(x::Vector{Float64}) = "[" * join(toml.(x), ", ") * "]"
toml(x::AbstractString) = repr(String(x))

function main(directory, output)
    version = strip(read(joinpath(directory, "VERSION"), String))
    io = IOBuffer()
    println(io, "[metadata]")
    println(io, "source = ",
        toml("the files <ion>.diparams and <ion>.easplom of the CHIANTI atomic database, " *
             "SolarSoft mirror, https://soho.nascom.nasa.gov/solarsoft/packages/chianti/dbase/; " *
             "the ionization data of K. P. Dere, Astron. Astrophys. 466, 771 (2007), " *
             "doi:10.1051/0004-6361:20066728"))
    println(io, "version_or_date = ",
        toml("CHIANTI $version (the VERSION file of the SolarSoft mirror), read on $READ_ON"))
    println(io, "licence = ",
        toml("CHIANTI database: work that uses these data should cite Dere et al., Astron. " *
             "Astrophys. Suppl. Ser. 125, 149 (1997), Dufresne et al., Astrophys. J. 974, 71 " *
             "(2024), and Dere, Astron. Astrophys. 466, 771 (2007); see THIRD_PARTY_NOTICES.md"))
    println(io, "generator = ", toml("scripts/generate_chianti_ionization.jl"))
    println(io, "description = ",
        toml("per element, one ion per charge state Z0 = 0, ..., Z - 1 in the order of Z0: " *
             "the shells of the direct ionization (scale f, threshold [eV], scaled energies U, " *
             "scaled cross sections [1e-14 cm² eV²]) and the transitions of the " *
             "excitation-autoionization (Burgess-Tully type, weighted oscillator strength, " *
             "excitation energy [Ry], scale C, five scaled collision strengths)"))
    for (element, Z) in ELEMENTS
        for stage in 1:Z
            ion = lowercase(element) * "_$stage"
            di = read_diparams(joinpath(directory, ion * ".diparams"))
            (di.Z, di.stage) == (Z, stage) ||
                error("$ion: the file is for Z = $(di.Z), stage $(di.stage)")
            ea = di.flag > 0 ? read_easplom(joinpath(directory, ion * ".easplom")) : []
            println(io)
            println(io, "[[data.$element]]")
            println(io, "ion = ", toml(ion))
            println(io, "autoionization_scale = ", toml(di.scale))
            isempty(ea) && println(io, "autoionization = []")
            for s in di.direct
                println(io)
                println(io, "[[data.$element.direct]]")
                println(io, "scale = ", toml(s.scale))
                println(io, "threshold = ", toml(s.threshold))
                println(io, "knots = ", toml(s.knots))
                println(io, "values = ", toml(s.values))
            end
            for t in ea
                println(io)
                println(io, "[[data.$element.autoionization]]")
                println(io, "kind = ", toml(t.kind))
                println(io, "oscillator_strength = ", toml(t.oscillator_strength))
                println(io, "energy = ", toml(t.energy))
                println(io, "scale = ", toml(t.scale))
                println(io, "values = ", toml(t.values))
            end
        end
    end
    write(output, take!(io))
    println("wrote $output")
end

length(ARGS) in (1, 2) ||
    error("usage: julia scripts/generate_chianti_ionization.jl <chianti directory> [output file]")
main(ARGS[1],
    length(ARGS) == 2 ? ARGS[2] :
    joinpath(@__DIR__, "..", "data", "chianti_ionization.toml"))
