# The parameters of the ions of the data file `data/chianti_ionization.toml`, one array of
# tables per element in the order of the charge state, built by the constructors that derive
# the spline curvatures and the Bethe coefficients
function _chianti_ionization_fits(content)
    fits = Dict{String, Vector{ChiantiIonization}}()
    for (element, ions) in content["data"]
        element_fits = ChiantiIonization[]
        for (stage, ion) in enumerate(ions)
            name = ion["ion"]::String
            if !(name == lowercase(element) * "_$stage")
                error("data/chianti_ionization.toml: ion $name at the place of stage $stage of $element")
            end
            direct = ChiantiDirectShell[ChiantiDirectShell(
                                            s["scale"]::Float64, s["threshold"]::Float64,
                                            float_vector(s["knots"]), float_vector(s["values"]))
                                        for s in ion["direct"]]
            autoionization = ChiantiAutoionization[ChiantiAutoionization(
                                                       t["kind"]::Int64,
                                                       t["oscillator_strength"]::Float64,
                                                       t["energy"]::Float64, t["scale"]::Float64,
                                                       float_vector(t["values"]))
                                                   for t in ion["autoionization"]]
            push!(element_fits,
                ChiantiIonization(
                    name, direct, ion["autoionization_scale"]::Float64, autoionization))
        end
        fits[element] = element_fits
    end
    return fits
end

"""
    CHIANTI_IONIZATION_FITS

The parameters of the cross sections of the ionization of each element, one
[`ChiantiIonization`](@ref) per charge state `Z0 = 0, …, Z − 1`: the direct ionization and the
excitation-autoionization of the CHIANTI database, version 11.0.2 (the `VERSION` file of its
SolarSoft mirror, https://soho.nascom.nasa.gov/solarsoft/packages/chianti/dbase/VERSION, read on
2026-10-13), from its files `<ion>.diparams` and `<ion>.easplom`, unchanged since K. P. Dere,
Astron. Astrophys. 466, 771 (2007), doi:10.1051/0004-6361:20066728. Dere fits measured cross
sections where they exist, mostly for the neutral and singly charged ions, and his distorted-wave
calculations with the Flexible Atomic Code otherwise; the final errors of his Tables 1 to 28 have
the median 10 % and the mean 13 % (his Sect. 3.31). The sources of the ions of the table, from his Sect. 3:

- H I: the measurements of Shah et al. (1987), error 7 %. Deuterium and tritium take the
  parameters of hydrogen.
- Ne I, Ar I and Ar II among others: fits to measurements (Ar I to Straub et al. 1995 and five
  other sets).
- Ne IX, Ne X, Ar XVII, Ar XVIII (the one- and two-electron ions): Dere's text takes the
  relativistic distorted-wave fits of Fontes et al. (1999) for these, which ChiantiPy evaluates
  instead of the files; the table holds the parameters of the files, as fiasco evaluates them.
  Against the relativistic distorted-wave calculations of R. Bao and B. Li (At. Data Nucl.
  Data Tables 159, 101663 (2024)), the files of Ar¹⁶⁺ and Ar¹⁷⁺ give 1.24 times their cross
  section at 5 times the threshold and 1.35 and 1.38 times it at 10 times.
- the other ions: the calculations, with the excitation-autoionization of Ne VII, Ne VIII, Ar III
  to Ar VIII, Ar XV and Ar XVI.

The parameters are in the data file `data/chianti_ionization.toml`, which
`scripts/generate_chianti_ionization.jl` writes from the files of CHIANTI.
"""
const CHIANTI_IONIZATION_FITS = _chianti_ionization_fits(read_data_file("chianti_ionization"))
