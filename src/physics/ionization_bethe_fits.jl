# The parameters of the kinetic ionization cross section in the form with 1/β² in its
# relativistic Bethe term, read from data/kinetic_ionization_bethe_fits.toml, which
# scripts/fit_kinetic_ionization.py writes.

"""
    KINETIC_IONIZATION_BETHE_FITS

The parameters `C`, `I` [eV] and `β*` of [`ionization_cross_section`](@ref) with `form =
:bethe`, one row per charge state `Z0 = 0, …, Z − 1`, as [`KINETIC_IONIZATION_FITS`](@ref)
for DREAM's form. They are DREAM's fit run unchanged (`tools/fit_kineticionization.py` with
`tools/ADAS/fit.py`, method "single", MIT License, Copyright (c) 2020 Mathias Hoppe and Ola
Embreus) on the cross section with the factor `1/β²`: per charge state, a least-squares fit
of the logarithm of the Maxwell-Jüttner average of `v σ` to the ADAS SCD coefficient at the
lowest density over the temperature window of DREAM's `tools/kineticCrossSectionParams.json`
(2 to 100 eV where it has none), with `C` and `β*` free in `[1e-4, 200]` and `[0, 1]` and `I`
the NIST ionization energy of DREAM's table. The constants are those of SciPy's
`scipy.constants` in the pixi environment of `reference/dream` (CODATA 2022 from SciPy 1.15
on, 1e-9 from CODATA 2018), where DREAM's tool takes `a₀ = 5.29e-11 m`, `R_y = 13.6 eV` and
`α = 1/137`; with DREAM's constants and form the same script reproduces
[`KINETIC_IONIZATION_FITS`](@ref) (D: `C = 3.05406`, `β* = 0.296804`). Fitted by
`scripts/fit_kinetic_ionization.py` on 2026-10-12 from the ADAS files of DREAM's cache
(`tools/cache`). The values are in `data/kinetic_ionization_bethe_fits.toml`, which the script
writes: in `[data]`, one array per element with one row `[C, I, β*]` per charge state.

Against the SCD coefficient, the Maxwellian average lies between 0.71 and 1.15 at the two
ends and the middle of the window of each state of argon (DREAM's form: 0.64 to 1.13). `C` is
0.72 to 0.74 of DREAM's for hydrogen and the states of argon up to Ar⁵⁺, near the weight `1 − S(0) = 0.73`
of the Burgess-Chidichimo term at thermal energies that DREAM's `C` absorbs. `β*` reaches its
upper bound 1 for Ne⁰ to Ne²⁺, Ar⁰ and Ar⁷⁺, as some of DREAM's fits approach it.
"""
const KINETIC_IONIZATION_BETHE_FITS = let
    data = read_data_file("kinetic_ionization_bethe_fits")["data"]
    Dict{String, Matrix{Float64}}(element => float_matrix(rows) for (element, rows) in data)
end
