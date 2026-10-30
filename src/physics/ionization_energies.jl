# Ionization energies of the elements of the ADF11 tables, from the NIST
# Atomic Spectra Database (Kramida, Ralchenko, Reader and the NIST ASD Team,
# https://physics.nist.gov/asd), as DREAM's `src/Atomics/nistdata_ionization.cpp`, which its
# `tools/get_nist.py` generated on 2023-04-28 (DREAM is under the MIT license).

"""
    IONIZATION_ENERGIES

The ionization energy [eV] of each charge state, `IONIZATION_ENERGIES[element][Z0 + 1]` for
the ion of charge `Z0 = 0, …, Z − 1`: A. Kramida, Yu. Ralchenko, J. Reader and the NIST ASD
Team, NIST Atomic Spectra Database, https://physics.nist.gov/asd, doi:10.18434/T4W30F, as
retrieved on 2023-04-28 by DREAM's `tools/get_nist.py` into its
`src/Atomics/nistdata_ionization.cpp`. Deuterium and tritium take the energy of hydrogen.
The values are read from `data/ionization_energies.toml`.
"""
const IONIZATION_ENERGIES = let data = read_data_file("ionization_energies")["data"]
    energies = Dict{String, Vector{Float64}}(element => float_vector(values)
    for (element, values) in data)
    energies["D"] = copy(energies["H"])
    energies["T"] = copy(energies["H"])
    energies
end

"""
    ionization_energy(element, Z0)

The energy [eV] to ionize the ion of `element` with charge `Z0` ([`IONIZATION_ENERGIES`](@ref)).
"""
function ionization_energy(element::AbstractString, Z0::Integer)
    energies = get(IONIZATION_ENERGIES, atomic_element(element), nothing)
    energies === nothing &&
        throw(ArgumentError("no ionization energies for the element $element"))
    0 <= Z0 < length(energies) ||
        throw(ArgumentError("$element has no ion of charge $Z0 to ionize"))
    return energies[Z0 + 1]
end

"""
    ATOMIC_MASS_UNIT

The unified atomic mass unit [kg], CODATA 2018 (1.66053906660e-27) rounded as DREAM's
`src/Constants.cpp` (`mu`).
"""
const ATOMIC_MASS_UNIT = 1.660539067e-27

"""
    ATOMIC_MASSES

The standard atomic weights [u] of the elements Z = 1 to 92, `NaN` for the elements without
one (technetium, promethium, polonium to actinium).

- Z = 1 to 40: DREAM's `atomicMassInMu` (`src/IonHandler.cpp:35-46`), the IUPAC standard
  atomic weights, which DREAM does not name; they agree with those of CIAAW 2024 to 2.2e-5
  (zirconium) or better.
- Z = 41 to 92: the standard atomic weights of the Commission on Isotopic Abundances and
  Atomic Weights of IUPAC (CIAAW, https://www.ciaaw.org/atomic-weights.htm, read
  2026-10-12), and for thallium and lead, which have an interval, its conventional values
  204.38 and 207.2.

The values are read from `data/atomic_masses.toml`.
"""
const ATOMIC_MASSES = float_vector(read_data_file("atomic_masses")["data"]["masses"])

"""
    HYDROGEN_ISOTOPE_MASSES

The masses [kg] of the nuclei of the hydrogen isotopes, CODATA 2018 (E. Tiesinga, P. J. Mohr,
D. B. Newell, B. N. Taylor, Rev. Mod. Phys. 93, 025010 (2021),
doi:10.1103/RevModPhys.93.025010): the proton, the deuteron and the triton. DREAM's `mH`
and `mD` (`src/Constants.cpp`) are the first two; its `mT` is the atomic mass of ³H
(3.01604928 u, 5.00826765e-27 kg), which exceeds the triton by one electron mass
([`DREAM_HYDROGEN_ISOTOPE_MASSES`](@ref)).
"""
const HYDROGEN_ISOTOPE_MASSES = Dict(
    "H" => 1.67262192369e-27, "D" => 3.3435837724e-27, "T" => 5.0073567446e-27)

"""
    DREAM_HYDROGEN_ISOTOPE_MASSES

DREAM's masses [kg] of the hydrogen isotopes, `mH`, `mD` and `mT` of `src/Constants.cpp`:
the proton and the deuteron of CODATA 2018 and the atomic mass of ³H (3.01604928 u).
"""
const DREAM_HYDROGEN_ISOTOPE_MASSES = Dict(
    "H" => 1.67262192369e-27, "D" => 3.3435837724e-27, "T" => 5.00826765e-27)

"""
    ion_mass(species; table=:standard)

The mass [kg] of an ion species. For nuclear charge 1 it is the mass of hydrogen, deuterium
(the default, as in DREAM) or tritium by the name `"H"`, `"D"` or `"T"`; a species of charge
1 with another name takes the mass of deuterium.

- `table = :standard`: the nuclei of [`HYDROGEN_ISOTOPE_MASSES`](@ref) and the standard
  atomic weight of [`ATOMIC_MASSES`](@ref) for the other elements, up to uranium. An element
  without a standard atomic weight needs its mass given (`masses` of
  [`IonEnergyExchangeTerm`](@ref)). The weight is that of the neutral atom: an ion lacks up
  to Z electron masses, which this neglects, at most 2.7e-4 of its mass for the elements
  beyond hydrogen.
- `table = :dream`: DREAM's rule (`src/IonHandler.cpp:153-170`), the masses of
  [`DREAM_HYDROGEN_ISOTOPE_MASSES`](@ref), the standard atomic weights up to Z = 40, and
  `2.3 Z` atomic mass units beyond, which puts tungsten at 170 u against 183.84 u and xenon
  at 124 u against 131.3 u.
"""
function ion_mass(species::IonSpecies; table::Symbol = :standard)
    table in (:standard, :dream) ||
        throw(ArgumentError("table must be :standard or :dream, not $table"))
    Z = species.nuclear_charge
    if Z == 1
        isotopes = if table === :standard
            HYDROGEN_ISOTOPE_MASSES
        else
            DREAM_HYDROGEN_ISOTOPE_MASSES
        end
        return get(isotopes, atomic_element(species.name), isotopes["D"])
    end
    table === :dream && Z > 40 && return 2.3 * Z * ATOMIC_MASS_UNIT
    if !(Z <= length(ATOMIC_MASSES) && !isnan(ATOMIC_MASSES[Z]))
        throw(ArgumentError("$(species.name) (Z = $Z) has no standard atomic weight: give its mass"))
    end
    return ATOMIC_MASSES[Z] * ATOMIC_MASS_UNIT
end
