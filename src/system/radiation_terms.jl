# The heating and the radiative losses of the cold electrons: the ohmic
# heating j_ohm E, and the radiated power of the ions with the binding energy they take on
# ionization and return on recombination, from the OpenADAS coefficients (physics/adas.jl)
# and the ionization energies of NIST. The terms sit in the row of the temperature, which
# holds the time derivative of the energy W_cold, in W/m³.

"""
    ohmic_heating_term(; electric_field=:electric_field,
                       current_density=:ohmic_current_density)

The ohmic heating `j_ohm E` [W/m³] of the cold electrons, for the row of their temperature
(DREAM's `OhmicHeatingTerm`, whose geometric weight is 1 in a cylinder): a
[`ProductTerm`](@ref) of the field `electric_field` [V/m] and the ohmic current density
`current_density` [A/m²]. The heating takes the treatment of the equation that sets the
current (the conductivity of a [`ConductivityTerm`](@ref)), so the two agree by
construction.
"""
function ohmic_heating_term(;
        electric_field::Symbol = :electric_field,
        current_density::Symbol = :ohmic_current_density)
    ProductTerm(electric_field, current_density)
end

"""
    GOULD_BREMSSTRAHLUNG_PREFACTOR

The prefactor of the bremsstrahlung power of a Maxwellian plasma in the mode `:dream` of
[`RadiatedPowerTerm`](@ref), `(32/3) α r₀² c √(2 mₑc² e/π)` [W m³ eV^-1/2], the 1.69e-38 of
the NRL Plasma Formulary: R. J. Gould, "Thermal bremsstrahlung from high-temperature
plasmas", Astrophys. J. 238, 1026 (1980), ADS 1980ApJ...238.1026G (erratum Astrophys. J. 243,
677 (1981)), as DREAM's `src/Equations/Fluid/RadiatedPowerTerm.cpp` (lines 47-52 and 174-189,
the 1.69e-38 named at line 179). Gould's expansion holds for 10⁷ to 10⁹ K, about 1 to 100 keV.
Neither Runes2P nor DREAM has its first-order Born correction, `(ln 2 − 1/2) π (Σ Z³ n/Σ Z² n)
(π Ry/kT)^{1/2}` (eq. (41) with the erratum's π), which at 10 eV is 1.25 times the leading
term for charge 1: at the temperatures of a disruption the formula is outside its range, where
the line radiation of the impurities dominates the losses.
"""
const GOULD_BREMSSTRAHLUNG_PREFACTOR = (32 / 3) * Constants.FINE_STRUCTURE_CONSTANT *
                                       Constants.CLASSICAL_ELECTRON_RADIUS^2 *
                                       Constants.SPEED_OF_LIGHT *
                                       sqrt(2 * Constants.ELECTRON_MASS *
                                            Constants.SPEED_OF_LIGHT^2 *
                                            Constants.ELEMENTARY_CHARGE / π)
"""
    GOULD_ION_CORRECTION

The first-order relativistic correction `11/24` of the electron-ion bremsstrahlung of a
Maxwellian plasma, the coefficient of `T/mₑc²` in the power of
[`GOULD_BREMSSTRAHLUNG_PREFACTOR`](@ref): R. J. Gould, Astrophys. J. 238, 1026 (1980),
eqs. (18) and (41), as corrected in the erratum, Astrophys. J. 243, 677 (1981) ("In equations
(18) and (41), 19/24 should be replaced by 11/24", found by E. Haug). It is the slope in `T`
of the Jüttner average of the cross section 4BN of H. W. Koch and J. W. Motz, Rev. Mod.
Phys. 31, 920 (1959), [`bremsstrahlung_cross_section`](@ref), which gives 11/24 to 1e-5
(`test_independent_references.jl`). DREAM's
`src/Equations/Fluid/RadiatedPowerTerm.cpp:51` keeps the 19/24 of the paper
([`DREAM_GOULD_ION_CORRECTION`](@ref)).
"""
const GOULD_ION_CORRECTION = 11 / 24

"""
    DREAM_GOULD_ION_CORRECTION

DREAM's relativistic correction of the electron-ion bremsstrahlung, the `19/24` of Gould 1980
before its erratum (`src/Equations/Fluid/RadiatedPowerTerm.cpp:51`, "relativistic-maxwellian
correction"); [`GOULD_ION_CORRECTION`](@ref) holds the corrected value.
"""
const DREAM_GOULD_ION_CORRECTION = 19 / 24

"""
    GOULD_ELECTRON_CORRECTION

The electron-electron bremsstrahlung of a Maxwellian plasma relative to the electron-ion
one of charge 1, `5(44 − 3π²)/(24√2) ≈ 2.12` times `T/mₑc²`: the ratio of eqs. (22) and (18)
of R. J. Gould, Astrophys. J. 238, 1026 (1980), the Born approximation of the
electron-electron and the electron-ion power, `(20/9)(44 − 3π²) π^{-1/2} α³Λ² m^{-1/2} n_e²
(kT)^{3/2}` over `(32/3)(2/π)^{1/2} α³Λ² m c³ n_e Σ Z² n_Z τ^{1/2}`; also eq. (58) of H. Xie,
"Bremsstrahlung radiation power in fusion plasmas revisited", arXiv:2404.11540 (2024).
Gould's summary, eq. (41), which says it combines eqs. (18) and (22), prints `5(44 −
3π²)/(8√2) ≈ 6.36`, three times larger, and the erratum (Astrophys. J. 243, 677 (1981)) does
not correct it; DREAM takes that value ([`DREAM_GOULD_ELECTRON_CORRECTION`](@ref)). The
factor rests on the consistency of the paper's own equations and on Xie, not on an erratum.
"""
const GOULD_ELECTRON_CORRECTION = 5 / (24 * sqrt(2)) * (44 - 3 * π^2)

"""
    DREAM_GOULD_ELECTRON_CORRECTION

DREAM's electron-electron bremsstrahlung relative to the electron-ion one, `5(44 −
3π²)/(8√2) ≈ 6.36` times `T/mₑc²`, eq. (41) of Gould 1980 as printed
(`src/Equations/Fluid/RadiatedPowerTerm.cpp:52`, "e-e brems correction");
[`GOULD_ELECTRON_CORRECTION`](@ref) holds the value of eqs. (18) and (22).
"""
const DREAM_GOULD_ELECTRON_CORRECTION = 5 / (8 * sqrt(2)) * (44 - 3 * π^2)

"""
    RadiatedPowerTerm(plasma; radiation=:recombination, temperature=:cold_electron_temperature,
                      density=:cold_electron_density, ions=nothing,
                      kinetic_ionization=nothing, extrapolation=:dream,
                      bremsstrahlung_corrections=:gould,
                      directory=adas_cache_directory(), initial_time=0.0)

The power that the cold electrons lose to the ions, `−P` [W/m³] in the row of their
temperature (which holds the time derivative of their energy), at every radius:

```math
P = n_e \\sum_{i, Z_0} n_{i, Z_0} \\left(L_{i, Z_0} + B_{i, Z_0}\\right)
```

with the electron density `n_e` and temperature from the unknowns `density` and
`temperature`, and the densities of the charge states `n_{i,Z_0}` from the ions of
`plasma`, a [`PlasmaState`](@ref) or a function of time that returns one, or from the
radial unknown `ions` of the charge-state densities of the species of `plasma`
([`ion_densities_unknown`](@ref)). The species are named by their element
(`"Ar"`, `"D"`), whose ADF11 tables are read from `directory` when the term is built
([`read_adas`](@ref)). With `ions` the term assembles its Jacobian with the columns of the
charge states ([`add_local_jacobian!`](@ref)). With `kinetic_ionization`, the
[`KineticIonizationTerm`](@ref) of the split of [`IonRateTerm`](@ref), the binding energy
of the ionization is that of the cold electrons below its threshold, at the fluid
coefficient of [`fluid_ionization_coefficient`](@ref).

- `radiation = :recombination` (the default, a choice of this package; see the docs page
  *Choices of this package*): the line radiation `L = PLT + PRB` with the recombination
  radiation and bremsstrahlung, the binding energy that ionization takes, `I(Z_0) SCD`, and
  the one recombination returns, `−I(Z_0 − 1) ACD`, with the ionization energies `I` of NIST
  ([`ionization_energy`](@ref)). DREAM's mode with recombination takes `I(Z_0)` in the gain
  from its second radial cell on.
- `radiation = :dream`: DREAM's default, `L = PLT`, the ionization loss alone, and the
  bremsstrahlung of Gould with a relativistic correction, `C √T (Σ n Z_0² + (a Σ n Z_0²
  + κ n_e) T/mₑc²) n_e`, `C = 1.69e-38 W m³ eV^{-1/2}`
  ([`GOULD_BREMSSTRAHLUNG_PREFACTOR`](@ref)). With `bremsstrahlung_corrections = :gould`
  (default) `a = 11/24` and `κ = 5(44 − 3π²)/(24√2)`, Gould's values after his erratum and by
  his own eqs. (18) and (22) ([`GOULD_ION_CORRECTION`](@ref),
  [`GOULD_ELECTRON_CORRECTION`](@ref)); with `:dream` DREAM's `19/24` and `5(44 − 3π²)/(8√2)`.
  DREAM's radiation therefore needs both `radiation = :dream` and
  `bremsstrahlung_corrections = :dream`; the second has no effect with `:recombination`. The
  ion sum is that of the charge states the term reads, `n_e` the unknown.

The coefficients are those of ADAS (H. P. Summers, The ADAS User Manual, version 2.6 (2004), http://www.adas.ac.uk, with the data of OpenADAS, https://open.adas.ac.uk), DREAM's
`src/Equations/Fluid/RadiatedPowerTerm.cpp`. `extrapolation` goes to
[`adas_coefficient`](@ref); its default `:dream` takes DREAM's extrapolation below the tables
(a choice of this package; see the docs page *Choices of this package*). The derivatives with
respect to the temperature and the density are exact through ForwardDiff.
[`radiated_power`](@ref) gives the radiation and the binding energy apart.
"""
struct RadiatedPowerTerm{
    S <: PrescribedPlasma, K <: Union{Nothing, KineticIonizationTerm}} <: EquationTerm
    plasma::S
    temperature::Symbol
    density::Symbol
    radiation::Symbol
    extrapolation::Symbol
    bremsstrahlung_corrections::Symbol
    tables::Dict{String, NamedTuple}
    ions::Union{Nothing, Symbol}
    kinetic_ionization::K         # nothing, or the KineticIonizationTerm of the split of the ionization
end

function RadiatedPowerTerm(plasma; radiation::Symbol = :recombination,
        temperature::Symbol = :cold_electron_temperature,
        density::Symbol = :cold_electron_density, ions::Union{Nothing, Symbol} = nothing,
        kinetic_ionization = nothing, extrapolation::Symbol = :dream, bremsstrahlung_corrections::Symbol = :gould,
        directory::AbstractString = adas_cache_directory(), initial_time::Real = 0.0)
    radiation in (:recombination, :dream) ||
        throw(ArgumentError("radiation must be :recombination or :dream, not $radiation"))
    extrapolation in (:dream, :clamp_density) ||
        throw(ArgumentError("extrapolation must be :dream or :clamp_density, not $extrapolation"))
    bremsstrahlung_corrections in (:gould, :dream) ||
        throw(ArgumentError("bremsstrahlung_corrections must be :gould or :dream, not $bremsstrahlung_corrections"))
    plasma isa PlasmaSource && !(plasma isa PrescribedPlasma) &&
        throw(ArgumentError("RadiatedPowerTerm takes a prescribed plasma and reads the temperature, the density " *
                            "and the ion densities from unknowns of its own: give it the prescribed part and " *
                            "the keywords temperature, density and ions"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    tables = Dict{String, NamedTuple}()
    for species in plasma_values(source, nothing, nothing, initial_time).ions
        element = species.name
        haskey(tables, element) && continue
        haskey(ADAS_FILES, atomic_element(element)) ||
            throw(ArgumentError("the ion species $element is not an element with ADF11 tables " *
                                "($(join(sort!(collect(keys(ADAS_FILES))), ", ")))"))
        ionization_energy(element, 0)
        element_tables = read_adas(element; directory)
        element_tables.plt.nuclear_charge == species.nuclear_charge ||
            throw(ArgumentError("the species $element has nuclear charge $(species.nuclear_charge), its ADF11 " *
                                "tables $(element_tables.plt.nuclear_charge)"))
        tables[element] = element_tables
    end
    kinetic_ionization === nothing || kinetic_ionization isa KineticIonizationTerm ||
        throw(ArgumentError("kinetic_ionization must be a KineticIonizationTerm or nothing"))
    kinetic_ionization === nothing ||
        _check_split_threshold(kinetic_ionization, "a RadiatedPowerTerm")
    return RadiatedPowerTerm(source, temperature, density, radiation,
        extrapolation, bremsstrahlung_corrections, tables,
        ions, kinetic_ionization)
end

function dependencies(term::RadiatedPowerTerm)
    unique([term.temperature; term.density; term.ions === nothing ? Symbol[] : [term.ions];
            term.kinetic_ionization === nothing ? Symbol[] :
            collect(Symbol, _region_unknowns(term.kinetic_ionization.envelopes))])
end
# the index of a species in the composition of the kinetic ionization of the split
function _species_of(term::RadiatedPowerTerm, name)
    species_index(term.kinetic_ionization.composition, name)
end
jacobian_assembly(term::RadiatedPowerTerm) = term.ions === nothing ? :automatic : :direct
function add_jacobian!(J, term::RadiatedPowerTerm, u, layout, row, t)
    add_local_jacobian!(J, term, u, layout, row, t)
end

function validate(term::RadiatedPowerTerm, layout, row, t)
    for id in (unknown_id(layout, term.temperature), unknown_id(layout, term.density), row)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("RadiatedPowerTerm takes and adds to radial unknowns with one component, not " *
                                "$(layout.unknowns[id].name)"))
    end
    term.ions === nothing ||
        _check_ion_unknown(layout, unknown_id(layout, term.ions),
            IonComposition(plasma_values(term.plasma, nothing, nothing, t)))
    return nothing
end
validate(term::RadiatedPowerTerm, layout, row) = validate(term, layout, row, 0.0)

# the radiation and the binding energy of the ions at radius ir, for the electron density
# n_e [m⁻³] and temperature T [eV], in W/m³
function _radiated_power(
        term::RadiatedPowerTerm, plasma::PlasmaState, n_e, T, ir, split = nothing)
    radiation = binding = zero(n_e * T)
    recombination = term.radiation === :recombination
    ion_sum = zero(eltype(first(plasma.ions).charge_state_densities))    # Σ n Z₀²
    for species in plasma.ions
        tables = term.tables[species.name]
        Z = species.nuclear_charge
        coefficient(table, Z0) = adas_coefficient(
            table, Z0, n_e, T; extrapolation = term.extrapolation)
        for Z0 in 0:Z
            density = species.charge_state_densities[Z0 + 1, ir]
            iszero(density) && continue
            ion_sum += density * Z0^2
            line = coefficient(tables.plt, Z0)
            energy = zero(line)
            recombination && (line += coefficient(tables.prb, Z0))
            if Z0 < Z
                ionization = coefficient(tables.scd, Z0)
                split === nothing ||
                    (ionization = _fluid_ionization(split, _species_of(term, species.name), Z0, ionization))
                energy += Constants.ELEMENTARY_CHARGE *
                          ionization_energy(species.name, Z0) * ionization
            end
            recombination && Z0 > 0 &&
                (energy -= Constants.ELEMENTARY_CHARGE *
                           ionization_energy(species.name, Z0 - 1) *
                           coefficient(tables.acd, Z0))
            radiation += density * line
            binding += density * energy
        end
    end
    if !recombination && T > 0
        theta = T / Constants.ELECTRON_REST_ENERGY
        a, κ = term.bremsstrahlung_corrections === :gould ?
               (GOULD_ION_CORRECTION, GOULD_ELECTRON_CORRECTION) :
               (DREAM_GOULD_ION_CORRECTION, DREAM_GOULD_ELECTRON_CORRECTION)
        radiation += GOULD_BREMSSTRAHLUNG_PREFACTOR * sqrt(T) *
                     (ion_sum + (a * ion_sum + κ * n_e) * theta)
    end
    return (radiation = n_e * radiation, binding = n_e * binding)
end

function rhs!(out, term::RadiatedPowerTerm, u, layout, row, t)
    plasma = _term_plasma(plasma_values(term.plasma, u, layout, t), term.ions, u, layout)
    radial_cell_count(plasma) == radial_cell_count(layout) ||
        throw(DimensionMismatch("the plasma of the radiated power and the layout have different radii"))
    T = unknown_values(u, layout, unknown_id(layout, term.temperature))
    n = unknown_values(u, layout, unknown_id(layout, term.density))
    @inbounds for ir in 1:radial_cell_count(layout)
        split = term.kinetic_ionization === nothing ? nothing :
                threshold_split(term.kinetic_ionization, ir, u, layout, t)
        power = _radiated_power(term, plasma, n[ir], T[ir], ir, split)
        out[first(unknown_range(layout, row, ir))] -= power.radiation + power.binding
    end
    return out
end

function sparsity!(entries, term::RadiatedPowerTerm, layout, row)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for name in dependencies(term)
            append!(entries, ((r, c)
            for c in unknown_range(layout, unknown_id(layout, name), ir)))
        end
    end
    return entries
end

"""
    radiated_power(term::RadiatedPowerTerm, u, layout, t)

The two parts of the power of [`RadiatedPowerTerm`](@ref) at the state `u`, per radius
[W/m³]: `radiation`, which leaves the plasma (line radiation, recombination radiation and
bremsstrahlung), and `binding`, the binding energy that the ions take on ionization, net of
what recombination returns, which stays in the plasma.
"""
function radiated_power(term::RadiatedPowerTerm, u, layout, t)
    plasma = _term_plasma(plasma_values(term.plasma, u, layout, t), term.ions, u, layout)
    T = unknown_values(u, layout, unknown_id(layout, term.temperature))
    n = unknown_values(u, layout, unknown_id(layout, term.density))
    parts = [_radiated_power(term, plasma, n[ir], T[ir], ir,
                 term.kinetic_ionization === nothing ? nothing :
                 threshold_split(term.kinetic_ionization, ir, u, layout, t))
             for ir in 1:radial_cell_count(layout)]
    return (radiation = [p.radiation for p in parts], binding = [p.binding for p in parts])
end
