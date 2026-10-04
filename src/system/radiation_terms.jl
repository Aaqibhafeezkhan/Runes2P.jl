# The heating and the radiative losses of the cold electrons: the ohmic
# heating j_ohm E, and the radiated power of the ions with the binding energy they take on
# ionization and return on recombination, from the OpenADAS coefficients (physics/adas.jl)
# and the ionization energies of NIST. The terms sit in the row of the temperature, which
# holds the time derivative of the energy W_cold, in W/m³.

"""
    ohmic_heating_term(; electric_field=:electric_field, current=:ohmic_current_density)

The ohmic heating `j_ohm E` [W/m³] of the cold electrons, for the row of their temperature
(DREAM's `OhmicHeatingTerm`, whose geometric weight is 1 in a cylinder): a
[`ProductTerm`](@ref) of the field [V/m] and the ohmic current density [A/m²]. The heating
takes the treatment of the equation that sets the current (the conductivity of a
[`ConductivityTerm`](@ref)), so the two agree by construction.
"""
ohmic_heating_term(; electric_field::Symbol=:electric_field, current::Symbol=:ohmic_current_density) =
    ProductTerm(electric_field, current)

# the bremsstrahlung of Gould (Astrophys. J. 238, 1980) with a relativistic correction, as
# DREAM's RadiatedPowerTerm.cpp: (32/3) α r₀² c √(2 mₑc² e/π), the 1.69e-38 W m³ eV^-1/2
# of the NRL formulary
const GOULD_BREMSSTRAHLUNG_PREFACTOR =
    (32 / 3) * Constants.FINE_STRUCTURE_CONSTANT * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT *
    sqrt(2 * Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT^2 * Constants.ELEMENTARY_CHARGE / π)
const GOULD_ION_CORRECTION = 19 / 24
const GOULD_ELECTRON_CORRECTION = 5 / (8 * sqrt(2)) * (44 - 3 * π^2)

"""
    RadiatedPowerTerm(plasma; radiation=:recombination, temperature=:cold_electron_temperature,
                      density=:cold_electron_density, extrapolation=:dream,
                      directory=adas_cache_directory(), initial_time=0.0)

The power that the cold electrons lose to the ions, `−P` [W/m³] in the row of their
temperature (which holds the time derivative of their energy), at every radius:

```math
P = n_e \\sum_{i, Z_0} n_{i, Z_0} \\left(L_{i, Z_0} + B_{i, Z_0}\\right)
```

with the electron density `n_e` and temperature from the unknowns `density` and
`temperature`, and the densities of the charge states `n_{i,Z_0}` from the ions of
`plasma`, a [`PlasmaState`](@ref) or a function of time that returns one (the ions are
prescribed). The species are named by their element (`"Ar"`, `"D"`), whose
ADF11 tables are read from `directory` when the term is built ([`read_adas`](@ref)).

- `radiation = :recombination` (the default of Runes2P): the line radiation `L = PLT + PRB`
  with the recombination radiation and bremsstrahlung, the binding energy that ionization
  takes, `I(Z_0) SCD`, and the one recombination returns, `−I(Z_0 − 1) ACD`, with the
  ionization energies `I` of NIST ([`ionization_energy`](@ref)). DREAM's mode with
  recombination takes `I(Z_0)` in the gain from its second radial cell on.
- `radiation = :dream`: DREAM's default, `L = PLT`, the ionization loss alone, and the
  bremsstrahlung of Gould with a relativistic correction, `C √T (Σ n Z_0² + (19/24 Σ n Z_0²
  + κ n_e) T/mₑc²) n_e`, `C = 1.69e-38 W m³ eV^{-1/2}`, `κ = 5(44 − 3π²)/(8√2)`. The ion sum
  is that of the prescribed ions, `n_e` the unknown.

`extrapolation` goes to [`adas_coefficient`](@ref). The derivatives with respect to the
temperature and the density are exact through ForwardDiff. [`radiated_power`](@ref) gives
the radiation and the binding energy apart.
"""
struct RadiatedPowerTerm{S<:PrescribedPlasma} <: EquationTerm
    plasma::S
    temperature::Symbol
    density::Symbol
    radiation::Symbol
    extrapolation::Symbol
    tables::Dict{String,NamedTuple}
end

function RadiatedPowerTerm(plasma; radiation::Symbol=:recombination, temperature::Symbol=:cold_electron_temperature,
                           density::Symbol=:cold_electron_density, extrapolation::Symbol=:dream,
                           directory::AbstractString=adas_cache_directory(), initial_time::Real=0.0)
    radiation in (:recombination, :dream) ||
        throw(ArgumentError("radiation must be :recombination or :dream, not $radiation"))
    extrapolation in (:dream, :clamp_density) ||
        throw(ArgumentError("extrapolation must be :dream or :clamp_density, not $extrapolation"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    tables = Dict{String,NamedTuple}()
    for species in plasma_values(source, nothing, nothing, initial_time).ions
        element = species.name
        haskey(tables, element) && continue
        haskey(ADAS_FILES, element) ||
            throw(ArgumentError("the ion species $element is not an element with ADF11 tables " *
                                "($(join(sort!(collect(keys(ADAS_FILES))), ", ")))"))
        ionization_energy(element, 0)
        element_tables = read_adas(element; directory)
        element_tables.plt.nuclear_charge == species.nuclear_charge ||
            throw(ArgumentError("the species $element has nuclear charge $(species.nuclear_charge), its ADF11 " *
                                "tables $(element_tables.plt.nuclear_charge)"))
        tables[element] = element_tables
    end
    return RadiatedPowerTerm(source, temperature, density, radiation, extrapolation, tables)
end

dependencies(term::RadiatedPowerTerm) = unique([term.temperature, term.density])

function validate(term::RadiatedPowerTerm, layout, row)
    for id in (unknown_id(layout, term.temperature), unknown_id(layout, term.density), row)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("RadiatedPowerTerm takes and adds to radial unknowns with one component, not " *
                                "$(layout.unknowns[id].name)"))
    end
    return nothing
end

# the radiation and the binding energy of the ions at radius ir, for the electron density
# n_e [m⁻³] and temperature T [eV], in W/m³
function _radiated_power(term::RadiatedPowerTerm, plasma::PlasmaState, n_e, T, ir)
    radiation = binding = zero(n_e * T)
    recombination = term.radiation === :recombination
    ion_sum = zero(eltype(first(plasma.ions).charge_state_densities))    # Σ n Z₀²
    for species in plasma.ions
        tables = term.tables[species.name]
        Z = species.nuclear_charge
        coefficient(table, Z0) = adas_coefficient(table, Z0, n_e, T; extrapolation=term.extrapolation)
        for Z0 in 0:Z
            density = species.charge_state_densities[Z0+1, ir]
            iszero(density) && continue
            ion_sum += density * Z0^2
            line = coefficient(tables.plt, Z0)
            energy = zero(line)
            recombination && (line += coefficient(tables.prb, Z0))
            Z0 < Z && (energy += Constants.ELEMENTARY_CHARGE * ionization_energy(species.name, Z0) *
                                 coefficient(tables.scd, Z0))
            recombination && Z0 > 0 &&
                (energy -= Constants.ELEMENTARY_CHARGE * ionization_energy(species.name, Z0 - 1) *
                           coefficient(tables.acd, Z0))
            radiation += density * line
            binding += density * energy
        end
    end
    if !recombination && T > 0
        theta = T / Constants.ELECTRON_REST_ENERGY
        radiation += GOULD_BREMSSTRAHLUNG_PREFACTOR * sqrt(T) *
                     (ion_sum + (GOULD_ION_CORRECTION * ion_sum + GOULD_ELECTRON_CORRECTION * n_e) * theta)
    end
    return (radiation=n_e * radiation, binding=n_e * binding)
end

function rhs!(out, term::RadiatedPowerTerm, u, layout, row, t)
    plasma = plasma_values(term.plasma, u, layout, t)
    radial_cell_count(plasma) == radial_cell_count(layout) ||
        throw(DimensionMismatch("the plasma of the radiated power and the layout have different radii"))
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    @inbounds for ir in 1:radial_cell_count(layout)
        power = _radiated_power(term, plasma, n[ir], T[ir], ir)
        out[first(unknown_range(layout, row, ir))] -= power.radiation + power.binding
    end
    return out
end

function sparsity!(entries, term::RadiatedPowerTerm, layout, row)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for name in dependencies(term)
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, name), ir))))
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
    plasma = plasma_values(term.plasma, u, layout, t)
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    parts = [_radiated_power(term, plasma, n[ir], T[ir], ir) for ir in 1:radial_cell_count(layout)]
    return (radiation=[p.radiation for p in parts], binding=[p.binding for p in parts])
end
