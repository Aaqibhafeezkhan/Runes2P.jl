# value of a number without its derivative parts (a method for ForwardDiff's dual numbers
# is defined in system/jacobian.jl)
_value(x::Real) = x

"""
    atomic_element(name)

The element of an ion species named `name`, whose atomic data the species takes (ADAS tables,
ionization energies, cross sections, masses): the name itself, or for a label of the form
`"<element>_<label>"` the part before the underscore. A label keeps two species of one element
apart, such as the injected deuterium `"D_inj"` of DREAM's disruption example next to the
background deuterium, which DREAM calls `Dinj` and evolves with an ion energy of its own
(`examples/Disruptions/generate.py:439`).
"""
function atomic_element(name::AbstractString)
    i = findfirst('_', name)
    return i === nothing ? String(name) : String(name[1:prevind(name, i)])
end

"""
    IonSpecies{T}

Ion species with nuclear charge `nuclear_charge` (Z) and charge-state densities
`charge_state_densities[Z0 + 1, ir]` [m⁻³] for charge states `Z0 = 0, …, Z` at each radius.
"""
struct IonSpecies{T <: Real}
    name::String
    nuclear_charge::Int
    charge_state_densities::Matrix{T}
    function IonSpecies(name::AbstractString, Z::Integer, densities::AbstractMatrix{T}) where {T <:
                                                                                               Real}
        size(densities, 1) == Z + 1 ||
            throw(ArgumentError("need Z + 1 charge-state rows for $name"))
        return new{T}(String(name), Int(Z), Matrix(densities))
    end
end

"""
    IonSpecies(name, Z, Z0, density)

Species in a single charge state `Z0` with density profile `density` (vector over radius).
"""
function IonSpecies(name::AbstractString, Z::Integer, Z0::Integer, density::AbstractVector{T}) where {T <:
                                                                                                      Real}
    0 <= Z0 <= Z || throw(ArgumentError("charge state must lie in 0:Z"))
    densities = zeros(T, Z + 1, length(density))
    densities[Z0 + 1, :] .= density
    return IonSpecies(name, Z, densities)
end

"""
    IonComposition(ions)

The species of a plasma and their charge states, without densities: the names, the nuclear
charges `Z`, and the components of each species in an unknown of the ion densities
([`ion_densities_unknown`](@ref)), where species `s` holds the charge states
`Z0 = 0, …, Z_s` at the components `species_components(composition, s)`, in the order of
`ions`. `ions` is a list of [`IonSpecies`](@ref), a [`PlasmaState`](@ref), or pairs
`name => Z`.
"""
struct IonComposition
    names::Vector{String}
    nuclear_charges::Vector{Int}
    offsets::Vector{Int}    # per species: the component before its neutral state
    function IonComposition(names::AbstractVector{<:AbstractString}, nuclear_charges::AbstractVector{<:Integer})
        length(names) == length(nuclear_charges) ||
            throw(DimensionMismatch("one nuclear charge per species name"))
        all(>=(1), nuclear_charges) ||
            throw(ArgumentError("a nuclear charge is at least 1"))
        allunique(names) ||
            throw(ArgumentError("the species names of an ion composition must differ"))
        offsets = cumsum([0; nuclear_charges[1:(end - 1)] .+ 1])
        return new(String.(names), Int.(nuclear_charges), offsets)
    end
end
function IonComposition(ions::AbstractVector{<:IonSpecies})
    IonComposition([s.name for s in ions], [s.nuclear_charge for s in ions])
end
function IonComposition(ions::AbstractVector{<:Pair})
    IonComposition([String(first(p)) for p in ions], [last(p) for p in ions])
end

Base.length(c::IonComposition) = length(c.names)
function Base.:(==)(a::IonComposition, b::IonComposition)
    a.names == b.names && a.nuclear_charges == b.nuclear_charges
end

"""
    component_count(composition::IonComposition)

The number of charge states of all species, `Σ (Z_s + 1)`: the components of the unknown of
the ion densities.
"""
function component_count(c::IonComposition)
    isempty(c.offsets) ? 0 : last(c.offsets) + last(c.nuclear_charges) + 1
end

"""
    species_components(composition, s)

The components of the charge states `Z0 = 0, …, Z` of species `s` (an index or a name) in
the unknown of the ion densities.
"""
function species_components(c::IonComposition, s::Integer)
    (c.offsets[s] + 1):(c.offsets[s] + c.nuclear_charges[s] + 1)
end
function species_components(c::IonComposition, name::AbstractString)
    species_components(c, species_index(c, name))
end

"""
    species_index(composition, name)

The index of the species `name` in `composition`.
"""
function species_index(c::IonComposition, name::AbstractString)
    s = findfirst(==(name), c.names)
    s === nothing &&
        throw(ArgumentError("no ion species $name in the composition ($(join(c.names, ", ")))"))
    return s
end

"""
    PlasmaState{T}

Background ("cold") plasma seen by the kinetic electrons at each radius: cold-electron
temperature [eV] and density [m⁻³], ion species, the electric field [V/m] (DREAM's
`E_field`, ⟨E·B⟩/√⟨B²⟩) and the minimum magnetic field on each flux surface [T].
`cold_density_given` records whether the cold-electron density was given, or is the
free-electron density of the ions by default: a system that holds the electrons of the
grids as unknowns refuses that default as the target of the e-e collisions.
"""
struct PlasmaState{T <: Real}
    cold_electron_temperature::Vector{T}
    cold_electron_density::Vector{T}
    ions::Vector{IonSpecies{T}}
    electric_field::Vector{T}
    magnetic_field::Vector{T}
    cold_density_given::Bool
end
function PlasmaState{T}(
        temperature, density, ions, electric_field, magnetic_field) where {T}
    PlasmaState{T}(temperature, density, ions, electric_field, magnetic_field, true)
end

"""
    PlasmaState(; cold_electron_temperature, ions, electric_field, magnetic_field,
                cold_electron_density=nothing)

Keyword constructor; without `cold_electron_density` the cold-electron density is the
free-electron density from quasi-neutrality ([`free_electron_density`](@ref)) at each
radius.
"""
function PlasmaState(; cold_electron_temperature, ions, electric_field, magnetic_field,
        cold_electron_density = nothing)
    ions = collect(ions)
    density = cold_electron_density === nothing ?
              [free_electron_density(ions, ir)
               for ir in eachindex(cold_electron_temperature)] :
              cold_electron_density
    T = promote_type(
        eltype(cold_electron_temperature), eltype(density), eltype(electric_field),
        eltype(magnetic_field), (eltype(s.charge_state_densities) for s in ions)...)
    #! format: off
    return PlasmaState{T}(collect(T, cold_electron_temperature), collect(T, density),
                          IonSpecies{T}[IonSpecies(s.name, s.nuclear_charge, T.(s.charge_state_densities))
                                        for s in ions],
                          collect(T, electric_field), collect(T, magnetic_field), cold_electron_density !== nothing)
    #! format: on
end

radial_cell_count(plasma::PlasmaState) = length(plasma.cold_electron_temperature)

"""
    with_electric_field(plasma, electric_field)

The plasma state with another electric field (one value per radius), e.g. the field of
the state of a run in place of a prescribed one.
"""
function with_electric_field(plasma::PlasmaState, electric_field::AbstractVector)
    length(electric_field) == radial_cell_count(plasma) ||
        throw(DimensionMismatch("the electric field needs one value per radius"))
    state = PlasmaState(;
        plasma.cold_electron_temperature, plasma.cold_electron_density, plasma.ions,
        electric_field, plasma.magnetic_field)
    return PlasmaState{eltype(state.cold_electron_temperature)}(
        state.cold_electron_temperature,
        state.cold_electron_density, state.ions,
        state.electric_field, state.magnetic_field,
        plasma.cold_density_given)
end

#! format: off
"""
    free_electron_density(ions, ir)

`Σ Z0 n_{Z0}` over all ion charge states (DREAM `GetFreeElectronDensityFromQuasiNeutrality`).
"""
free_electron_density(ions, ir) =
    sum((Z0 * s.charge_state_densities[Z0+1, ir] for s in ions for Z0 in 0:s.nuclear_charge);
        init=zero(eltype(first(ions).charge_state_densities)))
#! format: on
free_electron_density(plasma::PlasmaState, ir) = free_electron_density(plasma.ions, ir)

#! format: off
"""
    bound_electron_density(plasma, ir)

`Σ (Z - Z0) n_{Z0}` over all ion charge states.
"""
bound_electron_density(plasma::PlasmaState, ir) =
    sum(((s.nuclear_charge - Z0) * s.charge_state_densities[Z0+1, ir]
         for s in plasma.ions for Z0 in 0:s.nuclear_charge);
        init=zero(eltype(plasma.cold_electron_temperature)))
#! format: on

IonComposition(plasma::PlasmaState) = IonComposition(plasma.ions)
