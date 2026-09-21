"""
    IonSpecies{T}

Ion species with nuclear charge `Z` and charge-state densities `n[Z0 + 1, ir]` [m⁻³]
for charge states `Z0 = 0, …, Z` at each radius.
"""
struct IonSpecies{T<:Real}
    name::String
    Z::Int
    n::Matrix{T}
    function IonSpecies(name::AbstractString, Z::Integer, n::AbstractMatrix{T}) where {T<:Real}
        size(n, 1) == Z + 1 || throw(ArgumentError("need Z + 1 charge-state rows for $name"))
        return new{T}(String(name), Int(Z), Matrix(n))
    end
end

"""
    IonSpecies(name, Z, Z0, n)

Species in a single charge state `Z0` with density profile `n` (vector over radius).
"""
function IonSpecies(name::AbstractString, Z::Integer, Z0::Integer, n::AbstractVector{T}) where {T<:Real}
    0 <= Z0 <= Z || throw(ArgumentError("charge state must lie in 0:Z"))
    dens = zeros(T, Z + 1, length(n))
    dens[Z0+1, :] .= n
    return IonSpecies(name, Z, dens)
end

"""
    PlasmaState{T}

Background ("cold") plasma seen by the kinetic electrons at each radius: cold-electron
temperature `T_cold` [eV] and density `n_cold` [m⁻³], ion species, the electric field
`E` [V/m] (DREAM's `E_field`, ⟨E·B⟩/√⟨B²⟩) and the minimum magnetic field on each flux
surface `B` [T].
"""
struct PlasmaState{T<:Real}
    T_cold::Vector{T}
    n_cold::Vector{T}
    ions::Vector{IonSpecies{T}}
    E::Vector{T}
    B::Vector{T}
end

"""
    PlasmaState(; T_cold, ions, E, B, n_cold=free_electron_density(ions))

Keyword constructor; `n_cold` defaults to the free-electron density from
quasi-neutrality.
"""
function PlasmaState(; T_cold, ions, E, B, n_cold=nothing)
    ions = collect(ions)
    nc = n_cold === nothing ? [free_electron_density(ions, ir) for ir in eachindex(T_cold)] :
         n_cold
    T = promote_type(eltype(T_cold), eltype(nc), eltype(E), eltype(B),
                     (eltype(s.n) for s in ions)...)
    return PlasmaState{T}(collect(T, T_cold), collect(T, nc), IonSpecies{T}[
        IonSpecies(s.name, s.Z, T.(s.n)) for s in ions], collect(T, E), collect(T, B))
end

n_radii(ps::PlasmaState) = length(ps.T_cold)

"""
    free_electron_density(ions, ir)

`Σ Z0 n_{Z0}` over all ion charge states (DREAM `GetFreeElectronDensityFromQuasiNeutrality`).
"""
free_electron_density(ions, ir) =
    sum((Z0 * s.n[Z0+1, ir] for s in ions for Z0 in 0:s.Z); init=zero(eltype(first(ions).n)))
free_electron_density(ps::PlasmaState, ir) = free_electron_density(ps.ions, ir)

"""
    bound_electron_density(ps, ir)

`Σ (Z - Z0) n_{Z0}` over all ion charge states.
"""
bound_electron_density(ps::PlasmaState, ir) =
    sum(((s.Z - Z0) * s.n[Z0+1, ir] for s in ps.ions for Z0 in 0:s.Z); init=zero(eltype(ps.T_cold)))
