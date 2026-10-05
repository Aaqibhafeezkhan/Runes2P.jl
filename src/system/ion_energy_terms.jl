# The thermal energy of the ions and its collisional exchange with the cold electrons and
# between the ion species (DREAM's W_i and
# MaxwellianCollisionalEnergyTransferTerm). The ion densities are prescribed or read
# from the unknown of the charge states; the energy of each species is a
# component of one radial unknown.

"""
    IonEnergyExchangeTerm(plasma; side, ion_energy=:ion_thermal_energy,
                          temperature=:cold_electron_temperature, density=:cold_electron_density,
                          ion_ion=true, masses=nothing, ions=nothing, initial_time=0.0)

The collisional exchange of thermal energy between Maxwellian species (DREAM's
`src/Equations/Fluid/MaxwellianCollisionalEnergyTransferTerm.cpp`, the prefactor at line
49), in W/m³: the power that species `a` receives
from species `b`,

```math
Q_{ab} = \\sqrt{3/π}\\,\\frac{(e^2/ε_0)^2 \\sqrt{m_a m_b}}{4π} \\ln Λ\\,
         \\sqrt{N_a N_b}\\, (nZ^2)_a (nZ^2)_b\\, \\frac{N_a W_b − N_b W_a}{(m_b N_b W_a + m_a N_a W_b)^{3/2}},
```

the two-temperature relaxation of Spitzer (L. Spitzer, Physics of Fully Ionized Gases, 2nd
ed., Interscience, 1962), `Q_{ab} = (3/2) N_a ν_ε (T_b − T_a)` with the energy relaxation
rate of the NRL Plasma Formulary (J. D. Huba, Naval Research Laboratory). For the electrons `N = (nZ²) = n_e` and `W =
3/2 e n_e T_e` from the unknowns `density` and `temperature`; for an ion species `N` is its
total density, `(nZ²) = Σ n_{Z₀} Z₀²` (neutrals hold energy and exchange none), and `W` the
component of the unknown `ion_energy` (J/m³, one component per species of `plasma`, in its
order). The logarithms are DREAM's: the thermal one for electrons and ions, and for ions
among themselves `17.3 − ½ ln(n_free/10²⁰) + 3/2 ln(T_e/1 keV)` with the electron
temperature and the free-electron density (`src/Equations/CoulombLogarithm.cpp:107-113`,
with no source named there; checked against DREAM only).

- `side = :ions`, in the row of `ion_energy`: each component receives `Q_{i,e}` and, with
  `ion_ion`, `Σ_j Q_{ij}`.
- `side = :electrons`, in the row of the temperature (which holds the time derivative of
  the electron energy): `−Σ_i Q_{i,e}`.

A system needs both sides on the same unknowns, so the exchange conserves energy (refused
otherwise). `masses` overrides the masses of [`ion_mass`](@ref) per species.
The densities of the ions come from the prescribed `plasma`, a [`PlasmaState`](@ref) or a
function of time, or with `ions` from that radial unknown of the charge states, whose
columns the term then assembles directly; a species of zero density is
skipped. `ion_ion = true` adds the exchange among the ion species, as DREAM does whenever
the ion energies are evolved (`src/Settings/Equations/T_i.cpp:109`). A species of zero total density
exchanges nothing and gets no derivative there: the exchange goes as the square root of the
density, whose derivative at zero is singular. A Newton iterate with a negative
ion energy makes the denominator non-positive, where the exchange is zero with a zero
derivative: declare the unknown non-negative, `RadialUnknown(:ion_thermal_energy;
components, nonnegative = true)`.
"""
struct IonEnergyExchangeTerm{S<:PrescribedPlasma} <: EquationTerm
    plasma::S
    side::Symbol
    ion_energy::Symbol
    temperature::Symbol
    density::Symbol
    ion_ion::Bool
    masses::Vector{Float64}
    ions::Union{Nothing,Symbol}
end

function IonEnergyExchangeTerm(plasma; side::Symbol, ion_energy::Symbol=:ion_thermal_energy,
                               temperature::Symbol=:cold_electron_temperature,
                               density::Symbol=:cold_electron_density, ion_ion::Bool=true, masses=nothing,
                               ions::Union{Nothing,Symbol}=nothing, initial_time::Real=0.0)
    side in (:ions, :electrons) || throw(ArgumentError("side must be :ions or :electrons, not $side"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    species_list = plasma_values(source, nothing, nothing, initial_time).ions
    mass_values = masses === nothing ? [ion_mass(species) for species in species_list] : collect(Float64, masses)
    length(mass_values) == length(species_list) ||
        throw(DimensionMismatch("need one mass per ion species ($(length(species_list))), got $(length(mass_values))"))
    all(>(0), mass_values) || throw(ArgumentError("the ion masses must be positive"))
    return IonEnergyExchangeTerm(source, side, ion_energy, temperature, density, ion_ion, mass_values, ions)
end

dependencies(term::IonEnergyExchangeTerm) =
    [term.ion_energy; term.temperature; term.density; term.ions === nothing ? Symbol[] : [term.ions]]
jacobian_assembly(term::IonEnergyExchangeTerm) = term.ions === nothing ? :automatic : :direct
add_jacobian!(J, term::IonEnergyExchangeTerm, u, layout, row, t) = add_local_jacobian!(J, term, u, layout, row, t)

function validate(term::IonEnergyExchangeTerm, layout, row, t)
    plasma = plasma_values(term.plasma, nothing, nothing, t)
    species = length(plasma.ions)
    term.ions === nothing || _check_ion_unknown(layout, unknown_id(layout, term.ions), IonComposition(plasma))
    id = unknown_id(layout, term.ion_energy)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == species ||
        throw(ArgumentError("the ion energy $(term.ion_energy) needs one component per ion species ($species)"))
    for name in (term.temperature, term.density)
        k = unknown_id(layout, name)
        layout.kinds[k] === :radial && layout.sizes[k, 1] == 1 ||
            throw(ArgumentError("IonEnergyExchangeTerm reads $name as a radial unknown with one component"))
    end
    if term.side === :ions
        row == id || throw(ArgumentError("the side :ions of the energy exchange belongs to the row of $(term.ion_energy)"))
    else
        layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
            throw(ArgumentError("the side :electrons of the energy exchange adds to a radial unknown with one component"))
    end
    return nothing
end

const _EXCHANGE_PREFACTOR = sqrt(3 / π) * (Constants.ELEMENTARY_CHARGE^2 / Constants.VACUUM_PERMITTIVITY)^2 / (4π)

# Q_ab of DREAM's form with the species a = (N, nZ², W, m) and b
function _energy_exchange(lnL, a, b)
    Na, nZ2a, Wa, ma = a
    Nb, nZ2b, Wb, mb = b
    denominator = mb * Nb * Wa + ma * Na * Wb
    denominator > 0 || return zero(lnL * Wa * Wb)
    return _EXCHANGE_PREFACTOR * sqrt(ma * mb) * lnL * sqrt(Na * Nb) * nZ2a * nZ2b * (Na * Wb - Nb * Wa) /
           (denominator * sqrt(denominator))
end

# the power each ion species receives at radius ir, and the power the electrons lose
function _ion_exchange(term::IonEnergyExchangeTerm, plasma::PlasmaState, W, T_e, n_e, ir)
    e = Constants.ELEMENTARY_CHARGE
    n_free = free_electron_density(plasma, ir)
    lnL = thermal_coulomb_logarithm(T_e, n_free)
    lnL_ions = n_free == 0 ? zero(lnL) : 17.3 - 0.5 * log(n_free / 1e20) + 1.5 * log(T_e / 1e3)
    electrons = (n_e, n_e, 1.5 * e * n_e * T_e, Constants.ELECTRON_MASS)
    ions = map(enumerate(plasma.ions)) do (s, species)
        densities = view(species.charge_state_densities, :, ir)
        (sum(densities), sum(densities[Z0+1] * Z0^2 for Z0 in 0:species.nuclear_charge), W[s], term.masses[s])
    end
    received = [zero(lnL * W[s] * T_e) for s in eachindex(ions)]
    lost = zero(eltype(received))
    for (s, ion) in enumerate(ions)
        ion[1] > 0 || continue
        Q = _energy_exchange(lnL, ion, electrons)
        received[s] += Q
        lost += Q
        term.ion_ion || continue
        for (k, other) in enumerate(ions)
            k == s || other[1] <= 0 || (received[s] += _energy_exchange(lnL_ions, ion, other))
        end
    end
    return received, lost
end

function rhs!(out, term::IonEnergyExchangeTerm, u, layout, row, t)
    plasma = _term_plasma(plasma_values(term.plasma, u, layout, t), term.ions, u, layout)
    id = unknown_id(layout, term.ion_energy)
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    for ir in 1:radial_cell_count(layout)
        W = view(u, unknown_range(layout, id, ir))
        received, lost = _ion_exchange(term, plasma, W, T[ir], n[ir], ir)
        rows = unknown_range(layout, row, ir)
        if term.side === :ions
            for (s, r) in enumerate(rows)
                out[r] += received[s]
            end
        else
            out[first(rows)] -= lost
        end
    end
    return out
end

function sparsity!(entries, term::IonEnergyExchangeTerm, layout, row)
    id = unknown_id(layout, term.ion_energy)
    for ir in 1:radial_cell_count(layout), r in unknown_range(layout, row, ir)
        for c in unknown_range(layout, id, ir)
            push!(entries, (r, c))
        end
        for name in (term.temperature, term.density)
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, name), ir))))
        end
        term.ions === nothing || append!(entries, ((r, c) for c in unknown_range(layout, unknown_id(layout, term.ions), ir)))
    end
    return entries
end

"""
    set_ion_thermal_energy!(u, layout, plasma, temperature; ion_energy=:ion_thermal_energy, t=0.0)

Set the thermal energy of every ion species in the state `u`, `W = 3/2 e N T` with the total
density `N` of the species in `plasma` at time `t` and the temperature `temperature` [eV]:
a number, one value per species, or a matrix of species × radii (DREAM's initial `W_i` from
`T_i`, `T_i.cpp`).
"""
function set_ion_thermal_energy!(u::AbstractVector, layout::UnknownLayout, plasma, temperature;
                                 ion_energy::Symbol=:ion_thermal_energy, t::Real=0.0)
    state = plasma isa PlasmaState ? plasma : plasma_values(plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma),
                                                            nothing, nothing, t)
    id = unknown_id(layout, ion_energy)
    for ir in 1:radial_cell_count(layout), (s, r) in enumerate(unknown_range(layout, id, ir))
        T = temperature isa Real ? temperature : temperature isa AbstractVector ? temperature[s] : temperature[s, ir]
        u[r] = THERMAL_ENERGY_FACTOR * sum(view(state.ions[s].charge_state_densities, :, ir)) * T
    end
    return u
end

"""
    ion_temperatures(u, layout, plasma; ion_energy=:ion_thermal_energy, t=0.0)

The temperature [eV] of every ion species at every radius (species × radii), `T = W/(3/2 e
N)`, from the state `u`; `NaN` where a species has no density.
"""
function ion_temperatures(u::AbstractVector, layout::UnknownLayout, plasma; ion_energy::Symbol=:ion_thermal_energy,
                          t::Real=0.0)
    state = plasma isa PlasmaState ? plasma : plasma_values(plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma),
                                                            nothing, nothing, t)
    id = unknown_id(layout, ion_energy)
    nr = radial_cell_count(layout)
    T = fill(NaN, length(state.ions), nr)
    for ir in 1:nr, (s, r) in enumerate(unknown_range(layout, id, ir))
        N = sum(view(state.ions[s].charge_state_densities, :, ir))
        N > 0 && (T[s, ir] = u[r] / (THERMAL_ENERGY_FACTOR * N))
    end
    return T
end

validate(term::IonEnergyExchangeTerm, layout, row) = validate(term, layout, row, 0.0)
