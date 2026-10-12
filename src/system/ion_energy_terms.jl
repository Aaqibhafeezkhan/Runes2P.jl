# The thermal energy of the ions and its collisional exchange with the cold electrons and
# between the ion species (DREAM's W_i and
# MaxwellianCollisionalEnergyTransferTerm). The ion densities are prescribed or read
# from the unknown of the charge states; the energy of each species is a
# component of one radial unknown.

"""
    IonEnergyExchangeTerm(plasma; side, ion_energy=:ion_thermal_energy,
                          temperature=:cold_electron_temperature, density=:cold_electron_density,
                          ion_ion=true, ion_coulomb_logarithm=:nrl, masses=nothing, mass_table=:standard,
                          ions=nothing, initial_time=0.0)

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
rate of the NRL Plasma Formulary (A. S. Richardson, NRL/PU/6770--19-652, Naval Research
Laboratory, 2019). For the electrons `N = (nZ²) = n_e` and `W =
3/2 e n_e T_e` from the unknowns `density` and `temperature`; for an ion species `N` is its
total density, `(nZ²) = Σ n_{Z₀} Z₀²` (neutrals hold energy and exchange none), and `W` the
component of the unknown `ion_energy` (J/m³, one component per species of `plasma`, in its
order). The logarithm of electrons and ions is the thermal one, as DREAM's. Among the ions it is

- `ion_coulomb_logarithm = :nrl` (default): the mixed ion-ion logarithm of the NRL Plasma
  Formulary (A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019, p. 34),
  `λ_ab = 23 − ln[Z_a Z_b (μ_a + μ_b)/(μ_a T_b + μ_b T_a) (Σ_γ n_γ Z_γ²/T_γ)^{1/2}]` with the
  densities in cm⁻³, the temperatures in eV and `μ = m/m_p`. Formula (c) of the formulary sums
  over the two colliding ions; its definition of the screening length on the same page sums
  over every species whose thermal speed exceeds the relative speed of the pair, `ū² <
  kT_γ/m_γ`, which holds for the electrons unless the ions are hotter by the mass ratio, so the
  sum here takes every ion species and the electrons (`n_e/T_e`). Each ion species takes its own temperature
  `T = W/(3/2 e N)` and the electrons theirs, at least the 0.5 eV of
  [`DEFAULT_TEMPERATURE_FLOOR`](@ref) (below it the model of a plasma does not hold), and
  for `Z` its mean charge `(nZ²)/(nZ)`, `(nZ) = Σ n_{Z₀} Z₀`. The formula holds for λ ≫ 1
  (its corrections are O(1/λ)); for
  cold, dense, highly charged ions it falls towards zero and below (D with Ar¹⁶⁺ at 1 eV and
  10¹⁹ m⁻³ with electrons at 10 eV: 2.5; Ar¹⁶⁺ with itself at 0.5 eV: −1.3), and it is cut at 0
  so that the exchange
  keeps its sign. A lower bound is an open question.
- `ion_coulomb_logarithm = :dream`: DREAM's `17.3 − ½ ln(n_free/10²⁰) + 3/2 ln(T_e/1 keV)`
  with the electron temperature and the free-electron density, the same for every pair
  (`src/Equations/CoulombLogarithm.cpp:107-113`, with no source named there;
  `BootstrapCurrent.cpp:181` calls the form Wesson's, which takes the ion temperature). With
  T_e in place of T_i it is too small where the ions are hotter than the electrons and too
  large where they are colder: 10.4 at T_e = 10 eV and 10²⁰ m⁻³ against 14.9 of the NRL form for
  two deuterium species at T_i = 1 keV (the exchange among them 1.4 times too slow after the
  thermal quench), and against 3.8 for deuterium with Ar⁸⁺ at T_i = 1 eV (the exchange with
  freshly injected argon 2.7 times too fast).

- `side = :ions`, in the row of `ion_energy`: each component receives `Q_{i,e}` and, with
  `ion_ion`, `Σ_j Q_{ij}`.
- `side = :electrons`, in the row of the temperature (which holds the time derivative of
  the electron energy): `−Σ_i Q_{i,e}`.

A system needs both sides on the same unknowns, so the exchange conserves energy (refused
otherwise). The masses are those of [`ion_mass`](@ref) with `mass_table`
(`:standard`, the standard atomic weights and the nuclei of the hydrogen isotopes, or
`:dream`, DREAM's rule); `masses` overrides them per species.
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
    ion_coulomb_logarithm::Symbol
    masses::Vector{Float64}
    ions::Union{Nothing,Symbol}
end

function IonEnergyExchangeTerm(plasma; side::Symbol, ion_energy::Symbol=:ion_thermal_energy,
                               temperature::Symbol=:cold_electron_temperature,
                               density::Symbol=:cold_electron_density, ion_ion::Bool=true,
                               ion_coulomb_logarithm::Symbol=:nrl, masses=nothing, mass_table::Symbol=:standard,
                               ions::Union{Nothing,Symbol}=nothing, initial_time::Real=0.0)
    side in (:ions, :electrons) || throw(ArgumentError("side must be :ions or :electrons, not $side"))
    ion_coulomb_logarithm in (:nrl, :dream) ||
        throw(ArgumentError("ion_coulomb_logarithm must be :nrl or :dream, not $ion_coulomb_logarithm"))
    plasma isa PlasmaSource && !(plasma isa PrescribedPlasma) &&
        throw(ArgumentError("IonEnergyExchangeTerm takes a prescribed plasma and reads the temperature, the density " *
                            "and the ion densities from unknowns of its own: give it the prescribed part and " *
                            "the keywords temperature, density and ions (issue #182, item 9)"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    species_list = plasma_values(source, nothing, nothing, initial_time).ions
    mass_values = masses === nothing ? [ion_mass(species; table=mass_table) for species in species_list] :
                  collect(Float64, masses)
    length(mass_values) == length(species_list) ||
        throw(DimensionMismatch("need one mass per ion species ($(length(species_list))), got $(length(mass_values))"))
    all(>(0), mass_values) || throw(ArgumentError("the ion masses must be positive"))
    return IonEnergyExchangeTerm(source, side, ion_energy, temperature, density, ion_ion, ion_coulomb_logarithm,
                                 mass_values, ions)
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
    _plain_value(denominator) > 0 || return zero(lnL * Wa * Wb)
    return _EXCHANGE_PREFACTOR * sqrt(ma * mb) * lnL * sqrt(Na * Nb) * nZ2a * nZ2b * (Na * Wb - Nb * Wa) /
           (denominator * sqrt(denominator))
end

# the value of a number without its dual parts: a dual of value 0 with a positive derivative
# compares greater than 0 in ForwardDiff
_plain_value(x::ForwardDiff.Dual) = _plain_value(ForwardDiff.value(x))
_plain_value(x::Real) = x

# the temperature [eV] of the species x = (N, nZ², W, m, nZ), at least the floor of a plasma
_species_temperature(x) = max(x[3] / (THERMAL_ENERGY_FACTOR * x[1]), DEFAULT_TEMPERATURE_FLOOR)

# the mixed ion-ion logarithm of the NRL Plasma Formulary (2019, p. 34) of the ion species
# a = (N, nZ², W, m, nZ) and b, each at its own temperature and with its mean charge nZ²/nZ,
# with the screening Σ_γ n_γ Z_γ²/T_γ [cm⁻³ eV⁻¹] of every species
function _nrl_ion_logarithm(a, b, screening)
    Na, nZ2a, Wa, ma, nZa = a
    Nb, nZ2b, Wb, mb, nZb = b
    Ta, Tb = _species_temperature(a), _species_temperature(b)
    μa, μb = ma / Constants.PROTON_MASS, mb / Constants.PROTON_MASS
    # a species without charged states takes the charge 1, the limit as its first ionization
    # appears (its exchange is zero there, its derivative is not)
    Za = _plain_value(nZa) > 0 ? nZ2a / nZa : one(nZ2a)
    Zb = _plain_value(nZb) > 0 ? nZ2b / nZb : one(nZ2b)
    λ = 23 - log(Za * Zb * (μa + μb) / (μa * Tb + μb * Ta) * sqrt(screening))
    # cold, dense, highly charged ions take the formula below zero: the exchange keeps its sign
    return max(λ, zero(λ))
end

# the screening of the NRL logarithm, Σ_γ n_γ Z_γ²/T_γ over the ion species and the electrons,
# in cm⁻³ eV⁻¹
_nrl_screening(ions, n_e, T_e) =
    1e-6 * (sum(x -> _plain_value(x[1]) > 0 ? x[2] / _species_temperature(x) : zero(x[2]), ions) +
            n_e / max(T_e, DEFAULT_TEMPERATURE_FLOOR))

# the power each ion species receives at radius ir, and the power the electrons lose
function _ion_exchange(term::IonEnergyExchangeTerm, plasma::PlasmaState, W, T_e, n_e, ir)
    e = Constants.ELEMENTARY_CHARGE
    n_free = free_electron_density(plasma, ir)
    lnL = thermal_coulomb_logarithm(T_e, n_free)
    lnL_dream = n_free == 0 ? zero(lnL) : 17.3 - 0.5 * log(n_free / 1e20) + 1.5 * log(T_e / 1e3)
    electrons = (n_e, n_e, 1.5 * e * n_e * T_e, Constants.ELECTRON_MASS)
    ions = map(enumerate(plasma.ions)) do (s, species)
        densities = view(species.charge_state_densities, :, ir)
        (sum(densities), sum(densities[Z0+1] * Z0^2 for Z0 in 0:species.nuclear_charge), W[s], term.masses[s],
         sum(densities[Z0+1] * Z0 for Z0 in 0:species.nuclear_charge))
    end
    received = [zero(lnL * W[s] * T_e) for s in eachindex(ions)]
    lost = zero(eltype(received))
    screening = term.ion_ion && term.ion_coulomb_logarithm === :nrl ? _nrl_screening(ions, n_e, T_e) : zero(lnL)
    for (s, ion) in enumerate(ions)
        _plain_value(ion[1]) > 0 || continue
        Q = _energy_exchange(lnL, ion[1:4], electrons)
        received[s] += Q
        lost += Q
        term.ion_ion || continue
        for (k, other) in enumerate(ions)
            # two species of neutrals alone exchange nothing
            (k == s || _plain_value(other[1]) <= 0 || (_plain_value(ion[2]) <= 0 && _plain_value(other[2]) <= 0)) && continue
            lnL_ions = term.ion_coulomb_logarithm === :nrl ? _nrl_ion_logarithm(ion, other, screening) : lnL_dream
            received[s] += _energy_exchange(lnL_ions, ion[1:4], other[1:4])
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
    set_ion_thermal_energy!(u, layout, plasma, temperature; ion_energy=:ion_thermal_energy, ions=nothing, t=0.0)

Set the thermal energy of every ion species in the state `u`, `W = 3/2 e N T` with the total
density `N` of the species in `plasma` at time `t` and the temperature `temperature` [eV]:
a number, one value per species, or a matrix of species × radii (DREAM's initial `W_i` from
`T_i`, `T_i.cpp`). With `ions`, the radial unknown of the charge states, `N` is that of the
state `u`, as the energy exchange reads it: after [`deposit_neutrals!`](@ref) the prescribed
densities no longer hold.
"""
function set_ion_thermal_energy!(u::AbstractVector, layout::UnknownLayout, plasma, temperature;
                                 ion_energy::Symbol=:ion_thermal_energy, ions::Union{Nothing,Symbol}=nothing,
                                 t::Real=0.0)
    totals = _ion_totals(u, layout, plasma, ions, t)
    id = unknown_id(layout, ion_energy)
    for ir in 1:radial_cell_count(layout), (s, r) in enumerate(unknown_range(layout, id, ir))
        T = temperature isa Real ? temperature : temperature isa AbstractVector ? temperature[s] : temperature[s, ir]
        u[r] = THERMAL_ENERGY_FACTOR * totals[s, ir] * T
    end
    return u
end

# the total density of every ion species at every radius (species × radii): of the prescribed
# plasma at time t, or with `ions` of that unknown of the state u
function _ion_totals(u, layout, plasma, ions, t)
    state = plasma isa PlasmaState ? plasma : plasma_values(plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma),
                                                            nothing, nothing, t)
    ions === nothing && return reduce(vcat, [sum(s.charge_state_densities; dims=1) for s in state.ions])
    c = IonComposition(state.ions)
    id = unknown_id(layout, ions)
    _check_ion_unknown(layout, id, c)
    return [sum(radial_view(u, layout, id, ir)[species_components(c, s)]) for s in 1:length(c),
            ir in 1:radial_cell_count(layout)]
end

"""
    ion_temperatures(u, layout, plasma; ion_energy=:ion_thermal_energy, ions=nothing, t=0.0)

The temperature [eV] of every ion species at every radius (species × radii), `T = W/(3/2 e
N)`, from the state `u`; `NaN` where a species has no density. With `ions` the densities `N`
are those of that unknown of `u`, as in [`set_ion_thermal_energy!`](@ref).
"""
function ion_temperatures(u::AbstractVector, layout::UnknownLayout, plasma; ion_energy::Symbol=:ion_thermal_energy,
                          ions::Union{Nothing,Symbol}=nothing, t::Real=0.0)
    totals = _ion_totals(u, layout, plasma, ions, t)
    id = unknown_id(layout, ion_energy)
    nr = radial_cell_count(layout)
    T = fill(NaN, size(totals, 1), nr)
    for ir in 1:nr, (s, r) in enumerate(unknown_range(layout, id, ir))
        N = totals[s, ir]
        N > 0 && (T[s, ir] = u[r] / (THERMAL_ENERGY_FACTOR * N))
    end
    return T
end

validate(term::IonEnergyExchangeTerm, layout, row) = validate(term, layout, row, 0.0)
