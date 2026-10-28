# Coulomb logarithms following DREAM `CoulombLogarithm.cpp` (Hesslow et al.,
# J. Plasma Phys. 84 (2018), eqs. 2.7-2.10).

const COULOMB_LOGARITHM_INTERPOLATION_EXPONENT = 5  # between thermal and relativistic limits

"""
    thermal_coulomb_logarithm(temperature, free_electron_density)

Thermal Coulomb logarithm `14.9 - ½ ln(n/10²⁰) + ln(T/keV)` for the temperature
`temperature` (`T`) [eV] and the free-electron density `free_electron_density` (`n`) [m⁻³],
zero for a vanishing density.
"""
function thermal_coulomb_logarithm(temperature, free_electron_density)
    free_electron_density == 0 ? zero(temperature) :
    14.9 - 0.5 * log(free_electron_density / 1e20) + log(temperature / 1e3)
end

"""
    relativistic_coulomb_logarithm(temperature, free_electron_density)

Relativistic Coulomb logarithm `14.6 + ½ ln(T/(n/10²⁰))` (DREAM `lnLambda_c`) for the
temperature `temperature` (`T`) [eV] and the free-electron density `free_electron_density`
(`n`) [m⁻³], zero for a vanishing density.
"""
function relativistic_coulomb_logarithm(temperature, free_electron_density)
    free_electron_density == 0 ? zero(temperature) :
    14.6 + 0.5 * log(temperature / (free_electron_density / 1e20))
end

"""
    coulomb_logarithm(collision, model, p, temperature, free_electron_density)

Coulomb logarithm for `:electron_electron` or `:electron_ion` collisions at momentum `p`
(units of mₑc), the temperature `temperature` [eV] and the free-electron density
`free_electron_density` [m⁻³]. `model` is `:energy_dependent`, `:thermal` or
`:relativistic` (the constant relativistic value, DREAM's `CONSTANT`).
"""
function coulomb_logarithm(collision::Symbol, model::Symbol, p, temperature,
        free_electron_density)
    model === :relativistic &&
        return relativistic_coulomb_logarithm(temperature, free_electron_density)
    thermal = thermal_coulomb_logarithm(temperature, free_electron_density)
    model === :thermal && return thermal
    model === :energy_dependent ||
        throw(ArgumentError("unknown Coulomb logarithm model $model"))
    # the thermal value at p = 0, the limit of the formula below: its square root of zero has
    # no derivative when p carries one (the dual numbers of the plasma columns at the face p = 0)
    _value(p) == 0 && return thermal
    thermal_p = sqrt(2 * temperature / Constants.ELECTRON_REST_ENERGY)
    x = if collision === :electron_electron
        gamma_minus_one = p^2 / (sqrt(1 + p^2) + 1)
        sqrt(2 * gamma_minus_one) / thermal_p
    elseif collision === :electron_ion
        2p / thermal_p
    else
        throw(ArgumentError("collision must be :electron_electron or :electron_ion"))
    end
    k = COULOMB_LOGARITHM_INTERPOLATION_EXPONENT
    return thermal + log(1 + x^k) / k
end
