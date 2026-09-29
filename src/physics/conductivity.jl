# Electric conductivity of the thermal plasma: the conductivity of Braams
# and Karney as DREAM evaluates it (`RunawayFluid::evaluateBraamsElectricConductivity`), the
# effective charge, and DREAM's fit of the conductivity of the test-particle operator
# (`PredictedOhmicCurrentFromDistributionTerm::FractionOfBraamsConductivity`).

# The normalized conductivity σ̄(Θ, X) of a relativistic plasma, with Θ = T/mₑc² and
# X = 1/(1 + Z_eff), as tabulated in DREAM's `RunawayFluid.cpp` (`conductivityBraams`, after
# Braams and Karney, Phys. Fluids B 1, 1355 (1989)). Rows: X, columns: Θ.
const BRAAMS_KARNEY_TEMPERATURES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
const BRAAMS_KARNEY_CHARGES = (0.0, 0.090909090909091, 0.166666666666667, 0.333333333333333, 0.5, 1.0)
const BRAAMS_KARNEY_TABLE = [
    12.7661 12.2972 11.8737 10.8120 9.5075 7.8269 5.4760 3.9694 2.8247 1.7887 1.2649 0.8944 0.5657 0.4000
    11.3301 10.9587 10.6195 9.7540 8.6631 7.2156 5.1138 3.7221 2.6483 1.6738 1.1826 0.8359 0.5286 0.3738
    10.3912 10.0778 9.7896 9.0462 8.0936 6.8043 4.8805 3.5730 2.5484 1.6116 1.1386 0.8047 0.5089 0.3598
    8.7546 8.5328 8.3265 7.7844 7.0689 6.0624 4.4724 3.3261 2.3920 1.5180 1.0731 0.7585 0.4797 0.3392
    7.4290 7.2736 7.1277 6.7381 6.2095 5.4367 4.1373 3.1347 2.2786 1.4538 1.0288 0.7274 0.4600 0.3253
    3.7599 3.7549 3.7492 3.7285 3.6842 3.5713 3.1821 2.6501 2.0313 1.3301 0.9465 0.6704 0.4242 0.3000
]

# the interval of a sorted tuple that holds x, as the index of its lower end
function _table_interval(points, x)
    first(points) <= x <= last(points) ||
        throw(DomainError(x, "outside the table of the conductivity ($(first(points)) to $(last(points)))"))
    i = 1
    while i < length(points) - 1 && x > points[i+1]
        i += 1
    end
    return i
end

"""
    normalized_conductivity(theta, x)

The normalized conductivity `σ̄(Θ, X)` of Braams and Karney, with `Θ = T/mₑc²` and
`X = 1/(1 + Z_eff)`, interpolated bilinearly in DREAM's table (14 temperatures from 0 to
100, 6 charges from `Z_eff = ∞` to 0). `σ̄(0, 0) = 32/√(2π)` is the Lorentz gas, and
`σ̄(0, 1/2)/σ̄(0, 0) = 0.5819` the factor of Spitzer and Härm at `Z = 1`. Between `Θ = 0`
and `Θ = 0.01` (5.1 keV) the table is linear in the temperature.
"""
function normalized_conductivity(theta, x)
    i = _table_interval(BRAAMS_KARNEY_TEMPERATURES, theta)
    j = _table_interval(BRAAMS_KARNEY_CHARGES, x)
    t = (theta - BRAAMS_KARNEY_TEMPERATURES[i]) / (BRAAMS_KARNEY_TEMPERATURES[i+1] - BRAAMS_KARNEY_TEMPERATURES[i])
    u = (x - BRAAMS_KARNEY_CHARGES[j]) / (BRAAMS_KARNEY_CHARGES[j+1] - BRAAMS_KARNEY_CHARGES[j])
    return (1 - t) * (1 - u) * BRAAMS_KARNEY_TABLE[j, i] + t * (1 - u) * BRAAMS_KARNEY_TABLE[j, i+1] +
           (1 - t) * u * BRAAMS_KARNEY_TABLE[j+1, i] + t * u * BRAAMS_KARNEY_TABLE[j+1, i+1]
end

"""
    effective_charge(plasma, ir)
    effective_charge(ions, ir)

Effective charge `Z_eff = Σ Z₀² n / Σ Z₀ n` over the charge states of all ions at the
radius index `ir`, weighted with the free electrons, and at least 1 (DREAM
`IonHandler::GetZeff`). The bound electrons of partially ionized ions do not enter. 1 for
a plasma without free electrons.
"""
function effective_charge(ions, ir::Integer)
    free = free_electron_density(ions, ir)
    free == 0 && return one(float(typeof(free)))
    squares = sum((Z0^2 * s.charge_state_densities[Z0+1, ir] for s in ions for Z0 in 0:s.nuclear_charge);
                  init=zero(free))
    return max(squares / free, one(squares / free))
end
effective_charge(plasma::PlasmaState, ir::Integer) = effective_charge(plasma.ions, ir)

"""
    braams_karney_conductivity(temperature, effective_charge, free_electron_density)

Electric conductivity [S/m] of a relativistic plasma after Braams and Karney,

    σ = 4π ε₀² T^{3/2} / (Z_eff √mₑ e² lnΛ) σ̄(T/mₑc², 1/(1 + Z_eff)),

with the temperature in eV (in joules in the formula), the thermal Coulomb logarithm
([`thermal_coulomb_logarithm`](@ref)) at the free-electron density, and the table of
[`normalized_conductivity`](@ref). This is DREAM's `evaluateBraamsElectricConductivity`. In
cylindrical geometry DREAM's conductivity of Sauter and Redl equals it, since the
neoclassical correction is 1 without trapped particles. At low temperature and high charge
it is the Lorentz conductivity `64 √(2π) ε₀² T^{3/2} / (√mₑ e² Z lnΛ)`.
"""
function braams_karney_conductivity(temperature, effective_charge, free_electron_density)
    temperature > 0 || throw(DomainError(temperature, "the temperature must be positive"))
    effective_charge > 0 || throw(DomainError(effective_charge, "the effective charge must be positive"))
    energy = temperature * Constants.ELEMENTARY_CHARGE
    theta = energy / (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT^2)
    normalized = normalized_conductivity(theta, 1 / (1 + effective_charge))
    logarithm = thermal_coulomb_logarithm(temperature, free_electron_density)
    return 4π * Constants.VACUUM_PERMITTIVITY^2 * energy * sqrt(energy) /
           (effective_charge * sqrt(Constants.ELECTRON_MASS) * Constants.ELEMENTARY_CHARGE^2 * logarithm) * normalized
end

"""
    braams_karney_conductivity(plasma, ir)

The conductivity of Braams and Karney at the radius index `ir` of `plasma`, with its cold
temperature, its effective charge and its free-electron density.
"""
braams_karney_conductivity(plasma::PlasmaState, ir::Integer) =
    braams_karney_conductivity(plasma.cold_electron_temperature[ir], effective_charge(plasma, ir),
                               free_electron_density(plasma, ir))

"""
    test_particle_conductivity_fraction(effective_charge)

DREAM's fit of the conductivity of its kinetic operator over the conductivity of Braams
and Karney, `1 − 1.406/(1.888 + Z_eff)`
(`PredictedOhmicCurrentFromDistributionTerm::FractionOfBraamsConductivity`). The kinetic
operator is the test-particle operator, which does not return the momentum of the
electron-electron collisions, and the fit is to conductivities of CODE, with a largest
error of 1 % as stated in DREAM. It does not know the grid, the collision settings or the
temperature: 0.513, 0.638, 0.761 and 0.921 at `Z_eff` = 1, 2, 4 and 16.
"""
test_particle_conductivity_fraction(effective_charge) = 1 - 1.406 / (1.888 + effective_charge)
