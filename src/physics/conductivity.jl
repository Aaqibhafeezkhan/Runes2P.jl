# Electric conductivity of the thermal plasma: the conductivity of Braams
# and Karney as DREAM evaluates it (`RunawayFluid::evaluateBraamsElectricConductivity`), the
# effective charge, and DREAM's fit of the conductivity of the test-particle operator
# (`PredictedOhmicCurrentFromDistributionTerm::FractionOfBraamsConductivity`).

# the parsed table of data/braams_karney_conductivity.toml, read once
const _BRAAMS_KARNEY_DATA = read_data_file("braams_karney_conductivity")["data"]

"""
    BRAAMS_KARNEY_TEMPERATURES

The temperatures Θ = T/mₑc² of the columns of `BRAAMS_KARNEY_TABLE`, as tabulated in
DREAM's `RunawayFluid.cpp` (`conductivityTmc2`). Read from
`data/braams_karney_conductivity.toml`.
"""
const BRAAMS_KARNEY_TEMPERATURES = Tuple(float_vector(_BRAAMS_KARNEY_DATA["temperatures"]))

"""
    BRAAMS_KARNEY_CHARGES

The values X = 1/(1 + Z_eff) of the rows of `BRAAMS_KARNEY_TABLE`, as tabulated in
DREAM's `RunawayFluid.cpp` (`conductivityX`). Read from
`data/braams_karney_conductivity.toml`.
"""
const BRAAMS_KARNEY_CHARGES = Tuple(float_vector(_BRAAMS_KARNEY_DATA["charges"]))

"""
    BRAAMS_KARNEY_TABLE

The normalized conductivity σ̄(Θ, X) of a relativistic plasma, with Θ = T/mₑc² and
X = 1/(1 + Z_eff), of B. J. Braams and C. F. F. Karney, Phys. Fluids B 1, 1355 (1989), as
tabulated in DREAM's `RunawayFluid.cpp` (`conductivityBraams`). Rows: X
(`BRAAMS_KARNEY_CHARGES`), columns: Θ (`BRAAMS_KARNEY_TEMPERATURES`). Read from
`data/braams_karney_conductivity.toml`.
"""
const BRAAMS_KARNEY_TABLE = float_matrix(_BRAAMS_KARNEY_DATA["table"])

# the interval of a sorted tuple that holds x, as the index of its lower end
function _table_interval(points, x)
    first(points) <= x <= last(points) ||
        throw(DomainError(x, "outside the table of the conductivity ($(first(points)) to $(last(points)))"))
    i = 1
    while i < length(points) - 1 && x > points[i + 1]
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
    t = (theta - BRAAMS_KARNEY_TEMPERATURES[i]) /
        (BRAAMS_KARNEY_TEMPERATURES[i + 1] - BRAAMS_KARNEY_TEMPERATURES[i])
    u = (x - BRAAMS_KARNEY_CHARGES[j]) /
        (BRAAMS_KARNEY_CHARGES[j + 1] - BRAAMS_KARNEY_CHARGES[j])
    return (1 - t) * (1 - u) * BRAAMS_KARNEY_TABLE[j, i] +
           t * (1 - u) * BRAAMS_KARNEY_TABLE[j, i + 1] +
           (1 - t) * u * BRAAMS_KARNEY_TABLE[j + 1, i] +
           t * u * BRAAMS_KARNEY_TABLE[j + 1, i + 1]
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
    squares = zero(free)
    for s in ions
        for Z0 in 0:s.nuclear_charge
            squares += Z0^2 * s.charge_state_densities[Z0 + 1, ir]
        end
    end
    return max(squares / free, one(squares / free))
end
effective_charge(plasma::PlasmaState, ir::Integer) = effective_charge(plasma.ions, ir)

"""
    braams_karney_conductivity(free_electron_density, temperature, effective_charge)

Electric conductivity [S/m] of a relativistic plasma after Braams and Karney,

```math
\\sigma = \\frac{4\\pi \\varepsilon_0^2 T^{3/2}}{Z_\\mathrm{eff} \\sqrt{m_e} \\, e^2 \\ln\\Lambda} \\,
    \\bar\\sigma\\left(\\frac{T}{m_e c^2}, \\frac{1}{1 + Z_\\mathrm{eff}}\\right),
```

with the temperature in eV (in joules in the formula), the thermal Coulomb logarithm
([`thermal_coulomb_logarithm`](@ref)) at the free-electron density, and the table of
[`normalized_conductivity`](@ref). This is DREAM's `evaluateBraamsElectricConductivity`. In
cylindrical geometry DREAM's conductivity of Sauter and Redl equals it, since the
neoclassical correction is 1 without trapped particles. At low temperature and high charge
it is the Lorentz conductivity
``64 \\sqrt{2\\pi} \\, \\varepsilon_0^2 T^{3/2} / (\\sqrt{m_e} \\, e^2 Z \\ln\\Lambda)``.
"""
function braams_karney_conductivity(free_electron_density, temperature, effective_charge)
    temperature > 0 || throw(DomainError(temperature, "the temperature must be positive"))
    effective_charge > 0 ||
        throw(DomainError(effective_charge, "the effective charge must be positive"))
    energy = temperature * Constants.ELEMENTARY_CHARGE
    theta = energy / (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT^2)
    normalized = normalized_conductivity(theta, 1 / (1 + effective_charge))
    logarithm = thermal_coulomb_logarithm(free_electron_density, temperature)
    return 4π * Constants.VACUUM_PERMITTIVITY^2 * energy * sqrt(energy) /
           (effective_charge * sqrt(Constants.ELECTRON_MASS) *
            Constants.ELEMENTARY_CHARGE^2 * logarithm) * normalized
end

"""
    braams_karney_conductivity(plasma, ir)

The conductivity of Braams and Karney at the radius index `ir` of `plasma`, with its
free-electron density, its cold temperature and its effective charge.
"""
function braams_karney_conductivity(plasma::PlasmaState, ir::Integer)
    braams_karney_conductivity(
        free_electron_density(plasma, ir), plasma.cold_electron_temperature[ir],
        effective_charge(plasma, ir))
end

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
function test_particle_conductivity_fraction(effective_charge)
    1 - 1.406 / (1.888 + effective_charge)
end
