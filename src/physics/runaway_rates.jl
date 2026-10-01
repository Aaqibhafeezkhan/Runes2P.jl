# Fluid runaway rate formulas used as references for the kinetic operator
# and, later, as fluid source terms. Each formula states which critical momentum or
# field it uses, since the alternatives differ by factors of order one (DREAM #40).
#
# Ported from DREAM `RunawayFluid.cpp` and `ConnorHastie.cpp` (branch boltzmann-eqterm),
# MIT licensed, see THIRD_PARTY_NOTICES.md.

"""
    dreicer_field(density, temperature, coulomb_log)

Dreicer field `E_D = E_c(mₑc²/T)` [V/m] for an electron density `density` [m⁻³], a
temperature `temperature` [eV] and the thermal Coulomb logarithm `coulomb_log`, with
`E_c` the Connor-Hastie critical field ([`critical_field`](@ref)). Electrons with
`E > E_D` all run away; the Dreicer rate is a function of `E/E_D`.
"""
dreicer_field(density, temperature, coulomb_log) =
    critical_field(density, coulomb_log) * Constants.ELECTRON_REST_ENERGY / temperature

"""
    thermal_collision_time(density, temperature, coulomb_log)

Thermal electron collision time `τ_ee = β_th³ / (lnΛ_T n 4π r₀² c)` [s] with
`β_th = √(2T/mₑc²)`, the time scale of the Connor-Hastie rate (DREAM
`RunawayFluid::tauEETh`).
"""
function thermal_collision_time(density, temperature, coulomb_log)
    beta_thermal = thermal_momentum(temperature)
    return beta_thermal^3 /
           (coulomb_log * density * 4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT)
end

"""
    connor_hastie_rate(density, temperature, electric_field, effective_charge;
                       thermal_coulomb_log, relativistic_coulomb_log=thermal_coulomb_log,
                       corrections=true, undetermined_factor=1)

Connor-Hastie Dreicer runaway rate [m⁻³ s⁻¹]: the rate at which electrons cross the
critical momentum by diffusing up the electric field, for a field `electric_field` [V/m],
an electron density `density` [m⁻³], a temperature `temperature` [eV] and an effective
charge `effective_charge`. Zero for `E ≤ E_c`.

```math
\\mathrm{rate} = C \\frac{n}{\\tau_{ee}} \\left(\\frac{E}{E_D}\\right)^{-3 (1 + Z_\\mathrm{eff}) h / 16}
    \\exp\\left(-\\frac{\\lambda}{4 E / E_D} - \\sqrt{\\frac{\\eta (1 + Z_\\mathrm{eff})}{E / E_D}}\\right)
```

with `E_c` from the relativistic Coulomb logarithm (complete screening) and `E_D` and
`τ_ee` from the thermal one. `corrections = true` includes the `h`, `η` and `λ`
corrections in `E/E_c`; with `corrections = false` all three are 1. `C` is the
undetermined factor of the theory, which DREAM sets to 1 ("~1 is usually good according to
simulations"); a kinetic rate is expected to agree in scaling rather than exactly.

Ported from DREAM `ConnorHastie::RunawayRate`.
"""
function connor_hastie_rate(density, temperature, electric_field, effective_charge;
                            thermal_coulomb_log, relativistic_coulomb_log=thermal_coulomb_log,
                            corrections::Bool=true, undetermined_factor=1)
    field = abs(electric_field)
    critical = critical_field(density, relativistic_coulomb_log)
    field > critical || return zero(float(field))
    dreicer = dreicer_field(density, temperature, thermal_coulomb_log)
    collision_time = thermal_collision_time(density, temperature, thermal_coulomb_log)
    over_critical = field / critical
    over_dreicer = field / dreicer

    h, eta, lambda = 1.0, 1.0, 1.0
    if corrections
        h = 1 / (3 * (over_critical - 1)) *
            (over_critical + 2 * (over_critical - 2) * sqrt(over_critical / (over_critical - 1)) -
             (effective_charge - 7) / (effective_charge + 1))
        eta_field = pi / 2 - asin(1 - 2 / over_critical)
        eta = over_critical^2 / (4 * (over_critical - 1)) * eta_field^2
        lambda = 8 * over_critical^2 * (1 - 1 / (2 * over_critical) - sqrt(1 - 1 / over_critical))
    end
    exponent = -3 / 16 * (1 + effective_charge) * h
    return undetermined_factor * density / collision_time * over_dreicer^exponent *
           exp(-lambda / (4 * over_dreicer) - sqrt(eta * (1 + effective_charge) / over_dreicer))
end

"""
    avalanche_growth_rate(total_density, critical_momentum)

Avalanche growth rate `Γ = 4π r₀² c n_tot / p_c²` [s⁻¹] of a runaway population, the
Rosenbluth-Putvinski estimate written with an explicit critical momentum `p_c` (in units
of `mₑc`): every knock-on above `p_c` runs away, and the production per runaway is the
Møller rate above `p_c`, which is `4π r₀² c n_tot / p_c²` in the limit of an infinitely
energetic primary.

`p_c` is the ambiguous part of the formula (DREAM #40), so it is an argument, not computed
here: [`critical_momentum`](@ref) gives the test-particle value `1/√(E/E_c - 1)` used by
Rosenbluth-Putvinski, while DREAM's fluid mode uses the effective critical momentum of
Hesslow et al. 2019, which includes screening and pitch-angle scattering and is larger, so
its growth rate is lower. State which one a comparison uses.

DREAM: `RunawayFluid::CalculateGrowthRates`, `avalancheGrowthRate = n_tot constPreFactor /
p_c²` with `constPreFactor = 4π r₀² c`.
"""
avalanche_growth_rate(total_density, critical_momentum) =
    4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT * total_density /
    critical_momentum^2
