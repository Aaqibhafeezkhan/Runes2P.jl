# Fluid runaway rate formulas used as references for the kinetic operator and as fluid
# source terms. Each formula states which critical momentum or field it uses, since the
# alternatives differ by factors of order one.
#
# Ported from DREAM `RunawayFluid.cpp` and `ConnorHastie.cpp` (branch boltzmann-eqterm),
# MIT licensed, see THIRD_PARTY_NOTICES.md.

"""
    dreicer_field(density, temperature, coulomb_logarithm)

Dreicer field `E_D = E_c(mₑc²/T)` [V/m] for an electron density `density` [m⁻³], a
temperature `temperature` [eV] and the thermal Coulomb logarithm `coulomb_logarithm`, with
`E_c` the Connor-Hastie critical field ([`critical_field`](@ref)). Electrons with
`E > E_D` all run away; the Dreicer rate is a function of `E/E_D`.
"""
function dreicer_field(density, temperature, coulomb_logarithm)
    critical_field(density, coulomb_logarithm) * Constants.ELECTRON_REST_ENERGY /
    temperature
end

"""
    thermal_collision_time(density, temperature, coulomb_logarithm)

Thermal electron collision time `τ_ee = β_th³ / (lnΛ_T n 4π r₀² c)` [s] with
`β_th = √(2T/mₑc²)`, the time scale of the Connor-Hastie rate (DREAM
`RunawayFluid::tauEETh`).
"""
function thermal_collision_time(density, temperature, coulomb_logarithm)
    beta_thermal = thermal_momentum(temperature)
    return beta_thermal^3 /
           (coulomb_logarithm * density * 4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 *
            Constants.SPEED_OF_LIGHT)
end

"""
    connor_hastie_rate(density, temperature, electric_field, effective_charge;
                       thermal_coulomb_logarithm,
                       relativistic_coulomb_logarithm=thermal_coulomb_logarithm,
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
        thermal_coulomb_logarithm, relativistic_coulomb_logarithm = thermal_coulomb_logarithm,
        corrections::Bool = true, undetermined_factor = 1)
    field = abs(electric_field)
    critical = critical_field(density, relativistic_coulomb_logarithm)
    field > critical || return zero(float(field))
    dreicer = dreicer_field(density, temperature, thermal_coulomb_logarithm)
    collision_time = thermal_collision_time(density, temperature, thermal_coulomb_logarithm)
    over_critical = field / critical
    over_dreicer = field / dreicer

    h, eta, lambda = 1.0, 1.0, 1.0
    if corrections
        h = 1 / (3 * (over_critical - 1)) *
            (over_critical +
             2 * (over_critical - 2) * sqrt(over_critical / (over_critical - 1)) -
             (effective_charge - 7) / (effective_charge + 1))
        eta_field = pi / 2 - asin(1 - 2 / over_critical)
        eta = over_critical^2 / (4 * (over_critical - 1)) * eta_field^2
        lambda = 8 * over_critical^2 *
                 (1 - 1 / (2 * over_critical) - sqrt(1 - 1 / over_critical))
    end
    exponent = -3 / 16 * (1 + effective_charge) * h
    return undetermined_factor * density / collision_time * over_dreicer^exponent *
           exp(-lambda / (4 * over_dreicer) -
               sqrt(eta * (1 + effective_charge) / over_dreicer))
end

"""
    avalanche_growth_rate(total_density, critical_momentum)

Avalanche growth rate `Γ = 4π r₀² c n_tot / p_c²` [s⁻¹] of a runaway population, the
Rosenbluth-Putvinski estimate written with an explicit critical momentum `p_c` (in units
of `mₑc`): every knock-on above `p_c` runs away, and the production per runaway is the
Møller rate above `p_c`. For an infinitely energetic primary that rate is
`2π r₀² c n_tot / (γ_c − 1) = (γ_c + 1)/2 · 4π r₀² c n_tot / p_c²`
([`moller_total_rate`](@ref) with `p1 = Inf`; Embréus, Stahl and Fülöp, J. Plasma Phys. 84,
905840102 (2018), doi:10.1017/S002237781700099X), so the formula keeps its leading term
for `p_c ≪ 1`; at `p_c = 1` the exact rate is 21 % above the formula.

`p_c` is the ambiguous part of the formula, so it is an argument, not computed here:
[`critical_momentum`](@ref) gives the test-particle value `1/√(E/E_c - 1)` used by
Rosenbluth-Putvinski, while DREAM's fluid mode uses the effective critical momentum of
Hesslow et al. 2019, which includes screening and pitch-angle scattering and is larger, so
its growth rate is lower. State which one a comparison uses.

DREAM: `avalancheGrowthRate = n_tot constPreFactor / p_c²` (`src/Equations/RunawayFluid.cpp:353`)
with `constPreFactor = 4π r₀² c` (`include/DREAM/Equations/RunawayFluid.hpp:25-26`).
"""
function avalanche_growth_rate(total_density, critical_momentum)
    4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT * total_density /
    critical_momentum^2
end

# critical momenta of a plasma state

"""
    free_critical_momentum(plasma, ir)

Critical momentum at radius `ir` from the Connor-Hastie critical field of the cold
electrons with the relativistic Coulomb logarithm (DREAM's `Ec_free`,
`RunawayFluid.cpp`), [`critical_momentum`](@ref) of `|E|`. Screening and pitch-angle
scattering raise the critical momentum, so this is a lower estimate.

The Coulomb logarithm takes the free electron density, the cold, hot and runaway
electrons together, as DREAM's `lnLambdaC` does (agreement to 1e-8).
Before the thermal quench of a fully kinetic run most electrons are hot, and the cold
density alone overstates the logarithm.
"""
function free_critical_momentum(plasma::PlasmaState, ir::Integer)
    logarithm = relativistic_coulomb_logarithm(plasma.cold_electron_temperature[ir],
        free_electron_density(plasma, ir))
    field = critical_field(plasma.cold_electron_density[ir], logarithm)
    return critical_momentum(abs(plasma.electric_field[ir]), field)
end

"""
    total_critical_momentum(plasma, ir)

Critical momentum at radius `ir` from the Connor-Hastie critical field of all electrons,
free and bound, with the relativistic Coulomb logarithm of the free electrons (DREAM's
`Ec_tot`, `RunawayFluid.cpp`), [`critical_momentum`](@ref) of `|E|`. `Inf` where the
field is below that critical field.

It is not DREAM's effective critical momentum (`other/fluid/pCrit`, Hesslow et al.), which
takes the partial screening of the ions into account in the slowing down and the
pitch-angle scattering of the fast electrons. The free critical field ([`free_critical_momentum`](@ref)) can give a threshold
in a hot plasma where the effective one has none, below the thermal momentum. Runes2P has DREAM's effective critical field and
`p⋆` ([`effective_critical_field`](@ref), [`avalanche_critical_momentum`](@ref)), from which
`pCrit = 1/√((Ê − Ê_ceff)/(p⋆² Ê_eff))` follows in DREAM's FLUID and KINETIC avalanche modes
(`RunawayFluid.cpp:647-653`; not in `FLUID_HESSLOW`, whose `p⋆` differs); no function returns
it, and it is not the default of `critical_momentum_at`.
"""
function total_critical_momentum(plasma::PlasmaState, ir::Integer)
    free = free_electron_density(plasma, ir)
    logarithm = relativistic_coulomb_logarithm(plasma.cold_electron_temperature[ir], free)
    field = critical_field(free + bound_electron_density(plasma, ir), logarithm)
    return critical_momentum(abs(plasma.electric_field[ir]), field)
end
