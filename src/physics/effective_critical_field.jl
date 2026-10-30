# DREAM's fluid avalanche growth rate with the effective critical field: the effective
# critical field of Hesslow et al. in DREAM's mode FULL
# (src/Equations/EffectiveCriticalField.cpp), the critical momentum p⋆ of the avalanche
# (src/Equations/RunawayFluid.cpp, pStarFunction) and the growth rate, in a cylinder. The
# iterations run on the values; one Newton correction in dual numbers then gives the exact
# first derivatives with respect to every partial the plasma, the field or the density carry.

import ForwardDiff

"""
    runaway_fluid_settings(settings; role)

The collision settings of DREAM's `RunawayFluid` (`src/Settings/RunawayFluid.cpp:27-39`),
which take only the screening, the Coulomb logarithm and the tables of `settings`: the
superthermal mode always, and for `role = :critical_field` the energy-dependent Coulomb
logarithm and the bremsstrahlung stopping power, for `role = :critical_momentum` the
Coulomb logarithm of `settings` and no bremsstrahlung. The collision settings of the kinetic
equation (the mode `:full`, the bremsstrahlung of `f`) therefore do not reach the growth rate.
"""
function runaway_fluid_settings(s::CollisionSettings; role::Symbol)
    role in (:critical_field, :critical_momentum) ||
        throw(ArgumentError("role must be :critical_field or :critical_momentum, not $role"))
    field = role === :critical_field
    return CollisionSettings(; mode = :superthermal, screening = s.screening,
        coulomb_logarithm = field ? :energy_dependent : s.coulomb_logarithm,
        screened_diffusion = s.screened_diffusion, ion_size_table = s.ion_size_table,
        screening_window = s.screening_window, screening_window_start = s.screening_window_start,
        screening_window_end = s.screening_window_end,
        bremsstrahlung = field ? :stopping_power : :neglect,
        bremsstrahlung_constant = s.bremsstrahlung_constant)
end

"""
    SYNCHROTRON_RATE_PER_FIELD_SQUARED

`e⁴/(6π ε₀ mₑ³ c³)` [s⁻¹ T⁻²]: the inverse synchrotron time `τ_s⁻¹ = e⁴B²/(6π ε₀ mₑ³ c³)`
per square of the magnetic field, about 0.194 s⁻¹ T⁻² (DREAM's `EffectiveCriticalField.cpp:44-52`;
the radiation reaction of the Abraham-Lorentz-Dirac force, L. Hesslow et al., Plasma Phys.
Control. Fusion 60, 074010 (2018), doi:10.1088/1361-6587/aac33e, eq. (5)).
"""
const SYNCHROTRON_RATE_PER_FIELD_SQUARED = Constants.ELEMENTARY_CHARGE^4 /
                                           (6π * Constants.VACUUM_PERMITTIVITY *
                                            Constants.ELECTRON_MASS^3 *
                                            Constants.SPEED_OF_LIGHT^3)

# the field in units of a frequency, e|E|/(mₑc) [s⁻¹]
function _field_frequency(E)
    Constants.ELEMENTARY_CHARGE * abs(E) /
    (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT)
end

# the Langevin function L(A) = coth A − 1/A, the mean of ξ₀ in the pitch distribution exp(−A(1 − ξ₀)),
# and L(A)/A; series below A = 1e-2, where the difference cancels
function _langevin(A)
    if A < 1e-2
        return A / 3 - A^3 / 45 + 2A^5 / 945
    else
        return coth(A) - 1 / A
    end
end
function _langevin_over(A)
    if A < 1e-2
        return 1 / 3 - A^2 / 45 + 2A^4 / 945
    else
        return (coth(A) - 1 / A) / A
    end
end

"""
    critical_field_advection(model, ir, p, field_frequency)

The momentum advection of a runaway electron averaged over its pitch distribution, in s⁻¹
(units of `mₑc` per second), which defines DREAM's effective critical field in a cylinder
(mode FULL, `EffectiveCriticalField.cpp:360-372`, Hesslow et al. 2018, eq. (21)):

```math
U(p) = \\hat E ⟨ξ_0⟩_A − p ν_s(p) − p γ τ_s^{-1} ⟨1 − ξ_0^2⟩_A, \\qquad
A = \\frac{2 \\hat E}{p ν_D(p)},
```

with `Ê = e|E|/(mₑc)`, the slowing-down and deflection frequencies of `model` (superthermal,
with the bremsstrahlung stopping, [`runaway_fluid_settings`](@ref)), and the synchrotron time of
the magnetic field of the plasma ([`SYNCHROTRON_RATE_PER_FIELD_SQUARED`](@ref)). In a cylinder
the pitch distribution `exp(−A(1 − ξ₀))` gives `⟨ξ₀⟩ = coth A − 1/A` and `⟨1 − ξ₀²⟩ = 2⟨ξ₀⟩/A`
in closed form, where DREAM interpolates both on 50 points
(`REPitchDistributionAveragedBACoeff.cpp`), with differences of about 1e-3.
"""
function critical_field_advection(model::CollisionModel, ir, p, field_frequency)
    nu_s = slowing_down_frequency(model, ir, p)
    nu_d = deflection_frequency(model, ir, p)
    A = 2field_frequency / (p * nu_d)
    tau_inverse = SYNCHROTRON_RATE_PER_FIELD_SQUARED * model.plasma.magnetic_field[ir]^2
    return field_frequency * _langevin(A) - p * nu_s -
           p * sqrt(1 + p^2) * tau_inverse * 2 * _langevin_over(A)
end

# the maximum over p of the advection U at the field frequency Ê, and its momentum: a scan
# of log p from 1e-3 to 1e3, then a golden section around the best point; the momentum is
# capped at 1000 as in DREAM (EffectiveCriticalField.cpp:268-271). The search is global, where
# DREAM brackets a local extremum warm-started from its last one (EffectiveCriticalField.cpp:
# 309-337): where U(p) has two maxima the two codes can take different ones.
function _advection_maximum(model, ir, field_frequency)
    U(x) = critical_field_advection(model, ir, exp(x), field_frequency)
    xs = range(log(1e-3), log(1e3); length = 61)
    values = U.(xs)
    k = argmax(values)
    k == length(xs) && return values[k], exp(xs[k])
    a, b = xs[max(k - 1, 1)], xs[min(k + 1, length(xs))]
    golden = (sqrt(5) - 1) / 2
    c, d = b - golden * (b - a), a + golden * (b - a)
    Uc, Ud = U(c), U(d)
    while b - a > 1e-10
        if Uc > Ud
            b, d, Ud = d, c, Uc
            c = b - golden * (b - a)
            Uc = U(c)
        else
            a, c, Uc = c, d, Ud
            d = a + golden * (b - a)
            Ud = U(d)
        end
    end
    x = (a + b) / 2
    return U(x), exp(x)
end

# the root of the increasing function g on [a, b] with g(a) < 0 < g(b), by the Illinois
# variant of the regula falsi, to the relative tolerance tol of the argument
function _illinois(g, a, b, ga, gb; tol = 1e-13, iterations = 200)
    side = 0
    for _ in 1:iterations
        c = (a * gb - b * ga) / (gb - ga)
        gc = g(c)
        (gc == 0 || abs(b - a) <= tol * abs(c)) && return c
        if gc * gb > 0
            b, gb = c, gc
            if side == -1
                ga /= 2
            end
            side = -1
        else
            a, ga = c, gc
            if side == +1
                gb /= 2
            end
            side = +1
        end
    end
    return (a + b) / 2
end

# the values of a plasma state in dual numbers
function _plain_plasma(plasma::PlasmaState{<:ForwardDiff.Dual})
    PlasmaState(;
        cold_electron_temperature = ForwardDiff.value.(plasma.cold_electron_temperature),
        cold_electron_density = ForwardDiff.value.(plasma.cold_electron_density),
        ions = [IonSpecies(s.name, s.nuclear_charge, ForwardDiff.value.(s.charge_state_densities))
                for s in plasma.ions],
        electric_field = ForwardDiff.value.(plasma.electric_field),
        magnetic_field = ForwardDiff.value.(plasma.magnetic_field))
end
_plain_plasma(plasma::PlasmaState) = plasma

"""
    effective_critical_field(plasma, ir; settings=CollisionSettings())

DREAM's effective critical field [V/m] at radius `ir` of `plasma` in mode FULL in a
cylinder: the field `|E|` at which the largest momentum advection over `p`,
[`critical_field_advection`](@ref), is zero (Hesslow et al., Plasma Phys. Control. Fusion
60, 074010 (2018), doi:10.1088/1361-6587/aac33e, eq. (21); DREAM's `EffectiveCriticalField.cpp:124-169`
and `249-372`). Above it some electrons gain momentum at every pitch of their distribution.
The collisions are those of [`runaway_fluid_settings`](@ref)`(settings; role =
:critical_field)`: superthermal, partially screened as `settings`, with the bremsstrahlung
stopping and the synchrotron loss of the magnetic field of the plasma. DREAM solves to a
relative 2e-3 in the field and 1e-3 in the momentum with a secant step warm-started from the
last call; this solves to 1e-13 and 1e-10. Returns the field and the momentum of the largest
advection. A `plasma` in dual numbers ([`dual_plasma`](@ref)) gives the field with the exact
derivatives: the envelope of the maximum, `dE_ceff/dx = −(∂U/∂x)/(∂U/∂E)` at that momentum.
"""
function effective_critical_field(
        plasma::PlasmaState, ir::Integer; settings::CollisionSettings = CollisionSettings())
    s = runaway_fluid_settings(settings; role = :critical_field)
    dual = if eltype(plasma.cold_electron_temperature) <: ForwardDiff.Dual
        CollisionModel(plasma, s)
    else
        nothing
    end
    return effective_critical_field(CollisionModel(_plain_plasma(plasma), s), ir; dual)
end

"""
    effective_critical_field(model::CollisionModel, ir; dual=nothing)

The effective critical field of the collisions of `model` as they are, without the settings of
[`runaway_fluid_settings`](@ref): the method behind the one of a plasma, and the way to the
limits with other collisions (a constant Coulomb logarithm in the tests). `dual`, a model of
the same plasma in dual numbers, gives the derivatives.
"""
function effective_critical_field(plain::CollisionModel, ir::Integer; dual = nothing)
    plasma = plain.plasma
    g(w) = first(_advection_maximum(plain, ir, w))
    # bracket from the Connor-Hastie field of all electrons, as DREAM's first guess
    guess = _field_frequency(_connor_hastie_total_field(plasma, ir))
    lower, upper = 0.0, max(guess, 1e-30)
    g_lower = g(lower + 1e-300)
    g_upper = g(upper)
    while g_upper <= 0
        lower, g_lower = upper, g_upper
        upper *= 2
        g_upper = g(upper)
    end
    w = _illinois(g, lower, upper, g_lower, g_upper)
    p_opt = last(_advection_maximum(plain, ir, w))
    if dual !== nothing
        # one Newton correction in the partials of the plasma at the optimum
        slope = ForwardDiff.derivative(v -> critical_field_advection(plain, ir, p_opt, v), w)
        w = w - critical_field_advection(dual, ir, p_opt, w) / slope
    end
    #! format: off
    return w * Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT / Constants.ELEMENTARY_CHARGE, p_opt
    #! format: on
end

# the Connor-Hastie field of all electrons, free and bound, with the relativistic Coulomb
# logarithm of the free electrons (DREAM's Ec_tot, RunawayFluid.cpp:200-210), also where
# DREAM takes the thermal logarithm (lines 205-207; a choice of this package, see the docs
# page Choices of this package); it only brackets the root of effective_critical_field
function _connor_hastie_total_field(plasma::PlasmaState, ir)
    free = free_electron_density(plasma, ir)
    logarithm = relativistic_coulomb_logarithm(free, plasma.cold_electron_temperature[ir])
    return critical_field(free + bound_electron_density(plasma, ir), logarithm)
end

"""
    avalanche_critical_momentum(plasma, ir, field_frequency; settings=CollisionSettings())

DREAM's critical momentum `p⋆` of the avalanche in the kinetic and fluid modes
(`RunawayFluid.cpp:536-589`, the branch of `pStarFunction`; L. Hesslow et al., Nucl. Fusion
59, 084004 (2019), doi:10.1088/1741-4326/ab26c2): the root of

```math
p^2 \\hat E_\\mathrm{eff} = \\sqrt{\\hat ν_s(p) (\\hat ν_D(p) + 4 \\hat ν_s(p))}, \\qquad
\\hat ν_s = \\frac{p^3 ν_s}{γ^2}, \\quad \\hat ν_D = \\frac{p^3 ν_D}{γ},
```

with `Ê_eff = e max(|E|, E_ceff)/(mₑc)` (`field_frequency`) and the collisions of
[`runaway_fluid_settings`](@ref)`(settings; role = :critical_momentum)` (no bremsstrahlung),
in a cylinder, where the passing fraction is 1. Solved to 1e-12 relative (DREAM: 1e-3); with
dual numbers in the plasma or the field the derivatives follow from one Newton correction.
"""
function avalanche_critical_momentum(plasma::PlasmaState, ir::Integer, field_frequency;
        settings::CollisionSettings = CollisionSettings())
    s = runaway_fluid_settings(settings; role = :critical_momentum)
    dual = eltype(plasma.cold_electron_temperature) <: ForwardDiff.Dual ||
           field_frequency isa ForwardDiff.Dual
    return avalanche_critical_momentum(
        CollisionModel(_plain_plasma(plasma), s), ir, field_frequency;
        dual = dual ? CollisionModel(plasma, s) : nothing)
end

"""
    avalanche_critical_momentum(model::CollisionModel, ir, field_frequency; dual=nothing)

`p⋆` of the collisions of `model` as they are, without [`runaway_fluid_settings`](@ref): the
method behind the one of a plasma. `dual`, a model of the same plasma in dual numbers, or a
field frequency in dual numbers, gives the derivatives.
"""
function avalanche_critical_momentum(plain::CollisionModel, ir::Integer, field_frequency; dual = nothing)
    function G(model, p, w)
        q = _collision_hats(model, ir, p)
        return sqrt(sqrt(q.s * (q.d + 4q.s))) / sqrt(w) - p
    end
    w0 = if field_frequency isa ForwardDiff.Dual
        ForwardDiff.value(field_frequency)
    else
        field_frequency
    end
    g(p) = -G(plain, p, w0)     # increasing in p
    lower, upper = 1.0, 1.0
    while g(lower) >= 0
        lower *= 0.7
    end
    while g(upper) <= 0
        upper *= 1.4
    end
    p = _illinois(g, lower, upper, g(lower), g(upper); tol = 1e-12)
    dual === nothing && !(field_frequency isa ForwardDiff.Dual) && return p
    slope = ForwardDiff.derivative(x -> G(plain, x, w0), p)
    return p - G(dual === nothing ? plain : dual, p, field_frequency) / slope
end

# the hats ν̂_s = p³ν_s/γ² and ν̂_D = p³ν_D/γ of DREAM's RunawayFluid.cpp:665-673
function _collision_hats(model, ir, p)
    gamma = sqrt(1 + p^2)
    return (s = p^3 * slowing_down_frequency(model, ir, p) / gamma^2,
        d = p^3 * deflection_frequency(model, ir, p) / gamma)
end

"""
    fluid_avalanche_growth_rate(plasma, ir, total_density; settings=CollisionSettings())

DREAM's avalanche growth rate [s⁻¹] of the fluid runaways at radius `ir`
(`RunawayFluid.cpp:595-660` and `353`; Hesslow et al. 2019), in a cylinder:

```math
Γ = n_\\mathrm{tot}\\, 4π r_0^2 c\\, \\frac{\\hat E − \\hat E_\\mathrm{ceff}}{p_⋆^2 \\hat E_\\mathrm{eff}},
\\qquad \\hat E_\\mathrm{eff} = \\max(\\hat E, \\hat E_\\mathrm{ceff}),
```

with the field `|E|` of the plasma, the [`effective_critical_field`](@ref) and the
[`avalanche_critical_momentum`](@ref) at `Ê_eff`. It is negative below the effective
critical field, linear in `|E|` there, and takes the sign of the field away. Returns the rate,
the effective critical field [V/m] and `p⋆`. Dual numbers in `plasma` (also in its
`electric_field`) or in `total_density` give its exact derivatives.
"""
function fluid_avalanche_growth_rate(plasma::PlasmaState, ir::Integer, total_density;
        settings::CollisionSettings = CollisionSettings())
    field_settings = runaway_fluid_settings(settings; role = :critical_field)
    momentum_settings = runaway_fluid_settings(settings; role = :critical_momentum)
    plain = _plain_plasma(plasma)
    dual = eltype(plasma.cold_electron_temperature) <: ForwardDiff.Dual
    return fluid_avalanche_growth_rate(
        CollisionModel(plain, field_settings), CollisionModel(plain, momentum_settings),
        ir, total_density;
        field_dual = dual ? CollisionModel(plasma, field_settings) : nothing,
        momentum_dual = dual ? CollisionModel(plasma, momentum_settings) : nothing)
end

"""
    fluid_avalanche_growth_rate(field_model, momentum_model, ir, total_density; field_dual=nothing,
                                momentum_dual=nothing)

The growth rate of the collisions of `field_model` (for the effective critical field) and
`momentum_model` (for `p⋆`) as they are, without [`runaway_fluid_settings`](@ref): the method
behind the one of a plasma, and the way to the limits with other collisions. The field is that
of the plasma of the models; the models in dual numbers give the derivatives.
"""
function fluid_avalanche_growth_rate(
        field_model::CollisionModel, momentum_model::CollisionModel, ir::Integer,
        total_density; field_dual = nothing, momentum_dual = nothing)
    field, _ = effective_critical_field(field_model, ir; dual = field_dual)
    source = field_dual === nothing ? field_model.plasma : field_dual.plasma
    w = _field_frequency(source.electric_field[ir])
    w_c = _field_frequency(field)
    w_eff = ForwardDiff.value(w) >= ForwardDiff.value(w_c) ? w : w_c
    p_star = avalanche_critical_momentum(momentum_model, ir, w_eff; dual = momentum_dual)
    rate = total_density * 4π * Constants.CLASSICAL_ELECTRON_RADIUS^2 *
           Constants.SPEED_OF_LIGHT * (w - w_c) /
           (p_star^2 * w_eff)
    return rate, field, p_star
end
