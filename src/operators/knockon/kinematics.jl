# Kinematics and Møller cross section of knock-on collisions between an incident
# electron with momentum p1 and a stationary electron, ported from DREAM
# `KnockOnUtilities.hpp` (namespace `Kinematics`, branch `boltzmann-eqterm`).
# Momenta in units of mₑc. `p` denotes the knock-on (lower-energy) outgoing electron,
# valid for p ≤ max_knockon_momentum(p1). `p1 = Inf` gives the limit of an infinitely
# energetic incident electron.
#
# γ - 1 is evaluated as p²/(γ + 1), which avoids cancellation at small p and differs
# from DREAM only by round-off.

@inline _gm1(p) = p^2 / (sqrt(1 + p^2) + 1)

"""
    xi_star(p, p1)

Cosine of the angle between the incident electron (momentum `p1`, colliding with a
stationary electron) and the knock-on electron (momentum `p`),
`ξ* = √[(γ₁ + 1)(γ - 1) / ((γ₁ - 1)(γ + 1))]`.

Ported from DREAM `KnockOnUtilities.hpp` (`Kinematics::EvaluateXiStar`).
"""
function xi_star(p, p1)
    gm1 = _gm1(p)
    isinf(p1) && return sqrt(gm1 / (gm1 + 2))
    g1m1 = _gm1(p1)
    return sqrt((g1m1 + 2) * gm1 / (g1m1 * (gm1 + 2)))
end

"""
    moller_cross_section(p, p1)

Møller differential cross section `Σ = c dσ/dγ` [m³/s] for producing a knock-on
electron of momentum `p` from an incident electron of momentum `p1` on a stationary
target. The rate coefficient per target electron is `(p1/γ1) Σ dγ`.

Ported from DREAM `KnockOnUtilities.hpp`
(`Kinematics::EvaluateMollerDifferentialCrossSection`).
"""
function moller_cross_section(p, p1)
    pre = 2π * Constants.r0^2 * Constants.c
    gm1 = _gm1(p)
    isinf(p1) && return pre / gm1^2
    g = gm1 + 1
    g1 = _gm1(p1) + 1
    d = g1 - g
    term = (g1 - 1)^2 - gm1 * d / g1^2 * (2g1^2 + 2g1 - 1 - gm1 * d)
    return pre * term * g1^2 / (p1^2 * gm1^2 * d^2)
end

"""
    moller_flux(p, p1)

Antiderivative `S` of the Møller cross section in the knock-on momentum,
`∂S/∂p = (p/γ) Σ`, i.e. `∂S/∂γ = Σ`. Integrating the knock-on source over a
momentum cell reduces to differences of `S`.

Ported from DREAM `KnockOnUtilities.hpp` (`Kinematics::EvaluateMollerFlux`).
"""
function moller_flux(p, p1)
    pre = 2π * Constants.r0^2 * Constants.c
    gm1 = _gm1(p)
    isinf(p1) && return -pre / gm1
    g = gm1 + 1
    g1m1 = _gm1(p1)
    g1 = g1m1 + 1
    d = g1 - g
    return pre / p1^2 * (g1^2 * (1 / d - 1 / gm1) + g - (2g1 - 1) / g1m1 * log(gm1 / d))
end

"""
    max_knockon_momentum(p1)

Largest knock-on momentum from an incident electron of momentum `p1`: the momentum at
half the incident kinetic energy, `γ_max = (γ₁ + 1)/2`. Above it, the outgoing electron
with momentum `p` is the primary, not the knock-on.

Ported from DREAM `KnockOnUtilities.hpp` (`Kinematics::MaximumKnockOnMomentum`).
"""
function max_knockon_momentum(p1)
    isinf(p1) && return oftype(float(p1), Inf)
    gmax = (_gm1(p1) + 2) / 2
    return sqrt(max(zero(gmax), gmax^2 - 1))
end

"""
    moller_total_rate(p1, p_cut)

Rate coefficient [m³/s] for an incident electron of momentum `p1` to produce a
knock-on electron with momentum above `p_cut`,
`(p1/γ1) ∫ Σ dγ = (p1/γ1) [S(p_max) - S(p_cut)]` with `p_max = max_knockon_momentum(p1)`;
zero when `p_cut ≥ p_max`. Multiplied by the target density it gives the production
rate per incident electron. Corresponds to DREAM's
`EvaluateMollerFluxIntegratedOverKnockonGrid` on an unbounded knock-on grid.
"""
function moller_total_rate(p1, p_cut)
    p_cut > 0 || throw(ArgumentError("p_cut must be positive"))
    T = float(promote_type(typeof(p1), typeof(p_cut)))
    pmax = max_knockon_momentum(p1)
    p_cut < pmax || return zero(T)
    Smax = isinf(p1) ? zero(T) : moller_flux(pmax, p1)
    v1 = isinf(p1) ? one(T) : p1 / sqrt(1 + p1^2)
    return v1 * (Smax - moller_flux(p_cut, p1))
end
