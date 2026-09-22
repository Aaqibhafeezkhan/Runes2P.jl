# Kinematics and Møller cross section of knock-on collisions between an incident
# electron with momentum p1 and a stationary electron, ported from DREAM
# `KnockOnUtilities.hpp` (namespace `Kinematics`, branch `boltzmann-eqterm`).
# Momenta in units of mₑc. `p` denotes the knock-on (lower-energy) outgoing electron,
# valid for p ≤ max_knockon_momentum(p1). `p1 = Inf` gives the limit of an infinitely
# energetic incident electron.
#
# γ - 1 is evaluated as p²/(γ + 1), which avoids cancellation at small p and differs
# from DREAM only by round-off.

@inline _gamma_minus_one(p) = p^2 / (sqrt(1 + p^2) + 1)

const _MOLLER_PREFACTOR = 2pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT

"""
    xi_star(p, p1)

Cosine ξ* of the angle between the incident electron (momentum `p1`, colliding with a
stationary electron) and the knock-on electron (momentum `p`),
`ξ* = √[(γ₁ + 1)(γ - 1) / ((γ₁ - 1)(γ + 1))]` (notation of Embréus et al. 2018).

Ported from DREAM `KnockOnUtilities.hpp` (`Kinematics::EvaluateXiStar`).
"""
function xi_star(p, p1)
    knockon_energy = _gamma_minus_one(p)
    isinf(p1) && return sqrt(knockon_energy / (knockon_energy + 2))
    incident_energy = _gamma_minus_one(p1)
    return sqrt((incident_energy + 2) * knockon_energy / (incident_energy * (knockon_energy + 2)))
end

"""
    momentum_at_xi_star(xi_s, p1)

Knock-on momentum `p` with [`xi_star`](@ref)`(p, p1) = xi_s`, the inverse of `xi_star` in
its first argument (`ξ*` grows monotonically with `p`): from
`ξ*² = (γ₁ + 1)(γ - 1)/((γ₁ - 1)(γ + 1))`,

    γ - 1 = 2 ξ*² (γ₁ - 1) / (γ₁ + 1 - ξ*² (γ₁ - 1)),    p = √((γ - 1)(γ + 1)).

Used to split the receiving momentum interval of a pair at the `ξ*` values of the pitch
table.
"""
function momentum_at_xi_star(xi_s, p1)
    x2 = xi_s^2
    incident_energy = _gamma_minus_one(p1)
    knockon_energy = isinf(p1) ? 2x2 / (1 - x2) :
                     2x2 * incident_energy / (incident_energy + 2 - x2 * incident_energy)
    knockon_energy > 0 || return zero(float(knockon_energy))
    return sqrt(knockon_energy * (knockon_energy + 2))
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
    knockon_energy = _gamma_minus_one(p)
    isinf(p1) && return _MOLLER_PREFACTOR / knockon_energy^2
    gamma = knockon_energy + 1
    gamma1 = _gamma_minus_one(p1) + 1
    difference = gamma1 - gamma
    term = (gamma1 - 1)^2 - knockon_energy * difference / gamma1^2 *
                            (2gamma1^2 + 2gamma1 - 1 - knockon_energy * difference)
    return _MOLLER_PREFACTOR * term * gamma1^2 / (p1^2 * knockon_energy^2 * difference^2)
end

"""
    moller_flux(p, p1)

Antiderivative `S` of the Møller cross section in the knock-on momentum,
`∂S/∂p = (p/γ) Σ`, i.e. `∂S/∂γ = Σ`. Integrating the knock-on source over a
momentum cell reduces to differences of `S`.

Ported from DREAM `KnockOnUtilities.hpp` (`Kinematics::EvaluateMollerFlux`).
"""
function moller_flux(p, p1)
    knockon_energy = _gamma_minus_one(p)
    isinf(p1) && return -_MOLLER_PREFACTOR / knockon_energy
    gamma = knockon_energy + 1
    incident_energy = _gamma_minus_one(p1)
    gamma1 = incident_energy + 1
    difference = gamma1 - gamma
    return _MOLLER_PREFACTOR / p1^2 *
           (gamma1^2 * (1 / difference - 1 / knockon_energy) + gamma -
            (2gamma1 - 1) / incident_energy * log(knockon_energy / difference))
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
    gamma_max = (_gamma_minus_one(p1) + 2) / 2
    return sqrt(max(zero(gamma_max), gamma_max^2 - 1))
end

"""
    moller_total_rate(p1, p_cutoff)

Rate coefficient [m³/s] for an incident electron of momentum `p1` to produce a
knock-on electron with momentum above `p_cutoff`,
`(p1/γ1) ∫ Σ dγ = (p1/γ1) [S(p_max) - S(p_cutoff)]` with `p_max = max_knockon_momentum(p1)`;
zero when `p_cutoff ≥ p_max`. Multiplied by the target density it gives the production
rate per incident electron. Corresponds to DREAM's
`EvaluateMollerFluxIntegratedOverKnockonGrid` on an unbounded knock-on grid.
"""
function moller_total_rate(p1, p_cutoff)
    p_cutoff > 0 || throw(ArgumentError("p_cutoff must be positive"))
    T = float(promote_type(typeof(p1), typeof(p_cutoff)))
    p_max = max_knockon_momentum(p1)
    p_cutoff < p_max || return zero(T)
    flux_at_max = isinf(p1) ? zero(T) : moller_flux(p_max, p1)
    beta1 = isinf(p1) ? one(T) : p1 / sqrt(1 + p1^2)
    return beta1 * (flux_at_max - moller_flux(p_cutoff, p1))
end
