# Choice of the knock-on cutoff momentum p_c that follows the plasma.
#
# Upper bound (DREAM doc/sphinx/source/frontend/api/Settings/eqsys/distribution.rst,
# "Runaway avalanching": the cutoff "must be chosen to be smaller than the critical momentum
# for runaway generation, otherwise the cutoff will act as the effective critical momentum";
# O. Embréus, A. Stahl and T. Fülöp, J. Plasma Phys. 84, 905840102 (2018),
# doi:10.1017/S002237781700099X, §3.1: the growth rate is independent of p_c for
# p_c ≲ p_crit): p_c < p_crit.
# Lower bound (Embréus 2018, §2.1: stationary targets need p_c ≫ p_Te, and the grid must
# resolve p_c; DREAM's docs give no lower bound): p_c ≳ k p_Te. The reason, that the
# momentum and energy that the gain-only source creates stay small, and the factor
# k = thermal_factor = 4 are choices of this package (see the docs page Choices of this
# package).
# In a disruption the lower bound is high early in the thermal quench and the upper bound
# low in the current quench, so a fixed p_c may not satisfy both at all times; DREAM runs
# have to change pCutAvalanche between restarts.
# Non-relativistically p_crit²/p_Te² ≈ E_D/(2E), so the window closes at E/E_D ≈ 1/(2k²),
# where Dreicer generation becomes strong.

"""
    thermal_momentum(temperature)

Thermal momentum `√(2T/mₑc²)` in units of `mₑc` for the temperature `temperature` (`T`) in
eV.
"""
thermal_momentum(temperature) = sqrt(2 * temperature / Constants.ELECTRON_REST_ENERGY)

"""
    critical_field(density, coulomb_logarithm)

Connor-Hastie critical field `E_c = n e³ lnΛ / (4π ε₀² mₑ c²)` [V/m] (Connor and Hastie, Nucl.
Fusion 15, 415 (1975), doi:10.1088/0029-5515/15/3/007) for an electron density `n` [m⁻³]
and a Coulomb logarithm `lnΛ`. In the limit of complete screening `n` is the density of the
free electrons, and without screening the free plus the bound electrons (Hesslow et al.,
Nucl. Fusion 59, 084004 (2019), doi:10.1088/1741-4326/ab26c2; DREAM's `Ec_free` and
`Ec_tot`, `src/Equations/RunawayFluid.cpp:208-209`).
"""
function critical_field(density, coulomb_logarithm)
    density * coulomb_logarithm * Constants.ELEMENTARY_CHARGE^3 /
    (4pi * Constants.VACUUM_PERMITTIVITY^2 * Constants.ELECTRON_MASS *
     Constants.SPEED_OF_LIGHT^2)
end

"""
    critical_momentum(electric_field, critical_electric_field)

Critical momentum `1/√(E/E_c - 1)` in units of `mₑc` (relativistic test particle, no pitch
scattering or radiation), `Inf` for `E ≤ E_c`. DREAM's effective critical momentum
(`fluid.pCrit`) includes screening and pitch-angle scattering and is larger; a caller with
a better estimate passes it to [`cutoff_window`](@ref) directly.
"""
function critical_momentum(electric_field, critical_electric_field)
    if electric_field > critical_electric_field
        return 1 / sqrt(electric_field / critical_electric_field - 1)
    else
        return oftype(float(electric_field / critical_electric_field), Inf)
    end
end

"""
    cutoff_window(thermal_p, critical_p; thermal_factor=4)

Admissible range `(lower, upper) = (thermal_factor · thermal_p, critical_p)` of the
knock-on cutoff momentum: above the thermal momentum by a safety factor, below the
critical momentum. The window is empty when `lower ≥ upper`. The factor `thermal_factor = 4`
is a choice of this package (see the docs page *Choices of this package*).
"""
function cutoff_window(thermal_p, critical_p; thermal_factor = 4)
    (lower = thermal_factor * thermal_p, upper = critical_p)
end

"""
    choose_cutoff(p_faces, lower, upper; current_cutoff=nothing)

Cutoff momentum on a momentum-grid face (`p_faces`) inside the window `[lower, upper)`,
returned as `(p_cutoff, inside_window)`. A `current_cutoff` that is a face of the grid and
still inside the window is kept, so the kernels are rebuilt only when the window moves
past it; a value that is not a face (another grid's cutoff) is replaced. Otherwise the
face closest (in log p) to `√(lower·upper)` is taken, with `lower` raised to the smallest
positive face when it lies below it (the smallest face `≥ lower` when `upper = Inf`),
which leaves room for the bounds to move in either direction. If no
positive face lies inside the window (window closed or narrower than a cell), the largest
positive face below `upper` is returned (keeping knock-ons above the critical momentum,
at the cost of the lower bound), or the smallest positive face if none is below `upper`,
with `inside_window = false`.

Snapping to faces also keeps the cutoff off the interior of a cell, so the energy kernel
has no partial cell at the cutoff.
"""
function choose_cutoff(p_faces::AbstractVector{T}, lower::Real, upper::Real;
        current_cutoff = nothing) where {T}
    if current_cutoff !== nothing && lower <= current_cutoff < upper &&
       current_cutoff > 0 &&
       any(==(current_cutoff), p_faces)
        return (p_cutoff = T(current_cutoff), inside_window = true)
    end
    faces = filter(>(0), p_faces)
    if isempty(faces)
        throw(ArgumentError("the momentum grid has no positive face"))
    end
    candidates = filter(p -> lower <= p < upper, faces)
    if isempty(candidates)
        below = filter(<(upper), faces)
        return (
            p_cutoff = isempty(below) ? first(faces) : last(below), inside_window = false)
    end
    isinf(upper) && return (p_cutoff = first(candidates), inside_window = true)
    target = log(sqrt(max(lower, first(faces)) * upper))
    log_distances = abs.(log.(candidates) .- target)
    return (p_cutoff = candidates[argmin(log_distances)],
        inside_window = true)
end

# The cutoff rule of `cutoff_target`:
# both bounds as energies, the cutoff at the logarithmic midpoint between them, which
# makes the smaller of the two margins as large as it can be. DREAM keeps one cutoff
# (`pCutAvalanche`) for a run; Runes2P applies the rule after every accepted step and per
# radius (`KnockOnCutoffUpdate`).

"""
    kinetic_energy(p)

Kinetic energy `(γ - 1) mₑc²` in eV of an electron with the momentum `p` in units of
`mₑc`, in a form that keeps its digits for small `p`.
"""
kinetic_energy(p) = Constants.ELECTRON_REST_ENERGY * p^2 / (sqrt(1 + p^2) + 1)

"""
    momentum_at_energy(energy)

Momentum in units of `mₑc` of an electron with the kinetic energy `energy` in eV, the
inverse of [`kinetic_energy`](@ref).
"""
function momentum_at_energy(energy)
    x = energy / Constants.ELECTRON_REST_ENERGY
    return sqrt(x * (x + 2))
end

"""
    cutoff_target(temperature, critical_p; no_ceiling_factor=10)

Knock-on cutoff by the following rule (a choice of this package; see the docs page *Choices
of this package*), returned as `(p_cutoff, energy, policy)`:

- with a runaway threshold (`critical_p` finite), the logarithmic midpoint in energy
  between the thermal energy, which is the temperature [eV], and the kinetic energy at
  the critical momentum, `E = √(T E_crit)` (`policy = :log_midpoint`);
- without one (`critical_p = Inf`, the field below the critical field, where no avalanche
  is physical), `no_ceiling_factor` times the temperature (`:no_ceiling`);
- where the threshold is not above the thermal energy, which an estimate of the critical
  momentum can give in a hot plasma, the lower bound counts, since the operator heats the
  bulk when the cutoff comes near the thermal momentum: `no_ceiling_factor` times the
  temperature again (`:closed`).

`no_ceiling_factor = 10` is a choice of this package (see the docs page
*Choices of this package*).
"""
function cutoff_target(temperature::Real, critical_p::Real; no_ceiling_factor::Real = 10)
    temperature > 0 || throw(ArgumentError("the temperature must be positive"))
    no_ceiling_factor > 1 || throw(ArgumentError("no_ceiling_factor must be above 1"))
    policy = if isinf(critical_p)
        :no_ceiling
    elseif kinetic_energy(critical_p) > temperature
        :log_midpoint
    else
        :closed
    end
    energy = if policy === :log_midpoint
        sqrt(temperature * kinetic_energy(critical_p))
    else
        no_ceiling_factor * temperature
    end
    return (p_cutoff = momentum_at_energy(energy), energy = energy, policy = policy)
end

"""
    choose_cutoff_face(p_faces, target, lower, upper; current_cutoff=nothing)

The face of the momentum grid for a cutoff with the target `target`, inside the open
window `(lower, upper)`, returned as `(p_cutoff, inside_window)`: the face nearest to the
target in the logarithm of the energy. A `current_cutoff` that is a face inside the
window is kept while the target lies between its two neighbouring faces, so a target that
hovers between two faces does not rebuild the kernels at every time point; the cutoff
is then at most one cell from the target. Without a face inside the window, the face
nearest to the target is taken and `inside_window` is false.
"""
function choose_cutoff_face(
        p_faces::AbstractVector{T}, target::Real, lower::Real, upper::Real;
        current_cutoff = nothing) where {T}
    faces = filter(>(0), p_faces)
    isempty(faces) && throw(ArgumentError("the momentum grid has no positive face"))
    distance(p) = abs(log(kinetic_energy(p) / kinetic_energy(target)))
    inside = filter(p -> lower < p < upper, faces)
    isempty(inside) &&
        return (p_cutoff = T(faces[argmin(distance.(faces))]), inside_window = false)
    if current_cutoff !== nothing
        k = findfirst(==(current_cutoff), faces)
        if k !== nothing && lower < current_cutoff < upper
            below = k > 1 ? faces[k - 1] : zero(T)
            above = k < length(faces) ? faces[k + 1] : T(Inf)
            below < target < above &&
                return (p_cutoff = T(current_cutoff), inside_window = true)
        end
    end
    return (p_cutoff = T(inside[argmin(distance.(inside))]), inside_window = true)
end
