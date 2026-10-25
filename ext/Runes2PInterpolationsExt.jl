"""
Interpolators of Interpolations.jl as radial profiles: evaluated on the points
of the radial grid, and recorded by their points, the values there, the method and the
rule beyond the last point, from which the run can be rebuilt.
"""
module Runes2PInterpolationsExt

using Runes2P
using Runes2P: SettingsRecord, _plain
using Interpolations: Interpolations, AbstractInterpolation, AbstractExtrapolation,
                      FilledExtrapolation

# An interpolator is an AbstractArray of its values at the knots: without this method it
# would pass as a vector with one value per radial point.
function Runes2P.radial_values(profile::AbstractInterpolation, points::AbstractVector)
    map(profile, points)
end
Runes2P._is_radial_profile(::AbstractInterpolation) = true

# `delta_b_over_b_at` is a function of time: an interpolator in r passed in its place would
# be called with the time.
function Runes2P.RechesterRosenbluthUpdate(
        operator::Runes2P.AdvectionDiffusion, profile::AbstractInterpolation; kwargs...)
    throw(ArgumentError("the level of RechesterRosenbluthUpdate is a function of time; give a radial " *
                        "profile that is constant in time as `t -> profile`"))
end

# The values of a PrescribedValueTerm are a function of time; an interpolator, an array of
# its knot values, would pass as one value per radius where the knot count matches.
function Runes2P.PrescribedValueTerm(profile::AbstractInterpolation)
    throw(ArgumentError("the values of a PrescribedValueTerm are numbers or a function of time; give a " *
                        "radial profile as `t -> profile`, a profile in time as `t -> profile(t)`"))
end

# Runes2P records an AbstractArray as its values; an interpolator gets its own record.
Runes2P._describe_value(profile::AbstractInterpolation) = Runes2P.describe(profile)

# a value of Interpolations as text without the module prefix: printed relative to Main, the
# text would depend on whether the caller's script loaded Interpolations with `using`
_text(x) = sprint(show, x; context = :module => Interpolations)

# the interpolation method, from the innermost interpolation (a scaled interpolation wraps a
# B-spline)
function _method(itp)
    hasproperty(itp, :it) ? _text(itp.it) :
    hasproperty(itp, :itp) ? _method(itp.itp) : "unknown"
end

_extrapolation(itp::AbstractInterpolation) = "Throw()"   # an error beyond the last point
_extrapolation(etp::FilledExtrapolation) = "fill"
_extrapolation(etp::AbstractExtrapolation) = _text(Interpolations.etpflag(etp))

function Runes2P.describe(profile::AbstractInterpolation)
    inner = profile isa AbstractExtrapolation ? parent(profile) : profile
    knots = Tuple(collect(Float64, k) for k in Interpolations.getknots(inner))
    values = [inner(x...) for x in Iterators.product(knots...)]
    record = SettingsRecord("type" => "Interpolations." * String(nameof(typeof(inner))),
        "method" => _method(inner), "extrapolation" => _extrapolation(profile),
        "values" => _plain(values))
    if length(knots) == 1
        record["knots"] = _plain(only(knots))
    else
        for (k, points) in enumerate(knots)
            record["knots_$k"] = _plain(points)
        end
    end
    profile isa FilledExtrapolation && (record["fill_value"] = _plain(profile.fillvalue))
    return record
end

end
