# Radial profiles on points of their own: a number, one value per point, or a
# callable of r, evaluated on the cell centres or the faces of the run's radial grid when
# the run is built. Interpolator objects of Interpolations.jl are callables and work here;
# the extension Runes2PInterpolationsExt evaluates them and records their points. The levels
# that are piecewise constant in time, whose values may be such profiles, are at the end.

"""
    radial_values(profile, points)

Values of a radial profile at `points` [m], the cell centres or the faces of a radial grid:

- a number is the same at every point,
- a vector holds one value per point and is returned as it is,
- anything else is called at every point, `profile(r)`: an analytic function or an
  interpolator of a package (Interpolations.jl: `linear_interpolation(r_data, values)`).

The evaluation happens once, when the run is built or at the start of an interval, never
inside the Newton iteration. A profile that changes in time is a function of time that
returns one of these forms.
"""
radial_values(profile::Real, points::AbstractVector) = fill(profile, length(points))

function radial_values(profile::AbstractVector, points::AbstractVector)
    length(profile) == length(points) ||
        throw(DimensionMismatch("need one value per radial point ($(length(points))), got $(length(profile))"))
    return profile
end

radial_values(profile, points::AbstractVector) = map(profile, points)

# a radial profile given as a callable, not as a number or values; the extension adds the
# interpolators, which are AbstractVectors of their values at the knots
_is_radial_profile(value) = !(value isa Real) && !(value isa AbstractVector)

"""
    PlasmaState(radial_grid; cold_electron_temperature, ions, electric_field, magnetic_field,
                cold_electron_density=nothing)

The plasma state with profiles given in any form of [`radial_values`](@ref), evaluated on the
cell centres of `radial_grid` (a `RadialGrid` or a `KineticGrid`). Ion densities on points
of their own take the forms `IonSpecies(name, Z, Z0, profile, radial_grid)` and
`IonSpecies(name, Z, profiles, radial_grid)`. The state holds the evaluated values, so its
record holds them too.
"""
function PlasmaState(radial_grid::RadialGrid; cold_electron_temperature,
        ions, electric_field, magnetic_field,
        cold_electron_density = nothing)
    centers = radial_grid.r_centers
    if cold_electron_density === nothing
        density = nothing
    else
        density = radial_values(cold_electron_density, centers)
    end
    return PlasmaState(;
        cold_electron_temperature = radial_values(cold_electron_temperature, centers),
        ions,
        electric_field = radial_values(electric_field, centers),
        magnetic_field = radial_values(magnetic_field, centers), cold_electron_density = density)
end

PlasmaState(grid::KineticGrid; kwargs...) = PlasmaState(grid.radial_grid; kwargs...)

"""
    IonSpecies(name, Z, Z0, profile, radial_grid)
    IonSpecies(name, Z, profiles, radial_grid)

Ion species with densities [m⁻³] in any form of [`radial_values`](@ref), evaluated on the
cell centres of `radial_grid` (a `RadialGrid` or a `KineticGrid`): all in the charge state
`Z0`, or one profile per charge state `0:Z` in `profiles`.
"""
function IonSpecies(
        name::AbstractString, Z::Integer, Z0::Integer, profile, radial_grid::RadialGrid)
    IonSpecies(name, Z, Z0, radial_values(profile, radial_grid.r_centers))
end

function IonSpecies(name::AbstractString, Z::Integer, profiles::AbstractVector, radial_grid::RadialGrid)
    length(profiles) == Z + 1 ||
        throw(ArgumentError("need Z + 1 charge-state profiles for $name"))
    centers = radial_grid.r_centers
    #! format: off
    return IonSpecies(name, Z, permutedims(stack(radial_values(p, centers) for p in profiles)))
    #! format: on
end

function IonSpecies(
        name::AbstractString, Z::Integer, Z0::Integer, profile, grid::KineticGrid)
    IonSpecies(name, Z, Z0, profile, grid.radial_grid)
end
function IonSpecies(name::AbstractString, Z::Integer, profiles::AbstractVector, grid::KineticGrid)
    IonSpecies(name, Z, profiles, grid.radial_grid)
end

# levels that are piecewise constant in time

"""
    PiecewiseConstant(values, switch_times)

Function of time that holds `values[k]` from `switch_times[k-1]` up to `switch_times[k]`
(`values[1]` before the first switch time, the last value from the last switch time on).
At a switch time it has the value of the interval that starts there. `values` are numbers
or vectors. For example, a level of δB/B for the first millisecond and a hundredth of it
afterwards is `PiecewiseConstant([1e-3, 1e-5], [1e-3])`.

An update that takes such a function requires its switch times to be time points of the
run ([`forced_time_points`](@ref)), so that no interval holds a switch.
"""
struct PiecewiseConstant{V}
    values::Vector{V}
    switch_times::Vector{Float64}
    function PiecewiseConstant(values::AbstractVector, switch_times::AbstractVector{<:Real})
        length(values) == length(switch_times) + 1 ||
            throw(ArgumentError("need one value more than switch times, got $(length(values)) values and " *
                                "$(length(switch_times)) switch times"))
        issorted(switch_times; lt = <=) ||
            throw(ArgumentError("the switch times must be strictly increasing"))
        return new{eltype(values)}(collect(values), collect(Float64, switch_times))
    end
end

# a time point within round-off below a switch time counts as the switch time
function (history::PiecewiseConstant)(t::Real)
    history.values[searchsortedlast(history.switch_times, t + 1e-12 * abs(t)) + 1]
end

"""
    level_over_step(level, t)

The level of a function of time over a step that ends at `t`, for a term that evaluates it
in the residual of an implicit step: for a [`PiecewiseConstant`](@ref) the value of the
interval that ends at `t` (its limit from the left, so the step that ends at a switch time
keeps the value before the switch, as the interval updates of the kinetic transport do),
for any other function its value at `t`.
"""
level_over_step(level, t) = level(t)
function level_over_step(history::PiecewiseConstant, t::Real)
    history.values[searchsortedfirst(history.switch_times, t - 1e-12 * abs(t))]
end

"""
    forced_time_points(update)
    forced_time_points(function_of_time)

Times at which the setup of an update changes abruptly. [`integrate`](@ref) requires those
inside the run to be among its time points. Empty by default, the switch times of a
[`PiecewiseConstant`](@ref).
"""
forced_time_points(x) = Float64[]
forced_time_points(history::PiecewiseConstant) = history.switch_times
