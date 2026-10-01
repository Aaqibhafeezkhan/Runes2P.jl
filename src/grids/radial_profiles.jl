# Radial profiles on points of their own: a number, one value per point, or a
# callable of r, evaluated on the cell centres or the faces of the run's radial grid when
# the run is built. Interpolator objects of Interpolations.jl are callables and work here;
# the extension Runes2PInterpolationsExt evaluates them and records their points.

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
function PlasmaState(radial_grid::RadialGrid; cold_electron_temperature, ions, electric_field, magnetic_field,
                     cold_electron_density=nothing)
    centers = radial_grid.r_centers
    density = cold_electron_density === nothing ? nothing : radial_values(cold_electron_density, centers)
    return PlasmaState(; cold_electron_temperature=radial_values(cold_electron_temperature, centers), ions,
                       electric_field=radial_values(electric_field, centers),
                       magnetic_field=radial_values(magnetic_field, centers), cold_electron_density=density)
end

PlasmaState(grid::KineticGrid; kwargs...) = PlasmaState(grid.radial_grid; kwargs...)

"""
    IonSpecies(name, Z, Z0, profile, radial_grid)
    IonSpecies(name, Z, profiles, radial_grid)

Ion species with densities [m⁻³] in any form of [`radial_values`](@ref), evaluated on the
cell centres of `radial_grid` (a `RadialGrid` or a `KineticGrid`): all in the charge state
`Z0`, or one profile per charge state `0:Z` in `profiles`.
"""
IonSpecies(name::AbstractString, Z::Integer, Z0::Integer, profile, radial_grid::RadialGrid) =
    IonSpecies(name, Z, Z0, radial_values(profile, radial_grid.r_centers))

function IonSpecies(name::AbstractString, Z::Integer, profiles::AbstractVector, radial_grid::RadialGrid)
    length(profiles) == Z + 1 || throw(ArgumentError("need Z + 1 charge-state profiles for $name"))
    centers = radial_grid.r_centers
    return IonSpecies(name, Z, permutedims(stack(radial_values(p, centers) for p in profiles)))
end

IonSpecies(name::AbstractString, Z::Integer, Z0::Integer, profile, grid::KineticGrid) =
    IonSpecies(name, Z, Z0, profile, grid.radial_grid)
IonSpecies(name::AbstractString, Z::Integer, profiles::AbstractVector, grid::KineticGrid) =
    IonSpecies(name, Z, profiles, grid.radial_grid)
