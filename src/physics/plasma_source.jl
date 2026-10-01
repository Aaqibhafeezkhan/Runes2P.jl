# The source of the plasma values of a term: the terms read the plasma
# through one interface, so that their coefficients can later follow unknowns of the system.
# The caches of the coefficients are keyed on the plasma values of each radius, not on
# the time.

"""
    PlasmaSource

Source of the [`PlasmaState`](@ref) that the coefficients of a term are built from.
[`plasma_values`](@ref)`(source, u, layout, t)` gives the state at the state vector `u` of
the system and the time `t`. [`PrescribedPlasma`](@ref) is the only implementation: a
plasma prescribed as a function of time. A source that reads the temperature and the
densities from the unknowns is to follow.
"""
abstract type PlasmaSource end

"""
    plasma_values(source::PlasmaSource, u, layout, t)

The [`PlasmaState`](@ref) of `source` at the state `u` of the system with the layout
`layout`, and the time `t`. A prescribed source ignores `u` and `layout`, which may then be
`nothing`.
"""
function plasma_values end

"""
    PrescribedPlasma(plasma_at)

A [`PlasmaSource`](@ref) prescribed in time: `plasma_at` is a [`PlasmaState`](@ref) or a
function of time that returns one. The state of the last time is kept, since the
evaluations of a term at one time share it.
"""
struct PrescribedPlasma{F} <: PlasmaSource
    plasma_at::F
    time::Base.RefValue{Float64}    # the time of `state`, NaN before the first evaluation
    state::Base.RefValue{Any}
end
PrescribedPlasma(plasma_at) = PrescribedPlasma(plasma_at, Ref(NaN), Ref{Any}(nothing))
PrescribedPlasma(source::PrescribedPlasma) = source

plasma_values(source::PrescribedPlasma{<:PlasmaState}, u, layout, t) = source.plasma_at
function plasma_values(source::PrescribedPlasma, u, layout, t)
    time = Float64(_value(t))
    if source.time[] != time
        source.state[] = source.plasma_at(time)
        source.time[] = time
    end
    return source.state[]::PlasmaState
end

# a source of the given plasma: a PlasmaState, a function of time or a source
plasma_source(source::PlasmaSource) = source
plasma_source(plasma_at) = PrescribedPlasma(plasma_at)

# the plasma values of a radius that coefficients are built from, for the key of a cache:
# the temperature, the density, the magnetic field, the electric field when the
# coefficients hold it, and the densities of all ion charge states
function _radius_values!(values::Vector{Float64}, plasma::PlasmaState, ir; electric_field::Bool)
    empty!(values)
    push!(values, _value(plasma.cold_electron_temperature[ir]), _value(plasma.cold_electron_density[ir]),
          _value(plasma.magnetic_field[ir]))
    electric_field && push!(values, _value(plasma.electric_field[ir]))
    for species in plasma.ions
        push!(values, species.nuclear_charge)
        for Z0 in 0:species.nuclear_charge
            push!(values, _value(species.charge_state_densities[Z0+1, ir]))
        end
    end
    return values
end

"""
    PlasmaKey()

The plasma values per radius that a cache of coefficients was built from.
[`update_key!`](@ref) tells whether a new plasma differs from them at any radius.
"""
struct PlasmaKey
    values::Vector{Vector{Float64}}   # per radius
    scratch::Vector{Float64}
    time::Base.RefValue{Float64}      # the time of the last comparison
end
PlasmaKey() = PlasmaKey(Vector{Float64}[], Float64[], Ref(NaN))

"""
    update_key!(key::PlasmaKey, plasma::PlasmaState; electric_field=true)
    update_key!(key::Vector{Float64}, values)

Store the values of `plasma` (or the numbers `values`, one per radius) in `key`, and return
whether they differ from the stored ones at any radius (`true` for an empty key). The
values are compared with `isequal`. `electric_field = false` leaves the electric field out,
for coefficients that do not hold it.
"""
function update_key!(key::PlasmaKey, plasma::PlasmaState; electric_field::Bool=true)
    nr = radial_cell_count(plasma)
    changed = length(key.values) != nr
    changed && (resize!(key.values, nr); foreach(ir -> key.values[ir] = Float64[], 1:nr))
    for ir in 1:nr
        values = _radius_values!(key.scratch, plasma, ir; electric_field)
        stored = key.values[ir]
        if !isequal(values, stored)
            changed = true
            resize!(stored, length(values))
            copyto!(stored, values)
        end
    end
    return changed
end

function update_key!(key::Vector{Float64}, values)
    changed = length(key) != length(values)
    changed && resize!(key, length(values))
    for (k, value) in enumerate(values)
        v = Float64(_value(value))
        if !isequal(key[k], v) || changed
            changed = true
            key[k] = v
        end
    end
    return changed
end

# whether a key is current at the time without a comparison: a prescribed plasma depends
# on the time alone, and a constant one on nothing
_key_current(key::PlasmaKey, source::PrescribedPlasma{<:PlasmaState}, time) = !isempty(key.values)
_key_current(key::PlasmaKey, source::PrescribedPlasma, time) = !isempty(key.values) && key.time[] == time
_key_current(key::PlasmaKey, source::PlasmaSource, time) = false

"""
    changed_plasma!(key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field=true)

The plasma of `source` at the state `u` and the time `t` when its values differ from those
of `key` ([`update_key!`](@ref), which stores them), and `nothing` when they do not. A
prescribed source is compared once per time, so the evaluations of a term at one time
cost no comparison.
"""
function changed_plasma!(key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field::Bool=true)
    time = Float64(_value(t))
    _key_current(key, source, time) && return nothing
    plasma = plasma_values(source, u, layout, t)
    changed = update_key!(key, plasma; electric_field)
    key.time[] = time
    return changed ? plasma : nothing
end

"""
    reset_key!(key)

Empty `key`, so that the next [`update_key!`](@ref) reports a change: after a rebuild that
failed.
"""
reset_key!(key::PlasmaKey) = (empty!(key.values); key)
reset_key!(key::Vector{Float64}) = empty!(key)
