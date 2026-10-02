# The source of the plasma values of a term: the terms read the plasma
# through one interface, so that their coefficients can later follow unknowns of the system.
# The caches of the coefficients are keyed on the plasma values of each radius, not on
# the time.

"""
    PlasmaSource

Source of the [`PlasmaState`](@ref) that the coefficients of a term are built from.
[`plasma_values`](@ref)`(source, u, layout, t)` gives the state at the state vector `u` of
the system and the time `t`. [`PrescribedPlasma`](@ref) is a plasma prescribed as a
function of time, [`PlasmaFromUnknowns`](@ref) takes the cold-electron temperature and
density from unknowns of the system.
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
evaluations of a term at one time share it. A `PlasmaState` given directly is constant for
the whole run, and a term reads it once.
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

"""
    PlasmaFromUnknowns(prescribed; temperature=:cold_electron_temperature,
                       density=:cold_electron_density, temperature_floor=0.01)

A [`PlasmaSource`](@ref) whose cold-electron temperature [eV] and density [m⁻³] are the
values of radial unknowns of the system at each radius, with the rest of
the plasma (ions, fields) from `prescribed`, a [`PlasmaState`](@ref), a function of time
that returns one, or a [`PrescribedPlasma`](@ref). `density = nothing` keeps the density of
`prescribed`, which a [`PlasmaState`](@ref) built without `cold_electron_density` takes as
the free-electron density: in a fully kinetic case the target of the e-e collisions then
counts the hot and runaway electrons as well, where DREAM's target is the closure
`n_cold = n_free − n_hot − n_re`. The values are taken without their derivative parts, so the coefficients of a
term hold numbers. Their derivatives enter the Jacobian as columns of their own.

A temperature at or below `temperature_floor` [eV] raises a `DomainError`, which the
Newton iteration reports as a failed solve and the stepper answers with a shorter step.
The collision model needs the scaled Bessel functions of `mₑc²/T`, which fail above an
argument of about 1e9, near 5e-4 eV. The default floor of 0.01 eV keeps a margin of 20 and
lies below the temperatures of a disruption.
"""
struct PlasmaFromUnknowns{P<:PrescribedPlasma} <: PlasmaSource
    prescribed::P
    temperature::Symbol
    density::Union{Nothing,Symbol}
    temperature_floor::Float64
end
function PlasmaFromUnknowns(prescribed; temperature::Symbol=:cold_electron_temperature,
                            density::Union{Nothing,Symbol}=:cold_electron_density, temperature_floor::Real=0.01)
    temperature_floor >= 0 || throw(ArgumentError("the temperature floor must not be negative"))
    return PlasmaFromUnknowns(PrescribedPlasma(prescribed), temperature, density, Float64(temperature_floor))
end

"""
    is_state_dependent(source)

Whether a [`PlasmaSource`](@ref) reads the state of the system: true for
[`PlasmaFromUnknowns`](@ref), false for a prescribed plasma and for `nothing`.
"""
is_state_dependent(::PlasmaSource) = false
is_state_dependent(::Nothing) = false
is_state_dependent(::PlasmaFromUnknowns) = true

"""
    source_dependencies(source)

The names of the unknowns a [`PlasmaSource`](@ref) reads.
"""
source_dependencies(::Union{Nothing,PlasmaSource}) = ()
source_dependencies(source::PlasmaFromUnknowns) =
    source.density === nothing ? (source.temperature,) : (source.temperature, source.density)

function plasma_values(source::PlasmaFromUnknowns, u, layout, t)
    u === nothing && throw(ArgumentError("a plasma from the unknowns $(join(source_dependencies(source), " and ")) " *
                                         "needs the state of the system"))
    base = plasma_values(source.prescribed, u, layout, t)
    T = eltype(base.cold_electron_temperature)
    nr = radial_cell_count(base)
    temperature = radial_values(u, layout, unknown_id(layout, source.temperature))
    temperatures = Vector{T}(undef, nr)
    for ir in 1:nr
        value = _value(temperature[ir])
        value > source.temperature_floor ||
            throw(DomainError(value, "the cold-electron temperature $(source.temperature) at radius $ir is at or " *
                                     "below the floor $(source.temperature_floor) eV"))
        temperatures[ir] = value
    end
    densities = if source.density === nothing
        base.cold_electron_density
    else
        density = radial_values(u, layout, unknown_id(layout, source.density))
        T[_value(density[ir]) for ir in 1:nr]
    end
    return PlasmaState{T}(temperatures, densities, base.ions, base.electric_field, base.magnetic_field)
end

# the values of the unknowns a source reads, for the key of a cache
function _source_state!(values::Vector{Float64}, source::PlasmaFromUnknowns, u, layout)
    empty!(values)
    for name in source_dependencies(source), x in radial_values(u, layout, unknown_id(layout, name))
        push!(values, _value(x))
    end
    return values
end

function _same_source_state(stored::Vector{Float64}, source::PlasmaFromUnknowns, u, layout)
    k = 0
    for name in source_dependencies(source), x in radial_values(u, layout, unknown_id(layout, name))
        k += 1
        (k <= length(stored) && isequal(stored[k], Float64(_value(x)))) || return false
    end
    return k == length(stored)
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
    state::Vector{Float64}            # the values of the unknowns a source reads, at the last comparison
end
PlasmaKey() = PlasmaKey(Vector{Float64}[], Float64[], Ref(NaN), Float64[])

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
_key_current(key::PlasmaKey, source::PlasmaSource, time, u, layout) = _key_current(key, source, time)
# a plasma from the unknowns: its prescribed part depends on the time, the rest on the
# values of the unknowns it reads
_key_current(key::PlasmaKey, source::PlasmaFromUnknowns, time, u, layout) =
    !isempty(key.values) && key.time[] == time && _same_source_state(key.state, source, u, layout)

"""
    changed_plasma!(key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field=true)

The plasma of `source` at the state `u` and the time `t` when its values differ from those
of `key` ([`update_key!`](@ref), which stores them), and `nothing` when they do not. A
prescribed source is compared once per time, so the evaluations of a term at one time
cost no comparison.
"""
function changed_plasma!(key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field::Bool=true)
    time = Float64(_value(t))
    _key_current(key, source, time, u, layout) && return nothing
    plasma = plasma_values(source, u, layout, t)
    changed = update_key!(key, plasma; electric_field)
    key.time[] = time
    is_state_dependent(source) && _source_state!(key.state, source, u, layout)
    return changed ? plasma : nothing
end

"""
    reset_key!(key)

Empty `key`, so that the next [`update_key!`](@ref) reports a change: after a rebuild that
failed.
"""
reset_key!(key::PlasmaKey) = (empty!(key.values); key)
reset_key!(key::Vector{Float64}) = empty!(key)

# tag of the dual numbers of the plasma columns
struct PlasmaColumnTag end

"""
    dual_plasma(plasma::PlasmaState, with_density::Bool)

The plasma with the cold-electron temperature as a dual number of partial 1 at every
radius, and with `with_density` the cold-electron density as partial 2; the ions and the
fields hold zero partials. The coefficients at a radius depend only on the plasma of that
radius, so partial `k` of a coefficient at radius `ir` is its derivative with respect to
plasma scalar `k` at `ir`, for all radii in one evaluation.
"""
function dual_plasma(plasma::PlasmaState{Float64}, with_density::Bool)
    K = with_density ? 2 : 1
    D = ForwardDiff.Dual{PlasmaColumnTag,Float64,K}
    seeded(x, k) = D(x, ForwardDiff.Partials(ntuple(i -> Float64(i == k), K)))
    temperature = [seeded(x, 1) for x in plasma.cold_electron_temperature]
    density = with_density ? [seeded(x, 2) for x in plasma.cold_electron_density] : D.(plasma.cold_electron_density)
    ions = IonSpecies{D}[IonSpecies(s.name, s.nuclear_charge, D.(s.charge_state_densities)) for s in plasma.ions]
    return PlasmaState{D}(temperature, density, ions, D.(plasma.electric_field), D.(plasma.magnetic_field))
end
