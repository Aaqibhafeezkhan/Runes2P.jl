# The source of the plasma values of a term: the terms read the plasma through one
# interface, so that their coefficients can follow unknowns of the system. The caches of the
# coefficients are keyed on the plasma values of each radius, not on the time.

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
    DEFAULT_TEMPERATURE_FLOOR

The default floor [eV] of a temperature unknown, 0.5 eV (a choice of this package; see the
docs page *Choices of this package*). The default of [`PlasmaFromUnknowns`](@ref) and of
[`fluid_temperature_floors`](@ref).
"""
const DEFAULT_TEMPERATURE_FLOOR = 0.5

"""
    PlasmaFromUnknowns(prescribed; temperature=:cold_electron_temperature,
                       density=:cold_electron_density, ions=nothing, temperature_floor=0.5)

A [`PlasmaSource`](@ref) whose cold-electron temperature [eV] and density [m⁻³] are the
values of radial unknowns of the system at each radius, with the rest of
the plasma (ions, fields) from `prescribed`, a [`PlasmaState`](@ref), a function of time
that returns one, or a [`PrescribedPlasma`](@ref). `density = nothing` keeps the density of
`prescribed`, which a [`PlasmaState`](@ref) built without `cold_electron_density` takes as
the free-electron density: in a fully kinetic case the target of the e-e collisions then
counts the hot and runaway electrons as well, where DREAM's target is the closure
`n_cold = n_free − n_hot − n_re`. The values are taken without their derivative parts, so the coefficients of a
term hold numbers. Their derivatives enter the Jacobian as columns of their own.

`ions` names the radial unknown of the densities of the ion charge states,
whose components follow the species of `prescribed` in their order
([`IonComposition`](@ref), [`ion_densities_unknown`](@ref)); the densities of the prescribed ions are
then ignored. A plasma whose ions are unknowns takes its density from an unknown too: the
free-electron density changes with the charge states, and the prescribed density would not
follow it.

A temperature at or below `temperature_floor` [eV] raises a `DomainError`, which the
Newton iteration reports as a failed solve and the stepper answers with a shorter step:
a step that would end below the floor is refused. The default is 0.5 eV
([`DEFAULT_TEMPERATURE_FLOOR`](@ref)). DREAM has no floor, so a comparison that follows
DREAM below 0.5 eV sets a lower value. The floor is also the lower limit of
the temperature unknown in the Newton iteration, which damps an update that would cross
it ([`LowerLimit`](@ref), `NewtonSolver`). The numerical limit lies far below: the
collision model needs the scaled Bessel functions of `mₑc²/T`, which fail above an
argument of about 1e9, near 5e-4 eV.
"""
struct PlasmaFromUnknowns{P <: PrescribedPlasma} <: PlasmaSource
    prescribed::P
    temperature::Symbol
    density::Union{Nothing, Symbol}
    temperature_floor::Float64
    ions::Union{Nothing, Symbol}
    composition::Base.RefValue{Any}   # (layout, the species of the ions) after the check against that layout
end
function PlasmaFromUnknowns(prescribed; temperature::Symbol = :cold_electron_temperature,
        density::Union{Nothing, Symbol} = :cold_electron_density,
        ions::Union{Nothing, Symbol} = nothing, temperature_floor::Real = DEFAULT_TEMPERATURE_FLOOR)
    temperature_floor >= 0 ||
        throw(ArgumentError("the temperature floor must not be negative"))
    ions !== nothing && density === nothing &&
        throw(ArgumentError("a plasma whose ions are the unknown $ions takes its density from an unknown: the " *
                            "prescribed density does not follow the charge states"))
    return PlasmaFromUnknowns(PrescribedPlasma(prescribed), temperature,
        density, Float64(temperature_floor), ions,
        Ref{Any}(nothing))
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
source_dependencies(::Union{Nothing, PlasmaSource}) = ()
# the ions need the density as an unknown (constructor)
function source_dependencies(source::PlasmaFromUnknowns)
    if source.density === nothing
        read_unknowns = (source.temperature,)
    elseif source.ions === nothing
        read_unknowns = (source.temperature, source.density)
    else
        read_unknowns = (source.temperature, source.density, source.ions)
    end
    return read_unknowns
end

"""
    ion_dependency(source)

The name of the unknown of the ion densities that a [`PlasmaSource`](@ref) reads, or
`nothing`.
"""
ion_dependency(::Union{Nothing, PlasmaSource}) = nothing
ion_dependency(source::PlasmaFromUnknowns) = source.ions

"""
    with_state_ions(plasma::PlasmaState, u, layout, name)

The plasma with the densities of its ion charge states taken from the radial unknown
`name` of the state `u`, in the number type of `u`: a term that
evaluates its residual with dual numbers keeps their derivative parts, so its Jacobian
holds the columns of the ion densities. The species are those of `plasma`, in its order
([`IonComposition`](@ref)).
"""
function with_state_ions(plasma::PlasmaState, u::AbstractVector, layout, name::Symbol)
    id = unknown_id(layout, name)
    composition = IonComposition(plasma.ions)
    _check_ion_unknown(layout, id, composition)
    T = promote_type(eltype(plasma.cold_electron_temperature), eltype(u))
    nr = radial_cell_count(layout)
    ions = IonSpecies{T}[]
    for (s, species) in enumerate(plasma.ions)
        densities = Matrix{T}(undef, species.nuclear_charge + 1, nr)
        for ir in 1:nr
            values = radial_view(u, layout, id, ir)
            for (j, c) in enumerate(species_components(composition, s))
                densities[j, ir] = values[c]
            end
        end
        push!(ions, IonSpecies(species.name, species.nuclear_charge, densities))
    end
    return PlasmaState{T}(
        T.(plasma.cold_electron_temperature), T.(plasma.cold_electron_density), ions,
        T.(plasma.electric_field), T.(plasma.magnetic_field), plasma.cold_density_given)
end

# the plasma of a term with the ion densities of the state when it reads them
_term_plasma(plasma::PlasmaState, ::Nothing, u, layout) = plasma
function _term_plasma(plasma::PlasmaState, ions::Symbol, u, layout)
    with_state_ions(plasma, u, layout, ions)
end

# the ion species of `ions` with the charge-state densities of the unknown `name` of the
# state u, without their derivative parts
function _ions_from_unknowns(ions::Vector{IonSpecies{T}}, u, layout, name::Symbol,
        cache::Base.RefValue{Any}) where {T}
    id = unknown_id(layout, name)
    # the composition is checked against the unknown once per layout: the species of a
    # plasma do not change in time
    checked = cache[]
    if checked isa Tuple && first(checked) === layout
        composition = last(checked)::IonComposition
    else
        composition = IonComposition(ions)
        _check_ion_unknown(layout, id, composition)
        cache[] = (layout, composition)
    end
    nr = radial_cell_count(layout)
    species = IonSpecies{T}[]
    for (k, s) in enumerate(ions)
        densities = Matrix{T}(undef, s.nuclear_charge + 1, nr)
        for ir in 1:nr
            values = radial_view(u, layout, id, ir)
            for (j, c) in enumerate(species_components(composition, k))
                densities[j, ir] = _value(values[c])
            end
        end
        push!(species, IonSpecies(s.name, s.nuclear_charge, densities))
    end
    return species
end

# the unknown `id` holds the charge states of `composition`, one group per species
function _check_ion_unknown(layout, id, composition::IonComposition)
    u = layout.unknowns[id]
    u isa RadialUnknown && u.components == component_count(composition) ||
        throw(ArgumentError("the ion densities $(u.name) need $(component_count(composition)) components, one per " *
                            "charge state of $(join(composition.names, ", "))"))
    component_groups(u) ==
    [species_components(composition, s) for s in 1:length(composition)] ||
        throw(ArgumentError("the ion densities $(u.name) need one group of components per species " *
                            "(ion_densities_unknown)"))
    return nothing
end

function plasma_values(source::PlasmaFromUnknowns, u, layout, t; ion_state = u)
    u === nothing &&
        throw(ArgumentError("a plasma from the unknowns $(join(source_dependencies(source), " and ")) " *
                            "needs the state of the system"))
    base = plasma_values(source.prescribed, u, layout, t)
    T = eltype(base.cold_electron_temperature)
    nr = radial_cell_count(base)
    temperature = unknown_values(u, layout, unknown_id(layout, source.temperature))
    temperatures = Vector{T}(undef, nr)
    for ir in 1:nr
        value = _value(temperature[ir])
        if !(value > source.temperature_floor)
            throw(DomainError(value,
                "the cold-electron temperature $(source.temperature) at radius $ir is at or " *
                "below the floor $(source.temperature_floor) eV"))
        end
        temperatures[ir] = value
    end
    densities = if source.density === nothing
        base.cold_electron_density
    else
        density = unknown_values(u, layout, unknown_id(layout, source.density))
        T[_value(density[ir]) for ir in 1:nr]
    end
    if source.ions === nothing
        ions = base.ions
    else
        ions = _ions_from_unknowns(
            base.ions, ion_state, layout, source.ions, source.composition)
    end
    return PlasmaState{T}(
        temperatures, densities, ions, base.electric_field, base.magnetic_field,
        base.cold_density_given || source.density !== nothing)
end

# the values of the unknowns a source reads, all components, for the key of a cache
function _each_source_value(f, source::PlasmaFromUnknowns, u, layout)
    for name in source_dependencies(source)
        id = unknown_id(layout, name)
        for ir in 1:radial_cell_count(layout), x in radial_view(u, layout, id, ir)

            f(Float64(_value(x)))
        end
    end
    return nothing
end

function _source_state!(values::Vector{Float64}, source::PlasmaFromUnknowns, u, layout)
    empty!(values)
    _each_source_value(x -> push!(values, x), source, u, layout)
    return values
end

function _same_source_state(stored::Vector{Float64}, source::PlasmaFromUnknowns, u, layout)
    k = 0
    for name in source_dependencies(source)
        id = unknown_id(layout, name)
        for ir in 1:radial_cell_count(layout), x in radial_view(u, layout, id, ir)

            k += 1
            (k <= length(stored) && isequal(stored[k], Float64(_value(x)))) || return false
        end
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
    push!(values, _value(plasma.cold_electron_temperature[ir]),
        _value(plasma.cold_electron_density[ir]),
        _value(plasma.magnetic_field[ir]))
    if electric_field
        push!(values, _value(plasma.electric_field[ir]))
    end
    for species in plasma.ions
        push!(values, species.nuclear_charge)
        for Z0 in 0:species.nuclear_charge
            push!(values, _value(species.charge_state_densities[Z0 + 1, ir]))
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
function update_key!(key::PlasmaKey, plasma::PlasmaState; electric_field::Bool = true)
    nr = radial_cell_count(plasma)
    changed = length(key.values) != nr
    if changed
        resize!(key.values, nr)
        for ir in 1:nr
            key.values[ir] = Float64[]
        end
    end
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
    if changed
        resize!(key, length(values))
    end
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
function _key_current(key::PlasmaKey, source::PrescribedPlasma{<:PlasmaState}, time)
    !isempty(key.values)
end
function _key_current(key::PlasmaKey, source::PrescribedPlasma, time)
    !isempty(key.values) && key.time[] == time
end
_key_current(key::PlasmaKey, source::PlasmaSource, time) = false
function _key_current(key::PlasmaKey, source::PlasmaSource, time, u, layout)
    _key_current(key, source, time)
end
# a plasma from the unknowns: its prescribed part depends on the time, the rest on the
# values of the unknowns it reads
function _key_current(key::PlasmaKey, source::PlasmaFromUnknowns, time, u, layout)
    !isempty(key.values) && key.time[] == time &&
        _same_source_state(key.state, source, u, layout)
end

"""
    changed_plasma!(key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field=true)

The plasma of `source` at the state `u` and the time `t` when its values differ from those
of `key` ([`update_key!`](@ref), which stores them), and `nothing` when they do not. A
prescribed source is compared once per time, so the evaluations of a term at one time
cost no comparison. `ion_state` is the state the ion densities of a
[`PlasmaFromUnknowns`](@ref) are taken from, the start of the step for a term that lags them.
"""
function changed_plasma!(
        key::PlasmaKey, source::PlasmaSource, u, layout, t; electric_field::Bool = true,
        ion_state = u)
    time = Float64(_value(t))
    # ions of another state (lagged): the shortcut compares the unknowns of `u` alone
    ion_state === u && _key_current(key, source, time, u, layout) && return nothing
    if ion_state === u
        plasma = plasma_values(source, u, layout, t)
    else
        plasma = plasma_values(source, u, layout, t; ion_state)
    end
    changed = update_key!(key, plasma; electric_field)
    key.time[] = time
    if is_state_dependent(source)
        _source_state!(key.state, source, u, layout)
    end
    return changed ? plasma : nothing
end

"""
    reset_key!(key)

Empty `key`, so that the next [`update_key!`](@ref) reports a change: after a rebuild that
failed.
"""
function reset_key!(key::PlasmaKey)
    empty!(key.values)
    return key
end
reset_key!(key::Vector{Float64}) = empty!(key)

# tag of the dual numbers of the plasma columns
struct PlasmaColumnTag end

"""
    dual_plasma(plasma::PlasmaState, with_density::Bool, with_ions::Bool=false)

The plasma with the cold-electron temperature as a dual number of partial 1 at every
radius, with `with_density` the cold-electron density as partial 2, and with `with_ions`
the densities of the charge states as the partials after them, in the order of the
components of the ion densities ([`IonComposition`](@ref)); the rest
holds zero partials. The coefficients at a radius depend only on the plasma of that
radius, so partial `k` of a coefficient at radius `ir` is its derivative with respect to
plasma scalar `k` at `ir`, for all radii in one evaluation.
[`plasma_columns`](@ref) gives the entries of the state of the partials.
"""
function dual_plasma(plasma::PlasmaState{Float64}, with_density::Bool, with_ions::Bool = false)
    scalars = with_density ? 2 : 1
    K = scalars + (with_ions ? component_count(IonComposition(plasma.ions)) : 0)
    return _dual_plasma(plasma, with_density, with_ions, Val(K))
end

function _dual_plasma(plasma::PlasmaState{Float64}, with_density::Bool, with_ions::Bool, ::Val{K}) where {K}
    D = ForwardDiff.Dual{PlasmaColumnTag, Float64, K}
    seeded(x, k) = D(x, ForwardDiff.Partials(ntuple(i -> Float64(i == k), Val(K))))
    temperature = [seeded(x, 1) for x in plasma.cold_electron_temperature]
    if with_density
        density = [seeded(x, 2) for x in plasma.cold_electron_density]
    else
        density = D.(plasma.cold_electron_density)
    end
    ions = IonSpecies{D}[]
    k = with_density ? 2 : 1
    for s in plasma.ions
        densities = D.(s.charge_state_densities)
        if with_ions
            for Z0 in 0:s.nuclear_charge
                k += 1
                densities[Z0 + 1, :] .= [seeded(x, k)
                                         for x in s.charge_state_densities[Z0 + 1, :]]
            end
        end
        push!(ions, IonSpecies(s.name, s.nuclear_charge, densities))
    end
    return PlasmaState{D}(
        temperature, density, ions, D.(plasma.electric_field), D.(plasma.magnetic_field),
        plasma.cold_density_given)
end

"""
    plasma_columns(source::PlasmaFromUnknowns, layout, ir; with_ions=true)

The entries of the state at radius `ir` that the partials of [`dual_plasma`](@ref) belong
to: the temperature, the density when the source reads it, and with `with_ions` the
components of the ion densities when the source reads them, as the argument of
`dual_plasma`.
"""
function plasma_columns(source::PlasmaFromUnknowns, layout, ir; with_ions::Bool = true)
    columns = [first(unknown_range(layout, unknown_id(layout, source.temperature), ir))]
    if !(source.density === nothing)
        push!(columns, first(unknown_range(layout, unknown_id(layout, source.density), ir)))
    end
    if with_ions && source.ions !== nothing
        append!(columns, unknown_range(layout, unknown_id(layout, source.ions), ir))
    end
    return columns
end
