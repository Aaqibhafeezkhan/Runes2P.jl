# Updates of the setup during a run. `integrate` applies them at the start of the run and
# after every accepted step, at the state and the time where that step ends, before the step
# that follows. What an update changes in time is state of the run: a checkpoint stores it
# and a resume restores it.

"""
    IntervalUpdate

Supertype of the updates [`integrate`](@ref) applies at the start of the run and after
every accepted step (`updates`), at the state and the time where that step ends. An update implements

- `update_name(update)`: its name, unique among the updates of a run;
- `apply_update!(update, u, layout, t)`: apply it for the steps that start at the time `t`
  with the state `u`; returns whether the setup changed;
- `update_state(update)`: the numbers the update has set, as a vector, stored with every
  saved state and in every checkpoint;
- `restore_update!(update, state)`: set the setup back to a stored state;
- `describe(update, context)`: its record in the settings of the run.
"""
abstract type IntervalUpdate end

"""
    KnockOnCutoffUpdate(operator, plasma_at; name="knockon_cutoff", placement=:energy_midpoint,
                        no_ceiling_factor=10, per_radius=true, thermal_factor=4,
                        critical_momentum_at=total_critical_momentum, linked=[],
                        electric_field=nothing)

Knock-on cutoff that follows the plasma `plasma_at(t)`, or a [`PlasmaSource`](@ref) such as
[`PlasmaFromUnknowns`](@ref), which it reads at the state of the start of each step (a
system whose bulk reads the temperature from an unknown refuses a prescribed temperature
here): at the start of the run and after every accepted step the cutoff of each radius of
`operator` is set to a face of the knock-on momentum grid. The cutoff per radius is the
state of the update. `critical_momentum_at(plasma, ir)` gives the runaway threshold.

- `placement = :energy_midpoint` (the default, a choice of this package; see the docs page
  *Choices of this package*) takes the rule of
  [`cutoff_target`](@ref): the logarithmic midpoint in energy between the temperature and
  the energy at the critical momentum, and `no_ceiling_factor` times the temperature where
  there is no threshold. The face is the nearest to that target in the logarithm of the
  energy, kept while the target lies between its neighbouring faces
  ([`choose_cutoff_face`](@ref)).
- `placement = :window` takes a face inside `thermal_factor · p_Te ≤ p_c < p_crit`
  ([`cutoff_window`](@ref)), kept while it stays inside the window
  ([`choose_cutoff`](@ref)), which lets the cutoff come close to either bound.
- `per_radius = false` gives every radius the cutoff of the largest temperature and the
  smallest critical momentum over the radii, the one value DREAM can take.

`inside_window` holds, per radius, whether the last choice was inside the window,
`policy` the rule that gave the target (`:log_midpoint`, `:no_ceiling`, `:closed`, or
`:window`), and `rebuilds` counts the kernel sets built.

The cutoff of the first interval replaces the one the operator was built with, unless that
one is a face the rule keeps. Without the update the cutoff is fixed, as in DREAM
(`pCutAvalanche`), where a run can change it only between restarts.

`electric_field` names the radial unknown of the electric field when the field is an unknown
of the system: the runaway threshold then takes the field of the state at the start of the
step in place of the field of the prescribed plasma.

`linked` lists further knock-on operators that take the cutoff of `operator` at every
change. With a hot-tail and a runaway grid `operator` is the pairing hot-tail from hot-tail,
whose knock-on grid holds the cutoff, and the pairings hot-tail from runaway and runaway
from runaway are linked, so that every knock-on above the one cutoff is counted once, on one
of the two grids.
"""
struct KnockOnCutoffUpdate{T, F, C} <: IntervalUpdate
    name::String
    operator::KnockOnOperator{T}
    plasma_at::F
    critical_momentum_at::C
    placement::Symbol
    no_ceiling_factor::Float64
    per_radius::Bool
    thermal_factor::Float64
    linked::Vector{KnockOnOperator{T}}
    electric_field::Union{Nothing, Symbol}
    inside_window::Vector{Bool}
    policy::Vector{Symbol}
    rebuilds::Base.RefValue{Int}
end

function KnockOnCutoffUpdate(
        operator::KnockOnOperator, plasma_at; name::AbstractString = "knockon_cutoff",
        placement::Symbol = :energy_midpoint, no_ceiling_factor::Real = 10,
        per_radius::Bool = true, thermal_factor::Real = 4,
        critical_momentum_at = total_critical_momentum, linked = KnockOnOperator[],
        electric_field::Union{Nothing, Symbol} = nothing)
    placement in (:energy_midpoint, :window) ||
        throw(ArgumentError("placement must be :energy_midpoint or :window, not $placement"))
    thermal_factor > 0 || throw(ArgumentError("thermal_factor must be positive"))
    no_ceiling_factor > 1 || throw(ArgumentError("no_ceiling_factor must be above 1"))
    nr = radial_cell_count(operator.knockon)
    linked_operators = typeof(operator)[other for other in linked]
    for other in linked_operators
        radial_cell_count(other.knockon) == nr ||
            throw(DimensionMismatch("a linked knock-on operator has $(radial_cell_count(other.knockon)) " *
                                    "radii, the operator of the update $nr"))
        other === operator &&
            throw(ArgumentError("the operator of the update is linked to itself"))
    end
    return KnockOnCutoffUpdate(
        String(name), operator, plasma_at, critical_momentum_at, placement,
        Float64(no_ceiling_factor), per_radius, Float64(thermal_factor),
        linked_operators, electric_field, fill(true, nr),
        fill(placement === :window ? :window : :no_ceiling, nr), Ref(0))
end

update_name(update::KnockOnCutoffUpdate) = update.name

function apply_update!(update::KnockOnCutoffUpdate, u, layout, t)
    operator = update.operator
    source = update.plasma_at
    plasma = source isa PlasmaSource ? plasma_values(source, u, layout, t) : source(t)
    nr = radial_cell_count(operator.knockon)
    radial_cell_count(plasma) == nr ||
        throw(DimensionMismatch("plasma and knock-on grid radii differ"))
    if update.electric_field !== nothing
        field = radial_values(u, layout, unknown_id(layout, update.electric_field))
        plasma = with_electric_field(plasma, collect(Float64, field))
    end
    temperatures = Float64.(plasma.cold_electron_temperature)
    thresholds = Float64[update.critical_momentum_at(plasma, ir) for ir in 1:nr]
    if !update.per_radius
        fill!(temperatures, maximum(temperatures))
        fill!(thresholds, minimum(thresholds))
    end
    cutoffs = copy(operator.p_cutoff)
    for ir in 1:nr
        faces = momentum_grid(operator.knockon, ir).p_faces
        choice = if update.placement === :window
            window = cutoff_window(thermal_momentum(temperatures[ir]), thresholds[ir];
                thermal_factor = update.thermal_factor)
            choose_cutoff(faces, window.lower, window.upper; current_cutoff = operator.p_cutoff[ir])
        else
            target = cutoff_target(temperatures[ir], thresholds[ir];
                no_ceiling_factor = update.no_ceiling_factor)
            update.policy[ir] = target.policy
            # a closed window has no upper bound to keep: the lower one counts
            upper = target.policy === :closed ? Inf : thresholds[ir]
            chosen = choose_cutoff_face(
                faces, target.p_cutoff, momentum_at_energy(temperatures[ir]), upper;
                current_cutoff = operator.p_cutoff[ir])
            (p_cutoff = chosen.p_cutoff,
                inside_window = chosen.inside_window && target.policy !== :closed)
        end
        cutoffs[ir] = choice.p_cutoff
        update.inside_window[ir] = choice.inside_window
    end
    changed = cutoffs != operator.p_cutoff
    changed && (update.rebuilds[] += set_cutoff!(operator, cutoffs))
    changed |= _set_linked_cutoffs!(update, cutoffs)
    return changed
end

# the cutoff of the update's operator on every linked operator; whether any changed
function _set_linked_cutoffs!(update::KnockOnCutoffUpdate, cutoffs)
    changed = false
    for other in update.linked
        cutoffs == other.p_cutoff && continue
        update.rebuilds[] += set_cutoff!(other, cutoffs)
        changed = true
    end
    return changed
end

update_state(update::KnockOnCutoffUpdate) = collect(Float64, update.operator.p_cutoff)

function restore_update!(update::KnockOnCutoffUpdate, state::AbstractVector)
    length(state) == length(update.operator.p_cutoff) ||
        throw(DimensionMismatch("the stored cutoff of $(update.name) has $(length(state)) radii, " *
                                "the operator $(length(update.operator.p_cutoff))"))
    update.rebuilds[] += set_cutoff!(update.operator, state)
    _set_linked_cutoffs!(update, update.operator.p_cutoff)
    return update
end

function settings_fields(::Type{<:KnockOnCutoffUpdate})
    (
        described = (:name, :operator, :plasma_at, :critical_momentum_at,
            :placement, :no_ceiling_factor,
            :per_radius, :thermal_factor, :linked, :electric_field),
        derived = (:inside_window, :policy, :rebuilds))
end

function describe(update::KnockOnCutoffUpdate, context::DescriptionContext)
    record = SettingsRecord("type" => "KnockOnCutoffUpdate", "name" => update.name,
        "operator" => describe(update.operator, context),
        "placement" => String(update.placement),
        "no_ceiling_factor" => update.no_ceiling_factor,
        "per_radius" => update.per_radius,
        "thermal_factor" => update.thermal_factor,
        "critical_momentum" => _callable_name(update.critical_momentum_at))
    if update.plasma_at isa PlasmaSource
        _describe_plasma_source!(record, update.plasma_at, context)
    else
        record["plasma"] = _callable_name(update.plasma_at)
        record["plasma_table"] = prescribed_table!(context, update.plasma_at)
    end
    update.electric_field === nothing ||
        (record["electric_field"] = String(update.electric_field))
    isempty(update.linked) ||
        (record["linked_operators"] = SettingsRecord("operator_$k" =>
                                                         describe(other, context)
        for (k, other) in enumerate(update.linked)))
    return record
end

# an update without a record of its own
function describe(update::IntervalUpdate, context::DescriptionContext)
    SettingsRecord(
        "type" => _type_name(update), "name" => update_name(update), "described" => false,
        "reason" => "no describe method")
end

"""
    check_update(update, system)

Refuse an update that does not fit the equation system of the run, before the first step.
Nothing by default.
"""
check_update(update::IntervalUpdate, system) = nothing

# the cutoff follows the temperature of the bulk: the unknown of the temperature when the
# terms of the bulk read one
function check_update(update::KnockOnCutoffUpdate, system)
    temperature = _bulk_temperature_unknown(system.equations)
    temperature === nothing && return nothing
    source = update.plasma_at
    source isa PlasmaFromUnknowns && source.temperature === temperature ||
        throw(ArgumentError("the knock-on cutoff update $(update.name) takes a prescribed temperature, while the " *
                            "terms of the cold bulk read the unknown $temperature: give it " *
                            "PlasmaFromUnknowns(plasma_at; temperature = :$temperature), which it reads at the " *
                            "start of each step"))
    return nothing
end

# the updates of a run as a tuple, with unique names
function _checked_updates(updates)
    list = updates isa IntervalUpdate ? (updates,) : Tuple(updates)
    all(update -> update isa IntervalUpdate, list) ||
        throw(ArgumentError("updates must be IntervalUpdate objects"))
    names = [update_name(update) for update in list]
    allunique(names) ||
        throw(ArgumentError("the updates of a run need different names, got $(join(names, ", "))"))
    for name in names
        isempty(name) && throw(ArgumentError("an update needs a name"))
        occursin('/', name) &&
            throw(ArgumentError("the name \"$name\" of an update must not contain '/'"))
    end
    return list
end
