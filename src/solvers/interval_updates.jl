# Updates of the setup between the intervals of a run. `integrate`
# applies them at every requested time point, before the interval that starts there, so
# a step that is halved keeps the setup of its interval. What an update changes in time is
# state of the run: a checkpoint stores it and a resume restores it.

"""
    IntervalUpdate

Supertype of the updates [`integrate`](@ref) applies at the start of every interval
(`updates`). An update implements

- `update_name(update)`: its name, unique among the updates of a run;
- `apply_update!(update, u, layout, t)`: apply it for the interval that starts at the time
  `t` with the state `u`; returns whether the setup changed;
- `update_state(update)`: the numbers the update has set, as a vector, stored with every
  saved state and in every checkpoint;
- `restore_update!(update, state)`: set the setup back to a stored state;
- `describe(update, context)`: its record in the settings of the run.
"""
abstract type IntervalUpdate end

"""
    free_critical_momentum(plasma, ir)

Critical momentum at radius `ir` from the Connor-Hastie critical field of the cold
electrons with the relativistic Coulomb logarithm (DREAM's `Ec_free`,
`RunawayFluid.cpp`), [`critical_momentum`](@ref) of `|E|`. Screening and pitch-angle
scattering raise the critical momentum, so this is a lower estimate, which keeps the
knock-on cutoff below the critical momentum.
"""
function free_critical_momentum(plasma::PlasmaState, ir::Integer)
    temperature, density = plasma.cold_electron_temperature[ir], plasma.cold_electron_density[ir]
    field = critical_field(density, relativistic_coulomb_logarithm(temperature, density))
    return critical_momentum(abs(plasma.electric_field[ir]), field)
end

"""
    KnockOnCutoffUpdate(operator, plasma_at; name="knockon_cutoff", thermal_factor=4,
                        critical_momentum_at=free_critical_momentum)

Knock-on cutoff that follows the plasma: at the start of every
interval the cutoff of each radius of `operator` is chosen inside the window
`thermal_factor · p_Te ≤ p_c < p_crit` ([`cutoff_window`](@ref)) of the plasma
`plasma_at(t)`, on a face of the knock-on momentum grid ([`choose_cutoff`](@ref)). The
cutoff in force is kept while it stays inside the window, so the kernels are rebuilt only
when the window moves past it; that value is the state of the update.
`critical_momentum_at(plasma, ir)` gives the upper bound. `inside_window` holds, per
radius, whether the last choice was inside the window, and `rebuilds` counts the kernel
sets built.

The operator's cutoff at the start of the run is the one it was built with, which is
kept when it is a face inside the window. Without the update the cutoff is fixed, as in
DREAM (`pCutAvalanche`).
"""
struct KnockOnCutoffUpdate{T,F,C} <: IntervalUpdate
    name::String
    operator::KnockOnOperator{T}
    plasma_at::F
    critical_momentum_at::C
    thermal_factor::Float64
    inside_window::Vector{Bool}
    rebuilds::Base.RefValue{Int}
end

function KnockOnCutoffUpdate(operator::KnockOnOperator, plasma_at; name::AbstractString="knockon_cutoff",
                             thermal_factor::Real=4, critical_momentum_at=free_critical_momentum)
    thermal_factor > 0 || throw(ArgumentError("thermal_factor must be positive"))
    return KnockOnCutoffUpdate(String(name), operator, plasma_at, critical_momentum_at, Float64(thermal_factor),
                               fill(true, radial_cell_count(operator.knockon)), Ref(0))
end

update_name(update::KnockOnCutoffUpdate) = update.name

function apply_update!(update::KnockOnCutoffUpdate, u, layout, t)
    operator = update.operator
    plasma = update.plasma_at(t)
    nr = radial_cell_count(operator.knockon)
    radial_cell_count(plasma) == nr || throw(DimensionMismatch("plasma and knock-on grid radii differ"))
    cutoffs = copy(operator.p_cutoff)
    for ir in 1:nr
        window = cutoff_window(thermal_momentum(plasma.cold_electron_temperature[ir]),
                               update.critical_momentum_at(plasma, ir); thermal_factor=update.thermal_factor)
        choice = choose_cutoff(momentum_grid(operator.knockon, ir).p_faces, window.lower, window.upper;
                               current_cutoff=operator.p_cutoff[ir])
        cutoffs[ir] = choice.p_cutoff
        update.inside_window[ir] = choice.inside_window
    end
    cutoffs == operator.p_cutoff && return false
    update.rebuilds[] += set_cutoff!(operator, cutoffs)
    return true
end

update_state(update::KnockOnCutoffUpdate) = collect(Float64, update.operator.p_cutoff)

function restore_update!(update::KnockOnCutoffUpdate, state::AbstractVector)
    length(state) == length(update.operator.p_cutoff) ||
        throw(DimensionMismatch("the stored cutoff of $(update.name) has $(length(state)) radii, " *
                                "the operator $(length(update.operator.p_cutoff))"))
    update.rebuilds[] += set_cutoff!(update.operator, state)
    return update
end

settings_fields(::Type{<:KnockOnCutoffUpdate}) =
    (described=(:name, :operator, :plasma_at, :critical_momentum_at, :thermal_factor),
     derived=(:inside_window, :rebuilds))

function describe(update::KnockOnCutoffUpdate, context::DescriptionContext)
    return SettingsRecord("type" => "KnockOnCutoffUpdate", "name" => update.name,
                          "operator" => describe(update.operator, context),
                          "thermal_factor" => update.thermal_factor,
                          "critical_momentum" => _callable_name(update.critical_momentum_at),
                          "plasma" => _callable_name(update.plasma_at),
                          "plasma_table" => prescribed_table!(context, update.plasma_at))
end

# an update without a record of its own
describe(update::IntervalUpdate, context::DescriptionContext) =
    SettingsRecord("type" => _type_name(update), "name" => update_name(update), "described" => false,
                   "reason" => "no describe method")

# the updates of a run as a tuple, with unique names
function _checked_updates(updates)
    list = updates isa IntervalUpdate ? (updates,) : Tuple(updates)
    all(update -> update isa IntervalUpdate, list) ||
        throw(ArgumentError("updates must be IntervalUpdate objects"))
    names = [update_name(update) for update in list]
    allunique(names) || throw(ArgumentError("the updates of a run need different names, got $(join(names, ", "))"))
    for name in names
        isempty(name) && throw(ArgumentError("an update needs a name"))
        occursin('/', name) && throw(ArgumentError("the name \"$name\" of an update must not contain '/'"))
    end
    return list
end
