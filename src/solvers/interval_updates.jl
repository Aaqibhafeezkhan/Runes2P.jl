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
scattering raise the critical momentum, so this is a lower estimate.

The Coulomb logarithm takes the free electron density, the cold, hot and runaway
electrons together, as DREAM's `lnLambdaC` does (agreement to 1e-8).
Before the thermal quench of a fully kinetic run most electrons are hot, and the cold
density alone overstates the logarithm.
"""
function free_critical_momentum(plasma::PlasmaState, ir::Integer)
    logarithm = relativistic_coulomb_logarithm(plasma.cold_electron_temperature[ir],
                                               free_electron_density(plasma, ir))
    field = critical_field(plasma.cold_electron_density[ir], logarithm)
    return critical_momentum(abs(plasma.electric_field[ir]), field)
end

"""
    total_critical_momentum(plasma, ir)

Critical momentum at radius `ir` from the Connor-Hastie critical field of all electrons,
free and bound, with the relativistic Coulomb logarithm of the free electrons (DREAM's
`Ec_tot`, `RunawayFluid.cpp`), [`critical_momentum`](@ref) of `|E|`. `Inf` where the
field is below that critical field.

It is not DREAM's effective critical momentum (`other/fluid/pCrit`, Hesslow et al.), which
takes the partial screening of the ions into account in the slowing down and the
pitch-angle scattering of the fast electrons. The free critical field ([`free_critical_momentum`](@ref)) can give a threshold
in a hot plasma where the effective one has none, below the thermal momentum. DREAM's effective critical momentum is not
in Runes2P yet.
"""
function total_critical_momentum(plasma::PlasmaState, ir::Integer)
    free = free_electron_density(plasma, ir)
    logarithm = relativistic_coulomb_logarithm(plasma.cold_electron_temperature[ir], free)
    field = critical_field(free + bound_electron_density(plasma, ir), logarithm)
    return critical_momentum(abs(plasma.electric_field[ir]), field)
end

"""
    KnockOnCutoffUpdate(operator, plasma_at; name="knockon_cutoff", placement=:energy_midpoint,
                        no_ceiling_factor=10, per_radius=true, thermal_factor=4,
                        critical_momentum_at=total_critical_momentum)

Knock-on cutoff that follows the plasma `plasma_at(t)`:
at the start of every interval the cutoff of each radius of `operator` is set to a face
of the knock-on momentum grid. The cutoff per radius is the state of the update.
`critical_momentum_at(plasma, ir)` gives the runaway threshold.

- `placement = :energy_midpoint`, the default and a choice of this package, takes the rule
  of [`cutoff_target`](@ref): the logarithmic midpoint in energy between the temperature
  and the energy at the critical momentum, and `no_ceiling_factor` times the temperature
  where there is no threshold. The face is the nearest to that target in the logarithm of
  the energy, kept while the target lies between its neighbouring faces
  ([`choose_cutoff_face`](@ref)).
- `placement = :window` is the earlier rule: a face inside
  `thermal_factor · p_Te ≤ p_c < p_crit` ([`cutoff_window`](@ref)), kept while it stays
  inside the window ([`choose_cutoff`](@ref)), which lets the cutoff come close to either
  bound.
- `per_radius = false` gives every radius the cutoff of the largest temperature and the
  smallest critical momentum over the radii, the one value DREAM can take.

`inside_window` holds, per radius, whether the last choice was inside the window,
`policy` the rule that gave the target (`:log_midpoint`, `:no_ceiling`, `:closed`, or
`:window`), and `rebuilds` counts the kernel sets built.

The cutoff of the first interval replaces the one the operator was built with, unless
that one is a face the rule keeps. Without the update the cutoff is fixed, as in DREAM
(`pCutAvalanche`), where a run can change it only between restarts.
"""
struct KnockOnCutoffUpdate{T,F,C} <: IntervalUpdate
    name::String
    operator::KnockOnOperator{T}
    plasma_at::F
    critical_momentum_at::C
    placement::Symbol
    no_ceiling_factor::Float64
    per_radius::Bool
    thermal_factor::Float64
    inside_window::Vector{Bool}
    policy::Vector{Symbol}
    rebuilds::Base.RefValue{Int}
end

function KnockOnCutoffUpdate(operator::KnockOnOperator, plasma_at; name::AbstractString="knockon_cutoff",
                             placement::Symbol=:energy_midpoint, no_ceiling_factor::Real=10,
                             per_radius::Bool=true, thermal_factor::Real=4,
                             critical_momentum_at=total_critical_momentum)
    placement in (:energy_midpoint, :window) ||
        throw(ArgumentError("placement must be :energy_midpoint or :window, not $placement"))
    thermal_factor > 0 || throw(ArgumentError("thermal_factor must be positive"))
    no_ceiling_factor > 1 || throw(ArgumentError("no_ceiling_factor must be above 1"))
    nr = radial_cell_count(operator.knockon)
    return KnockOnCutoffUpdate(String(name), operator, plasma_at, critical_momentum_at, placement,
                               Float64(no_ceiling_factor), per_radius, Float64(thermal_factor), fill(true, nr),
                               fill(placement === :window ? :window : :no_ceiling, nr), Ref(0))
end

update_name(update::KnockOnCutoffUpdate) = update.name

function apply_update!(update::KnockOnCutoffUpdate, u, layout, t)
    operator = update.operator
    plasma = update.plasma_at(t)
    nr = radial_cell_count(operator.knockon)
    radial_cell_count(plasma) == nr || throw(DimensionMismatch("plasma and knock-on grid radii differ"))
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
                                   thermal_factor=update.thermal_factor)
            choose_cutoff(faces, window.lower, window.upper; current_cutoff=operator.p_cutoff[ir])
        else
            target = cutoff_target(temperatures[ir], thresholds[ir]; no_ceiling_factor=update.no_ceiling_factor)
            update.policy[ir] = target.policy
            # a closed window has no upper bound to keep: the lower one counts
            upper = target.policy === :closed ? Inf : thresholds[ir]
            chosen = choose_cutoff_face(faces, target.p_cutoff, momentum_at_energy(temperatures[ir]), upper;
                                        current_cutoff=operator.p_cutoff[ir])
            (p_cutoff=chosen.p_cutoff, inside_window=chosen.inside_window && target.policy !== :closed)
        end
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
    (described=(:name, :operator, :plasma_at, :critical_momentum_at, :placement, :no_ceiling_factor,
                :per_radius, :thermal_factor),
     derived=(:inside_window, :policy, :rebuilds))

function describe(update::KnockOnCutoffUpdate, context::DescriptionContext)
    return SettingsRecord("type" => "KnockOnCutoffUpdate", "name" => update.name,
                          "operator" => describe(update.operator, context),
                          "placement" => String(update.placement),
                          "no_ceiling_factor" => update.no_ceiling_factor,
                          "per_radius" => update.per_radius,
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
