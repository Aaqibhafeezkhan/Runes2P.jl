# Radial transport that changes during a run: the level of the magnetic
# perturbation as a function of time, set at the start of every interval.

"""
    PiecewiseConstant(values, switch_times)

Function of time that holds `values[k]` from `switch_times[k-1]` up to `switch_times[k]`
(`values[1]` before the first switch time, the last value from the last switch time on).
At a switch time it has the value of the interval that starts there. `values` are numbers
or vectors. A typical transport history of a disruption run is of this form: a level of δB/B
for the first milliseconds and a hundredth of it afterwards,
`PiecewiseConstant([1e-3, 1e-5], [1e-3])`.

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
        issorted(switch_times; lt=<=) || throw(ArgumentError("the switch times must be strictly increasing"))
        return new{eltype(values)}(collect(values), collect(Float64, switch_times))
    end
end

# a time point within round-off below a switch time counts as the switch time
(history::PiecewiseConstant)(t::Real) =
    history.values[searchsortedlast(history.switch_times, t + 1e-12 * abs(t)) + 1]

"""
    forced_time_points(update)
    forced_time_points(function_of_time)

Times at which the setup of an update changes abruptly. [`integrate`](@ref) requires those
inside the run to be among its time points. Empty by default, the switch times of a
[`PiecewiseConstant`](@ref).
"""
forced_time_points(x) = Float64[]
forced_time_points(history::PiecewiseConstant) = history.switch_times

"""
    RechesterRosenbluthUpdate(operator, delta_b_over_b_at; major_radius, safety_factor=1,
                              name="radial_transport")

Radial diffusion of `operator` that follows the level of the magnetic perturbation in time:
at the start of every interval of the run, at the time `t`,
`operator.diffusion_r_r` is set to the Rechester-Rosenbluth coefficient of
`delta_b_over_b_at(t)` ([`add_rechester_rosenbluth_diffusion!`](@ref)): a number, one value
per radial face, or a callable of r such as an interpolator, evaluated on the faces
([`radial_face_values`](@ref)). A profile on points of its own that is constant in time is
`t -> profile`. The safety factor takes the same forms and is evaluated once. The interval from `t_k` to `t_{k+1}` takes the level at `t_k`. The
levels of the faces are the state of the update, stored with the checkpoints and restored
on resume.

`operator` is the operator of the radial transport alone: the update owns its
`diffusion_r_r`. A [`PiecewiseConstant`](@ref) level needs its switch times among the time
points of the run. DREAM interpolates the table of its `dBB` linearly in time and takes the
value at the end of a step.
"""
struct RechesterRosenbluthUpdate{T,F} <: IntervalUpdate
    name::String
    operator::AdvectionDiffusion{T}
    delta_b_over_b_at::F
    major_radius::Float64
    safety_factor::Vector{Float64}     # per radial face
    safety_factor_profile::Any         # as given, for the record
    delta_b_over_b::Vector{Float64}    # per radial face, the levels of the interval
end

function RechesterRosenbluthUpdate(operator::AdvectionDiffusion, delta_b_over_b_at; major_radius::Real,
                                   safety_factor=1.0, name::AbstractString="radial_transport")
    major_radius > 0 || throw(ArgumentError("the major radius must be positive"))
    q = radial_face_values(safety_factor, operator.grid)
    all(>(0), q) || throw(ArgumentError("the safety factor must be positive"))
    return RechesterRosenbluthUpdate(String(name), operator, delta_b_over_b_at, Float64(major_radius), q,
                                     safety_factor, fill(NaN, length(q)))
end

update_name(update::RechesterRosenbluthUpdate) = update.name
forced_time_points(update::RechesterRosenbluthUpdate) = forced_time_points(update.delta_b_over_b_at)

function _set_levels!(update::RechesterRosenbluthUpdate, levels)
    fill!(update.operator.diffusion_r_r, 0)
    add_rechester_rosenbluth_diffusion!(update.operator, levels; major_radius=update.major_radius,
                                        safety_factor=update.safety_factor)
    copyto!(update.delta_b_over_b, levels)
    return update
end

function apply_update!(update::RechesterRosenbluthUpdate, u, layout, t)
    levels = radial_face_values(update.delta_b_over_b_at(t), update.operator.grid)
    levels == update.delta_b_over_b && return false
    _set_levels!(update, levels)
    return true
end

# The rebuild of a prescribed term clears every coefficient of its operator, the radial ones
# too, and the update writes them only when the level changes: on a shared operator the
# transport would be gone without an error.
function check_update(update::RechesterRosenbluthUpdate, system)
    # a prescribed term is reached as a term of an equation or through the coefficients that
    # an interface or a boundary flux reads
    for equation in system.equations, outer in equation.terms, term in (outer, _coefficient_sources(outer)...)
        has_plasma_source(term) && term.operator === update.operator &&
            throw(ArgumentError("the update $(update.name) sets the radial transport of the operator of the " *
                                "Fokker-Planck term of $(equation.unknown) with a plasma source, whose rebuild " *
                                "clears it. Give the " *
                                "radial transport an operator and a FokkerPlanckTerm of its own."))
    end
    return nothing
end

update_state(update::RechesterRosenbluthUpdate) = copy(update.delta_b_over_b)

function restore_update!(update::RechesterRosenbluthUpdate, state::AbstractVector)
    length(state) == length(update.delta_b_over_b) ||
        throw(DimensionMismatch("the stored levels of $(update.name) have $(length(state)) radial faces, " *
                                "the operator $(length(update.delta_b_over_b))"))
    # a run that stopped before its first interval has no levels yet
    any(isnan, state) || _set_levels!(update, collect(Float64, state))
    return update
end

settings_fields(::Type{<:PiecewiseConstant}) = (described=(:values, :switch_times), derived=())
settings_fields(::Type{<:RechesterRosenbluthUpdate}) =
    (described=(:name, :operator, :delta_b_over_b_at, :major_radius, :safety_factor, :safety_factor_profile),
     derived=(:delta_b_over_b,))

function describe(history::PiecewiseConstant)
    record = SettingsRecord("type" => "PiecewiseConstant", "switch_times" => _plain(history.switch_times))
    if any(_is_radial_profile, history.values)
        # profiles on points of their own, one record per interval
        record["profiles"] = SettingsRecord("interval_$k" => _is_radial_profile(v) ? _describe_profile(v) :
                                                             _describe_value(v)
                                            for (k, v) in enumerate(history.values))
    else
        record["values"] = _plain(stack(history.values))
    end
    return record
end

"""
    describe_prescribed_levels(levels_at, context; points=nothing)

Record of a level prescribed as a function of time (a number or one value per radial
face): the record of a [`PiecewiseConstant`](@ref), or the values at the time points of
`context`, thinned as the tables of the prescribed plasma are. With `points`, the function
may return a callable of r ([`radial_values`](@ref)): the values are those at `points`, and
`profile` records the callable (an interpolator by its points), or `profiles` one per
sampled time where it changes in time.
"""
function describe_prescribed_levels(levels_at, context::DescriptionContext; points=nothing)
    levels_at isa PiecewiseConstant && return describe(levels_at)
    record = SettingsRecord("type" => "prescribed levels", "source" => _callable_name(levels_at))
    function incomplete(reason)
        record["described"] = false
        record["reason"] = reason
        return record
    end
    isempty(context.times) && return incomplete("no time points to sample the function at")
    values = try
        first_value = levels_at(first(context.times))
        width = points !== nothing && _is_radial_profile(first_value) ? length(points) : length(first_value)
        max_samples = context.max_table_bytes ÷ (8 * width)
        indices = _sample_indices(length(context.times), max_samples)
        record["times"] = context.times[indices]
        record["thinned"] = length(indices) < length(context.times)
        sampled = [levels_at(t) for t in context.times[indices]]
        profiles = points === nothing ? [] : filter(_is_radial_profile, sampled)
        if !isempty(profiles)
            if all(p -> p == first(profiles), profiles)
                record["profile"] = _describe_profile(first(profiles))
            else
                record["profiles"] = SettingsRecord("sample_$k" => _is_radial_profile(v) ? _describe_profile(v) :
                                                                   _describe_value(v)
                                                    for (k, v) in enumerate(sampled))
            end
        end
        # numbers and values are recorded as given, profiles by their values at the points
        [collect(Float64, isempty(profiles) ? vcat(v) : radial_values(v, points)) for v in sampled]
    catch error
        return incomplete("the function failed at a time point: " * sprint(showerror, error))
    end
    record["time_count"] = length(context.times)
    record["values"] = _plain(stack(values))
    return record
end

# record of a radial profile given as a callable: an interpolator by its points (extension),
# a function by its name
_describe_profile(profile) = hasmethod(describe, Tuple{typeof(profile)}) ? describe(profile) :
                             SettingsRecord("type" => "callable", "source" => _callable_name(profile))

function describe(update::RechesterRosenbluthUpdate, context::DescriptionContext)
    record = SettingsRecord("type" => "RechesterRosenbluthUpdate", "name" => update.name,
                            "operator" => describe(update.operator, context; coefficients=false),
                            "major_radius" => update.major_radius,
                            "safety_factor" => _plain(update.safety_factor),
                            "delta_b_over_b" => describe_prescribed_levels(update.delta_b_over_b_at, context;
                                                                           points=update.operator.grid.radial_grid.r_faces))
    _is_radial_profile(update.safety_factor_profile) &&
        (record["safety_factor_profile"] = _describe_profile(update.safety_factor_profile))
    return record
end
