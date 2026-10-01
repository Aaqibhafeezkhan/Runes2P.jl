# Legs of a run: the time points and the save times of a run from one arithmetic, so that
# every save time of a prescribed-step run is a time point.

"""
    Leg(duration; steps, saves)
    Leg(duration; dt, save_dt)
    Leg(duration; save_dt)

One leg of a run, `duration` seconds long, in one of three keyword forms:

- `steps` time steps of equal length with `saves` saved states, at the time points nearest
  to the multiples of `duration / saves` (DREAM's `nSaveSteps`);
- steps of length `dt` with a saved state every `save_dt`, where `dt` and `save_dt` divide
  the duration (to 1e-12 relative) and the saved states are again the time points nearest
  to the multiples of `save_dt`;
- for a [`StepController`](@ref) alone, no prescribed steps and a saved state every
  `save_dt`, which divides the duration.

The last time point of a leg is its end, and it is saved. Any other combination of
keywords, a positional form or a non-positive value throws an `ArgumentError` that names
the leg. [`time_points`](@ref) turns a list of legs into the `times` and `save_times` of
[`integrate`](@ref).
"""
struct Leg
    duration::Float64
    steps::Int            # 0 in the controller form
    saves::Int
    # the only constructor with positional fields, so that a positional call reaches the error below
    Leg(::Val{:checked}, duration, steps, saves) = new(duration, steps, saves)
end

Leg(duration, args...) =
    throw(ArgumentError("Leg takes keywords: Leg(duration; steps=, saves=), Leg(duration; dt=, save_dt=) " *
                        "or Leg(duration; save_dt=)"))

function Leg(duration::Real; steps::Union{Nothing,Integer}=nothing, saves::Union{Nothing,Integer}=nothing,
             dt::Union{Nothing,Real}=nothing, save_dt::Union{Nothing,Real}=nothing)
    duration > 0 || throw(ArgumentError("Leg($duration): the duration must be positive"))
    given = (steps !== nothing, saves !== nothing, dt !== nothing, save_dt !== nothing)
    if given == (true, true, false, false)
        steps >= 1 || throw(ArgumentError("Leg($duration; steps=$steps, saves=$saves): steps must be at least 1"))
        1 <= saves <= steps ||
            throw(ArgumentError("Leg($duration; steps=$steps, saves=$saves): saves must lie between 1 and steps"))
        return Leg(Val(:checked), Float64(duration), Int(steps), Int(saves))
    elseif given == (false, false, true, true)
        name = "Leg($duration; dt=$dt, save_dt=$save_dt)"
        dt > 0 && save_dt > 0 || throw(ArgumentError("$name: dt and save_dt must be positive"))
        save_dt >= dt || throw(ArgumentError("$name: save_dt must not be shorter than dt"))
        return Leg(Val(:checked), Float64(duration), _count(duration, dt, "dt", name), _count(duration, save_dt, "save_dt", name))
    elseif given == (false, false, false, true)
        name = "Leg($duration; save_dt=$save_dt)"
        save_dt > 0 || throw(ArgumentError("$name: save_dt must be positive"))
        return Leg(Val(:checked), Float64(duration), 0, _count(duration, save_dt, "save_dt", name))
    end
    names = join([n for (n, g) in zip(("steps", "saves", "dt", "save_dt"), given) if g], ", ")
    throw(ArgumentError("Leg($duration; $names): give steps and saves, dt and save_dt, or save_dt alone"))
end

# the number of intervals of length `spacing` in `duration`, which must be a whole number
function _count(duration, spacing, what, name)
    ratio = duration / spacing
    count = round(Int, ratio)
    count >= 1 && abs(ratio - count) <= 1e-12 * ratio ||
        throw(ArgumentError("$name: $what does not divide the duration (duration / $what = $ratio)"))
    return count
end

"""
    time_points(legs; start=0.0)

The `times` and `save_times` of a run made of `legs` ([`Leg`](@ref)), one after the other
from `start`, as a pair of vectors. Both begin with `start`. A prescribed-step leg adds
its time points, with the saved ones among them; a controller leg adds only its end to
the time points and its save times in between, which a [`StepController`](@ref) makes
ceilings of its steps. The end of each leg is `start` plus the durations so far, in one
arithmetic for both vectors.
"""
function time_points(legs; start::Real=0.0)
    isempty(legs) && throw(ArgumentError("time_points needs at least one leg"))
    times, save_times = [Float64(start)], [Float64(start)]
    for leg in legs
        leg_start = last(times)
        leg_end = leg_start + leg.duration
        if leg.steps > 0
            points = [k == leg.steps ? leg_end : leg_start + k * leg.duration / leg.steps for k in 1:leg.steps]
            append!(times, points)
            for k in 1:leg.saves
                push!(save_times, points[round(Int, k * leg.steps / leg.saves)])
            end
        else
            push!(times, leg_end)
            for k in 1:leg.saves
                push!(save_times, k == leg.saves ? leg_end : leg_start + k * leg.duration / leg.saves)
            end
        end
    end
    return times, unique!(save_times)
end
