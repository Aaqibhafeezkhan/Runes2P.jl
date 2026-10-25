# Radial transport that changes during a run: the level of the magnetic perturbation as a
# function of time, set at the start of the run and after every accepted step.

"""
    RechesterRosenbluthUpdate(operator, delta_b_over_b_at; major_radius, safety_factor=1,
                              name="radial_transport")

Radial diffusion of `operator` that follows the level of the magnetic perturbation in time:
at the start of the run and after every accepted step, at the time `t` where that step ends,
`operator.diffusion_r_r` is set to the Rechester-Rosenbluth coefficient of
`delta_b_over_b_at(t)` ([`add_rechester_rosenbluth_diffusion!`](@ref)): a number, one value
per radial face, or a callable of r such as an interpolator, evaluated on the faces
([`radial_face_values`](@ref)). A profile on points of its own that is constant in time is
`t -> profile`. The safety factor takes the same forms and is evaluated once. Each step
takes the level at its start, so a [`PiecewiseConstant`](@ref) whose switch times are among
the time points gives every step from `t_k` to `t_{k+1}` the level at `t_k`. The levels of
the faces are the state of the update, stored with the checkpoints and restored on resume.

`operator` is the operator of the radial transport alone: the update owns its
`diffusion_r_r`. A [`PiecewiseConstant`](@ref) level needs its switch times among the time
points of the run. DREAM interpolates the table of its `dBB` linearly in time and takes the
value at the end of a step.
"""
struct RechesterRosenbluthUpdate{T, F} <: IntervalUpdate
    name::String
    operator::AdvectionDiffusion{T, T}
    delta_b_over_b_at::F
    major_radius::Float64
    safety_factor::Vector{Float64}     # per radial face
    safety_factor_profile::Any         # as given, for the record
    delta_b_over_b::Vector{Float64}    # per radial face, the levels of the interval
end

function RechesterRosenbluthUpdate(
        operator::AdvectionDiffusion, delta_b_over_b_at; major_radius::Real,
        safety_factor = 1.0, name::AbstractString = "radial_transport")
    major_radius > 0 || throw(ArgumentError("the major radius must be positive"))
    q = radial_face_values(safety_factor, operator.grid)
    all(>(0), q) || throw(ArgumentError("the safety factor must be positive"))
    return RechesterRosenbluthUpdate(
        String(name), operator, delta_b_over_b_at, Float64(major_radius), q,
        safety_factor, fill(NaN, length(q)))
end

update_name(update::RechesterRosenbluthUpdate) = update.name
function forced_time_points(update::RechesterRosenbluthUpdate)
    forced_time_points(update.delta_b_over_b_at)
end

function _set_levels!(update::RechesterRosenbluthUpdate, levels)
    fill!(update.operator.diffusion_r_r, 0)
    add_rechester_rosenbluth_diffusion!(
        update.operator, levels; major_radius = update.major_radius,
        safety_factor = update.safety_factor)
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
    for equation in system.equations, outer in equation.terms,
        term in (outer, _coefficient_sources(outer)...)
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

function settings_fields(::Type{<:RechesterRosenbluthUpdate})
    (
        described = (:name, :operator, :delta_b_over_b_at, :major_radius,
            :safety_factor, :safety_factor_profile),
        derived = (:delta_b_over_b,))
end

function describe(update::RechesterRosenbluthUpdate, context::DescriptionContext)
    record = SettingsRecord("type" => "RechesterRosenbluthUpdate", "name" => update.name,
        "operator" => describe(update.operator, context; coefficients = false),
        "major_radius" => update.major_radius,
        "safety_factor" => _plain(update.safety_factor),
        "delta_b_over_b" => describe_prescribed_levels(update.delta_b_over_b_at, context;
            points = update.operator.grid.radial_grid.r_faces))
    _is_radial_profile(update.safety_factor_profile) &&
        (record["safety_factor_profile"] = _describe_profile(update.safety_factor_profile))
    return record
end
