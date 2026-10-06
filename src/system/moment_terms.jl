# Moments of a distribution as terms of the equations: the
# density and the current of a kinetic unknown, over a momentum region, with the envelopes
# of the region kept per term. Moved out of terms.jl and momentum_region.jl.

# the limits per radius of a region that follows the temperature unknown, in the number type
# of the state
function _state_limits(region::MomentumRegion, u, layout)
    limit = region.limit
    temperature = radial_values(u, layout, unknown_id(layout, limit.temperature))
    return [limit.factor * sqrt(2 * temperature[ir] / Constants.ELECTRON_REST_ENERGY) for ir in eachindex(temperature)]
end

"""
    RegionEnvelopes(region, grid)

The envelopes of a region on a kinetic grid, per radius, kept by a term. A limit that
depends on time is evaluated once per time, and the envelopes are rebuilt when the limit of
a radius changes. The evaluations of a term share them, so a term with a
region allocates nothing between the changes.
"""
struct RegionEnvelopes{R<:MomentumRegion}
    region::R
    values::Vector{Vector{Float64}}   # per radius, per momentum cell
    limits::Vector{Float64}           # the limits of the values per radius, empty before the first evaluation
    time::Base.RefValue{Float64}      # the time of the last evaluation of a limit in time
end
function RegionEnvelopes(region::MomentumRegion, grid::KineticGrid)
    # The width Δp of the smooth step is that of the first cell above the limit, so the
    # envelope jumps where the limit passes a cell centre between cells of different
    # widths. With the temperature as an unknown the residual then jumps in it, and the
    # Newton iteration can cycle.
    region.envelope === :smooth && !isempty(region_temperature(region)) &&
        any(ir -> !_uniform_widths(momentum_grid(grid, ir)), 1:radial_cell_count(grid)) &&
        throw(ArgumentError("the :smooth envelope of a region that follows the temperature unknown " *
                            "$(only(region_temperature(region))) jumps in the temperature on a momentum grid with " *
                            "cells of different widths: use envelope = :interpolated or :cell_fraction"))
    return RegionEnvelopes(region, [zeros(momentum_cell_count(grid, ir)) for ir in 1:radial_cell_count(grid)], Float64[],
                           Ref(NaN))
end
_uniform_widths(m::MomentumGrid) = all(w -> isapprox(w, first(m.p_widths); rtol=1e-10), m.p_widths)
RegionEnvelopes(::Nothing, grid::KineticGrid) = nothing

# the envelope of the radius at the state u and the time t; nothing without a region
@inline _envelope_at(::Nothing, grid, ir, u, layout, t) = nothing
function _envelope_at(envelopes::RegionEnvelopes, grid::KineticGrid, ir, u, layout, t)
    _refresh!(envelopes, grid, u, layout, t)
    return envelopes.values[ir]
end

function _refresh!(envelopes::RegionEnvelopes, grid::KineticGrid, u, layout, t)
    time = Float64(_value(t))
    built = !isempty(envelopes.limits)
    radii = radial_cell_count(grid)
    if !isempty(region_temperature(envelopes.region))
        # a limit that follows the state is compared at every evaluation
        u === nothing && throw(ArgumentError("a region that follows the temperature unknown needs the state"))
        values = [_value(x) for x in _state_limits(envelopes.region, u, layout)]
    else
        built && (!is_time_dependent(envelopes.region) || envelopes.time[] == time) && return envelopes
        values = _limit_values(envelopes.region.limit, time)
    end
    limits = values isa Real ? Iterators.repeated(values, radii) : values
    values isa Real || length(values) == radii ||
        throw(DimensionMismatch("the limit of the momentum region has $(length(values)) values for $radii radii"))
    envelopes.time[] = time
    update_key!(envelopes.limits, limits) || return envelopes
    try
        for ir in 1:radii
            momentum_envelope!(envelopes.values[ir], envelopes.region, momentum_grid(grid, ir), envelopes.limits[ir])
        end
    catch
        reset_key!(envelopes.limits)
        rethrow()
    end
    return envelopes
end

"""
    DensityMomentTerm(kinetic; coefficient=1)
    DensityMomentTerm(kinetic, grid; coefficient=1, region=nothing)

`coefficient` times the density `n(r) = (1/V') Σ 𝒱' Δp Δξ f` of the kinetic unknown
`kinetic`, added to a radial row unknown with one component (e.g. a hot-electron density
defined by an algebraic equation). Works with different momentum grids at different radii.

With `region`, a [`MomentumRegion`](@ref), the moment covers that part of the momentum
grid: the hot electrons above the threshold of the hot region. The limit
of the region at the time of the evaluation counts, and `grid` is the grid of the unknown.
The limit is prescribed data: the term is linear in the distribution.
"""
struct DensityMomentTerm{E} <: EquationTerm
    kinetic::Symbol
    coefficient::Float64
    envelopes::E    # nothing, or the envelopes of the region per radius
end
DensityMomentTerm(kinetic::Symbol; coefficient::Real=1.0) = DensityMomentTerm(kinetic, Float64(coefficient), nothing)
DensityMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0,
                  region::Union{Nothing,MomentumRegion}=nothing) =
    DensityMomentTerm(kinetic, Float64(coefficient), RegionEnvelopes(region, grid))

_region_unknowns(envelopes) = envelopes === nothing ? () : region_temperature(envelopes.region)
dependencies(term::DensityMomentTerm) = [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
is_linear(term::DensityMomentTerm) = isempty(_region_unknowns(term.envelopes))

# the envelopes of a term fit the grid of its unknown
_check_envelopes(::Nothing, grid) = nothing
function _check_envelopes(envelopes::RegionEnvelopes, grid::KineticGrid)
    length(envelopes.values) == radial_cell_count(grid) &&
        all(ir -> length(envelopes.values[ir]) == momentum_cell_count(grid, ir), 1:radial_cell_count(grid)) ||
        throw(DimensionMismatch("the momentum region of the term was built for another grid"))
    return nothing
end

function validate(term::DensityMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("DensityMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("DensityMomentTerm adds to a radial unknown with one component"))
    _check_envelopes(term.envelopes, kinetic_grid(layout, id))
    return nothing
end

rhs!(out, term::DensityMomentTerm, u, layout, row, t) =
    _add_density_moment!(out, term, kinetic_grid(layout, unknown_id(layout, term.kinetic)), u, layout, row, t)

function _add_density_moment!(out, term::DensityMomentTerm, grid::KineticGrid, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = _moment_envelope(term.envelopes, grid, ir, u, layout, t)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            n += _envelope_value(envelope, i) * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        out[first(unknown_range(layout, row, ir))] +=
            term.coefficient * n / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::DensityMomentTerm) = :direct

function add_jacobian!(J, term::DensityMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        r = first(unknown_range(layout, row, ir))
        factor = term.coefficient / grid.radial_grid.spatial_jacobian[ir]
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                        factor * _envelope_value(envelope, i) * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j])
        end
    end
    _add_threshold_column!(J, term.envelopes, grid, u, layout, row, id) do ir, i, j
        m = momentum_grid(grid, ir)
        term.coefficient / grid.radial_grid.spatial_jacobian[ir] * phase_space_jacobian(grid, ir)[i, j] *
            m.p_widths[i] * m.xi_widths[j]
    end
    return J
end

sparsity!(entries, term::DensityMomentTerm, layout, row) =
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)

# the row of a moment holds every cell of its radius, and the temperature of the radius when
# the limit of its region follows that unknown
function _moment_sparsity!(entries, kinetic::Symbol, layout, row, envelopes=nothing)
    id = unknown_id(layout, kinetic)
    temperature = envelopes === nothing ? () : region_temperature(envelopes.region)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for s in unknown_range(layout, id, ir)
            push!(entries, (r, s))
        end
        for name in temperature
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, name), ir))))
        end
    end
    return entries
end

# the envelope for the residual of a moment: in the number type of a state with derivative
# parts when the limit follows the temperature, so that the AD of rhs! holds its column
function _moment_envelope(envelopes, grid, ir, u, layout, t)
    if envelopes !== nothing && !isempty(region_temperature(envelopes.region)) && eltype(u) <: ForwardDiff.Dual
        limit = _state_limits(envelopes.region, u, layout)[ir]
        return momentum_envelope!(zeros(typeof(limit), momentum_cell_count(grid, ir)), envelopes.region,
                                  momentum_grid(grid, ir), limit)
    end
    return _envelope_at(envelopes, grid, ir, u, layout, t)
end

# the column of the temperature of a moment over a region that follows it: the derivative of
# the envelope with respect to the temperature times the moment weight `weight(ir, i, j)` of
# each cell and f
function _add_threshold_column!(weight, J, envelopes, grid, u, layout, row, id)
    envelopes === nothing && return J
    temperature = region_temperature(envelopes.region)
    isempty(temperature) && return J
    T_id = unknown_id(layout, only(temperature))
    D = ForwardDiff.Dual{PlasmaColumnTag,Float64,1}
    T = radial_values(u, layout, T_id)
    limit = envelopes.region.limit
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        p0 = limit.factor * sqrt(2 * D(T[ir], ForwardDiff.Partials((1.0,))) / Constants.ELECTRON_REST_ENERGY)
        envelope = momentum_envelope!(zeros(D, momentum_cell_count(m)), envelopes.region, m, p0)
        f = kinetic_view(u, layout, id, ir)
        value = 0.0
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            value += ForwardDiff.partials(envelope[i], 1) * weight(ir, i, j) * f[i, j]
        end
        _add_entry!(J, first(unknown_range(layout, row, ir)), first(unknown_range(layout, T_id, ir)), value)
    end
    return J
end

"""
    CurrentMomentTerm(kinetic, grid; coefficient=1, pitch_rule=:linear, region=nothing)

`coefficient` times the current density of the kinetic unknown `kinetic` on `grid`
([`current_density`](@ref); `pitch_rule = :cell_center` is DREAM's `j_hot` moment), added
to a radial row unknown with one component (the ohmic or hot-electron current defined by
an algebraic equation). The pitch weights are built per radius when the term is created,
so different momentum grids at different radii work; `grid` must be the unknown's grid
(the same radii and momentum faces, checked when the system is built).

With `region`, a [`MomentumRegion`](@ref), the moment covers that part of the momentum
grid, with the limit of the region at the time of the evaluation: the
current of the hot electrons above the threshold of the hot region.
"""
struct CurrentMomentTerm{G<:KineticGrid,E} <: EquationTerm
    kinetic::Symbol
    grid::G
    coefficient::Float64
    pitch_rule::Symbol
    pitch_weights::Vector{Vector{Float64}}   # per radius
    envelopes::E                             # nothing, or the envelopes of the region per radius
end
function CurrentMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0, pitch_rule::Symbol=:linear,
                           region::Union{Nothing,MomentumRegion}=nothing)
    weights = [pitch_moment_weights(momentum_grid(grid, ir); pitch_rule) for ir in 1:radial_cell_count(grid)]
    return CurrentMomentTerm(kinetic, grid, Float64(coefficient), pitch_rule, weights, RegionEnvelopes(region, grid))
end

dependencies(term::CurrentMomentTerm) = [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
is_linear(term::CurrentMomentTerm) = isempty(_region_unknowns(term.envelopes))

function validate(term::CurrentMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("CurrentMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("CurrentMomentTerm adds to a radial unknown with one component"))
    # the pitch weights belong to the term's grid: equal cell counts are not enough
    _same_momentum_grids(term.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the term's grid does not match $(term.kinetic)"))
    return nothing
end

function rhs!(out, term::CurrentMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        j = _current_moment(momentum_grid(grid, ir), term.pitch_weights[ir], phase_space_jacobian(grid, ir),
                            kinetic_view(u, layout, id, ir), _moment_envelope(term.envelopes, grid, ir, u, layout, t))
        out[first(unknown_range(layout, row, ir))] += term.coefficient * j / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::CurrentMomentTerm) = :direct

function add_jacobian!(J, term::CurrentMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    charge = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        weights = term.pitch_weights[ir]
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        r = first(unknown_range(layout, row, ir))
        factor = term.coefficient * charge / grid.radial_grid.spatial_jacobian[ir]
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            p = m.p_centers[i]
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                        factor * _envelope_value(envelope, i) * weights[j] * jacobian[i, j] * m.p_widths[i] * p /
                        sqrt(1 + p^2))
        end
    end
    _add_threshold_column!(J, term.envelopes, grid, u, layout, row, id) do ir, i, j
        m = momentum_grid(grid, ir)
        p = m.p_centers[i]
        term.coefficient * charge / grid.radial_grid.spatial_jacobian[ir] * term.pitch_weights[ir][j] *
            phase_space_jacobian(grid, ir)[i, j] * m.p_widths[i] * p / sqrt(1 + p^2)
    end
    return J
end

sparsity!(entries, term::CurrentMomentTerm, layout, row) =
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)
