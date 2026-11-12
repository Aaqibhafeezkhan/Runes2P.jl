# Moments of a distribution as terms of the equations: the density and the current of a
# kinetic unknown, over a momentum region, with the envelopes of the region kept per term.

# the limits per radius of a region that follows the temperature unknown, in the number type
# of the state
function _state_limits(region::MomentumRegion, u, layout)
    limit = region.limit
    temperature = unknown_values(u, layout, unknown_id(layout, limit.temperature))
    return [limit.factor * sqrt(2 * temperature[ir] / Constants.ELECTRON_REST_ENERGY)
            for ir in eachindex(temperature)]
end

"""
    RegionEnvelopes(region, grid)

The envelopes of a region on a kinetic grid, per radius, kept by a term. A limit that
depends on time is evaluated once per time, and the envelopes are rebuilt when the limit of
a radius changes. The evaluations of a term share them, so a term with a
region allocates nothing between the changes.
"""
struct RegionEnvelopes{R <: MomentumRegion}
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
    return RegionEnvelopes(
        region, [zeros(momentum_cell_count(grid, ir)) for ir in 1:radial_cell_count(grid)],
        Float64[],
        Ref(NaN))
end
function _uniform_widths(m::MomentumGrid)
    all(w -> isapprox(w, first(m.p_widths); rtol = 1e-10), m.p_widths)
end
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
        if u === nothing
            throw(ArgumentError("a region that follows the temperature unknown needs the state"))
        end
        values = [_value(x) for x in _state_limits(envelopes.region, u, layout)]
    else
        built && (!is_time_dependent(envelopes.region) || envelopes.time[] == time) &&
            return envelopes
        values = _limit_values(envelopes.region.limit, time)
    end
    limits = values isa Real ? Iterators.repeated(values, radii) : values
    if !(values isa Real || length(values) == radii)
        throw(DimensionMismatch("the limit of the momentum region has $(length(values)) values for $radii radii"))
    end
    envelopes.time[] = time
    update_key!(envelopes.limits, limits) || return envelopes
    try
        for ir in 1:radii
            momentum_envelope!(envelopes.values[ir], envelopes.region,
                momentum_grid(grid, ir), envelopes.limits[ir])
        end
    catch
        reset_key!(envelopes.limits)
        rethrow()
    end
    return envelopes
end

# the dual numbers of the envelope of a region in the column of the temperature
const MomentEnvelopeDual = ForwardDiff.Dual{PlasmaColumnTag, Float64, 1}

# the threshold momentum of a limit that follows the temperature, in dual numbers of the
# column of the temperature `T` of one radius
function _dual_threshold(limit, T)
    temperature = MomentEnvelopeDual(T, ForwardDiff.Partials((1.0,)))
    return limit.factor * sqrt(2 * temperature / Constants.ELECTRON_REST_ENERGY)
end

# The rows of a moment term in the stored Jacobian, kept between its Jacobians: per radius
# the position in the stored values of J of the entry of each cell, in the order of the loop
# over the cells (pitch outer, momentum inner), and the weight of each cell, the product of the
# moment with the envelope of the region set to 1; with a region that follows the temperature,
# the position of the entry of the temperature of the radius and the envelope in dual numbers
# of its column. The positions belong to the pattern of J, the layout and the row of the last
# Jacobian, the pattern compared by the identity of its arrays and by its number of stored
# entries, and are found again when one of them differs. A term filled into two matrices in
# turn (the matrix of an integrator and the copy of a Jacobian-vector product) finds its
# positions again at each switch, into the vectors it already holds. The cache is not safe to
# use concurrently: the terms of a system add their Jacobians one after the other.
mutable struct MomentRowCache
    # the layout of the positions; nothing before the first Jacobian
    layout::Union{Nothing, UnknownLayout}
    row::Int
    colptr::Vector{Int}    # the arrays of the pattern of J of the positions
    rowval::Vector{Int}
    entry_count::Int       # the number of stored entries of that pattern
    const positions::Vector{Vector{Int}}
    const weights::Vector{Vector{Float64}}
    const temperature_positions::Vector{Int}
    const dual_envelopes::Vector{Vector{MomentEnvelopeDual}}
end
# the empty arrays of the pattern are never those of a J
function MomentRowCache()
    MomentRowCache(nothing, 0, Int[], Int[], 0, Vector{Int}[], Vector{Float64}[], Int[],
        Vector{MomentEnvelopeDual}[])
end

# the vector at `ir` of `vectors` with `entries` entries: the one already there resized, or
# a new one
function _reused_vector!(vectors::Vector{Vector{E}}, ir, entries) where {E}
    if isassigned(vectors, ir)
        return resize!(vectors[ir], entries)
    end
    vector = Vector{E}(undef, entries)
    vectors[ir] = vector
    return vector
end

"""
    DensityMomentTerm(kinetic; coefficient=1)
    DensityMomentTerm(kinetic, grid; coefficient=1, region=nothing)

`coefficient` times the density `n(r) = (1/V') Σ 𝒱' Δp Δξ f` of the kinetic unknown
`kinetic`, added to a radial row unknown with one component (e.g. a hot-electron density
defined by an algebraic equation). Works with different momentum grids at different radii.

With `region`, a [`MomentumRegion`](@ref), the moment covers that part of the momentum
grid: the hot electrons above the threshold of the hot region. The limit of the region at
the time of the evaluation counts, and `grid` is the grid of the unknown. A prescribed limit
makes the term linear in the distribution. A limit that follows the temperature unknown
([`ThermalMomentum`](@ref) with the unknown's name) makes it nonlinear, and its Jacobian holds
the column of the temperature through the threshold.

The term keeps the positions of its row in the stored Jacobian between calls, so its
[`add_jacobian!`](@ref) is not safe to call concurrently.
"""
struct DensityMomentTerm{E} <: EquationTerm
    kinetic::Symbol
    coefficient::Float64
    envelopes::E                  # nothing, or the envelopes of the region per radius
    row_cache::MomentRowCache     # the positions and weights of the row in the stored Jacobian
end
function DensityMomentTerm(kinetic::Symbol; coefficient::Real = 1.0)
    DensityMomentTerm(kinetic, Float64(coefficient), nothing, MomentRowCache())
end
function DensityMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real = 1.0,
        region::Union{Nothing, MomentumRegion} = nothing)
    DensityMomentTerm(
        kinetic, Float64(coefficient), RegionEnvelopes(region, grid), MomentRowCache())
end

function _region_unknowns(envelopes)
    envelopes === nothing ? () : region_temperature(envelopes.region)
end
function dependencies(term::DensityMomentTerm)
    [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
end
is_linear(term::DensityMomentTerm) = isempty(_region_unknowns(term.envelopes))

# the envelopes of a term fit the grid of its unknown
_check_envelopes(::Nothing, grid) = nothing
function _check_envelopes(envelopes::RegionEnvelopes, grid::KineticGrid)
    #! format: off
    length(envelopes.values) == radial_cell_count(grid) &&
        all(ir -> length(envelopes.values[ir]) == momentum_cell_count(grid, ir), 1:radial_cell_count(grid)) ||
        throw(DimensionMismatch("the momentum region of the term was built for another grid"))
    #! format: on
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

function rhs!(out, term::DensityMomentTerm, u, layout, row, t)
    _add_density_moment!(out, term, kinetic_grid(layout, unknown_id(layout, term.kinetic)),
        u, layout, row, t)
end

function _add_density_moment!(
        out, term::DensityMomentTerm, grid::KineticGrid, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        envelope = _moment_envelope(term.envelopes, grid, ir, u, layout, t)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            n += _envelope_value(envelope, i) * jacobian[i, j] * m.p_widths[i] *
                 m.xi_widths[j] * f[i, j]
        end
        r = first(unknown_range(layout, row, ir))
        out[r] += term.coefficient * n / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::DensityMomentTerm) = :direct

function add_jacobian!(J, term::DensityMomentTerm, u, layout, row, t)
    _add_moment_jacobian!(J, term, u, layout, row, t)
end

# the factor of a radius in the moment: the coefficient over the spatial Jacobian
function _moment_factor(term::DensityMomentTerm, grid::KineticGrid, ir)
    term.coefficient / grid.radial_grid.spatial_jacobian[ir]
end

# the entry of the cell (i, j) of the radius ir in the row of the moment, with the envelope
# value `envelope` of the cell as the second factor of the product
Base.@propagate_inbounds function _moment_weight(
        term::DensityMomentTerm, m::MomentumGrid, jacobian, factor, ir, i, j, envelope)
    factor * envelope * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j]
end

function sparsity!(entries, term::DensityMomentTerm, layout, row)
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)
end

# the row of a moment holds every cell of its radius, and the temperature of the radius when
# the limit of its region follows that unknown
function _moment_sparsity!(entries, kinetic::Symbol, layout, row, envelopes = nothing)
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
    if envelopes !== nothing && !isempty(region_temperature(envelopes.region)) &&
       eltype(u) <: ForwardDiff.Dual
        limit = _state_limits(envelopes.region, u, layout)[ir]
        return momentum_envelope!(
            zeros(typeof(limit), momentum_cell_count(grid, ir)), envelopes.region,
            momentum_grid(grid, ir), limit)
    end
    return _envelope_at(envelopes, grid, ir, u, layout, t)
end

# the positions and weights of the row of a moment term in J, found again when the pattern of
# J, the layout or the row differ from those of the last Jacobian
function _moment_rows!(cache::MomentRowCache, J::SparseMatrixCSC, term, layout, row)
    if cache.layout === layout && cache.row == row && cache.colptr === J.colptr &&
       cache.rowval === J.rowval && cache.entry_count == last(J.colptr) - 1
        return cache
    end
    # no key until the positions are complete, so that a missing entry throws at every call
    cache.layout = nothing
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    radii = radial_cell_count(layout)
    resize!(cache.positions, radii)
    resize!(cache.weights, radii)
    for ir in 1:radii
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        factor = _moment_factor(term, grid, ir)
        r = first(unknown_range(layout, row, ir))
        cells = momentum_cell_count(m) * pitch_cell_count(m)
        positions = _reused_vector!(cache.positions, ir, cells)
        weights = _reused_vector!(cache.weights, ir, cells)
        k = 0
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            k += 1
            positions[k] = _pattern_position(J, r, _kinetic_index(layout, id, ir, i, j))
            # the envelope true, a factor of the product in each moment, gives the entry at a
            # full envelope
            weights[k] = _moment_weight(term, m, jacobian, factor, ir, i, j, true)
        end
    end
    temperature = _region_unknowns(term.envelopes)
    temperature_radii = isempty(temperature) ? 0 : radii
    resize!(cache.temperature_positions, temperature_radii)
    resize!(cache.dual_envelopes, temperature_radii)
    if !isempty(temperature)
        T_id = unknown_id(layout, only(temperature))
        for ir in 1:radii
            r = first(unknown_range(layout, row, ir))
            column = first(unknown_range(layout, T_id, ir))
            cache.temperature_positions[ir] = _pattern_position(J, r, column)
            # every value is written by momentum_envelope! before it is read
            _reused_vector!(cache.dual_envelopes, ir, momentum_cell_count(grid, ir))
        end
    end
    cache.layout = layout
    cache.row = row
    cache.colptr = J.colptr
    cache.rowval = J.rowval
    cache.entry_count = last(J.colptr) - 1
    return cache
end

# the Jacobian of a density, current or energy moment from the positions and weights of its
# row: the stored weight where the envelope of the cell is 1, and the product with the
# envelope as its second factor elsewhere, so that every entry is the product of the
# residual's factors in their order. `_moment_factor` gives the part of a radius shared by
# its cells, a number or a tuple of numbers (the energy moment), which only
# `_moment_weight` of the same term reads.
function _add_moment_jacobian!(J, term, u, layout, row, t)
    cache = _moment_rows!(term.row_cache, J, term, layout, row)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    values = nonzeros(J)
    envelopes = term.envelopes
    if envelopes === nothing
        for ir in 1:radial_cell_count(layout)
            positions = cache.positions[ir]
            weights = cache.weights[ir]
            @inbounds for k in eachindex(positions, weights)
                values[positions[k]] += weights[k]
            end
        end
        return J
    end
    _refresh!(envelopes, grid, u, layout, t)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        factor = _moment_factor(term, grid, ir)
        envelope = envelopes.values[ir]
        positions = cache.positions[ir]
        weights = cache.weights[ir]
        k = 0
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            k += 1
            cell_envelope = envelope[i]
            entry = if cell_envelope == 1
                weights[k]
            else
                _moment_weight(term, m, jacobian, factor, ir, i, j, cell_envelope)
            end
            values[positions[k]] += entry
        end
    end
    _add_threshold_column!(values, cache, envelopes, grid, u, layout, id)
    return J
end

# the column of the temperature of a moment over a region that follows it, with the positions
# and weights of `cache`: the derivative of the envelope with respect to the temperature times
# the weight of each cell and f. The envelope in dual numbers is evaluated at every call, since
# the temperature changes at every Newton iteration.
function _add_threshold_column!(
        values, cache::MomentRowCache, envelopes::RegionEnvelopes, grid, u, layout, id)
    temperature = region_temperature(envelopes.region)
    isempty(temperature) && return values
    T = unknown_values(u, layout, unknown_id(layout, only(temperature)))
    limit = envelopes.region.limit
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        p0 = _dual_threshold(limit, T[ir])
        envelope = momentum_envelope!(cache.dual_envelopes[ir], envelopes.region, m, p0)
        f = kinetic_view(u, layout, id, ir)
        weights = cache.weights[ir]
        value = 0.0
        k = 0
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            k += 1
            value += ForwardDiff.partials(envelope[i], 1) * weights[k] * f[i, j]
        end
        values[cache.temperature_positions[ir]] += value
    end
    return values
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

The term keeps the positions of its row in the stored Jacobian between calls, so its
[`add_jacobian!`](@ref) is not safe to call concurrently.
"""
struct CurrentMomentTerm{G <: KineticGrid, E} <: EquationTerm
    kinetic::Symbol
    grid::G
    coefficient::Float64
    pitch_rule::Symbol
    pitch_weights::Vector{Vector{Float64}}   # per radius
    envelopes::E                             # nothing, or the envelopes of the region per radius
    row_cache::MomentRowCache                # the positions and weights of the row in the stored Jacobian
end
function CurrentMomentTerm(kinetic::Symbol, grid::KineticGrid;
        coefficient::Real = 1.0, pitch_rule::Symbol = :linear,
        region::Union{Nothing, MomentumRegion} = nothing)
    weights = [pitch_moment_weights(momentum_grid(grid, ir); pitch_rule)
               for ir in 1:radial_cell_count(grid)]
    return CurrentMomentTerm(kinetic, grid, Float64(coefficient), pitch_rule,
        weights, RegionEnvelopes(region, grid), MomentRowCache())
end

function dependencies(term::CurrentMomentTerm)
    [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
end
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
        #! format: off
        j = _current_moment(momentum_grid(grid, ir), term.pitch_weights[ir], phase_space_jacobian(grid, ir),
                            kinetic_view(u, layout, id, ir), _moment_envelope(term.envelopes, grid, ir, u, layout, t))
        #! format: on
        r = first(unknown_range(layout, row, ir))
        out[r] += term.coefficient * j / grid.radial_grid.spatial_jacobian[ir]
    end
    return out
end

jacobian_assembly(::CurrentMomentTerm) = :direct

function add_jacobian!(J, term::CurrentMomentTerm, u, layout, row, t)
    _add_moment_jacobian!(J, term, u, layout, row, t)
end

# the factor of a radius in the moment: the coefficient times e c over the spatial Jacobian
function _moment_factor(term::CurrentMomentTerm, grid::KineticGrid, ir)
    charge = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT
    return term.coefficient * charge / grid.radial_grid.spatial_jacobian[ir]
end

# the entry of the cell (i, j) of the radius ir in the row of the moment, with the envelope
# value `envelope` of the cell as the second factor of the product
Base.@propagate_inbounds function _moment_weight(
        term::CurrentMomentTerm, m::MomentumGrid, jacobian, factor, ir, i, j, envelope)
    p = m.p_centers[i]
    weight = factor * envelope * term.pitch_weights[ir][j] * jacobian[i, j] *
             m.p_widths[i] * p
    lorentz_factor = sqrt(1 + p^2)
    return weight / lorentz_factor
end

function sparsity!(entries, term::CurrentMomentTerm, layout, row)
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)
end
