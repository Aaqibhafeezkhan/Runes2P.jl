# Regions in momentum for the moments of a distribution: the hot region
# above a threshold that follows the cold temperature (DREAM's pThreshold of MomentQuantity),
# limits in units of m_e c, upper limits and a smooth step.

"""
    ThermalMomentum(temperature; factor=7)

Momentum limit that follows the cold-electron temperature: `factor` thermal momenta,
`p₀ = factor √(2 T / mₑc²)` in units of `mₑc` at each radius (DREAM's hot region,
`pThreshold = 7` in thermal units by default in its Python interface). `temperature` is
the temperature in eV at each radius, or a function of time that returns the temperatures
or a [`PlasmaState`](@ref). `limit(t)` gives the momenta at the time `t`.

`temperature` may also be the name of the radial unknown of the temperature. The limit
then follows the state, and the moments over the region hold the column of
the temperature in their Jacobian: the derivative of the envelope with respect to the
limit, piecewise in the cells as DREAM's (`MomentQuantity.cpp`).
"""
struct ThermalMomentum{S}
    temperature::S
    factor::Float64
end
function ThermalMomentum(temperature; factor::Real=7.0)
    factor > 0 || throw(ArgumentError("the factor of a thermal momentum limit must be positive"))
    return ThermalMomentum(temperature, Float64(factor))
end

_temperatures(temperature::AbstractVector, t) = temperature
_temperatures(name::Symbol, t) =
    throw(ArgumentError("a limit that follows the temperature unknown $name needs the state of the system"))
_temperatures(plasma::PlasmaState, t) = plasma.cold_electron_temperature
_temperatures(temperature_at, t) = _temperatures(temperature_at(t), t)

(limit::ThermalMomentum)(t) =
    [limit.factor * sqrt(2 * T / Constants.ELECTRON_REST_ENERGY) for T in _temperatures(limit.temperature, t)]

"""
    MomentumRegion(limit; side=:above, envelope=:interpolated, smooth_width=2)

Part of the momentum grid that a moment of a distribution covers
([`DensityMomentTerm`](@ref), [`CurrentMomentTerm`](@ref), [`density`](@ref),
[`current_density`](@ref)): the momenta above `limit` (`side = :above`) or below it
(`:below`). `limit` is a momentum in units of `mₑc`: one number, one number per radius, or
a function of time that returns either, e.g. a [`ThermalMomentum`](@ref).

A moment with a region multiplies the contribution of the momentum cell `i` by the
envelope of the region ([`momentum_envelope`](@ref)), 1 inside the region and 0 outside.
`envelope` selects its values in the cells at the limit, for `side = :above`:

- `:interpolated`: the part of the cell that holds the limit `p₀` counts with the integrand
  at the midpoint of that part, interpolated linearly between the centre of the cell and
  the next one (DREAM's `P_THRESHOLD_MODE_MIN_THERMAL`,
  `EvaluateMinThermalInterpolationEnvelope`). In the last cell of the grid the integrand is
  taken to fall to zero at `p_max`. The moment is continuous in `p₀` with a continuous
  derivative inside a cell.
- `:cell_fraction`: the fraction of the cell above the limit, with the integrand of the
  cell centre (DREAM's `P_THRESHOLD_MODE_MIN_MC`).
- `:smooth`: `(1 + tanh((p − p₀) / (smooth_width Δp))) / 2` at the cell centres, with `Δp`
  the width of the first cell whose centre lies above `p₀` (DREAM's
  `P_THRESHOLD_MODE_MIN_THERMAL_SMOOTH`, width 2).

The envelope of `side = :below` is one minus that of `:above`, so the two regions of one
limit add up to the whole grid (DREAM's `MAX` modes for the cell fraction and the smooth
step).
"""
struct MomentumRegion{L}
    limit::L
    side::Symbol
    envelope::Symbol
    smooth_width::Float64
end
function MomentumRegion(limit; side::Symbol=:above, envelope::Symbol=:interpolated, smooth_width::Real=2.0)
    side in (:above, :below) || throw(ArgumentError("side must be :above or :below, not $side"))
    envelope in (:interpolated, :cell_fraction, :smooth) ||
        throw(ArgumentError("envelope must be :interpolated, :cell_fraction or :smooth, not $envelope"))
    smooth_width > 0 || throw(ArgumentError("smooth_width must be positive"))
    limit isa Real && (limit >= 0 || throw(ArgumentError("the limit of a momentum region must not be negative")))
    return MomentumRegion(limit, side, envelope, Float64(smooth_width))
end

"""
    region_temperature(region)

The name of the temperature unknown that the limit of a region follows, as a tuple of one
name, or an empty tuple (also for `nothing`).
"""
region_temperature(region::MomentumRegion) = region.limit isa ThermalMomentum{Symbol} ? (region.limit.temperature,) : ()
region_temperature(::Nothing) = ()

"""
    is_time_dependent(region::MomentumRegion)

Whether the limit of the region is a function of time.
"""
is_time_dependent(region::MomentumRegion) = !(region.limit isa Union{Real,AbstractVector{<:Real}})

_limit_values(limit::Union{Real,AbstractVector{<:Real}}, t) = limit
_limit_values(limit, t) = limit(t)

"""
    momentum_limit(region, ir, t=0; radii=nothing)

Limit of the region in units of `mₑc` at the radius index `ir` and the time `t`. With
`radii`, the number of radii of the grid, the length of a limit per radius is checked.
"""
function momentum_limit(region::MomentumRegion, ir::Integer, t=0.0; radii::Union{Nothing,Integer}=nothing)
    values = _limit_values(region.limit, t)
    values isa Real && return Float64(values)
    radii === nothing || length(values) == radii ||
        throw(DimensionMismatch("the limit of the momentum region has $(length(values)) values for $radii radii"))
    return Float64(values[ir])
end

# the envelope of the momenta above p0 in the cell i
function _envelope_above(envelope::Symbol, smooth_width, m::MomentumGrid, i::Integer, p0::Real)
    p, p_lower, p_upper, dp = m.p_centers[i], m.p_faces[i], m.p_faces[i+1], m.p_widths[i]
    np = momentum_cell_count(m)
    if envelope === :cell_fraction
        p_lower >= p0 && return 1.0
        return p_upper >= p0 ? (p_upper - p0) / dp : 0.0
    elseif envelope === :smooth
        # the width of the first cell whose centre lies above the limit
        k = findfirst(>(p0), m.p_centers)
        step = k === nothing ? p0 - m.p_centers[np] : m.p_widths[k]
        return (1 + tanh((p - p0) / (smooth_width * step))) / 2
    end
    value = p0 <= p_lower ? 1.0 : 0.0
    if p_lower < p0 < p_upper
        # the part (p0, p_upper) of this cell: the share of this cell in the integrand at
        # the midpoint q, between this centre and the next one
        p_next = i < np ? m.p_centers[i+1] : p_upper
        q = (p0 + p_upper) / 2
        value += (p_upper - p0) / dp * (p_next - q) / (p_next - p)
    elseif i > 1 && m.p_faces[i-1] < p0 < p_lower
        # the part (p0, p_lower) of the cell below: the share of this cell in the integrand
        # at its midpoint. DREAM reads the centre of a cell below the first one for a limit
        # between 0 and the p_min of a grid that starts above 0; such a limit adds nothing here.
        p_previous = m.p_centers[i-1]
        q = (p0 + p_lower) / 2
        value += (p_lower - p0) / dp * (q - p_previous) / (p - p_previous)
    end
    return value
end

"""
    momentum_envelope!(values, region, momentum_grid, limit)
    momentum_envelope(region, momentum_grid, limit)

Envelope of the region on the momentum cells of one momentum grid, for the limit `limit`
in units of `mₑc`: the factor of the contribution of each momentum cell to a moment
([`MomentumRegion`](@ref)).
"""
function momentum_envelope!(values::AbstractVector, region::MomentumRegion, m::MomentumGrid, limit::Real)
    length(values) == momentum_cell_count(m) ||
        throw(DimensionMismatch("the envelope has one value per momentum cell"))
    limit >= 0 || throw(ArgumentError("the limit of a momentum region must not be negative, it is $limit"))
    for i in eachindex(values)
        above = _envelope_above(region.envelope, region.smooth_width, m, i, limit)
        values[i] = region.side === :above ? above : 1 - above
    end
    return values
end
momentum_envelope(region::MomentumRegion, m::MomentumGrid, limit::Real) =
    momentum_envelope!(zeros(momentum_cell_count(m)), region, m, limit)

@inline _envelope_value(::Nothing, i) = true
Base.@propagate_inbounds _envelope_value(envelope::AbstractVector, i) = envelope[i]
