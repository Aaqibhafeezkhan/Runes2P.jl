# Terms of the algebraic equations: a multiple of an unknown and a value prescribed in time.

"""
    LinearTerm(unknown; coefficient=1)

`coefficient` times the unknown `unknown`, added entry by entry to the row unknown, which
must have the same kind and sizes (e.g. `-1` times an unknown in its own algebraic
equation, or a decay rate). `coefficient` is a number, or one number per radius for a
radial or kinetic unknown: a conductivity in Ohm's law `0 = σ E − j`. The vector is
referenced, so an update of the run may change its values between the intervals.
"""
struct LinearTerm{C <: Union{Float64, Vector{Float64}}} <: EquationTerm
    unknown::Symbol
    coefficient::C
end
function LinearTerm(unknown::Symbol; coefficient::Union{Real, AbstractVector{<:Real}} = 1.0)
    LinearTerm(unknown,
        coefficient isa Real ? Float64(coefficient) :
        coefficient isa Vector{Float64} ? coefficient : collect(Float64, coefficient))
end

dependencies(term::LinearTerm) = [term.unknown]
is_linear(::LinearTerm) = true

function validate(term::LinearTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === layout.kinds[row] && layout.sizes[id, :] == layout.sizes[row, :] ||
        throw(DimensionMismatch("LinearTerm needs an unknown of the same kind and sizes as its row"))
    if term.coefficient isa Vector
        layout.kinds[row] !== :scalar &&
        length(term.coefficient) == radial_cell_count(layout) ||
            throw(DimensionMismatch("a coefficient per radius needs a radial or kinetic unknown and one value " *
                                    "per radius"))
    end
    return nothing
end

Base.@propagate_inbounds _coefficient_at(coefficient::Float64, k) = coefficient
Base.@propagate_inbounds _coefficient_at(coefficient::Vector{Float64}, k) = coefficient[k]

# the (row range, column range) pairs of a LinearTerm: one per radius, or one for scalars
_range_count(layout, row) = layout.kinds[row] === :scalar ? 1 : radial_cell_count(layout)
function _paired_ranges(layout, row, id, k)
    layout.kinds[row] === :scalar ?
    (unknown_range(layout, row), unknown_range(layout, id)) :
    (unknown_range(layout, row, k), unknown_range(layout, id, k))
end

function rhs!(out, term::LinearTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for k in 1:_range_count(layout, row)
        rows, columns = _paired_ranges(layout, row, id, k)
        c = _coefficient_at(term.coefficient, k)
        @inbounds for (r, s) in zip(rows, columns)
            out[r] += c * u[s]
        end
    end
    return out
end

function sparsity!(entries, term::LinearTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for k in 1:_range_count(layout, row)
        append!(entries, zip(_paired_ranges(layout, row, id, k)...))
    end
    return entries
end

"""
    PrescribedValueTerm(values)

A prescribed value added to the row of a radial unknown with one component or of a scalar
unknown: a number, one number per radius, or a function of time that returns either. For a
radial unknown the function may also return a callable of r, such as an interpolator:
it is evaluated on the cell centres once per time, and a profile on points of
its own that is constant in time is `t -> profile`. With
`LinearTerm(name; coefficient = -1)` in an algebraic equation it holds the unknown at the
prescribed value, `0 = value(t) - unknown`, which makes a prescribed profile an unknown of
the system: a prescribed electric field `E(r, t)` that the kinetic terms read from the
state.
"""
struct PrescribedValueTerm{V} <: EquationTerm
    values::V
    points::Base.RefValue{Vector{Float64}}                  # cell centres, set by validate
    evaluated::Base.RefValue{Tuple{Float64, Vector{Float64}}} # (time, values) of a profile
end

# The points and the cache belong to the term object: a term shared between systems on
# different radial grids takes the points of the last `validate`, and the cache is not safe
# for concurrent calls of `rhs!` at different times.
function PrescribedValueTerm(values)
    PrescribedValueTerm(values, Ref(Float64[]), Ref((NaN, Float64[])))
end

dependencies(::PrescribedValueTerm) = Symbol[]
is_linear(::PrescribedValueTerm) = true
sparsity!(entries, ::PrescribedValueTerm, layout, row) = entries

_prescribed_values(values, t) = values
_prescribed_values(values::Function, t) = values(t)

function _prescribed_values(term::PrescribedValueTerm, t)
    values = _prescribed_values(term.values, t)
    _is_radial_profile(values) || return values
    cached_time, cached = term.evaluated[]
    cached_time == t && return cached
    evaluated = collect(Float64, radial_values(values, term.points[]))
    term.evaluated[] = (Float64(t), evaluated)
    return evaluated
end

function validate(term::PrescribedValueTerm, layout, row)
    kind = layout.kinds[row]
    kind === :scalar || (kind === :radial && layout.sizes[row, 1] == 1) ||
        throw(ArgumentError("PrescribedValueTerm adds to a scalar unknown or to a radial unknown with one component"))
    kind === :radial && (term.points[] = collect(Float64, layout.radial_grid.r_centers))
    term.values isa Function && return nothing
    term.values isa Real ||
        (term.values isa AbstractVector{<:Real} && kind === :radial &&
         length(term.values) == radial_cell_count(layout)) ||
        throw(ArgumentError("the values of a PrescribedValueTerm are a number, one number per radius of a radial " *
                            "unknown, or a function of time"))
    return nothing
end

function rhs!(out, term::PrescribedValueTerm, u, layout, row, t)
    values = _prescribed_values(term, t)
    if layout.kinds[row] === :scalar
        out[first(unknown_range(layout, row))] += values
    else
        for ir in 1:radial_cell_count(layout)
            out[first(unknown_range(layout, row, ir))] += values isa Real ? values :
                                                          values[ir]
        end
    end
    return out
end
