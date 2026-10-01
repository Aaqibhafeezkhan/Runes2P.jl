# Terms of radial transport next to the Fokker-Planck term that holds the radial operator:
# the loss through the edge of the plasma as the rate of a scalar unknown, and a prescribed
# loss rate.

"""
    RadialEdgeLossTerm(kinetic, operator; coefficient=1)

`coefficient` times the particles per second that the kinetic unknown `kinetic` loses
through the edge of the plasma by the radial transport of `operator`
([`radial_particle_flux`](@ref) at the last radial face), added to a scalar row unknown.
With `TimeDerivative()` the row unknown is the number of particles lost so far, integrated
with the same time levels as the distribution, so that the content of the distribution
plus the lost particles stays constant to round-off when nothing else changes the content.
The term reads the coefficients of `operator` as they are, also when an update changes
them during the run.
"""
struct RadialEdgeLossTerm{T} <: EquationTerm
    kinetic::Symbol
    operator::AdvectionDiffusion{T}
    coefficient::Float64
end
RadialEdgeLossTerm(kinetic::Symbol, operator::AdvectionDiffusion; coefficient::Real=1.0) =
    RadialEdgeLossTerm(kinetic, operator, Float64(coefficient))

dependencies(term::RadialEdgeLossTerm) = [term.kinetic]
is_linear(::RadialEdgeLossTerm) = true

function validate(term::RadialEdgeLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic || throw(ArgumentError("RadialEdgeLossTerm needs a kinetic unknown"))
    layout.kinds[row] === :scalar || throw(ArgumentError("RadialEdgeLossTerm adds to a scalar unknown"))
    _same_momentum_grids(term.operator.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the grid of the operator does not match $(term.kinetic)"))
    return nothing
end

function rhs!(out, term::RadialEdgeLossTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    f = kinetic_array(u, layout, id)
    edge = radial_cell_count(layout) + 1
    out[first(unknown_range(layout, row))] += term.coefficient * radial_particle_flux(term.operator, f, edge)
    return out
end

jacobian_assembly(::RadialEdgeLossTerm) = :direct

# the flux through the edge is Σ 𝒱' Δp Δξ (A⁺ + D/distance) f over the cells of the last radius
function add_jacobian!(J, term::RadialEdgeLossTerm, u, layout, row, t)
    op = term.operator
    op.r_boundary === :zero_flux && return J
    id = unknown_id(layout, term.kinetic)
    rg = op.grid.radial_grid
    nr = radial_cell_count(rg)
    m = op.momentum_grids[nr]
    distance = _r_edge_distance(op)
    r = first(unknown_range(layout, row))
    @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
        A = op.advection_r[i, j, nr+1]
        rate = (A > 0 ? A : zero(A)) + op.diffusion_r_r[i, j, nr+1] / distance
        _add_entry!(J, r, _kinetic_index(layout, id, nr, i, j),
                    term.coefficient * m.p_widths[i] * m.xi_widths[j] * op.jacobian_r_faces[i, j, nr+1] * rate)
    end
    return J
end

function sparsity!(entries, term::RadialEdgeLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    r = first(unknown_range(layout, row))
    for s in unknown_range(layout, id, radial_cell_count(layout))
        push!(entries, (r, s))
    end
    return entries
end

"""
    LossRateTerm(unknown, rate)

Prescribed loss `-ν f` of the kinetic unknown `unknown` with the rate `ν` [1/s] of each
radius (`rate`, one value per radius), added to the row unknown, which has the same grid.
The exponential loss of a region with open field lines, as DREAM's loss outside the last
closed flux surface. The vector is referenced: an update of the run may change its values
between the intervals.
"""
struct LossRateTerm <: EquationTerm
    unknown::Symbol
    rate::Vector{Float64}
end
LossRateTerm(unknown::Symbol, rate::AbstractVector{<:Real}) = LossRateTerm(unknown, collect(Float64, rate))

dependencies(term::LossRateTerm) = [term.unknown]
is_linear(::LossRateTerm) = true

function validate(term::LossRateTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("LossRateTerm acts on and adds to kinetic unknowns"))
    layout.sizes[id, :] == layout.sizes[row, :] ||
        throw(DimensionMismatch("LossRateTerm needs an unknown with the sizes of its row"))
    length(term.rate) == radial_cell_count(layout) ||
        throw(DimensionMismatch("the loss rate needs one value per radius"))
    all(>=(0), term.rate) || throw(ArgumentError("the loss rate must not be negative"))
    return nothing
end

function rhs!(out, term::LossRateTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        rate = term.rate[ir]
        rate == 0 && continue
        @inbounds for (r, s) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
            out[r] -= rate * u[s]
        end
    end
    return out
end

function sparsity!(entries, term::LossRateTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end
