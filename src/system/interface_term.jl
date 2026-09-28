# Flux handover between two kinetic unknowns whose momentum grids meet at a momentum face:
# the hot-tail grid below, the runaway grid above. The particle flux
# through the upper momentum boundary of the lower grid enters the upper grid at its lower
# boundary, carried between the two pitch grids by their overlap.

const INTERFACE_SCHEMES = (:upwind, :central, :exponential_fitting)

"""
    MomentumInterfaceTerm(lower, upper, lower_term, upper_operator;
                          pitch_mapping=:overlap, scheme=<scheme of the lower operator>)

Particle flux through the momentum face shared by the kinetic unknowns `lower` and `upper`:
the last p face of the grid of `lower` (the hot-tail grid) is the first p face of the grid
of `upper` (the runaway grid) at every radius. The pitch grids of the two may differ. The
same term goes into both equations. In the equation of `lower` it removes the flux from
the last momentum cells, in the equation of `upper` it adds the flux to the first momentum
cells, and both are computed from the same numbers, so no particle is lost or created at
the interface.

`lower_term` is the Fokker-Planck term of `lower` ([`FokkerPlanckTerm`](@ref) or
[`PrescribedFokkerPlanckTerm`](@ref)), or its operator. The flux takes the coefficients
`Aᵖ` and `Dᵖᵖ` of that operator at its last p face, as DREAM's
`PXiExternalKineticKinetic` does, and the two operators add no flux of their own through
the face: the lower one has `p_upper_boundary = :interface`, `upper_operator` has
`p_lower_boundary = :interface`. The cross term `Dᵖᵠ` at the face is not included (none in
DREAM either, and zero for the collision operator).

`pitch_mapping`:

- `:overlap` (Runes2P): one flux per overlap cell of the two pitch grids
  ([`PitchOverlap`](@ref)), from the value of the lower cell and the value of the upper
  cell the overlap cell lies in,

      Γ_s = flux(Aᵖ_j, Dᵖᵖ_j, f_lower[end, j], f_upper[1, J]; scheme),

  with the two-point flux of `scheme` between the two cell centres (`:upwind`, `:central`
  or `:exponential_fitting`, by default the p scheme of the lower operator, and `:upwind`
  when that is a four-point scheme). The couplings are those of the scheme between two
  neighbouring cells, so exponential fitting and upwind keep their sign properties across
  the interface.
- `:dream`: the flux of DREAM, one per pitch cell of the lower grid. The upper value is
  interpolated linearly in the pitch to the centre of the lower cell, the advection is
  first-order upwind whatever the scheme of the interior, the diffusion is the difference
  of the two values over the distance of the centres, and the flux is given to the upper
  cells by their overlap with the lower cell. Ported as written, including the rule for a
  lower centre beyond the last upper centre, which takes the second-to-last upper cell
  (the nearest one at the other end). `scheme` does not apply. Where the pitch grids
  differ, the interpolation couples an upper cell to its neighbours in the pitch with
  negative weights in the diffusive part, so this mapping does not keep the sign of `f` by
  construction.

Both conserve particles to round-off. Linear in the state, stored sparse.
[`interface_flux`](@ref) gives the flux through the face as a rate per volume.
"""
struct MomentumInterfaceTerm{T,F} <: EquationTerm
    lower::Symbol
    upper::Symbol
    lower_term::F                          # the term or operator that holds the coefficients
    lower_grid::KineticGrid{T}
    upper_grid::KineticGrid{T}
    pitch_mapping::Symbol
    scheme::Symbol
    overlaps::Vector{PitchOverlap{T}}      # per radius, left: the lower grid, right: the upper grid
    interpolation_cells::Vector{Vector{Int}}  # :dream, per radius and lower pitch cell: the upper cell Jj
    interpolation_weights::Vector{Vector{T}}  # :dream: weight of cell Jj, the rest on cell Jj - 1
end

_operator_of(op::AdvectionDiffusion) = op
_operator_of(term::FokkerPlanckTerm) = term.operator
_operator_of(term::PrescribedFokkerPlanckTerm) = term.inner.operator

# the operator with its coefficients at the time t
_operator_at(source, t) = _operator_of(source)
_operator_at(term::PrescribedFokkerPlanckTerm, t) = _rebuild_at!(term, t)

function MomentumInterfaceTerm(lower::Symbol, upper::Symbol, lower_term, upper_operator;
                               pitch_mapping::Symbol=:overlap, scheme::Union{Nothing,Symbol}=nothing)
    lower_term isa Union{AdvectionDiffusion,FokkerPlanckTerm,PrescribedFokkerPlanckTerm} ||
        throw(ArgumentError("lower_term must be a Fokker-Planck term or an AdvectionDiffusion operator"))
    upper_operator isa Union{AdvectionDiffusion,FokkerPlanckTerm,PrescribedFokkerPlanckTerm} ||
        throw(ArgumentError("upper_operator must be an AdvectionDiffusion operator or a Fokker-Planck term"))
    lower != upper || throw(ArgumentError("the interface lies between two different unknowns"))
    pitch_mapping in (:overlap, :dream) ||
        throw(ArgumentError("pitch_mapping must be :overlap or :dream, not $pitch_mapping"))
    lower_operator, upper_op = _operator_of(lower_term), _operator_of(upper_operator)
    lower_operator.p_upper_boundary === :interface ||
        throw(ArgumentError("the lower operator needs p_upper_boundary = :interface, it has " *
                            ":$(lower_operator.p_upper_boundary) and would add its own flux through the face"))
    upper_op.p_lower_boundary === :interface ||
        throw(ArgumentError("the upper operator needs p_lower_boundary = :interface, it has " *
                            ":$(upper_op.p_lower_boundary)"))
    if scheme === nothing
        scheme = lower_operator.p_scheme in INTERFACE_SCHEMES ? lower_operator.p_scheme : :upwind
    end
    scheme in INTERFACE_SCHEMES ||
        throw(ArgumentError("the scheme of the interface flux must be one of $(join(INTERFACE_SCHEMES, ", ")), " *
                            "not $scheme"))
    lower_grid, upper_grid = lower_operator.grid, upper_op.grid
    T = eltype(lower_operator.advection_p)
    lower_grid.radial_grid.r_faces == upper_grid.radial_grid.r_faces ||
        throw(ArgumentError("the two grids of the interface must share the radial grid"))
    nr = radial_cell_count(lower_grid)
    for ir in 1:nr
        p_interface = last(momentum_grid(lower_grid, ir).p_faces)
        p_lower_face = first(momentum_grid(upper_grid, ir).p_faces)
        p_interface == p_lower_face ||
            throw(ArgumentError("at radius $ir the lower grid ends at p = $p_interface and the upper grid " *
                                "starts at p = $p_lower_face: the two must be the same face"))
    end
    built = Dict{Tuple{UInt,UInt},PitchOverlap{T}}()
    overlaps = map(1:nr) do ir
        left, right = momentum_grid(lower_grid, ir), momentum_grid(upper_grid, ir)
        get!(() -> PitchOverlap(left, right), built, (objectid(left), objectid(right)))
    end
    cells, weights = Vector{Vector{Int}}(undef, nr), Vector{Vector{T}}(undef, nr)
    for ir in 1:nr
        cells[ir], weights[ir] = _dream_pitch_interpolation(momentum_grid(lower_grid, ir).xi_centers,
                                                            momentum_grid(upper_grid, ir).xi_centers)
    end
    return MomentumInterfaceTerm{T,typeof(lower_term)}(lower, upper, lower_term, lower_grid, upper_grid,
                                                       pitch_mapping, scheme, overlaps, cells, weights)
end

# DREAM's interpolation of the upper distribution to the pitch centres of the lower grid
# (PXiExternalKineticKinetic.cpp): Jj is the first upper cell with its centre at or above
# the lower centre, and the value is delta1 f[Jj] + (1 - delta1) f[Jj - 1]. At the lower
# end (Jj the first cell) delta1 = 1. Where no upper centre reaches the lower centre, Jj is
# the last cell and delta1 = 0, which takes the cell before the last.
function _dream_pitch_interpolation(lower_centers::Vector{T}, upper_centers::Vector{T}) where {T}
    count = length(upper_centers)
    cells, weights = zeros(Int, length(lower_centers)), zeros(T, length(lower_centers))
    cell = 1
    for (j, xi) in enumerate(lower_centers)
        while upper_centers[cell] < xi && cell < count
            cell += 1
        end
        cells[j] = cell
        weights[j] = if cell == 1
            one(T)
        elseif upper_centers[cell] < xi
            zero(T)
        else
            (xi - upper_centers[cell-1]) / (upper_centers[cell] - upper_centers[cell-1])
        end
    end
    return cells, weights
end

dependencies(term::MomentumInterfaceTerm) = [term.lower, term.upper]
is_linear(::MomentumInterfaceTerm) = true

function validate(term::MomentumInterfaceTerm, layout, row)
    lower, upper = unknown_id(layout, term.lower), unknown_id(layout, term.upper)
    layout.kinds[lower] === :kinetic && layout.kinds[upper] === :kinetic ||
        throw(ArgumentError("MomentumInterfaceTerm lies between two kinetic unknowns"))
    row in (lower, upper) ||
        throw(ArgumentError("MomentumInterfaceTerm belongs to the equations of $(term.lower) and " *
                            "$(term.upper), not of $(layout.unknowns[row].name)"))
    _same_momentum_grids(term.lower_grid, kinetic_grid(layout, lower)) ||
        throw(DimensionMismatch("the grid of the lower operator does not match $(term.lower)"))
    _same_momentum_grids(term.upper_grid, kinetic_grid(layout, upper)) ||
        throw(DimensionMismatch("the grid of the upper operator does not match $(term.upper)"))
    return nothing
end

# value of the upper distribution at the pitch centre of lower cell j (:dream)
Base.@propagate_inbounds function _dream_upper_value(term::MomentumInterfaceTerm, f_upper, j, ir)
    cell, weight = term.interpolation_cells[ir][j], term.interpolation_weights[ir][j]
    return weight * f_upper[1, cell] + (1 - weight) * f_upper[1, max(cell - 1, 1)]
end

# Γᵖ through the face between lower pitch cell j and upper pitch cell J (not multiplied by 𝒱')
Base.@propagate_inbounds function _interface_gamma(term::MomentumInterfaceTerm, op, f_lower, f_upper, j, J, ir,
                                                   distance, weight_left)
    np = size(f_lower, 1)
    A = op.advection_p[np+1, j, ir]
    D = op.diffusion_p_p[np+1, j, ir]
    f_left = f_lower[np, j]
    if term.pitch_mapping === :dream
        f_right = _dream_upper_value(term, f_upper, j, ir)
        return A * (A > 0 ? f_left : f_right) + D * (f_left - f_right) / distance
    end
    return _face_flux(term.scheme, A, D, f_left, f_upper[1, J], distance, weight_left)
end

# 𝒱'Γᵖ Δξ through the part of the face in lower pitch cell j, divided by Δξ_j
Base.@propagate_inbounds function _lower_cell_flux(term::MomentumInterfaceTerm, op, f_lower, f_upper, j, ir,
                                                   distance, weight_left)
    np = size(f_lower, 1)
    V = op.jacobian_p_faces[np+1, j, ir]
    overlap = term.overlaps[ir]
    if term.pitch_mapping === :dream
        return V * _interface_gamma(term, op, f_lower, f_upper, j, 0, ir, distance, weight_left)
    end
    flux = zero(promote_type(eltype(f_lower), eltype(f_upper)))
    for s in overlap.left_ranges[j]
        flux += overlap.left_fractions[s] *
                (V * _interface_gamma(term, op, f_lower, f_upper, j, overlap.right_cells[s], ir, distance,
                                      weight_left))
    end
    return flux
end

# the same divided by Δξ_J of upper pitch cell J
Base.@propagate_inbounds function _upper_cell_flux(term::MomentumInterfaceTerm, op, f_lower, f_upper, J, ir,
                                                   distance, weight_left)
    np = size(f_lower, 1)
    overlap = term.overlaps[ir]
    flux = zero(promote_type(eltype(f_lower), eltype(f_upper)))
    for s in overlap.right_ranges[J]
        j = overlap.left_cells[s]
        V = op.jacobian_p_faces[np+1, j, ir]
        flux += overlap.right_fractions[s] *
                (V * _interface_gamma(term, op, f_lower, f_upper, j, J, ir, distance, weight_left))
    end
    return flux
end

function _interface_distance(term::MomentumInterfaceTerm, ir)
    lower, upper = momentum_grid(term.lower_grid, ir), momentum_grid(term.upper_grid, ir)
    distance = first(upper.p_centers) - last(lower.p_centers)
    weight_left = (first(upper.p_centers) - last(lower.p_faces)) / distance
    return distance, weight_left
end

function rhs!(out, term::MomentumInterfaceTerm, u, layout, row, t)
    op = _operator_at(term.lower_term, t)
    lower, upper = unknown_id(layout, term.lower), unknown_id(layout, term.upper)
    for ir in 1:radial_cell_count(layout)
        f_lower, f_upper = kinetic_view(u, layout, lower, ir), kinetic_view(u, layout, upper, ir)
        target = kinetic_view(out, layout, row, ir)
        distance, weight_left = _interface_distance(term, ir)
        if row == lower
            m = momentum_grid(term.lower_grid, ir)
            jacobian = phase_space_jacobian(term.lower_grid, ir)
            np = momentum_cell_count(m)
            @inbounds for j in 1:pitch_cell_count(m)
                jacobian[np, j] == 0 && continue
                flux = _lower_cell_flux(term, op, f_lower, f_upper, j, ir, distance, weight_left)
                target[np, j] -= flux / (jacobian[np, j] * m.p_widths[np])
            end
        else
            m = momentum_grid(term.upper_grid, ir)
            jacobian = phase_space_jacobian(term.upper_grid, ir)
            @inbounds for J in 1:pitch_cell_count(m)
                jacobian[1, J] == 0 && continue
                flux = _upper_cell_flux(term, op, f_lower, f_upper, J, ir, distance, weight_left)
                target[1, J] += flux / (jacobian[1, J] * m.p_widths[1])
            end
        end
    end
    return out
end

# upper pitch cells whose values enter the flux through the part of the face in lower cell j
function _upper_cells_of(term::MomentumInterfaceTerm, j, ir)
    if term.pitch_mapping === :dream
        cell = term.interpolation_cells[ir][j]
        return max(cell - 1, 1):cell
    end
    overlap = term.overlaps[ir]
    range = overlap.left_ranges[j]
    return overlap.right_cells[first(range)]:overlap.right_cells[last(range)]
end

function sparsity!(entries, term::MomentumInterfaceTerm, layout, row)
    lower, upper = unknown_id(layout, term.lower), unknown_id(layout, term.upper)
    for ir in 1:radial_cell_count(layout)
        np = layout.shapes[lower, ir][1]
        overlap = term.overlaps[ir]
        if row == lower
            for j in 1:layout.shapes[lower, ir][2]
                r = _kinetic_index(layout, lower, ir, np, j)
                push!(entries, (r, r))
                for J in _upper_cells_of(term, j, ir)
                    push!(entries, (r, _kinetic_index(layout, upper, ir, 1, J)))
                end
            end
        else
            for J in 1:layout.shapes[upper, ir][2]
                r = _kinetic_index(layout, upper, ir, 1, J)
                for s in overlap.right_ranges[J]
                    j = overlap.left_cells[s]
                    push!(entries, (r, _kinetic_index(layout, lower, ir, np, j)))
                    if term.pitch_mapping === :dream
                        for K in _upper_cells_of(term, j, ir)
                            push!(entries, (r, _kinetic_index(layout, upper, ir, 1, K)))
                        end
                    else
                        push!(entries, (r, r))
                    end
                end
            end
        end
    end
    return entries
end

"""
    interface_flux(term::MomentumInterfaceTerm, u, layout, ir, t=0)

Particle flux per unit volume [m⁻³ s⁻¹] through the momentum interface at radius index
`ir`, positive from the lower to the upper grid, for the state `u`: the rate at which the
upper grid gains particles from the lower one, in the normalization of
[`density`](@ref) and [`momentum_flux`](@ref). With the interface above the critical
momentum this is the runaway rate.
"""
function interface_flux(term::MomentumInterfaceTerm, u::AbstractVector, layout::UnknownLayout, ir::Integer,
                        t::Real=0.0)
    op = _operator_at(term.lower_term, t)
    lower, upper = unknown_id(layout, term.lower), unknown_id(layout, term.upper)
    f_lower, f_upper = kinetic_view(u, layout, lower, ir), kinetic_view(u, layout, upper, ir)
    distance, weight_left = _interface_distance(term, ir)
    m = momentum_grid(term.lower_grid, ir)
    total = zero(eltype(u))
    for j in 1:pitch_cell_count(m)
        total += m.xi_widths[j] * _lower_cell_flux(term, op, f_lower, f_upper, j, ir, distance, weight_left)
    end
    return total / term.lower_grid.radial_grid.spatial_jacobian[ir]
end
