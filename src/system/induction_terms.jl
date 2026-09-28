# Terms of the induction equations in cylindrical geometry (DREAM's psi_p.cpp, E_field.cpp
# and j_tot.cpp): Ampère's law for the poloidal flux with the gradient at the edge from the
# flux at the edge, and the current through the cross-section.
#
# The poloidal flux is DREAM's: divided by the major radius, in V s/m. Faraday's law is
# `dψ/dt = 2π E` and Ampère's law `2π μ₀ j = (1/r) ∂_r (r ∂_r ψ)`, so a field that is
# positive drives a current that is positive and a flux that grows towards the edge.

"""
    PoloidalFluxDiffusionTerm(poloidal_flux, edge_poloidal_flux; coefficient=1)

`coefficient` times `(1/V') ∂_r (V' ∂_r ψ)` of the radial unknown `poloidal_flux`, added
to the row of a radial unknown with one component: the side of Ampère's law that holds
the flux, `0 = (1/r) ∂_r (r ∂_r ψ) − 2π μ₀ j`. `V'` is the spatial Jacobian of the radial
grid (`r` in a cylinder).

The gradient at a face between two cells is the difference of the cell values over the
distance of the centres. No flux passes the axis. The gradient at the edge of the plasma
is `(ψ_edge − ψ_N)/(a − r_N)`, with the scalar unknown `edge_poloidal_flux` and the last
cell `N` (DREAM's `AmperesLawDiffusionTerm` and `AmperesLawBoundaryAtRMax`).
"""
struct PoloidalFluxDiffusionTerm <: EquationTerm
    poloidal_flux::Symbol
    edge_poloidal_flux::Symbol
    coefficient::Float64
end
PoloidalFluxDiffusionTerm(poloidal_flux::Symbol, edge_poloidal_flux::Symbol; coefficient::Real=1.0) =
    PoloidalFluxDiffusionTerm(poloidal_flux, edge_poloidal_flux, Float64(coefficient))

dependencies(term::PoloidalFluxDiffusionTerm) = [term.poloidal_flux, term.edge_poloidal_flux]
is_linear(::PoloidalFluxDiffusionTerm) = true

function validate(term::PoloidalFluxDiffusionTerm, layout, row)
    id, edge = unknown_id(layout, term.poloidal_flux), unknown_id(layout, term.edge_poloidal_flux)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("the poloidal flux is a radial unknown with one component"))
    layout.kinds[edge] === :scalar ||
        throw(ArgumentError("the poloidal flux at the edge is a scalar unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("PoloidalFluxDiffusionTerm adds to a radial unknown with one component"))
    return nothing
end

# V' times the gradient of the flux at the radial face k (1 the axis, nr + 1 the edge)
Base.@propagate_inbounds function _flux_gradient(rg::RadialGrid, psi, psi_edge, k)
    nr = radial_cell_count(rg)
    k == 1 && return zero(eltype(psi)) * zero(psi_edge)
    k == nr + 1 &&
        return rg.spatial_jacobian_faces[k] * (psi_edge - psi[nr]) / (rg.r_faces[k] - rg.r_centers[nr])
    return rg.spatial_jacobian_faces[k] * (psi[k] - psi[k-1]) / (rg.r_centers[k] - rg.r_centers[k-1])
end

function rhs!(out, term::PoloidalFluxDiffusionTerm, u, layout, row, t)
    rg = layout.radial_grid
    psi = radial_values(u, layout, unknown_id(layout, term.poloidal_flux))
    psi_edge = u[first(unknown_range(layout, unknown_id(layout, term.edge_poloidal_flux)))]
    @inbounds for ir in 1:radial_cell_count(rg)
        divergence = (_flux_gradient(rg, psi, psi_edge, ir + 1) - _flux_gradient(rg, psi, psi_edge, ir)) /
                     (rg.spatial_jacobian[ir] * rg.r_widths[ir])
        out[first(unknown_range(layout, row, ir))] += term.coefficient * divergence
    end
    return out
end

function sparsity!(entries, term::PoloidalFluxDiffusionTerm, layout, row)
    id, edge = unknown_id(layout, term.poloidal_flux), unknown_id(layout, term.edge_poloidal_flux)
    nr = radial_cell_count(layout)
    for ir in 1:nr
        r = first(unknown_range(layout, row, ir))
        for neighbour in max(ir - 1, 1):min(ir + 1, nr)
            push!(entries, (r, first(unknown_range(layout, id, neighbour))))
        end
    end
    push!(entries, (first(unknown_range(layout, row, nr)), first(unknown_range(layout, edge))))
    return entries
end

"""
    CrossSectionIntegralTerm(unknown; coefficient=1)

`coefficient` times the integral of the radial unknown `unknown` over the cross-section of
the plasma, `Σ 2π V' Δr u` with the spatial Jacobian `V'` of the radial grid (`r` in a
cylinder), added to the row of a scalar unknown: the plasma current from the current
density, `0 = Σ 2π r Δr j − I_p` (DREAM's `TotalPlasmaCurrentFromJTot`). The entries of
its row are assembled directly.
"""
struct CrossSectionIntegralTerm <: EquationTerm
    unknown::Symbol
    coefficient::Float64
end
CrossSectionIntegralTerm(unknown::Symbol; coefficient::Real=1.0) =
    CrossSectionIntegralTerm(unknown, Float64(coefficient))

dependencies(term::CrossSectionIntegralTerm) = [term.unknown]
is_linear(::CrossSectionIntegralTerm) = true
jacobian_assembly(::CrossSectionIntegralTerm) = :direct

function validate(term::CrossSectionIntegralTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("CrossSectionIntegralTerm integrates a radial unknown with one component"))
    layout.kinds[row] === :scalar ||
        throw(ArgumentError("CrossSectionIntegralTerm adds to a scalar unknown"))
    return nothing
end

"""
    cross_section_weights(radial_grid)

Weights `2π V' Δr` of the integral over the cross-section of the plasma, one per radius.
"""
cross_section_weights(rg::RadialGrid) = 2π .* rg.spatial_jacobian .* rg.r_widths

function rhs!(out, term::CrossSectionIntegralTerm, u, layout, row, t)
    rg = layout.radial_grid
    values = radial_values(u, layout, unknown_id(layout, term.unknown))
    total = zero(eltype(u))
    @inbounds for ir in 1:radial_cell_count(rg)
        total += 2π * rg.spatial_jacobian[ir] * rg.r_widths[ir] * values[ir]
    end
    out[first(unknown_range(layout, row))] += term.coefficient * total
    return out
end

function add_jacobian!(J, term::CrossSectionIntegralTerm, u, layout, row, t)
    rg = layout.radial_grid
    id = unknown_id(layout, term.unknown)
    r = first(unknown_range(layout, row))
    for ir in 1:radial_cell_count(rg)
        _add_entry!(J, r, first(unknown_range(layout, id, ir)),
                    term.coefficient * 2π * rg.spatial_jacobian[ir] * rg.r_widths[ir])
    end
    return J
end

function sparsity!(entries, term::CrossSectionIntegralTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    r = first(unknown_range(layout, row))
    for ir in 1:radial_cell_count(layout)
        push!(entries, (r, first(unknown_range(layout, id, ir))))
    end
    return entries
end

"""
    edge_to_wall_inductance(minor_radius, wall_radius)

Inductance per major radius between the edge of the plasma and the wall,
`μ₀ ln(b/a)`, of the flux at the edge `ψ_edge = ψ_wall − μ₀ ln(b/a) I_p` (DREAM's
`PlasmaEdgeToWallInductanceTerm`). Zero for a wall at the edge of the plasma.
"""
function edge_to_wall_inductance(minor_radius::Real, wall_radius::Real)
    wall_radius >= minor_radius > 0 ||
        throw(ArgumentError("the wall radius $wall_radius must not lie inside the plasma of radius $minor_radius"))
    return Constants.VACUUM_PERMEABILITY * log(wall_radius / minor_radius)
end

"""
    InductionNames(; electric_field=:electric_field, poloidal_flux=:poloidal_flux,
                   total_current_density=:total_current_density,
                   plasma_current=:plasma_current, edge_poloidal_flux=:edge_poloidal_flux,
                   wall_poloidal_flux=:wall_poloidal_flux)

Names of the unknowns of the induction equations ([`induction_unknowns`](@ref),
[`induction_equations`](@ref)).
"""
Base.@kwdef struct InductionNames
    electric_field::Symbol = :electric_field
    poloidal_flux::Symbol = :poloidal_flux
    total_current_density::Symbol = :total_current_density
    plasma_current::Symbol = :plasma_current
    edge_poloidal_flux::Symbol = :edge_poloidal_flux
    wall_poloidal_flux::Symbol = :wall_poloidal_flux
end

"""
    induction_unknowns(radial_grid; electric_field_scale=1, current_density_scale=1e6,
                       names=InductionNames())

The unknowns of the induction equations with their scales: the electric field [V/m], the
poloidal flux per major radius [V s/m] and the total current density [A/m²] at each
radius, and the scalars plasma current [A], flux at the edge and flux at the wall. The
scales of the flux and of the plasma current follow from that of the current density
(`current_density_scale`, 1 MA/m² by default) and the minor radius: `μ₀ I_p` and
`π a² j`. The total current density gets its equation from the caller, the sum of the
currents of the run, the others from [`induction_equations`](@ref).
"""
function induction_unknowns(rg::RadialGrid; electric_field_scale::Real=1.0, current_density_scale::Real=1e6,
                            names::InductionNames=InductionNames())
    current_scale = π * last(rg.r_faces)^2 * current_density_scale
    flux_scale = Constants.VACUUM_PERMEABILITY * current_scale
    return [RadialUnknown(names.electric_field; scale=electric_field_scale),
            RadialUnknown(names.poloidal_flux; scale=flux_scale),
            RadialUnknown(names.total_current_density; scale=current_density_scale),
            ScalarUnknown(names.plasma_current; scale=current_scale),
            ScalarUnknown(names.edge_poloidal_flux; scale=flux_scale),
            ScalarUnknown(names.wall_poloidal_flux; scale=flux_scale)]
end

"""
    induction_equations(radial_grid; wall_radius=minor radius, loop_voltage_at_wall=0,
                        current_density_scale=1e6, names=InductionNames())

The induction equations in cylindrical geometry, with DREAM's pairing of rows and
unknowns:

| Unknown | Equation |
|---|---|
| electric field | `dψ/dt = 2π E` (Faraday's law, the time derivative of the flux) |
| poloidal flux | `0 = (1/r) ∂_r (r ∂_r ψ) − 2π μ₀ j_tot` (Ampère's law) |
| plasma current | `0 = Σ 2π r Δr j_tot − I_p` |
| flux at the edge | `0 = ψ_wall − μ₀ ln(b/a) I_p − ψ_edge` |
| flux at the wall | `dψ_wall/dt = V_loop,wall` |

`wall_radius` is the radius `b` of the wall. `loop_voltage_at_wall` is the loop voltage
per major radius at the wall [V/m], a number or a function of time: zero for a perfectly
conducting wall, which holds the flux at the wall. The row of Ampère's law takes the scale
`2π μ₀` times `current_density_scale`. The equation of the total current density, the sum
of the currents of the run, comes from the caller.
"""
function induction_equations(rg::RadialGrid; wall_radius::Real=last(rg.r_faces), loop_voltage_at_wall=0.0,
                             current_density_scale::Real=1e6, names::InductionNames=InductionNames())
    inductance = edge_to_wall_inductance(last(rg.r_faces), wall_radius)
    mu0 = Constants.VACUUM_PERMEABILITY
    return [
        Equation(names.electric_field, TimeDerivative(names.poloidal_flux),
                 LinearTerm(names.electric_field; coefficient=2π)),
        Equation(names.poloidal_flux, Algebraic(),
                 PoloidalFluxDiffusionTerm(names.poloidal_flux, names.edge_poloidal_flux),
                 LinearTerm(names.total_current_density; coefficient=-2π * mu0);
                 scale=2π * mu0 * current_density_scale),
        Equation(names.plasma_current, Algebraic(), CrossSectionIntegralTerm(names.total_current_density),
                 LinearTerm(names.plasma_current; coefficient=-1)),
        Equation(names.edge_poloidal_flux, Algebraic(), LinearTerm(names.wall_poloidal_flux),
                 LinearTerm(names.plasma_current; coefficient=-inductance),
                 LinearTerm(names.edge_poloidal_flux; coefficient=-1)),
        Equation(names.wall_poloidal_flux, TimeDerivative(), PrescribedValueTerm(loop_voltage_at_wall))]
end

"""
    initial_poloidal_flux(radial_grid, total_current_density; wall_radius=minor radius,
                          wall_poloidal_flux=0)

The poloidal flux that satisfies the discrete Ampère's law of
[`PoloidalFluxDiffusionTerm`](@ref) for the current density `total_current_density` (one
value per radius), as DREAM initializes it (a steady solve of the equation of the flux,
with the flux at the wall zero). Returns the named tuple `(poloidal_flux, plasma_current,
edge_poloidal_flux, wall_poloidal_flux)`: the state of the induction unknowns at the start
of a run, next to the field and the current density.
"""
function initial_poloidal_flux(rg::RadialGrid, total_current_density::AbstractVector{<:Real};
                               wall_radius::Real=last(rg.r_faces), wall_poloidal_flux::Real=0.0)
    nr = radial_cell_count(rg)
    length(total_current_density) == nr ||
        throw(DimensionMismatch("the current density needs one value per radius"))
    mu0 = Constants.VACUUM_PERMEABILITY
    plasma_current = sum(cross_section_weights(rg) .* total_current_density)
    edge_flux = wall_poloidal_flux - edge_to_wall_inductance(last(rg.r_faces), wall_radius) * plasma_current
    # V' times the gradient at the upper face of a cell is the sum of 2π μ₀ j V' Δr up to the
    # cell: the flux follows from the edge inwards
    enclosed = cumsum(2π * mu0 .* total_current_density .* rg.spatial_jacobian .* rg.r_widths)
    flux = zeros(float(eltype(total_current_density)), nr)
    flux[nr] = edge_flux - (rg.r_faces[nr+1] - rg.r_centers[nr]) * enclosed[nr] / rg.spatial_jacobian_faces[nr+1]
    for ir in nr-1:-1:1
        flux[ir] = flux[ir+1] -
                   (rg.r_centers[ir+1] - rg.r_centers[ir]) * enclosed[ir] / rg.spatial_jacobian_faces[ir+1]
    end
    return (; poloidal_flux=flux, plasma_current, edge_poloidal_flux=edge_flux,
            wall_poloidal_flux=float(wall_poloidal_flux))
end
