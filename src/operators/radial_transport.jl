# Radial transport of a distribution: the Rechester-Rosenbluth diffusion
# coefficient from the level of the magnetic perturbation, and the particle flux through
# the radial faces. The coefficients belong to an `AdvectionDiffusion` operator of their
# own, next to the operator of the momentum-space terms, whose rebuild clears its arrays.

"""
    radial_face_values(values, grid)

One value per radial face of `grid` (radii + 1) from a number, which holds on all faces, a
vector with one value per face, or a callable of r evaluated on the faces
([`radial_values`](@ref)).
"""
radial_face_values(values, grid::KineticGrid) = collect(Float64, radial_values(values, grid.radial_grid.r_faces))

"""
    add_rechester_rosenbluth_diffusion!(op, delta_b_over_b; major_radius, safety_factor=1)

Radial diffusion of electrons along stochastic field lines (Rechester and Rosenbluth, Phys.
Rev. Lett. 40, 38, 1978), added to `op.diffusion_r_r`:

```math
D^{rr} = \\pi q R_0 \\left(\\frac{\\delta B}{B}\\right)^2 |v_\\parallel|, \\qquad v_\\parallel = \\frac{c p \\xi}{\\gamma}.
```

The form with `v_∥` and the factor π is that of Papp et al. (arXiv:1504.05074, section 2)
and, with `q = 1`, of Berger et al. (arXiv:2208.02117, section 2), the second from the DREAM
group.

`delta_b_over_b` and the safety factor `q` are numbers, have one value per radial face
(radii + 1, as DREAM's `dBB` on its radial flux grid) or are callables of r evaluated on the
faces ([`radial_face_values`](@ref)), `major_radius` is `R₀` in metres.
The pitch of the coefficient is that of the pitch cells of the radial face
([`radial_face_overlap`](@ref)), which are the pitch cells of the grid when all radii share
their pitch grid. A level of zero on a face leaves the face without transport.

DREAM (`RechesterRosenbluthTransport.cpp`) has `q = 1` fixed and sets `R₀ = 1` in
cylindrical geometry, where its major radius is infinite, so that the level of the
perturbation alone sets the coefficient. `major_radius = 1` with the default safety factor
reproduces it. A physical coefficient in a cylinder needs the major radius of the device,
and DREAM needs `dBB √R₀` for the same coefficient.
"""
function add_rechester_rosenbluth_diffusion!(op::AdvectionDiffusion{T}, delta_b_over_b; major_radius::Real,
                                             safety_factor=1.0) where {T}
    major_radius > 0 || throw(ArgumentError("the major radius must be positive"))
    grid = op.grid
    level = radial_face_values(delta_b_over_b, grid)
    q = radial_face_values(safety_factor, grid)
    all(>=(0), level) || throw(ArgumentError("δB/B must not be negative"))
    all(>(0), q) || throw(ArgumentError("the safety factor must be positive"))
    p_centers = momentum_grid(grid, 1).p_centers   # the p faces are the same at all radii
    for k in eachindex(level)
        overlap = op.radial_overlaps[k]
        prefactor = pi * q[k] * major_radius * level[k]^2 * Constants.SPEED_OF_LIGHT
        for s in 1:overlap_cell_count(overlap), (i, p) in enumerate(p_centers)
            op.diffusion_r_r[i, s, k] += prefactor * p * abs(overlap.xi_centers[s]) / sqrt(1 + p^2)
        end
    end
    return op
end

"""
    radial_particle_flux(op::AdvectionDiffusion, f, face)

Particles per second that cross the radial face `face` (between the radii `face - 1` and
`face`) towards larger radii, `Σ 𝒱' Γʳ Δp Δξ` over the momentum cells, in the normalization
of [`total_particles`](@ref): the content of the radii inside the face falls by this flux
and by nothing else when only radial transport acts. Face `radii + 1` is the edge of the
plasma, where the flux is the loss of the plasma (zero with `r_boundary = :zero_flux`). The
flux through the first face, the axis, is zero.
"""
function radial_particle_flux(op::AdvectionDiffusion, f::AbstractArray{<:Any,3}, face::Integer)
    nr = radial_cell_count(op.grid)
    1 <= face <= nr + 1 || throw(ArgumentError("radial face $face outside the grid (1 to $(nr + 1))"))
    size(f) == size(op.grid) || throw(DimensionMismatch("f must have the grid size"))
    total = zero(eltype(f))
    face == 1 && return total
    ir = face - 1   # the radius inside the face
    m = op.momentum_grids[ir]
    @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
        total += m.p_widths[i] * m.xi_widths[j] * _r_flux(op, f, i, j, ir, face)
    end
    return total
end

"""
    radial_particle_content(grid, f)

Particle content of every radius, `Σ 𝒱' Δp Δξ Δr f` over its momentum cells. The sum over
the radii is [`total_particles`](@ref).
"""
function radial_particle_content(grid::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    size(f) == size(grid) || throw(DimensionMismatch("f must have the grid size"))
    return density(grid, f) .* grid.radial_grid.spatial_jacobian .* grid.radial_grid.r_widths
end
