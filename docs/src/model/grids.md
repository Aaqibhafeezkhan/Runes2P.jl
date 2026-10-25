# Grids

```@meta
CurrentModule = Runes2P
```

A distribution function lives on a phase-space grid in momentum ``p``, pitch ``\xi`` and minor radius ``r``.
Every grid is defined by its faces, and cell centres are face midpoints.
The geometry is a cylinder with a homogeneous magnetic field: no trapping and no bounce averaging.

## Momentum and pitch grids

A [`MomentumGrid`](@ref) holds the faces of one ``(p, \xi)`` grid, with ``p`` the momentum in units of ``m_e c`` and ``\xi = \cos\theta`` with ``\theta`` the pitch angle.
The momentum faces start at ``p_\mathrm{min} \ge 0``, and the pitch faces lie in ``[-1, 1]``.

The momentum faces come from three generators:

- [`uniform_p_faces`](@ref): uniform spacing.
- [`biuniform_p_faces`](@ref): two uniform regions split at ``p_\mathrm{sep}``, typically finer below it to resolve the thermal bulk (DREAM's `PBiUniformGridGenerator`).
- [`tanh_p_faces`](@ref): the biuniform grid with its jump in spacing replaced by a smooth transition.

Its transition is a tanh in the cell index, centred three widths above ``p_\mathrm{sep}`` so that the bulk keeps its resolution.
The default width grows with the cell count, so the truncation error stays second order everywhere.
DREAM has uniform, biuniform and custom momentum grids, so these faces enter DREAM as a custom grid.

The pitch faces come from [`uniform_xi_faces`](@ref), uniform in ``\xi``, or from [`biuniform_theta_xi_faces`](@ref).
The latter is uniform in ``\theta`` in two regions, used with finer cells near ``\xi = 1``, where runaway electrons concentrate.
It is DREAM's `XiBiUniformThetaGridGenerator`, and `test/test_grids.jl` compares its faces with a transcription of DREAM's generator.

## Radial grid

A [`RadialGrid`](@ref) is defined by its radial faces, non-negative and increasing, from [`uniform_r_faces`](@ref) on ``[0, a]`` or given directly.
Its spatial Jacobian is ``V'(r) = r`` for the cylinder, the derivative ``dV/dr`` of the volume per unit length divided by ``2\pi``.
Radial profiles of the plasma are evaluated at its cell centres or faces by [`radial_values`](@ref).

## The kinetic grid

[`cylindrical_grid`](@ref) combines a radial grid with one momentum grid for all radii, or one per radius, into a [`KineticGrid`](@ref).
It holds the phase-space Jacobian ``\mathcal{V}'`` (DREAM's `Vp`) at the cell centres and on the faces of each direction:

```math
\mathcal{V}' = 2\pi\, p^2\, V'(r),
\qquad
n(r) = \frac{1}{V'} \sum_{ij} \mathcal{V}'_{ij}\, \Delta p_i\, \Delta\xi_j\, f_{ij}.
```

The accessors are [`phase_space_jacobian`](@ref), [`phase_space_jacobian_p_faces`](@ref), [`phase_space_jacobian_xi_faces`](@ref) and [`phase_space_jacobian_r_faces`](@ref).
The limit [`phase_space_jacobian_over_p2_at_zero`](@ref) enters the flux through ``p = 0``, where ``\mathcal{V}'`` vanishes but collisional friction still removes particles (see [`AdvectionDiffusion`](@ref)).
DREAM's cylindrical `VpVol` and `Vp` are ``4\pi^2`` times ``V'`` and ``\mathcal{V}'``.
`test/test_two_grids_dream.jl` checks this ratio for the `Vprime` of DREAM's output.

The moments [`density`](@ref), [`current_density`](@ref) and [`total_particles`](@ref) sum over the cells with these weights.
The pitch weights of the current are those of [`pitch_moment_weights`](@ref).
A [`MomentumRegion`](@ref) restricts a moment to the momenta above or below a limit.
An example is the hot region above a [`ThermalMomentum`](@ref), 7 thermal momenta of the cold electrons by default (DREAM's `pThreshold`).
How the cells near the limit count is set by the `envelope` of [`MomentumRegion`](@ref).
The `:interpolated` envelope (DREAM's `MIN_THERMAL` mode) is compared with a transcription of DREAM's function in `test/test_momentum_region.jl`, and the other envelopes with their formulas.

## Per-radius grids

Each radius of a [`KineticGrid`](@ref) has its own [`MomentumGrid`](@ref).
Grids with the same faces at all radii count as shared.
Three predicates say how far the radii agree, and each one gates a part of the code:

| Predicate | Holds when | Needed by |
|---|---|---|
| [`has_shared_momentum_grid`](@ref) | one momentum grid at all radii | [`momentum_grid`](@ref) without a radius index |
| [`has_same_cell_counts`](@ref) | the same numbers of ``p`` and ``\xi`` cells at all radii | the 3D array `f[i, j, ir]`, the cell counts without a radius index, [`FieldParticleOperator`](@ref) |
| [`has_same_p_faces`](@ref) | the same ``p`` faces at all radii | the Jacobian on the radial faces, with pitch grids that cover one interval |

[`kinetic_array`](@ref) gives the 3D array, and [`momentum_cell_count`](@ref), [`pitch_cell_count`](@ref) and [`cell_count`](@ref) the counts.

[`AdvectionDiffusion`](@ref), and with it every Fokker-Planck operator, needs the same cell counts and the same ``p`` faces at all radii.
The pitch faces may differ.
The knock-on operator ([`KnockOnOperator`](@ref)) also takes a different momentum grid at each radius.
In the state vector a [`KineticUnknown`](@ref) holds its cells radius by radius, ``p`` fastest, whatever its grids (see [Unknowns and the equation system](unknowns.md)).
A state is carried to another layout only onto the same momentum grids ([`copy_unknowns!`](@ref)), so a restart onto another momentum grid is refused.

### Radial fluxes between different pitch grids

A radial face lies between two radii whose pitch grids may differ.
Its pitch cells are the overlap cells of the two grids, the common refinement of their faces ([`PitchOverlap`](@ref), [`radial_face_overlap`](@ref)).
The radial coefficients are given per overlap cell, and [`radial_neighbour_pitch_cells`](@ref) gives the cells a radial flux couples.
The pitch grids of all radii must cover one interval, or the operator is refused.

The flux through an overlap cell is computed once, from the values of the two cells it lies in, and given to both with the share of each cell.
No particle is lost or created between different pitch grids, and upwind and exponential fitting keep their positivity across the face.
The two values are taken as constant over their cells, so where the grids differ the flux pairs values whose pitch centres are up to one cell width apart.
This acts as a numerical diffusion in the pitch.
On uniform pitch grids of cell width ``h``, with the grid of one radius shifted by ``s h`` against its neighbours, the radial operator at that radius gains, compared with a shared grid, the term

```math
\frac{D^{rr}}{\Delta r^2}\, s (1 - s)\, h^2\, \frac{\partial^2 f}{\partial \xi^2},
```

checked in `test/test_per_radius_pitch_grids.jl`.
With one pitch grid at all radii the overlap cells are the pitch cells, and the operator is the one of a shared grid.

A four-point radial scheme reads the same pitch cell at three consecutive radii.
An overlap relates only the two grids of one face, so such a stencil exists only when all radii share one pitch grid, and [`AdvectionDiffusion`](@ref) refuses a four-point `r_scheme` otherwise.

[`remap_to_right!`](@ref) and [`remap_to_left!`](@ref) move cell averages between the two grids of an overlap.
They conserve the content, keep the sign and are first-order accurate in the cell width for a smooth distribution.

## Hot-tail and runaway grids

Two kinetic unknowns can split the momentum range: the hot-tail grid below and the runaway grid above, on the same radii.
[`runaway_grid`](@ref) builds the runaway grid at each radius from the last ``p`` face of the hot-tail grid there, followed by the faces given above it, so the two grids share that face exactly (bitwise).
Its pitch grid is the same at all radii and independent of the pitch grid of the hot-tail grid.
The default pitch grid, `biuniform_theta_xi_faces(20, 12, 0.9)`, is a choice of this package (see [Choices of this package](../choices.md)).

### The interface

A [`MomentumInterfaceTerm`](@ref) carries the particle flux through the shared face.
The lower operator has `p_upper_boundary = :interface` and the upper operator `p_lower_boundary = :interface`, so neither adds a flux of its own there.
The same term goes into both equations.
It removes the flux from the last momentum cells of the hot-tail grid and adds the same numbers to the first momentum cells of the runaway grid, so particles are conserved to round-off.
The flux takes ``A^p`` and ``D^{pp}`` of the lower operator at the face, as DREAM's `PXiExternalKineticKinetic` does.
The flux itself is two-point, upwind when the ``p`` scheme of the lower operator is four-point.
Four-point schemes of the two [`AdvectionDiffusion`](@ref) operators see zero ghost cells beyond the interface, so the last interior face on each side sees the other grid as empty.

The pitch grids of the two sides meet through their overlap, in one of two mappings:

- `:overlap` (the default): one flux per overlap cell with the two-point flux of `scheme` between the two cell centres, so upwind and exponential fitting keep their sign properties across the interface.
- `:dream`: DREAM's flux, one per pitch cell of the lower grid, with the upper value interpolated linearly in the pitch and upwind advection.
  Where the pitch grids differ, the interpolation couples an upper cell to its pitch neighbours with negative weights in the diffusive part.
  This mapping therefore does not keep the sign of ``f`` by construction.
  It is compared with DREAM in `test/test_two_grids_dream.jl`.

[`interface_flux`](@ref) gives the flux through the face as a rate per volume.
With the face above the critical momentum it is the runaway rate.
The energy carried through the face enters the energy balance through [`InterfaceEnergyFluxTerm`](@ref) (see [Energy and current balance](energy-current.md)).
