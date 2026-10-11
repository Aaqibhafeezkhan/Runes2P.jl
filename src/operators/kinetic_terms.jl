# Small-angle kinetic terms in cylindrical geometry, following the DREAM equation terms
# (ElectricFieldTerm, SlowingDownTerm, PitchScatterTerm, EnergyDiffusionTerm,
# SynchrotronTerm). In a homogeneous field all bounce averages are unity.

"""
    add_electric_field!(op, plasma)

Acceleration by the parallel electric field: `Aᵖ = (eE/mc) ξ`,
`Aᵠ = (eE/mc)(1 - ξ²)/p`. A positive field accelerates electrons towards `ξ = +1`.
"""
function add_electric_field!(op::AdvectionDiffusion, plasma::PlasmaState)
    g = op.grid
    np, nxi, nr = size(g)
    for ir in 1:nr
        m = momentum_grid(g, ir)
        a = Constants.ELEMENTARY_CHARGE * plasma.electric_field[ir] /
            (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT)
        for j in 1:nxi, k in 1:np+1
            op.advection_p[k, j, ir] += a * m.xi_centers[j]
        end
        for k in 1:nxi+1, i in 1:np
            op.advection_xi[i, k, ir] += a * (1 - m.xi_faces[k]^2) / m.p_centers[i]
        end
    end
    return op
end

"""
    set_unit_field_coefficients!(op)

Advection of an electric field of 1 V/m on the faces of `op`, `Aᵖ = (e/mc) ξ` and
`Aᵠ = (e/mc)(1 - ξ²)/p`, into `op.unit_field_advection_p` and `op.unit_field_advection_xi`.
They depend on the grid alone. An operator applied with a field as an argument
([`apply!`](@ref)`(...; electric_field)`, [`FokkerPlanckTerm`](@ref)`(...; electric_field)`)
adds the field times these to its advection, in place of [`add_electric_field!`](@ref)
with a prescribed field.
"""
function set_unit_field_coefficients!(op::AdvectionDiffusion)
    g = op.grid
    np, nxi, nr = size(g)
    a = Constants.ELEMENTARY_CHARGE / (Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT)
    for ir in 1:nr
        m = momentum_grid(g, ir)
        for j in 1:nxi, k in 1:np+1
            op.unit_field_advection_p[k, j, ir] = a * m.xi_centers[j]
        end
        for k in 1:nxi+1, i in 1:np
            op.unit_field_advection_xi[i, k, ir] = a * (1 - m.xi_faces[k]^2) / m.p_centers[i]
        end
    end
    return op
end

"""
    add_slowing_down!(op, collisions)

Collisional friction `Aᵖ = -p ν_s(p)` on the p faces, and the particle flux through
`p = 0` from `lim p³ν_s`.
"""
function add_slowing_down!(op::AdvectionDiffusion, collisions::CollisionModel)
    g = op.grid
    np, nxi, nr = size(g)
    for ir in 1:nr
        m = momentum_grid(g, ir)
        for k in 1:np+1
            p = m.p_faces[k]
            A = -p * slowing_down_frequency(collisions, ir, p)
            for j in 1:nxi
                op.advection_p[k, j, ir] += A
            end
        end
        if m.p_faces[1] == 0
            op.advection_p_times_p2_at_zero[:, ir] .-= p3_slowing_down_frequency_at_zero(collisions, ir)
        end
    end
    return op
end

"""
    add_pitch_scattering!(op, collisions)

Pitch-angle scattering `Dᵠᵠ = ν_D(p)(1 - ξ²)/2` on the ξ faces.
"""
function add_pitch_scattering!(op::AdvectionDiffusion, collisions::CollisionModel)
    g = op.grid
    np, nxi, nr = size(g)
    nxi == 1 && return op
    for ir in 1:nr, i in 1:np
        m = momentum_grid(g, ir)
        nu = deflection_frequency(collisions, ir, m.p_centers[i])
        for k in 1:nxi+1
            op.diffusion_xi_xi[i, k, ir] += nu * (1 - m.xi_faces[k]^2) / 2
        end
    end
    return op
end

"""
    add_energy_diffusion!(op, collisions)

Energy diffusion `Dᵖᵖ = ν_∥(p)` on the p faces (zero unless `mode = :full`).
"""
function add_energy_diffusion!(op::AdvectionDiffusion, collisions::CollisionModel)
    collisions.settings.mode === :full || return op
    g = op.grid
    np, nxi, nr = size(g)
    for ir in 1:nr, k in 1:np+1
        m = momentum_grid(g, ir)
        D = parallel_diffusion_frequency(collisions, ir, m.p_faces[k])
        for j in 1:nxi
            op.diffusion_p_p[k, j, ir] += D
        end
    end
    return op
end

"""
    add_synchrotron!(op, plasma)

Synchrotron radiation reaction in a homogeneous field `B`:
`Aᵖ = -C B² p γ (1 - ξ²)`, `Aᵠ = C B² ξ (1 - ξ²)/γ`, `C = e⁴/(6π ε₀ mₑ³ c³)`.
"""
function add_synchrotron!(op::AdvectionDiffusion, plasma::PlasmaState)
    g = op.grid
    np, nxi, nr = size(g)
    C = Constants.ELEMENTARY_CHARGE^4 /
        (6pi * Constants.VACUUM_PERMITTIVITY * Constants.ELECTRON_MASS^3 * Constants.SPEED_OF_LIGHT^3)
    for ir in 1:nr
        m = momentum_grid(g, ir)
        prefactor = C * plasma.magnetic_field[ir]^2
        for j in 1:nxi, k in 1:np+1
            p = m.p_faces[k]
            op.advection_p[k, j, ir] -= prefactor * p * sqrt(1 + p^2) * (1 - m.xi_centers[j]^2)
        end
        for k in 1:nxi+1, i in 1:np
            xi = m.xi_faces[k]
            op.advection_xi[i, k, ir] += prefactor * (1 - xi^2) * xi / sqrt(1 + m.p_centers[i]^2)
        end
    end
    return op
end

"""
    fokker_planck_operator(grid, plasma, settings; synchrotron=true,
                           scheme=:exponential_fitting, p_scheme=scheme, xi_scheme=scheme,
                           p_lower_boundary=:zero_flux, p_upper_boundary=:f_zero,
                           r_boundary=:zero_flux, electric_field=true)

Kinetic operator for the hot electrons: electric field, slowing down, pitch-angle
scattering, energy diffusion (`:full` mode) and synchrotron losses, with
`f = 0` beyond `p_max` as in DREAM's hot-tail equation. The synchrotron loss
([`add_synchrotron!`](@ref)) is on by default (decision of the author, 2026-10-04), and
`synchrotron = false` leaves it out, as DREAM does by default. `scheme` is one of the schemes
of [`AdvectionDiffusion`](@ref). The default, exponential fitting (decision of the
author, 2026-09-25, replacing van Albada), keeps f non-negative at every resolution,
holds the collisional Maxwellian's tail to second order where every limiter
overpopulates it (f/f_MJ of 1 800 for van Albada at 6 p_th
on the production grid at 1 keV, and a Dreicer rate 3.5 times the converged one), is
linear in f (two Newton iterations per step, an exact stored Jacobian, about half the
cost of the limiters). Its price is first-order accuracy where a smooth
tail is driven by the field: the slide-away loss through `p_max` on the production grid
is 0.87%, 0.58% and 0.46% at 1×, 2× and 4× the momentum cells against 0.34% converged,
the hot-tail seed of a fast quench 4% high at 1×, so field-driven tails need more
momentum cells than a limiter, with the error roughly halving per doubling (README, DREAM
compatibility and departures). `:van_albada` (Runes2P) and DREAM's limiters stay available;
`scheme = :central` is DREAM's default (centred interpolation) and the setting of the
DREAM comparisons.
`p_scheme` and `xi_scheme` set the directions separately (DREAM's `adv_interp/p1` and
`p2`), e.g. `p_scheme = :tcdf` for DREAM's TCDF limiter.

A hot-tail grid with a runaway grid above it takes `p_upper_boundary = :interface`, and
the operator of the runaway grid `p_lower_boundary = :interface` (a grid with
`p_min > 0`, [`runaway_grid`](@ref)). The flux between the two is the
[`MomentumInterfaceTerm`](@ref) of the equation system.

`electric_field = false` leaves the field of the plasma out of the coefficients, for a
field that is an unknown of the system: the operator holds the collisions
and the coefficients of a unit field, and the term passes the field
([`FokkerPlanckTerm`](@ref)`(...; electric_field = :electric_field)`).
"""
function fokker_planck_operator(grid::KineticGrid, plasma::PlasmaState,
                                settings::CollisionSettings=CollisionSettings();
                                synchrotron::Bool=true, scheme::Symbol=:exponential_fitting,
                                p_scheme::Symbol=scheme, xi_scheme::Symbol=scheme,
                                p_lower_boundary::Symbol=:zero_flux,
                                p_upper_boundary::Symbol=:f_zero, r_boundary::Symbol=:zero_flux,
                                electric_field::Bool=true)
    # DREAM replaces the electric-field advection by a diffusion term in the reduced
    # (pitch-averaged) equation; that form is not implemented. A grid with different cell
    # counts or p faces at different radii is refused by AdvectionDiffusion below.
    has_same_cell_counts(grid) && pitch_cell_count(grid) == 1 &&
        throw(ArgumentError("the reduced equation (one pitch cell) is not implemented"))
    op = AdvectionDiffusion(grid; scheme, p_scheme, xi_scheme, p_lower_boundary, p_upper_boundary, r_boundary)
    set_unit_field_coefficients!(op)
    set_fokker_planck_coefficients!(op, plasma, settings; synchrotron, electric_field)
    return op
end

"""
    zero_coefficients!(op::AdvectionDiffusion)

Set every coefficient array of `op` to zero (not `p_flux_at_lower_boundary`, a boundary
value the caller sets independently of the physics terms). For rebuilding an operator's
coefficients from a new plasma state: the `add_*!` functions accumulate
into the arrays, so a rebuild needs them cleared first.
"""
function zero_coefficients!(op::AdvectionDiffusion)
    for A in (op.advection_p, op.diffusion_p_p, op.diffusion_p_xi, op.advection_xi, op.diffusion_xi_xi,
             op.diffusion_xi_p, op.advection_r, op.diffusion_r_r, op.advection_p_times_p2_at_zero)
        fill!(A, zero(eltype(A)))
    end
    op.coefficient_source[] = nothing
    return op
end

"""
    set_fokker_planck_coefficients!(op, plasma, settings=CollisionSettings(); synchrotron=true,
                                    electric_field=true)

Rebuild `op`'s coefficients in place from `plasma` and `settings`
([`zero_coefficients!`](@ref) then the same `add_*!` calls as
[`fokker_planck_operator`](@ref)), for a Fokker-Planck term whose plasma state changes
with time ([`PrescribedFokkerPlanckTerm`](@ref)). `op`'s scheme and
boundary conditions are unchanged. `electric_field = false` leaves the field of the plasma
out, for a field that is an unknown of the system. The synchrotron loss is on by default, as
in [`fokker_planck_operator`](@ref) (decision of the author, 2026-10-04).
"""
function set_fokker_planck_coefficients!(op::AdvectionDiffusion, plasma::PlasmaState,
                                         settings::CollisionSettings=CollisionSettings();
                                         synchrotron::Bool=true, electric_field::Bool=true)
    radial_cell_count(plasma) == radial_cell_count(op.grid) ||
        throw(DimensionMismatch("plasma and grid radii differ"))
    zero_coefficients!(op)
    collisions = CollisionModel(plasma, settings;
                                faces=[momentum_grid(op.grid, ir).p_faces for ir in 1:radial_cell_count(op.grid)])
    electric_field && add_electric_field!(op, plasma)
    add_slowing_down!(op, collisions)
    add_pitch_scattering!(op, collisions)
    add_energy_diffusion!(op, collisions)
    synchrotron && add_synchrotron!(op, plasma)
    op.coefficient_source[] = (; plasma, settings, synchrotron, electric_field)
    return op
end
