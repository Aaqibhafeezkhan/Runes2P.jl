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
    m = g.momentum_grid
    np, nxi, nr = size(g)
    for ir in 1:nr
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
    add_slowing_down!(op, collisions)

Collisional friction `Aᵖ = -p ν_s(p)` on the p faces, and the particle flux through
`p = 0` from `lim p³ν_s`.
"""
function add_slowing_down!(op::AdvectionDiffusion, collisions::CollisionModel)
    g = op.grid
    m = g.momentum_grid
    np, nxi, nr = size(g)
    for ir in 1:nr
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
    m = g.momentum_grid
    np, nxi, nr = size(g)
    nxi == 1 && return op
    for ir in 1:nr, i in 1:np
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
    m = g.momentum_grid
    np, nxi, nr = size(g)
    for ir in 1:nr, k in 1:np+1
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
    m = g.momentum_grid
    np, nxi, nr = size(g)
    C = Constants.ELEMENTARY_CHARGE^4 /
        (6pi * Constants.VACUUM_PERMITTIVITY * Constants.ELECTRON_MASS^3 * Constants.SPEED_OF_LIGHT^3)
    for ir in 1:nr
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
    fokker_planck_operator(grid, plasma, settings; synchrotron=false, scheme=:central,
                           p_upper_boundary=:f_zero, r_boundary=:zero_flux)

Kinetic operator for the hot electrons: electric field, slowing down, pitch-angle
scattering, energy diffusion (`:full` mode) and optionally synchrotron losses.
The defaults match DREAM's hot-tail equation with centred interpolation and
`f = 0` beyond `p_max`.
"""
function fokker_planck_operator(grid::KineticGrid, plasma::PlasmaState,
                                settings::CollisionSettings=CollisionSettings();
                                synchrotron::Bool=false, scheme::Symbol=:central,
                                p_upper_boundary::Symbol=:f_zero, r_boundary::Symbol=:zero_flux)
    radial_cell_count(plasma) == radial_cell_count(grid) ||
        throw(DimensionMismatch("plasma and grid radii differ"))
    # DREAM replaces the electric-field advection by a diffusion term in the reduced
    # (pitch-averaged) equation; that form is not implemented.
    pitch_cell_count(grid) > 1 ||
        throw(ArgumentError("the reduced equation (one pitch cell) is not implemented"))
    collisions = CollisionModel(plasma, settings)
    op = AdvectionDiffusion(grid; scheme, p_upper_boundary, r_boundary)
    add_electric_field!(op, plasma)
    add_slowing_down!(op, collisions)
    add_pitch_scattering!(op, collisions)
    add_energy_diffusion!(op, collisions)
    synchrotron && add_synchrotron!(op, plasma)
    return op
end
