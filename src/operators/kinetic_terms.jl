# Small-angle kinetic terms in cylindrical geometry, following the DREAM equation terms
# (ElectricFieldTerm, SlowingDownTerm, PitchScatterTerm, EnergyDiffusionTerm,
# SynchrotronTerm). In a homogeneous field all bounce averages are unity.

"""
    add_electric_field!(op, plasma)

Acceleration by the parallel electric field: `Aᵖ = (eE/mc) ξ`,
`Aᵠ = (eE/mc)(1 - ξ²)/p`. Positive `E` accelerates electrons towards `ξ = +1`.
"""
function add_electric_field!(op::AdvectionDiffusion, plasma::PlasmaState)
    g = op.grid
    m = g.mom
    n1, n2, n3 = size(g)
    for ir in 1:n3
        a = Constants.ec * plasma.E[ir] / (Constants.me * Constants.c)
        for j in 1:n2, k in 1:n1+1
            op.Ap[k, j, ir] += a * m.xi[j]
        end
        for k in 1:n2+1, i in 1:n1
            op.Axi[i, k, ir] += a * (1 - m.xi_f[k]^2) / m.p[i]
        end
    end
    return op
end

"""
    add_slowing_down!(op, cm)

Collisional friction `Aᵖ = -p ν_s(p)` on the p faces, and the particle flux through
`p = 0` from `lim p³ν_s`.
"""
function add_slowing_down!(op::AdvectionDiffusion, cm::CollisionModel)
    g = op.grid
    m = g.mom
    n1, n2, n3 = size(g)
    for ir in 1:n3
        for k in 1:n1+1
            pf = m.p_f[k]
            A = -pf * nu_s(cm, ir, pf)
            for j in 1:n2
                op.Ap[k, j, ir] += A
            end
        end
        if m.p_f[1] == 0
            op.Ap_p2_zero[:, ir] .-= p3nu_s_at_zero(cm, ir)
        end
    end
    return op
end

"""
    add_pitch_scattering!(op, cm)

Pitch-angle scattering `Dᵠᵠ = ν_D(p)(1 - ξ²)/2` on the ξ faces.
"""
function add_pitch_scattering!(op::AdvectionDiffusion, cm::CollisionModel)
    g = op.grid
    m = g.mom
    n1, n2, n3 = size(g)
    n2 == 1 && return op
    for ir in 1:n3, i in 1:n1
        ν = nu_D(cm, ir, m.p[i])
        for k in 1:n2+1
            op.Dxixi[i, k, ir] += ν * (1 - m.xi_f[k]^2) / 2
        end
    end
    return op
end

"""
    add_energy_diffusion!(op, cm)

Energy diffusion `Dᵖᵖ = ν_∥(p)` on the p faces (zero unless `mode = :full`).
"""
function add_energy_diffusion!(op::AdvectionDiffusion, cm::CollisionModel)
    cm.settings.mode === :full || return op
    g = op.grid
    m = g.mom
    n1, n2, n3 = size(g)
    for ir in 1:n3, k in 1:n1+1
        D = nu_par(cm, ir, m.p_f[k])
        for j in 1:n2
            op.Dpp[k, j, ir] += D
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
    m = g.mom
    n1, n2, n3 = size(g)
    C = Constants.ec^4 / (6π * Constants.eps0 * Constants.me^3 * Constants.c^3)
    for ir in 1:n3
        pre = C * plasma.B[ir]^2
        for j in 1:n2, k in 1:n1+1
            p = m.p_f[k]
            op.Ap[k, j, ir] -= pre * p * sqrt(1 + p^2) * (1 - m.xi[j]^2)
        end
        for k in 1:n2+1, i in 1:n1
            ξ = m.xi_f[k]
            op.Axi[i, k, ir] += pre * (1 - ξ^2) * ξ / sqrt(1 + m.p[i]^2)
        end
    end
    return op
end

"""
    fokker_planck_operator(grid, plasma, settings; synchrotron=false, scheme=:central,
                           pmax_bc=:f_zero)

Kinetic operator for the hot electrons: electric field, slowing down, pitch-angle
scattering, energy diffusion (`:full` mode) and optionally synchrotron losses.
The defaults match DREAM's hot-tail equation with centred interpolation and
`f = 0` beyond `p_max`.
"""
function fokker_planck_operator(grid::KineticGrid, plasma::PlasmaState,
                                settings::CollisionSettings=CollisionSettings();
                                synchrotron::Bool=false, scheme::Symbol=:central,
                                pmax_bc::Symbol=:f_zero, rbc::Symbol=:zero_flux)
    n_radii(plasma) == n_r(grid) || throw(DimensionMismatch("plasma and grid radii differ"))
    # DREAM replaces the electric-field advection by a diffusion term in the reduced
    # (pitch-averaged) equation; that form is not implemented.
    n_xi(grid) > 1 || throw(ArgumentError("the reduced equation (n_xi = 1) is not implemented"))
    cm = CollisionModel(plasma, settings)
    op = AdvectionDiffusion(grid; scheme, pmax_bc, rbc)
    add_electric_field!(op, plasma)
    add_slowing_down!(op, cm)
    add_pitch_scattering!(op, cm)
    add_energy_diffusion!(op, cm)
    synchrotron && add_synchrotron!(op, plasma)
    return op
end
