# Helpers of test_pitch_accuracy.jl.

const PITCH_LINEAR_OR_LIMITED = (:exponential_fitting, :van_albada, :ospre, :muscl, :tcdf)

pitch_uniform_theta_faces(cell_count) = cos.(range(pi, 0; length = cell_count + 1))
function pitch_biuniform_theta_faces(refinement)
    biuniform_theta_xi_faces(15refinement, 10refinement, 0.5)
end

function pitch_system(xi_faces, scheme, advection, diffusion)
    mom = MomentumGrid([1.0, 1.1], xi_faces)
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    op = AdvectionDiffusion(g; scheme, p_upper_boundary = :f_zero)
    for (k, xi) in enumerate(mom.xi_faces)
        op.advection_xi[1, k, 1] = advection(xi)
        op.diffusion_xi_xi[1, k, 1] = diffusion(xi)
    end
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
    return mom,
    EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
end

pitch_beam(xi) = 1 + 0.3xi + 0.2 * exp(10 * (xi - 1))
pitch_beam_derivative(xi) = 0.3 + 2.0 * exp(10 * (xi - 1))

# relative L1 error of the manufactured step u - L(u) = f_e - L_e f_e
function pitch_manufactured_error(xi_faces, scheme, strength)
    advection(xi) = strength * (1 - xi^2)
    diffusion(xi) = 0.4 * (1 - xi^2)
    mom, system = pitch_system(xi_faces, scheme, advection, diffusion)
    faces, widths = mom.xi_faces, mom.xi_widths
    flux(xi) = advection(xi) * pitch_beam(xi) - diffusion(xi) * pitch_beam_derivative(xi)
    exact = [-(flux(faces[j + 1]) - flux(faces[j])) / widths[j] for j in eachindex(widths)]
    f_exact = pitch_beam.(mom.xi_centers)
    u = copy(f_exact)
    result = newton_solve!(u, f_exact .- exact,
        NewtonWorkspace(system, NewtonSolver(max_iterations = 60)), 1.0, 1.0)
    return result.status, sum(abs.(u .- f_exact) .* widths) / sum(abs.(f_exact) .* widths)
end

# relative L1 error against the cell averages of exp(κ ξ), and the smallest value
function pitch_equilibrium_error(xi_faces, scheme, kappa)
    mom, system = pitch_system(xi_faces, scheme, xi -> kappa * (1 - xi^2), xi -> 1 - xi^2)
    faces, widths = mom.xi_faces, mom.xi_widths
    u_old = ones(length(widths))
    u = copy(u_old)
    result = newton_solve!(
        u, u_old, NewtonWorkspace(system, NewtonSolver(max_iterations = 100)), 1e8, 1e8)
    exact = [(exp(kappa * (faces[j + 1] - 1)) - exp(kappa * (faces[j] - 1))) /
             (kappa * widths[j]) for j in eachindex(widths)]
    exact .*= sum(u_old .* widths) / sum(exact .* widths)
    return (status = result.status,
        error = sum(abs.(u .- exact) .* widths) / sum(exact .* widths),
        minimum = minimum(u))
end

# pure pitch advection a(1 - ξ²) (electric field, no scattering): the exact solution is
# carried along ξ(t) = tanh(atanh ξ₀ + a t) and compressed towards ξ = 1, with cell
# averages from the antiderivative of the initial profile. Backward Euler at dt and dt/2
# combined by Richardson extrapolation, so that the time error is second order.
pitch_initial(xi) = 1 + 0.5 * sin(pi * xi)
pitch_initial_antiderivative(xi) = xi - 0.5 * cos(pi * xi) / pi
pitch_origin(xi, t) = xi >= 1 ? 1.0 : xi <= -1 ? -1.0 : tanh(atanh(xi) - t)
function pitch_advected_averages(faces, t)
    [(pitch_initial_antiderivative(pitch_origin(faces[j + 1], t)) -
      pitch_initial_antiderivative(pitch_origin(faces[j], t))) /
     (faces[j + 1] - faces[j]) for j in 1:(length(faces) - 1)]
end

function pitch_advection_error(xi_faces, scheme; dt = 2e-3, t_end = 1.0)
    mom, system = pitch_system(xi_faces, scheme, xi -> 1 - xi^2, xi -> 0.0)
    u0 = pitch_advected_averages(mom.xi_faces, 0.0)
    coarse = integrate(system, u0, collect(0:dt:t_end)).states[:, end]
    fine = integrate(system, u0, collect(0:(dt / 2):t_end)).states[:, end]
    exact = pitch_advected_averages(mom.xi_faces, t_end)
    widths = mom.xi_widths
    return sum(abs.((2fine .- coarse) .- exact) .* widths) / sum(abs.(exact) .* widths)
end
