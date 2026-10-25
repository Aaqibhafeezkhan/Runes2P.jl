# Accuracy of the advection schemes in the pitch direction alone, independent of DREAM:
# one momentum cell, only the ξ advection and diffusion set, on a uniform pitch grid,
# DREAM's uniform-θ grid and a biuniform-θ grid (`biuniform_theta` in `grids`: 10 cells to
# 0.5 rad and 5 to π) and its refinements. Three references:
# - a manufactured solution with a beam-like profile peaked at ξ = 1, one backward-Euler
#   step of unit length, at small and large cell Péclet number (strength 1 and 20);
# - the exact collisional equilibrium of electric-field advection a(1 - ξ²) against
#   pitch-angle scattering d(1 - ξ²), f ∝ exp(κ ξ) with κ = a/d: the flux vanishes, so it
#   is the steady state of the physical pitch operator at fixed p, a narrow beam at ξ = 1
#   for large κ. Compared as cell averages, which the moments assume, after a backward-Euler
#   step of 1e8 from a uniform state (the scheme conserves particles, so the total matches).
# - pure pitch advection against its exact characteristics (below).
# The equilibrium spans 9 decades at κ = 20; the solve leaves a floor near 1e-9 in the
# relative L1 error, the same for every scheme.

include("test_pitch_accuracy_helpers.jl")

@testset "pitch accuracy" begin
    grids = (uniform = (uniform_xi_faces, (64, 128)),
        uniform_theta = (pitch_uniform_theta_faces, (64, 128)),
        biuniform_theta = (pitch_biuniform_theta_faces, (4, 8)))

    @testset "manufactured solution in ξ" begin
        # observed orders at the two finest levels: 1.84-2.51 for every limiter and
        # exponential fitting at strength 1 and 20, except exponential fitting at strength
        # 20 on the biuniform-θ grid (1.64, and 1.15, 1.34 on the coarser levels: first order
        # at large cell Péclet number, with 10 times van Albada's error at 1x and 58 times at
        # 8x); upwind 0.96-0.99. TCDF and SMART agree exactly here: the profile keeps the
        # slope ratio where both limiter functions are QUICK.
        for (name, (faces, levels)) in pairs(grids), strength in (1.0, 20.0)

            for scheme in PITCH_LINEAR_OR_LIMITED
                (status1, e1), (status2, e2) = (pitch_manufactured_error(faces(n), scheme, strength)
                for n in levels)
                @test status1 === status2 === :converged
                if scheme === :exponential_fitting && strength == 20.0 &&
                   name === :biuniform_theta
                    @test log2(e1 / e2) > 1.5
                else
                    @test log2(e1 / e2) > 1.8
                end
            end
            (_, e1), (_, e2) = (pitch_manufactured_error(faces(n), :upwind, strength)
            for n in levels)
            @test 0.8 < log2(e1 / e2) < 1.1
        end
        _, fitted = pitch_manufactured_error(pitch_biuniform_theta_faces(8), :exponential_fitting, 20.0)
        _, limited = pitch_manufactured_error(pitch_biuniform_theta_faces(8), :van_albada, 20.0)
        @test fitted > 10limited
    end

    @testset "collisional equilibrium exp(κξ), κ = 20" begin
        kappa = 20.0
        # exponential fitting is exact for a constant A/D on a uniform grid (its flux
        # vanishes for exp(κξ) between any two cell centres): only the Newton floor remains
        for n in (16, 64)
            r = pitch_equilibrium_error(uniform_xi_faces(n), :exponential_fitting, kappa)
            @test r.status === :converged
            @test r.error < 1e-6
        end
        # on nonuniform grids the cell averages differ from the point values it matches by
        # O(Δξ²): second order at every level (biuniform-θ 0.137, 0.052, 0.014, 0.0036 at
        # 1x, 2x, 4x, 8x)
        e = [pitch_equilibrium_error(pitch_biuniform_theta_faces(k), :exponential_fitting, kappa).error
             for k in (4, 8)]
        @test log2(e[1] / e[2]) > 1.8
        # van Albada on the biuniform-θ grid is pre-asymptotic: 0.102, 0.050, 0.026,
        # 0.0090 at 1x, 2x, 4x, 8x (orders 1.03, 0.93, 1.54), smaller than exponential
        # fitting's error at 1x and 2x, larger at 4x and 8x; on the uniform-θ grid it reaches
        # second order from 32 cells (2.11, 2.26)
        e = [pitch_equilibrium_error(pitch_biuniform_theta_faces(k), :van_albada, kappa).error
             for k in (4, 8)]
        @test log2(e[1] / e[2]) > 1.4
        @test e[2] < 0.01
        e = [pitch_equilibrium_error(pitch_uniform_theta_faces(n), :van_albada, kappa).error
             for n in (64, 128)]
        @test log2(e[1] / e[2]) > 1.9
        # positivity in ξ: the schemes with a positivity property stay positive; central
        # goes negative on the biuniform-θ grid (min f -25.6 at 1x). SMART's Newton iteration
        # does not converge in this problem on the uniform-θ and biuniform-θ grids and its
        # iterate goes negative (not asserted: a solver failure, not a property)
        for scheme in (:upwind, PITCH_LINEAR_OR_LIMITED...), k in (1, 2)

            r = pitch_equilibrium_error(pitch_biuniform_theta_faces(k), scheme, kappa)
            @test r.status === :converged
            @test r.minimum > 0
        end
        @test pitch_equilibrium_error(pitch_biuniform_theta_faces(1), :central, kappa).minimum <
              0
    end

    @testset "SMART reaches the pitch equilibrium by time stepping" begin
        # From a distant start the steady Newton step of SMART cycles between the linear
        # pieces of its limiter on two of nine grids (period 2 and 4 on the uniform-θ grid of
        # 16 cells, no convergence at 2x the biuniform-θ grid). The steady state exists
        # and is positive: time steps of 0.01 reach it with 1.03 to 1.05 Newton iterations per
        # step, and the steady step started there converges in 2 to 3 iterations. Errors
        # measured: 0.0947 and 0.0482.
        kappa = 20.0
        for (faces, bound) in ((pitch_uniform_theta_faces(16), 0.11), (
            pitch_biuniform_theta_faces(2), 0.06))
            mom, system = pitch_system(faces, :smart, xi -> kappa * (1 - xi^2), xi -> 1 -
                                                                                      xi^2)
            widths = mom.xi_widths
            start = ones(length(widths))
            solution = integrate(system, start, collect(0:0.01:20))
            @test all(solution.statistics.accepted)
            @test sum(solution.statistics.newton_iterations) <
                  1.2 * length(solution.statistics.newton_iterations)
            @test minimum(solution.states) > 0
            state = solution.states[:, end]
            @test maximum(abs, state .- solution.states[:, end - 1]) <
                  1e-12 * maximum(state)
            exact = [(exp(kappa * (mom.xi_faces[j + 1] - 1)) -
                      exp(kappa * (mom.xi_faces[j] - 1))) / (kappa * widths[j])
                     for j in eachindex(widths)]
            exact .*= sum(start .* widths) / sum(exact .* widths)
            @test sum(abs.(state .- exact) .* widths) / sum(exact .* widths) < bound
            steady = copy(state)
            result = newton_solve!(steady, start,
                NewtonWorkspace(system, NewtonSolver(max_iterations = 100)), 1e8, 1e8)
            @test result.status === :converged
            @test result.iterations <= 4
            @test minimum(steady) > 0
        end
    end

    @testset "pure pitch advection against the characteristics" begin
        # biuniform-θ grid at 1x, 2x, 4x, 8x:
        # van Albada 0.18, 0.098, 0.029, 0.0064 (second order from 2x); exponential fitting,
        # which is upwind without diffusion, 0.30, 0.23, 0.15, 0.087 (first order); TCDF
        # 0.13 at 1x; central goes negative. On a uniform grid van Albada's order falls to
        # 1.1 between 64 and 128 cells: the limiter clips the extremum of the profile.
        va = [pitch_advection_error(pitch_biuniform_theta_faces(k), :van_albada)
              for k in (2, 4)]
        @test log2(va[1] / va[2]) > 1.5
        fitted = pitch_advection_error(pitch_biuniform_theta_faces(4), :exponential_fitting)
        @test fitted ≈ pitch_advection_error(pitch_biuniform_theta_faces(4), :upwind) rtol = 1e-10
        @test fitted > 3va[2]
    end
end
