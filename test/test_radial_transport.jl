# Radial transport of f: the Rechester-Rosenbluth coefficient, the edge of
# the plasma with the distance of Runes2P and of DREAM, the decay of the Bessel modes of a
# cylinder as the independent reference, the particle budget with the loss through the
# edge, a region without transport, and the transport history as an update of the run.
using HDF5
using LinearAlgebra: eigvals
using Random: Xoshiro

const BESSEL_J0_ZEROS = [2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281,
                         14.930917708487787, 18.071063967910924, 21.21163662987926, 24.352471530749302]

# lowest decay rate of the radial operator with a uniform coefficient, over D j₀₁²/a²
function lowest_decay_rate(radii, r_boundary; minor_radius=1.3, diffusion=0.7)
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(1, 0.5, 1.5), uniform_xi_faces(2)),
                            RadialGrid(uniform_r_faces(radii, minor_radius)))
    op = AdvectionDiffusion(grid; scheme=:central, r_boundary)
    op.diffusion_r_r .= diffusion
    rates = -real.(eigvals(Matrix(operator_matrix(op))))
    return minimum(rates) / (diffusion * BESSEL_J0_ZEROS[1]^2 / minor_radius^2)
end

# radial transport of one kinetic unknown with the particles lost through the edge
function radial_transport_system(grid, radial)
    layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid), ScalarUnknown(:lost_particles)])
    system = EquationSystem(layout, [
        Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, radial)),
        Equation(:lost_particles, TimeDerivative(), RadialEdgeLossTerm(:f, radial))])
    return layout, system
end

state_of(layout, f) = vcat(vec(f), 0.0)
distribution(layout, grid, u) = reshape(u[1:end-1], size(grid))

@testset "radial transport" begin
    rng = Xoshiro(51)
    momentum = MomentumGrid(biuniform_p_faces(6, 3, 0.0, 0.5, 3.0), uniform_xi_faces(4))

    @testset "Rechester-Rosenbluth coefficient" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(3, 0.6)))
        op = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        level, major_radius = 1e-3, 6.2
        add_rechester_rosenbluth_diffusion!(op, level; major_radius)
        c = Constants.SPEED_OF_LIGHT
        for k in 1:4, (j, xi) in enumerate(momentum.xi_centers), (i, p) in enumerate(momentum.p_centers)
            @test op.diffusion_r_r[i, j, k] ≈ pi * major_radius * level^2 * c * p * abs(xi) / sqrt(1 + p^2) rtol = 1e-14
        end
        @test all(==(0), op.advection_r)
        # it grows with the square of the level (DREAM #421) and with the safety factor
        double = AdvectionDiffusion(grid; scheme=:central)
        add_rechester_rosenbluth_diffusion!(double, 2level; major_radius)
        @test double.diffusion_r_r ≈ 4 .* op.diffusion_r_r rtol = 1e-14
        with_q = AdvectionDiffusion(grid; scheme=:central)
        add_rechester_rosenbluth_diffusion!(with_q, level; major_radius, safety_factor=[1.0, 1.5, 2.0, 3.0])
        for k in 1:4
            @test with_q.diffusion_r_r[:, :, k] ≈ [1.0, 1.5, 2.0, 3.0][k] .* op.diffusion_r_r[:, :, k] rtol = 1e-14
        end
        # DREAM in a cylinder: R₀ = 1 and q = 1, so dBB √R₀ there gives the coefficient of R₀ here
        dream = AdvectionDiffusion(grid; scheme=:central)
        add_rechester_rosenbluth_diffusion!(dream, level * sqrt(major_radius); major_radius=1)
        @test dream.diffusion_r_r ≈ op.diffusion_r_r rtol = 1e-14
        # one level per radial face, a face without transport, and the sum of two calls
        profile = AdvectionDiffusion(grid; scheme=:central)
        add_rechester_rosenbluth_diffusion!(profile, [0.0, 0.0, 1e-3, 2e-3]; major_radius)
        @test all(==(0), profile.diffusion_r_r[:, :, 1:2])
        @test profile.diffusion_r_r[:, :, 3] ≈ op.diffusion_r_r[:, :, 3] rtol = 1e-14
        @test profile.diffusion_r_r[:, :, 4] ≈ 4 .* op.diffusion_r_r[:, :, 4] rtol = 1e-14
        add_rechester_rosenbluth_diffusion!(profile, [0.0, 0.0, 1e-3, 2e-3]; major_radius)
        @test profile.diffusion_r_r[:, :, 3] ≈ 2 .* op.diffusion_r_r[:, :, 3] rtol = 1e-14
        @test radial_face_values(2.0, grid) == fill(2.0, 4)
        @test_throws DimensionMismatch add_rechester_rosenbluth_diffusion!(profile, [1e-3, 1e-3, 1e-3]; major_radius)
        @test_throws ArgumentError add_rechester_rosenbluth_diffusion!(profile, -1e-3; major_radius)
        @test_throws ArgumentError add_rechester_rosenbluth_diffusion!(profile, 1e-3; major_radius=0)
        @test_throws ArgumentError add_rechester_rosenbluth_diffusion!(profile, 1e-3; major_radius, safety_factor=0)

        # pitch grids that differ between radii: the pitch of the cells of the radial face
        faces = [uniform_xi_faces(4), biuniform_theta_xi_faces(4, 2, 0.9), uniform_xi_faces(4)]
        per_radius = cylindrical_grid([MomentumGrid(momentum.p_faces, xi) for xi in faces],
                                      RadialGrid(uniform_r_faces(3, 0.6)))
        op = AdvectionDiffusion(per_radius; scheme=:central)
        add_rechester_rosenbluth_diffusion!(op, level; major_radius)
        p = momentum.p_centers[2]
        for k in 1:4
            overlap = radial_face_overlap(per_radius, k)
            count = overlap_cell_count(overlap)
            @test op.diffusion_r_r[2, 1:count, k] ≈
                  pi * major_radius * level^2 * c * p / sqrt(1 + p^2) .* abs.(overlap.xi_centers) rtol = 1e-14
            @test all(==(0), op.diffusion_r_r[2, count+1:end, k])
        end

        # a runaway grid takes the radial operator as well
        runaway = runaway_grid(grid, uniform_p_faces(4, 3.0, 40.0)[2:end])
        op = AdvectionDiffusion(runaway; scheme=:central, r_boundary=:absorbing)
        add_rechester_rosenbluth_diffusion!(op, level; major_radius)
        @test all(>(0), op.diffusion_r_r[:, 1:20, 2:4])
        f = rand(rng, size(runaway)...)
        @test radial_particle_flux(op, f, 4) > 0
    end

    @testset "edge of the plasma" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(5, 1.0)))
        f = rand(rng, size(grid)...)
        fluxes = map((:absorbing, :absorbing_dream, :zero_flux)) do r_boundary
            op = AdvectionDiffusion(grid; scheme=:central, r_boundary)
            op.diffusion_r_r .= rand(Xoshiro(2), size(op.diffusion_r_r)...)
            radial_particle_flux(op, f, 6)
        end
        # DREAM's distance is that of the last two centres, twice the distance to the face
        @test fluxes[1] ≈ 2 * fluxes[2] rtol = 1e-14
        @test fluxes[3] == 0
        # the edge flux by hand, with Runes2P's distance
        op = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        op.diffusion_r_r .= 0.3
        V = phase_space_jacobian_r_faces(grid)
        expected = sum(V[i, j, 6] * momentum.p_widths[i] * momentum.xi_widths[j] * 0.3 * f[i, j, 5] / 0.1
                       for i in 1:6, j in 1:4)
        @test radial_particle_flux(op, f, 6) ≈ expected rtol = 1e-13
        # the advective part is the upwind outflow in both forms
        for r_boundary in (:absorbing, :absorbing_dream)
            op = AdvectionDiffusion(grid; scheme=:upwind, r_boundary)
            op.advection_r .= 2.0
            outflow = sum(V[i, j, 6] * momentum.p_widths[i] * momentum.xi_widths[j] * 2.0 * f[i, j, 5]
                          for i in 1:6, j in 1:4)
            @test radial_particle_flux(op, f, 6) ≈ outflow rtol = 1e-13
            op.advection_r .= -2.0
            @test radial_particle_flux(op, f, 6) == 0
        end
        # one radius: DREAM takes the distance 1
        single = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(1, 0.4)))
        op = AdvectionDiffusion(single; scheme=:central, r_boundary=:absorbing_dream)
        op.diffusion_r_r .= 0.3
        g = rand(rng, size(single)...)
        V = phase_space_jacobian_r_faces(single)
        @test radial_particle_flux(op, g, 2) ≈
              sum(V[i, j, 2] * momentum.p_widths[i] * momentum.xi_widths[j] * 0.3 * g[i, j, 1] for i in 1:6, j in 1:4) rtol = 1e-13
        @test_throws ArgumentError AdvectionDiffusion(grid; r_boundary=:open)
        @test_throws ArgumentError radial_particle_flux(op, g, 3)
    end

    @testset "fluxes through the radial faces and the content" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(5, 1.0)))
        op = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r)...)
        op.advection_r .= randn(rng, size(op.advection_r)...)
        f = rand(rng, size(grid)...)
        df = apply!(zeros(size(grid)), op, f)
        content_rate = radial_particle_content(grid, df)
        @test sum(radial_particle_content(grid, f)) ≈ total_particles(grid, f) rtol = 1e-14
        @test radial_particle_flux(op, f, 1) == 0
        # the content of a radius changes by the difference of the fluxes through its faces
        for ir in 1:5
            @test content_rate[ir] ≈ radial_particle_flux(op, f, ir) - radial_particle_flux(op, f, ir + 1) rtol = 1e-12
        end
        @test sum(content_rate) ≈ -radial_particle_flux(op, f, 6) rtol = 1e-12
    end

    @testset "decay of the lowest Bessel mode" begin
        # a uniform coefficient in a cylinder: the lowest mode J₀(j₀₁ r/a) decays at the rate
        # D j₀₁²/a². Runes2P's edge is second order, DREAM's distance first order
        runes2p = [lowest_decay_rate(n, :absorbing) for n in (5, 11, 20, 40)]
        dream = [lowest_decay_rate(n, :absorbing_dream) for n in (5, 11, 20, 40)]
        @test round.(runes2p; digits=3) == [0.987, 0.997, 0.999, 1.0]
        @test round.(dream; digits=3) == [0.811, 0.911, 0.951, 0.975]
        order(coarse, fine) = log2((1 - coarse) / (1 - fine))
        @test 1.9 < order(runes2p[3], runes2p[4]) < 2.1
        @test 1.9 < order(runes2p[2], lowest_decay_rate(22, :absorbing)) < 2.1
        @test 0.9 < order(dream[3], dream[4]) < 1.1

        # radial diffusion couples no momentum cells: with the Rechester-Rosenbluth
        # coefficient every momentum cell decays at its own rate π q R₀ (δB/B)² |v_∥| j₀₁²/a²
        minor_radius, major_radius, level = 2.0, 6.2, 1e-3
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(11, minor_radius)))
        op = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        add_rechester_rosenbluth_diffusion!(op, level; major_radius)
        M = Matrix(operator_matrix(op))
        index = LinearIndices(size(grid))
        for (j, xi) in enumerate(momentum.xi_centers), (i, p) in enumerate(momentum.p_centers)
            cells = [index[i, j, ir] for ir in 1:11]
            rate = minimum(-real.(eigvals(M[cells, cells])))
            exact = pi * major_radius * level^2 * Constants.SPEED_OF_LIGHT * p * abs(xi) / sqrt(1 + p^2) *
                    BESSEL_J0_ZEROS[1]^2 / minor_radius^2
            @test rate / exact ≈ runes2p[2] rtol = 1e-10
            # nothing leaves the cell in momentum space
            others = setdiff(1:length(index), cells)
            @test all(==(0), M[cells, others])
        end
    end

    @testset "manufactured solution with a coefficient that varies in r" begin
        # f = cos(π r/2a) with D = D₀ (1 + r²/a²) solves (1/r)(r D f')' = S.
        # The steady solution of the discrete operator with the exact S converges at second
        # order with Runes2P's edge on a uniform grid and on one refined towards the edge, at
        # first order with DREAM's distance on the uniform grid, and at second order with it
        # on the refined grid, where the edge cell is of order h².
        a, D0 = 1.3, 0.7
        coefficient(r) = D0 * (1 + (r / a)^2)
        exact(r) = cos(pi * r / (2a))
        k = pi / (2a)
        source(r) = -coefficient(r) * k^2 * cos(k * r) - (coefficient(r) / r + 2D0 * r / a^2) * k * sin(k * r)
        function solution_error(cells, refined, r_boundary)
            faces = refined ? a .* sin.(pi / 2 .* (0:cells) ./ cells) : uniform_r_faces(cells, a)
            g = cylindrical_grid(MomentumGrid(uniform_p_faces(1, 0.5, 1.5), uniform_xi_faces(2)), RadialGrid(faces))
            op = AdvectionDiffusion(g; scheme=:central, r_boundary)
            for (ir, r) in enumerate(g.radial_grid.r_faces)
                op.diffusion_r_r[:, :, ir] .= coefficient(r)
            end
            centers = g.radial_grid.r_centers
            u = operator_matrix(op) \ vec([source(r) for _ in 1:1, _ in 1:2, r in centers])
            return maximum(abs.(u .- vec([exact(r) for _ in 1:1, _ in 1:2, r in centers])))
        end
        order(args...) = log2(solution_error(20, args...) / solution_error(40, args...))
        @test 1.95 < order(false, :absorbing) < 2.05
        @test 1.95 < order(true, :absorbing) < 2.05
        @test 0.95 < order(false, :absorbing_dream) < 1.05
        @test 1.95 < order(true, :absorbing_dream) < 2.05
        @test solution_error(40, false, :absorbing) < 3e-5
    end

    @testset "particle budget and survival of a uniform profile" begin
        minor_radius, diffusion = 1.0, 2.0
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(2)),
                                RadialGrid(uniform_r_faces(40, minor_radius)))
        radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        radial.diffusion_r_r .= diffusion
        layout, system = radial_transport_system(grid, radial)
        f0 = ones(size(grid))
        times = collect(range(0, 0.2; length=401))
        solution = integrate(system, state_of(layout, f0), times; save_times=times[1:40:end])
        initial = total_particles(grid, f0)
        for k in axes(solution.states, 2)
            f = distribution(layout, grid, solution.states[:, k])
            lost = solution.states[end, k]
            @test total_particles(grid, f) + lost ≈ initial rtol = 1e-12
            # the content against the sum over the Bessel modes, Σ 4/j₀ₙ² exp(-D j₀ₙ² t/a²).
            # Eight modes hold the sum from the second saved time on
            k == 1 && continue
            t = solution.times[k]
            series = sum(4 / z^2 * exp(-diffusion * z^2 * t / minor_radius^2) for z in BESSEL_J0_ZEROS)
            @test total_particles(grid, f) / initial ≈ series rtol = 0.02
        end
        @test solution.states[end, end] > 0.8 * initial

        # no loss through a closed edge, and the unknown of the lost particles stays zero
        closed = AdvectionDiffusion(grid; scheme=:central, r_boundary=:zero_flux)
        closed.diffusion_r_r .= diffusion
        layout, system = radial_transport_system(grid, closed)
        peaked = [exp(-(r / 0.3)^2) for _ in 1:2, _ in 1:2, r in grid.radial_grid.r_centers]
        solution = integrate(system, state_of(layout, peaked), times[1:41])
        @test solution.states[end, end] == 0
        final = distribution(layout, grid, solution.states[:, end])
        @test total_particles(grid, final) ≈ total_particles(grid, peaked) rtol = 1e-12
        @test maximum(final) - minimum(final) < maximum(peaked) - minimum(peaked)
    end

    @testset "region without transport" begin
        # no transport inside the fourth radial face: the content of the three inner radii
        # stays as it is, to the bit
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(8, 1.0)))
        radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        add_rechester_rosenbluth_diffusion!(radial, [0, 0, 0, 0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]; major_radius=6.2)
        layout, system = radial_transport_system(grid, radial)
        f0 = rand(rng, size(grid)...)
        solution = integrate(system, state_of(layout, f0), collect(range(0, 1e-3; length=21)))
        final = distribution(layout, grid, solution.states[:, end])
        @test final[:, :, 1:3] == f0[:, :, 1:3]
        @test final[:, :, 4:8] != f0[:, :, 4:8]
        @test total_particles(grid, final) + solution.states[end, end] ≈ total_particles(grid, f0) rtol = 1e-12
    end

    @testset "prescribed loss rate" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(3, 1.0)))
        rate = [0.0, 50.0, 400.0]
        layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid)])
        term = LossRateTerm(:f, rate)
        @test is_linear(term) && dependencies(term) == [:f]
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)])
        f0 = rand(rng, size(grid)...)
        dt, steps = 1e-3, 10
        solution = integrate(system, vec(f0), collect(range(0, steps * dt; length=steps + 1)))
        final = reshape(solution.states[:, end], size(grid))
        for ir in 1:3
            # backward Euler of df/dt = -ν f
            @test final[:, :, ir] ≈ f0[:, :, ir] ./ (1 + rate[ir] * dt)^steps rtol = 1e-12
        end
        @test_throws DimensionMismatch EquationSystem(layout, [Equation(:f, TimeDerivative(), LossRateTerm(:f, [1.0]))])
        @test_throws ArgumentError EquationSystem(layout, [Equation(:f, TimeDerivative(),
                                                                    LossRateTerm(:f, [1.0, -1.0, 0.0]))])
        record = describe(term)
        @test record["type"] == "LossRateTerm" && record["rate"] == rate
    end

    @testset "transport history" begin
        history = PiecewiseConstant([1e-3, 1e-5], [1e-3])
        @test history(0.0) == 1e-3
        @test history(0.999e-3) == 1e-3
        @test history(1e-3) == 1e-5           # the interval that starts at the switch time
        @test history(1e-3 * (1 - 1e-14)) == 1e-5   # a time point within round-off of it
        @test history(5e-3) == 1e-5
        @test forced_time_points(history) == [1e-3]
        profiles = PiecewiseConstant([[1e-3, 1e-3, 2e-3], [0.0, 1e-5, 1e-5]], [3e-3])
        @test profiles(2e-3) == [1e-3, 1e-3, 2e-3] && profiles(3e-3) == [0.0, 1e-5, 1e-5]
        @test_throws ArgumentError PiecewiseConstant([1.0, 2.0], [1.0, 2.0])
        @test_throws ArgumentError PiecewiseConstant([1.0, 2.0, 3.0], [2.0, 1.0])

        minor_radius, major_radius = 2.0, 6.2
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(6, minor_radius)))
        function case(level_at)
            radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
            layout, system = radial_transport_system(grid, radial)
            update = RechesterRosenbluthUpdate(radial, level_at; major_radius)
            return (; radial, layout, system, update)
        end
        f0 = [1 + 0.2 * xi for _ in momentum.p_centers, xi in momentum.xi_centers, _ in 1:6]
        times = collect(range(0, 2e-3; length=21))

        run = case(history)
        @test Runes2P.update_name(run.update) == "radial_transport"
        @test forced_time_points(run.update) == [1e-3]
        @test all(isnan, Runes2P.update_state(run.update))
        # a checkpoint before the first interval holds no levels: the restore leaves the
        # operator without transport
        fresh = case(history)
        Runes2P.restore_update!(fresh.update, fill(NaN, 7))
        @test all(isnan, Runes2P.update_state(fresh.update)) && all(iszero, fresh.radial.diffusion_r_r)
        @test_throws DimensionMismatch Runes2P.restore_update!(fresh.update, fill(NaN, 6))
        u0 = state_of(run.layout, f0)
        @test Runes2P.apply_update!(run.update, u0, run.layout, 0.0) === true
        @test Runes2P.apply_update!(run.update, u0, run.layout, 0.5e-3) === false
        expected = AdvectionDiffusion(grid; scheme=:central)
        add_rechester_rosenbluth_diffusion!(expected, 1e-3; major_radius)
        @test run.radial.diffusion_r_r == expected.diffusion_r_r
        @test Runes2P.update_state(run.update) == fill(1e-3, 7)
        @test Runes2P.apply_update!(run.update, u0, run.layout, 1e-3) === true
        @test run.radial.diffusion_r_r ≈ 1e-4 .* expected.diffusion_r_r rtol = 1e-13

        # the operator of a prescribed term is rebuilt, which clears its radial coefficients:
        # the update refuses it
        let shared = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
            bulk = PlasmaState(; cold_electron_temperature=fill(100.0, 6), ions=[IonSpecies("D", 1, 1, fill(1e19, 6))],
                               electric_field=zeros(6), magnetic_field=fill(5.0, 6))
            layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid)])
            system = EquationSystem(layout, [Equation(:f, TimeDerivative(), PrescribedFokkerPlanckTerm(:f, shared, t -> bulk))])
            update = RechesterRosenbluthUpdate(shared, history; major_radius)
            @test_throws "rebuild clears it" integrate(system, vec(f0), times; updates=(update,))
        end

        run = case(history)
        whole = integrate(run.system, u0, times; updates=(run.update,))
        # a saved state carries the levels of the interval that ended at its time
        @test whole.update_states["radial_transport"][:, 1] == fill(1e-3, 7)
        @test whole.update_states["radial_transport"][:, 11] == fill(1e-3, 7)   # t = 1 ms
        @test whole.update_states["radial_transport"][:, 12] == fill(1e-5, 7)
        # the first ten steps are those of the high level, the step that ends at 1 ms among them
        high = case(t -> 1e-3)
        first_part = integrate(high.system, u0, times[1:11]; updates=(high.update,))
        @test whole.states[:, 1:11] == first_part.states
        low = case(t -> 1e-5)
        second_part = integrate(low.system, first_part.states[:, end], times[11:end]; updates=(low.update,))
        @test whole.states[:, 11:end] == second_part.states
        # the low level loses a hundredth of a hundredth as fast
        loss(solution, a, b) = solution.states[end, b] - solution.states[end, a]
        @test loss(whole, 11, 21) < 1e-3 * loss(whole, 1, 11)

        # a switch between two time points is refused
        run = case(PiecewiseConstant([1e-3, 1e-5], [1.05e-3]))
        @test_throws "not a time point" integrate(run.system, u0, times; updates=(run.update,))
        # a switch after the end of the run does not matter
        run = case(PiecewiseConstant([1e-3, 1e-5], [5e-3]))
        constant = case(t -> 1e-3)
        @test integrate(run.system, u0, times; updates=(run.update,)).states ==
              integrate(constant.system, u0, times; updates=(constant.update,)).states

        # a run in two segments with the switch in the second one is the run in one piece
        directory = mktempdir()
        checkpoint = CheckpointSettings("history"; directory, interval=0)
        run = case(history)
        integrate(run.system, u0, times[1:8]; updates=(run.update,), checkpoint)
        run = case(history)
        second = @test_logs (:warn, r"changed settings") integrate(run.system, u0, times; updates=(run.update,),
                                                                   checkpoint, resume=true)
        @test second.states == whole.states[:, 9:end]
        joined = read_output(checkpoint)
        @test joined.states == whole.states
        @test joined.update_states["radial_transport"] == whole.update_states["radial_transport"]
        # the stored levels set the coefficients of a new operator
        restored = case(history)
        Runes2P.restore_update!(restored.update, fill(1e-5, 7))
        @test restored.radial.diffusion_r_r ≈ 1e-4 .* expected.diffusion_r_r rtol = 1e-13
        @test_throws DimensionMismatch Runes2P.restore_update!(restored.update, fill(1e-5, 3))

        # records
        record = describe(run.update, DescriptionContext(; times))
        @test record["type"] == "RechesterRosenbluthUpdate"
        @test record["major_radius"] == major_radius
        @test record["safety_factor"] == ones(7)
        @test record["delta_b_over_b"] == Dict{String,Any}("type" => "PiecewiseConstant", "values" => [1e-3, 1e-5],
                                                          "switch_times" => [1e-3])
        @test !haskey(record["operator"], "coefficients")
        sampled = describe(high.update, DescriptionContext(; times))["delta_b_over_b"]
        @test sampled["values"] == fill(1e-3, 1, 21) && sampled["times"] == times
        @test describe(high.update, DescriptionContext())["delta_b_over_b"]["described"] === false
        for T in (RechesterRosenbluthUpdate, PiecewiseConstant, RadialEdgeLossTerm, LossRateTerm)
            fields = settings_fields(T)
            @test isempty(intersect(fields.described, fields.derived))
            @test Set(union(fields.described, fields.derived)) == Set(fieldnames(T))
        end
        loss_record = describe(RadialEdgeLossTerm(:f, run.radial))
        @test loss_record["type"] == "RadialEdgeLossTerm" && loss_record["kinetic"] == "f"
        @test whole.settings["updates"]["radial_transport"]["delta_b_over_b"]["switch_times"] == [1e-3]
    end

    @testset "four-point schemes in r" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(7, 1.0)))
        @testset "$r_scheme" for r_scheme in (:quick, :upwind_second_order, :tcdf, :muscl, :van_albada)
            for r_boundary in (:zero_flux, :absorbing)
                op = AdvectionDiffusion(grid; scheme=:central, r_scheme, r_boundary)
                @test op.r_scheme === r_scheme
                @test is_linear(op) == (r_scheme in (:quick, :upwind_second_order))
                op.advection_r .= randn(rng, size(op.advection_r)...)
                op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r)...)
                @test check_stencil_pattern(op)
                f = rand(rng, size(grid)...)
                df = apply!(zeros(size(grid)), op, f)
                scale = total_particles(grid, abs.(df))
                @test total_particles(grid, df) ≈ -radial_particle_flux(op, f, 8) atol = 1e-13 * scale
                r_boundary === :zero_flux && @test radial_particle_flux(op, f, 8) == 0
                # the limiter frozen at the state itself is the scheme
                frozen = apply!(zeros(size(grid)), op, f; limiter_state=f)
                @test frozen ≈ df rtol = 1e-13
                @test (@allocated apply!(df, op, f)) == 0
            end
        end
        # the second neighbours in r are in the pattern, and the matrix is the operator
        op = AdvectionDiffusion(grid; scheme=:central, r_scheme=:quick, r_boundary=:absorbing)
        op.advection_r .= randn(rng, size(op.advection_r)...)
        op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r)...)
        pattern = stencil_pattern(op)
        index = LinearIndices(size(grid))
        @test pattern[index[2, 3, 4], index[2, 3, 6]] && pattern[index[2, 3, 4], index[2, 3, 2]]
        @test !stencil_pattern(AdvectionDiffusion(grid; scheme=:central))[index[2, 3, 4], index[2, 3, 6]]
        f = rand(rng, size(grid)...)
        @test operator_matrix(op) * vec(f) ≈ vec(apply!(zeros(size(grid)), op, f)) rtol = 1e-12
        layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
        J = jacobian_prototype(system)
        jacobian!(J, system, vec(f), 0.0)
        @test J * vec(f) ≈ vec(apply!(zeros(size(grid)), op, f)) rtol = 1e-12

        # `scheme` alone leaves r central, and two radii take the central face value
        @test AdvectionDiffusion(grid; scheme=:tcdf).r_scheme === :central
        two = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(2, 1.0)))
        results = map((:quick, :central)) do r_scheme
            op = AdvectionDiffusion(two; scheme=:central, r_scheme)
            op.advection_r .= 1.5
            apply!(zeros(size(two)), op, reshape(collect(1.0:length(zeros(size(two)))), size(two)))
        end
        @test results[1] == results[2]
        # one pitch grid at all radii is required
        faces = [uniform_xi_faces(4), biuniform_theta_xi_faces(4, 2, 0.9), uniform_xi_faces(4)]
        per_radius = cylindrical_grid([MomentumGrid(momentum.p_faces, xi) for xi in faces],
                                      RadialGrid(uniform_r_faces(3, 0.6)))
        @test_throws "one pitch grid at all radii" AdvectionDiffusion(per_radius; scheme=:central, r_scheme=:quick)
        @test_throws ArgumentError AdvectionDiffusion(grid; r_scheme=:cubic)

        # order of the radial advection: -(1/r) d(r A f)/dr of a smooth profile with a
        # constant velocity outwards, in the middle of the radius
        shape(r) = exp(-(r / 0.4)^2)
        exact(r, A) = -A * (shape(r) / r - 2r / 0.4^2 * shape(r))
        function advection_error(radii, r_scheme)
            g = cylindrical_grid(MomentumGrid(uniform_p_faces(1, 0.5, 1.5), uniform_xi_faces(2)),
                                 RadialGrid(uniform_r_faces(radii, 1.0)))
            op = AdvectionDiffusion(g; scheme=:central, r_scheme, r_boundary=:absorbing)
            op.advection_r .= 2.0
            centers = g.radial_grid.r_centers
            f = [shape(r) for _ in 1:1, _ in 1:2, r in centers]
            df = apply!(zeros(size(g)), op, f)
            return maximum(abs(df[1, 1, ir] - exact(centers[ir], 2.0)) for ir in 1:radii if 0.2 < centers[ir] < 0.8)
        end
        order(r_scheme) = log2(advection_error(40, r_scheme) / advection_error(80, r_scheme))
        @test 0.8 < order(:upwind) < 1.2
        for r_scheme in (:central, :quick, :upwind_second_order)
            @test 1.8 < order(r_scheme) < 2.3
        end
        for r_scheme in (:tcdf, :van_albada, :muscl)
            @test order(r_scheme) > 1.5
            @test advection_error(80, r_scheme) < advection_error(80, :upwind) / 5
        end
    end

    @testset "arguments of the edge loss term" begin
        grid = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(3, 1.0)))
        radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
        other = cylindrical_grid(MomentumGrid(uniform_p_faces(6, 0.0, 2.0), uniform_xi_faces(4)), grid.radial_grid)
        layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid), KineticUnknown(:g, other),
                                                  ScalarUnknown(:lost_particles), RadialUnknown(:density)])
        equations(row, kinetic) = [
            Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, radial)),
            Equation(:g, TimeDerivative(), LinearTerm(:g; coefficient=-1)),
            Equation(:lost_particles, row === :lost_particles ? TimeDerivative() : Algebraic(),
                     (row === :lost_particles ? (RadialEdgeLossTerm(kinetic, radial),) :
                      (LinearTerm(:lost_particles; coefficient=-1),))...),
            Equation(:density, Algebraic(), LinearTerm(:density; coefficient=-1),
                     (row === :density ? (RadialEdgeLossTerm(kinetic, radial),) : ())...)]
        @test EquationSystem(layout, equations(:lost_particles, :f)) isa EquationSystem
        @test_throws DimensionMismatch EquationSystem(layout, equations(:lost_particles, :g))
        @test_throws ArgumentError EquationSystem(layout, equations(:density, :f))
        @test_throws ArgumentError EquationSystem(layout, equations(:lost_particles, :density))
        # the stored Jacobian holds the term
        system = EquationSystem(layout, equations(:lost_particles, :f))
        u = rand(rng, length(layout))
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        radial.diffusion_r_r .= 0.4
        rhs!(du, u, system, 0.0)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        @test J * u ≈ du rtol = 1e-12
        @test du[first(unknown_range(layout, unknown_id(layout, :lost_particles)))] ≈
              radial_particle_flux(radial, kinetic_array(u, layout, unknown_id(layout, :f)), 4) rtol = 1e-13

        # the row of the lost particles holds every cell of the last radius. The term
        # assembles it itself, so the coloring of the AD Jacobian does not grow with the cells
        @test jacobian_assembly(RadialEdgeLossTerm(:f, radial)) === :direct
        @test jacobian_assembly(FokkerPlanckTerm(:f, radial)) === :automatic
        alone = EquationSystem(UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid)]),
                               [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, radial))])
        @test jacobian_colors(system) <= jacobian_colors(alone) + 2 < cell_count(grid, 1)
        @test length(system.direct_terms) == 1
        @test isempty(alone.direct_terms)
        # with DREAM's distance and with advection through the edge, on one pitch grid and on
        # pitch grids that differ between the radii
        per_radius = cylindrical_grid([MomentumGrid(momentum.p_faces, faces)
                                       for faces in (uniform_xi_faces(4), biuniform_theta_xi_faces(4, 2, 0.9),
                                                     [-1.0, -0.3, 0.2, 0.7, 1.0])], grid.radial_grid)
        for edge_grid in (grid, per_radius), r_boundary in (:absorbing, :absorbing_dream, :zero_flux)
            op = AdvectionDiffusion(edge_grid; scheme=:upwind, r_boundary)
            op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r)...)
            op.advection_r .= randn(rng, size(op.advection_r)...)
            small = UnknownLayout(edge_grid.radial_grid, [KineticUnknown(:f, edge_grid), ScalarUnknown(:lost_particles)])
            budget = EquationSystem(small, [
                Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op)),
                Equation(:lost_particles, TimeDerivative(), RadialEdgeLossTerm(:f, op; coefficient=2.5))])
            v = rand(rng, length(small))
            dv = zeros(length(small))
            rhs!(dv, v, budget, 0.0)
            K = jacobian_prototype(budget)
            jacobian!(K, budget, rand(rng, length(small)), 0.0)
            @test K * v ≈ dv rtol = 1e-12
            @test dv[end] ≈ 2.5 * radial_particle_flux(op, reshape(v[1:end-1], size(edge_grid)), 4) rtol = 1e-13
            # what leaves the radii arrives in the lost particles
            change = total_particles(edge_grid, reshape(dv[1:end-1], size(edge_grid)))
            @test change + dv[end] / 2.5 ≈ 0 atol = 1e-12 * total_particles(edge_grid, abs.(reshape(dv[1:end-1], size(edge_grid))))
            @test (@allocated jacobian!(K, budget, v, 0.0)) == 0
        end
    end
end
