# The stored Jacobian and the factorization of a system that is linear in the state are
# kept between the Newton iterations of a step: the first iteration solves
# the step, the second confirms it with the same matrix. The results are those of a
# solver that assembles and factorizes in every iteration, bit for bit. The key of the
# reuse is the end time and the length of the step, which identify the start state.

function reuse_case(scheme; radii = 1)
    mom = MomentumGrid(biuniform_p_faces(24, 10, 0.0, 0.2, 1.5), uniform_xi_faces(6))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(radii, 1.0)))
    temperature(t) = 2000.0 * (100.0 / 2000.0)^(t / 1e-3)
    plasma_at(t) = PlasmaState(cold_electron_temperature = fill(temperature(t), radii),
        cold_electron_density = fill(1e20, radii),
        ions = [IonSpecies("D", 1, 1, fill(1e20, radii))],
        electric_field = fill(0.5, radii), magnetic_field = fill(5.0, radii))
    f0 = [exp(-p^2 / (2 * 2000.0 / Constants.ELECTRON_REST_ENERGY))
          for p in mom.p_centers,
    _ in 1:pitch_cell_count(mom), _ in 1:radii]
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = maximum(f0))])
    operator = AdvectionDiffusion(g; scheme, p_upper_boundary = :f_zero)
    system = EquationSystem(layout,
        [Equation(:f, TimeDerivative(),
            prescribed_fokker_planck_term(:f, operator, plasma_at))])
    return system, vec(f0)
end

@testset "linear systems keep the Jacobian within a step" begin
    dt = 1e-5
    @testset "direct solver" begin
        system, u0 = reuse_case(:exponential_fitting)
        @test system.linear
        kept = NewtonWorkspace(system, NewtonSolver())
        fresh = NewtonWorkspace(system, NewtonSolver(reuse_linear_matrix = false))
        @test kept.linear && fresh.linear
        u_kept, u_fresh = copy(u0), copy(u0)
        result_kept = newton_solve!(u_kept, u0, kept, dt, dt)
        result_fresh = newton_solve!(u_fresh, u0, fresh, dt, dt)
        @test result_kept.status === :converged && result_fresh.status === :converged
        @test result_kept.iterations == result_fresh.iterations == 2
        @test kept.jacobian_assemblies[] == 1
        @test fresh.jacobian_assemblies[] == 2
        @test kept.cache.factorizations == 1
        @test fresh.cache.factorizations == 2
        @test u_kept == u_fresh

        # the next time level has other coefficients: one more assembly
        v_kept, v_fresh = copy(u_kept), copy(u_fresh)
        newton_solve!(v_kept, u_kept, kept, 2dt, dt)
        newton_solve!(v_fresh, u_fresh, fresh, 2dt, dt)
        @test kept.jacobian_assemblies[] == 2
        @test kept.cache.factorizations == 2
        @test v_kept == v_fresh
        @test v_kept != u_kept

        # the same end time with another time step is a step from another start state
        # (step doubling): the Jacobian, the matrix and its factorization are new
        w_kept, w_fresh = copy(u_kept), copy(u_fresh)
        newton_solve!(w_kept, u_kept, kept, 2dt, dt / 2)
        newton_solve!(w_fresh, u_fresh, fresh, 2dt, dt / 2)
        @test kept.jacobian_assemblies[] == 3
        @test kept.cache.factorizations == 3
        @test w_kept == w_fresh
        @test w_kept != v_kept
        @test (kept.assembled_time[], kept.assembled_step[]) == (2dt, dt / 2)

        # the same step again from the same start: kept
        y = copy(u_kept)
        newton_solve!(y, u_kept, kept, 2dt, dt / 2)
        @test kept.jacobian_assemblies[] == 3
        @test kept.cache.factorizations == 3
        @test y == w_kept

        # discarded by hand
        invalidate_jacobian!(kept)
        @test isnan(kept.assembled_time[]) && isnan(kept.assembled_step[])
        x = copy(u_kept)
        newton_solve!(x, u_kept, kept, 2dt, dt / 2)
        @test kept.jacobian_assemblies[] == 4
        @test kept.cache.factorizations == 4
        @test x == w_kept
    end

    @testset "a nonlinear scheme assembles in every iteration" begin
        system, u0 = reuse_case(:van_albada)
        @test !system.linear
        workspace = NewtonWorkspace(system, NewtonSolver())
        u = copy(u0)
        result = newton_solve!(u, u0, workspace, dt, dt)
        @test result.status === :converged
        @test result.iterations > 2
        @test workspace.jacobian_assemblies[] == result.iterations
        @test workspace.cache.factorizations == result.iterations
        @test isnan(workspace.assembled_time[])
    end

    @testset "a run with coefficients that change in time" begin
        system, u0 = reuse_case(:exponential_fitting; radii = 2)
        times = collect(0:dt:20dt)
        for linear_solver in (SparseLU(), GMRES(), GMRES(preconditioner_refresh = :every_step))
            kept = integrate(system, u0, times; solver = NewtonSolver(; linear_solver))
            fresh = integrate(system, u0, times;
                solver = NewtonSolver(; linear_solver, reuse_linear_matrix = false))
            @test all(kept.statistics.accepted)
            @test kept.statistics.newton_iterations == fresh.statistics.newton_iterations
            @test kept.statistics.krylov_iterations == fresh.statistics.krylov_iterations
            @test kept.states == fresh.states
            if linear_solver isa GMRES
                @test all(==(1), kept.statistics.preconditioner_updates)
                expected = linear_solver.preconditioner_refresh === :every_iteration ?
                           fresh.statistics.newton_iterations : ones(Int, length(times) - 1)
                @test fresh.statistics.preconditioner_updates == expected
            end
        end
    end
end
