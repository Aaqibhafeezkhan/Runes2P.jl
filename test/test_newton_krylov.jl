# Newton-Krylov: GMRES preconditioned by the per-radius block LU or ILU of
# the stored matrix, with the implicit knock-on term matrix-free. Checked against a dense
# direct Newton iteration that assembles the knock-on block, including its coupling to a
# target density that is itself an unknown. Uses the test terms of test_time_stepping.jl
# and the example setups of example_cases.jl.

using LinearAlgebra: Diagonal, norm
using SparseArrays: SparseMatrixCSC

# Newton iterations with the dense Jacobian: the stored matrix plus every matrix-free
# column, assembled by products with unit vectors. Returns the iterates.
# an assertion on allocations holds for optimised code alone: it is skipped, and reported
# as broken, in a development run at -O0 or -O1
const OPTIMIZED = Base.JLOptions().opt_level >= 2

function dense_newton_iterates(system, u_old, t, dt, iterations)
    n = length(system.layout)
    u = copy(u_old)
    F = zeros(n)
    J = jacobian_prototype(system)
    iterates = Vector{Vector{Float64}}()
    for _ in 1:iterations
        residual!(F, u, u_old, system, t, dt)
        jacobian!(J, system, u, t)
        dense = Matrix(jacobian_operator(J, system, u, t))
        u = u - (Diagonal(system.mass) / dt - dense) \ F
        push!(iterates, copy(u))
    end
    return iterates
end

relative_difference(a, b) = norm(a - b, Inf) / norm(b, Inf)

# hot-tail distribution with the implicit knock-on term and, when `coupled`, the knock-on
# target density as an algebraic unknown: n_tot = n_background + n(f)
function knockon_system(case; treatment=:implicit, coupled=false, distribution_scale=1.0)
    g = case.grid
    nr = radial_cell_count(g)
    knockon = KnockOnOperator(g, g, case.cutoff)
    unknowns = Any[KineticUnknown(:hot_tail_distribution, g; scale=distribution_scale)]
    target = fill(case.total_electron_density, nr)
    equations = Any[]
    if coupled
        push!(unknowns, RadialUnknown(:total_electron_density; scale=case.total_electron_density))
        push!(equations, Equation(:total_electron_density, Algebraic(),
                                  DensityMomentTerm(:hot_tail_distribution),
                                  LinearTerm(:total_electron_density; coefficient=-1.0),
                                  ConstantSourceTerm(case.total_electron_density)))
        target = :total_electron_density
    end
    push!(equations, Equation(:hot_tail_distribution, TimeDerivative(),
                              FokkerPlanckTerm(:hot_tail_distribution, case.fokker_planck),
                              KnockOnTerm(:hot_tail_distribution, knockon; target_density=target, treatment)))
    layout = UnknownLayout(g.radial_grid, [unknowns...])
    return EquationSystem(layout, [equations...])
end

@testset "Newton-Krylov" begin
    # the grid of avalanche/generateCylindrical.py with a tenth of its momentum cells and a
    # fifth of its pitch cells, small enough for a dense reference
    small = avalanche_example(momentum_cells=20, pitch_cells=6, radial_cells=2)

    @testset "implicit knock-on coupled to its target density, against dense Newton" begin
        # A hot population of 1e20 m⁻³ at 100 keV doubles the target density, so the
        # knock-on source is bilinear in (f, n_tot) and the step is nonlinear. At the
        # example's step of 2 ms the dense iteration converges quadratically; from about
        # 10 ms on it diverges, since the knock-on growth of so hot a population times dt
        # exceeds one and M/dt - J loses its dominance.
        system = knockon_system(small; coupled=true, distribution_scale=1e20)
        layout = system.layout
        u_old = small.initial_state(layout, unknown_id(layout, :hot_tail_distribution); density=1e20)
        for ir in 1:2
            unknown_view(u_old, layout, :total_electron_density, ir)[1] = 2 * small.total_electron_density
        end
        t = dt = small.time_step
        @test_throws ArgumentError NewtonWorkspace(system)   # SparseLU needs every term stored
        reference = dense_newton_iterates(system, u_old, t, dt, 6)
        # quadratic convergence of the dense iteration (observed relative steps 3.7e-3,
        # 1.6e-6, 2.5e-13)
        steps = [relative_difference(reference[k+1], reference[k]) for k in 1:3]
        @test steps[2] < 1e-3 * steps[1] && steps[3] < 1e-5 * steps[2]

        # GMRES solves to a relative residual of 1e-10, so the first iterate differs at
        # that level (observed 6.4e-10, then 1.9e-12 and 1.4e-15)
        for (iterations, tolerance) in ((1, 1e-8), (2, 1e-10), (3, 1e-12))
            u = copy(u_old)
            ws = NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES(), max_iterations=iterations))
            newton_solve!(u, u_old, ws, t, dt)
            @test relative_difference(u, reference[iterations]) < tolerance
        end
        u = copy(u_old)
        result = newton_solve!(u, u_old, NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES())), t, dt)
        @test result.status === :converged
        @test relative_difference(u, reference[end]) < 1e-12
        # quadratic only with the matrix-free (f, n_tot) block in the Newton operator
        # (observed 2.03)
        @test result.convergence_order > 1.8
    end

    @testset "explicit knock-on: the block LU is the whole Newton matrix" begin
        system = knockon_system(small; treatment=:explicit)
        layout = system.layout
        u0 = small.initial_state(layout, 1)
        times = collect(range(0, 10 * small.time_step; length=11))
        direct = integrate(system, u0, times)
        lu_run = integrate(system, u0, times; solver=NewtonSolver(linear_solver=GMRES()))
        @test lu_run.statistics.krylov_iterations == lu_run.statistics.newton_iterations
        # the block factorization is timed once per step with the default refresh
        st = lu_run.statistics
        @test all(>(0), st.factorization_time) && all(st.factorization_time .<= st.linear_solve_time)
        @test all(st.jacobian_time .+ st.residual_time .+ st.linear_solve_time .<= st.wall_time)
        @test relative_difference(lu_run.states[:, end], direct.states[:, end]) < 1e-12
        ilu_run = integrate(system, u0, times;
                            solver=NewtonSolver(linear_solver=GMRES(preconditioner=RadiusBlockILU(drop_tolerance=1e-2))))
        @test all(ilu_run.statistics.accepted)
        @test sum(ilu_run.statistics.krylov_iterations) > sum(ilu_run.statistics.newton_iterations)
        @test relative_difference(ilu_run.states[:, end], direct.states[:, end]) < 1e-8
    end

    @testset "preconditioner refresh" begin
        system = knockon_system(small)
        layout = system.layout
        u0 = small.initial_state(layout, 1)
        times = collect(range(0, 10 * small.time_step; length=11))
        run(refresh) = integrate(system, u0, times;
                                 solver=NewtonSolver(linear_solver=GMRES(preconditioner_refresh=refresh)))
        every_iteration = run(:every_iteration)
        every_step = run(:every_step)
        on_demand = run(:on_demand)
        # the system is linear in the state: the matrix of the second iteration of a step is
        # that of the first, and its factors are kept
        @test all(==(1), every_iteration.statistics.preconditioner_updates)
        every_matrix = integrate(system, u0, times;
                                 solver=NewtonSolver(linear_solver=GMRES(), reuse_linear_matrix=false))
        stats = every_matrix.statistics
        @test stats.preconditioner_updates == stats.newton_iterations
        @test every_matrix.states == every_iteration.states
        @test all(==(1), every_step.statistics.preconditioner_updates)
        # prescribed coefficients and a constant step: one factorization for the whole run
        @test sum(on_demand.statistics.preconditioner_updates) == 1
        for other in (every_step, on_demand)
            @test relative_difference(other.states[:, end], every_iteration.states[:, end]) < 1e-9
        end
        @test_throws ArgumentError NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:never)))
        @test_throws ArgumentError NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES(refresh_step_change=-1.0)))
        @test_throws ArgumentError NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES(refresh_count_growth=0.5)))

        # the rules of :on_demand on the cache: the step
        # against that of the factors, the count against that of the first solve with them
        solver = NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:on_demand, refresh_step_change=0.5,
                                                  refresh_count_growth=2.0))
        ws = NewtonWorkspace(system, solver)
        cache = ws.cache
        @test Runes2P._needs_factorization(cache, small.time_step, 1)          # nothing factorized yet
        u = copy(u0)
        newton_solve!(u, u0, ws, small.time_step, small.time_step)
        @test cache.factorized && cache.factorized_dt == small.time_step
        @test cache.fresh_iterations == cache.last_iterations >= 1
        @test !Runes2P._needs_factorization(cache, small.time_step, 2)
        @test !Runes2P._needs_factorization(cache, 1.4 * small.time_step, 1)   # within the tolerance
        @test !Runes2P._needs_factorization(cache, 0.6 * small.time_step, 1)
        @test Runes2P._needs_factorization(cache, 1.6 * small.time_step, 1)    # outside
        @test Runes2P._needs_factorization(cache, 0.4 * small.time_step, 1)
        @test !Runes2P._needs_factorization(cache, 0.5 * small.time_step, 1)   # a halving is the boundary and keeps the factors
        @test Runes2P._needs_factorization(cache, 2.0 * small.time_step, 1)    # a doubling refreshes
        cache.last_iterations = 2 * cache.fresh_iterations
        @test !Runes2P._needs_factorization(cache, small.time_step, 1)         # at the growth, not above
        cache.last_iterations = 2 * cache.fresh_iterations + 1
        @test Runes2P._needs_factorization(cache, small.time_step, 1)
        # a first solve without an iteration (a zero right-hand side) is no reference
        fresh_before = cache.fresh_iterations
        cache.fresh_iterations, cache.last_iterations = 0, 1
        @test !Runes2P._needs_factorization(cache, small.time_step, 1)
        cache.last_iterations = 3
        @test Runes2P._needs_factorization(cache, small.time_step, 1)
        cache.fresh_iterations = fresh_before

        # under the step controller the factorizations are those the step rule asks for
        # (the count rule off), one per solve whose step differs from that of the factors
        # by more than the tolerance; the states are those of the run with fresh factors
        stepper = StepController(initial_step=small.time_step, relative_tolerance=1e-3)
        controlled(linear_solver) = integrate(system, u0, [0.0, 20 * small.time_step];
                                              solver=NewtonSolver(; linear_solver), stepper)
        reference = controlled(GMRES())
        on_demand = controlled(GMRES(preconditioner_refresh=:on_demand, refresh_step_change=0.5,
                                     refresh_count_growth=Inf))
        st = on_demand.statistics
        # stale factors change the solves within the Krylov tolerance, and the error
        # estimates with them, so the steps agree to about 1e-10 and not to the bit
        @test length(st.time_step) == length(reference.statistics.time_step)
        @test st.time_step ≈ reference.statistics.time_step rtol = 1e-6
        @test count(st.accepted) >= 6
        @test length(unique(st.time_step)) >= 4                # the step changed during the run
        expected = zeros(Int, length(st.time_step))
        factorized_dt = NaN
        for (k, dt) in enumerate(st.time_step)
            if isnan(factorized_dt) || abs(dt / factorized_dt - 1) > 0.5
                expected[k] = 1
                factorized_dt = dt
            end
        end
        @test st.preconditioner_updates == expected
        @test 1 < sum(expected) < length(expected)
        @test relative_difference(on_demand.states[:, end], reference.states[:, end]) < 1e-8
        # neither rule: one factorization serves the whole run, plus at most one refresh
        # after a solve that failed with factors of a step 100 times longer
        once = controlled(GMRES(preconditioner_refresh=:on_demand, refresh_step_change=Inf, refresh_count_growth=Inf))
        @test 1 <= sum(once.statistics.preconditioner_updates) <= 2
        @test once.statistics.time_step ≈ reference.statistics.time_step rtol = 1e-6
        @test relative_difference(once.states[:, end], reference.states[:, end]) < 1e-8
    end

    @testset "failures are reported, not thrown" begin
        system = knockon_system(small)
        layout = system.layout
        u0 = small.initial_state(layout, 1)
        u = copy(u0)
        one_iteration = NewtonSolver(linear_solver=GMRES(max_iterations=1))
        @test newton_solve!(u, u0, NewtonWorkspace(system, one_iteration), 0.1, 0.1).status === :linear_solve_failed
        singular = single_unknown_system(Algebraic(), LinearTerm(:x; coefficient=0.0))
        u = [1.0]
        @test newton_solve!(u, [1.0], NewtonWorkspace(singular, NewtonSolver(linear_solver=GMRES())), 0.0, 1.0).status ===
              :singular_matrix
    end

    @testset "the residual-floor waiver needs an accurate linear solve" begin
        # the stiff exchange step of test_time_stepping.jl, whose residual has a round-off
        # floor above the tolerance
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        layout = UnknownLayout(rad, [RadialUnknown(:x; components=2, scale=1.0)])
        system = EquationSystem(layout, [Equation(:x, TimeDerivative(), ExchangeTerm(:x, 1e10))])
        u_old = [1.0, 0.0]
        u = copy(u_old)
        accurate = newton_solve!(u, u_old, NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES())), 1.0, 1.0)
        @test accurate.status === :converged
        @test accurate.residual_floor
        # a loose Krylov tolerance converges linearly at best, and a small update then does
        # not bound the error, so the waiver must not apply
        loose = NewtonSolver(linear_solver=GMRES(relative_tolerance=1e-4), max_iterations=8)
        u = copy(u_old)
        result = newton_solve!(u, u_old, NewtonWorkspace(system, loose), 1.0, 1.0)
        @test result.status === :max_iterations
    end

    @testset "allocations per Newton step do not grow with the grid" begin
        # Factorizing allocates in UMFPACK, as much as for the direct solver; the rest of a
        # Newton step (residual, Jacobian, GMRES with the knock-on products, preconditioner
        # solves) allocates a few hundred bytes at any grid size (observed 208 to 240)
        for momentum_cells in (10, 40, 160)
            case = avalanche_example(; momentum_cells, pitch_cells=6)
            system = knockon_system(case)
            u0 = case.initial_state(system.layout, 1)
            ws = NewtonWorkspace(system, NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:on_demand)))
            u = copy(u0)
            newton_solve!(u, u0, ws, small.time_step, small.time_step)   # compiles and factorizes
            copyto!(u, u0)
            allocated = @allocated newton_solve!(u, u0, ws, small.time_step, small.time_step)
            @test allocated < 1024 skip = !OPTIMIZED
        end
    end
end
