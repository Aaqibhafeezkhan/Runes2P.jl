# Backward Euler with Newton and a sparse direct solver: convergence on update
# and residual, lower limits that shorten only steps towards the limit, the optional line
# search, failures handed to the time stepper, time points from any array, saved states
# independent of the steps, solver statistics and HDF5 output. Every test problem has a
# known backward-Euler solution.
using HDF5

# backward-Euler step of du/dt = -k √u: √u1 solves s² + k dt s - u0 = 0
square_root_step(u0, k, dt) = ((-k * dt + sqrt((k * dt)^2 + 4u0)) / 2)^2

# A sink that is constant above `top` and falls linearly to zero at `bottom`, refused at or
# below `floor` with a DomainError, as the plasma of a temperature below its floor is.
# The full Newton step from above `top` follows the constant sink and can
# land below the floor, while the solution of the step lies between `bottom` and `top`.
struct FloorSinkTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
    bottom::Float64
    top::Float64
    floor::Float64
end
Runes2P.dependencies(term::FloorSinkTerm) = [term.unknown]
function Runes2P.rhs!(out, term::FloorSinkTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))

        u[b] > term.floor || throw(DomainError(u[b], "at or below the floor of the sink"))
        out[a] -= term.rate * clamp((u[b] - term.bottom) / (term.top - term.bottom), 0, 1)
    end
    return out
end
function Runes2P.sparsity!(entries, term::FloorSinkTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end

@testset "Newton and backward-Euler time stepping" begin
    @testset "nonlinear step: quadratic convergence, lower limit" begin
        k, dt, u0 = 10.0, 1.0, [1.0]
        # the full Newton step from u0 lands at u0 - 5/3 < 0, where √ is undefined
        free = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, k))
        u = copy(u0)
        result = newton_solve!(u, u0, NewtonWorkspace(free), dt, dt)
        @test result.status === :domain_error
        @test result.iterations == 0

        limited = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, k); nonnegative = true)
        ws = NewtonWorkspace(limited)
        u = copy(u0)
        result = newton_solve!(u, u0, ws, dt, dt)
        @test result.status === :converged
        @test u[1] ≈ square_root_step(1.0, k, dt) rtol = 1e-12
        # DREAM's damping: the first step removes 90% of the distance to zero
        @test ws.step_lengths[1] ≈ 0.9 / (5 / 3)
        @test result.smallest_step_length == ws.step_lengths[1]
        @test all(==(1.0), ws.step_lengths[3:end])
        @test result.convergence_order > 1.8
        @test result.residual_norm <= NewtonSolver().residual_tolerance
        @test length(ws.update_norms) == length(ws.residual_norms) == result.iterations
    end

    @testset "lower limits shorten only steps towards the limit, never growth" begin
        dt, u0 = 1.0, [2.0]
        # growth by a factor 100 in one step: never shortened
        growth = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = 0.99); nonnegative = true)
        u = copy(u0)
        ws = NewtonWorkspace(growth)
        result = newton_solve!(u, u0, ws, dt, dt)
        @test result.status === :converged
        @test u[1] ≈ 200.0 rtol = 1e-12
        @test result.smallest_step_length == 1.0
        # a linear system: the solve and the check of its update, accepted without the
        # residual test
        @test result.iterations == 2 && result.linear_step

        # decay by a factor 1000: the limit lets each iteration remove at most 90% of the
        # distance to zero, so a few more iterations, never below zero
        decay = single_unknown_system(
            TimeDerivative(), LinearTerm(:x; coefficient = -999.0); nonnegative = true)
        u = copy(u0)
        ws = NewtonWorkspace(decay)
        result = newton_solve!(u, u0, ws, dt, dt)
        @test result.status === :converged
        @test u[1] ≈ 2e-3 rtol = 1e-10
        @test result.smallest_step_length < 1
        @test result.iterations > 2
        # a larger admissible reduction per iteration converges faster
        u = copy(u0)
        faster = newton_solve!(
            u, u0, NewtonWorkspace(decay, NewtonSolver(max_reduction = 0.999)), dt, dt)
        @test faster.status === :converged
        @test faster.iterations < result.iterations
        # without the limit the linear step takes one iteration plus the check
        unlimited = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -999.0))
        u = copy(u0)
        @test newton_solve!(u, u0, NewtonWorkspace(unlimited), dt, dt).iterations == 2
    end

    @testset "lower limit relative to a reference unknown" begin
        # du/dt = -x - 3 with the limit x ≥ -0.1 n: the backward-Euler solution -1 (dt = 1)
        # lies below the limit; the iterates approach the limit and stop there, and the
        # step fails
        reference = RadialUnknown(:n; scale = 1.0)
        n_equation = Equation(:n, TimeDerivative(), LinearTerm(:n; coefficient = 0.0))
        limit = LowerLimit(tolerance = 0.1, reference = :n)
        system = single_unknown_system(
            TimeDerivative(), LinearTerm(:x; coefficient = -1.0),
            ConstantSourceTerm(-3.0); nonnegative = limit,
            extra = ((reference, n_equation),))
        layout = system.layout
        u_old = zeros(length(layout))
        unknown_view(u_old, layout, :x, 1)[1] = 1.0
        unknown_view(u_old, layout, :n, 1)[1] = 1.0
        u = copy(u_old)
        result = newton_solve!(u, u_old, NewtonWorkspace(system), 1.0, 1.0)
        @test result.status === :below_lower_limit
        @test -0.1 <= unknown_view(u, layout, :x, 1)[1] < -0.0999
        @test describe(layout)["unknowns"]["x"]["lower_limit_reference"] == "n"
        # checks on declaration
        @test_throws ArgumentError LowerLimit(tolerance = -1.0)
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        @test_throws ArgumentError UnknownLayout(
            rad, [RadialUnknown(:x; nonnegative = LowerLimit(reference = :y))])
        @test_throws ArgumentError UnknownLayout(rad,
            [RadialUnknown(:x; nonnegative = LowerLimit(reference = :y)),
                RadialUnknown(:y; components = 2)])
        @test_throws ArgumentError UnknownLayout(rad,
            [ScalarUnknown(:x; nonnegative = LowerLimit(reference = :y)),
                RadialUnknown(:y)])
    end

    @testset "absolute lower limit: the floor of a temperature" begin
        # dx/dt = -4.7 s(x) from x = 4 with dt = 1, s the ramp from 0 at 0.8 to 1 at 1: the
        # backward-Euler solution is 22.8 / 24.5, above the floor 0.5 of the sink
        sink = FloorSinkTerm(:x, 4.7, 0.8, 1.0, 0.5)
        solution = 22.8 / 24.5
        u0 = [4.0]
        # the sink is not defined at the zero state, so the system takes the initial state
        floor_system(limit) = EquationSystem(
            UnknownLayout(RadialGrid(uniform_r_faces(1, 0.5)), [RadialUnknown(:x; nonnegative = limit)]),
            [Equation(:x, TimeDerivative(), sink)]; initial_state = u0)
        # the limit at zero lets the first step fall to 10 % of 4, below the floor: the
        # solve fails although the solution of the step is allowed
        at_zero = floor_system(true)
        u = copy(u0)
        @test newton_solve!(u, u0, NewtonWorkspace(at_zero), 1.0, 1.0).status ===
              :domain_error
        # the limit at the floor damps the first step to 90 % of the distance to the floor,
        # and the iteration converges
        at_floor = floor_system(LowerLimit(value = 0.5))
        ws = NewtonWorkspace(at_floor)
        u = copy(u0)
        result = newton_solve!(u, u0, ws, 1.0, 1.0)
        @test result.status === :converged
        @test u[1] ≈ solution rtol = 1e-12
        @test ws.step_lengths[1] ≈ 0.9 * 3.5 / 4.7
        @test all(==(1.0), ws.step_lengths[2:end])
        # dx/dt = -x - 3.6 from x = 4: the solution 0.2 of the step lies below the floor,
        # the iterates approach the floor from above and the step is refused
        below = single_unknown_system(
            TimeDerivative(), LinearTerm(:x; coefficient = -1.0), ConstantSourceTerm(-3.6);
            nonnegative = LowerLimit(value = 0.5))
        u = copy(u0)
        @test newton_solve!(u, u0, NewtonWorkspace(below), 1.0, 1.0).status ===
              :below_lower_limit
        @test 0.5 <= u[1] < 0.5001
        @test_throws TimeStepFailure integrate(below, u0, [0.0, 1.0]; stepper = FixedSteps(;
            max_halvings = 0))
        # the value with a tolerance: the limit is value - tolerance * scale
        loose = single_unknown_system(
            TimeDerivative(), LinearTerm(:x; coefficient = -1.0), ConstantSourceTerm(-3.6);
            scale = 2.0, nonnegative = LowerLimit(value = 0.5, tolerance = 0.1))
        u = copy(u0)
        @test newton_solve!(u, u0, NewtonWorkspace(loose), 1.0, 1.0).status ===
              :below_lower_limit
        @test 0.3 <= u[1] < 0.3001
        # the record holds the value only where one is set
        @test describe(at_floor.layout)["unknowns"]["x"]["lower_limit_value"] == 0.5
        @test !haskey(describe(at_zero.layout)["unknowns"]["x"], "lower_limit_value")
        @test_throws ArgumentError LowerLimit(value = Inf)
    end

    @testset "line search on the scaled residual" begin
        # 0 = atan(x - 1): plain Newton diverges from |x - 1| > 1.39
        system = single_unknown_system(Algebraic(), ArctangentTerm(:x, 1.0))
        u_old = [4.0]
        u = copy(u_old)
        plain = newton_solve!(u, u_old, NewtonWorkspace(system), 0.0, 1.0)
        @test plain.status !== :converged
        u = copy(u_old)
        ws = NewtonWorkspace(system, NewtonSolver(line_search = true))
        result = newton_solve!(u, u_old, ws, 0.0, 1.0)
        @test result.status === :converged
        @test u[1] ≈ 1.0 atol = 1e-10
        @test result.smallest_step_length < 1
        @test ws.step_lengths[end] == 1.0
        @test result.convergence_order > 2  # atan'' = 0 at the root: cubic
    end

    @testset "failures are reported, not thrown" begin
        singular = single_unknown_system(Algebraic(), LinearTerm(:x; coefficient = 0.0))
        u = [1.0]
        @test newton_solve!(u, [1.0], NewtonWorkspace(singular), 0.0, 1.0).status ===
              :singular_matrix
        nan_source = single_unknown_system(
            TimeDerivative(), LinearTerm(:x; coefficient = -1.0),
            ConstantSourceTerm(NaN))
        u = [1.0]
        @test newton_solve!(u, [1.0], NewtonWorkspace(nan_source), 1.0, 1.0).status ===
              :non_finite_residual
        # a finite residual with a non-finite Jacobian: √x at x = 0
        u = [0.0]
        steep = NewtonWorkspace(single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, 10.0)))
        @test newton_solve!(u, [0.0], steep, 1.0, 1.0).status === :non_finite_jacobian
        u = [1.0]
        few = NewtonWorkspace(
            single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, 10.0);
                nonnegative = true),
            NewtonSolver(max_iterations = 2))
        @test newton_solve!(u, [1.0], few, 1.0, 1.0).status === :max_iterations
    end

    @testset "direct solver refuses implicit matrix-free terms" begin
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        g = cylindrical_grid(MomentumGrid(uniform_p_faces(6, 0.0, 3.0), uniform_xi_faces(4)), rad)
        op = KnockOnOperator(g, g, 0.2)
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g)])
        term(treatment) = KnockOnTerm(:hot_tail_distribution, op; target_density = [1e20], treatment)
        decay = LinearTerm(:hot_tail_distribution; coefficient = -1.0)
        implicit = EquationSystem(layout,
            [Equation(:hot_tail_distribution, TimeDerivative(), decay,
                term(:implicit))])
        @test_throws ArgumentError NewtonWorkspace(implicit)
        explicit = EquationSystem(layout,
            [Equation(:hot_tail_distribution, TimeDerivative(), decay,
                term(:explicit))])
        @test NewtonWorkspace(explicit) isa NewtonWorkspace
    end

    @testset "time points from any array, saved states independent of the steps" begin
        # du/dt = -k u on two radii with different initial values
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k); nr = 2)
        u0 = [1.0, 2.0]
        times = [0.0, 0.01, 0.05, 0.06, 0.2, 0.7]
        save_times = [0.0, 0.05, 0.06, 0.7]
        solution = integrate(system, u0, times; save_times)
        exact = [u0 ./
                 prod(1 + k * (times[j + 1] - times[j]) for j in 1:(m - 1); init = 1.0)
                 for m in eachindex(times)]
        @test solution.times == save_times
        @test solution.states[:, 1] == u0
        @test solution.states[:, 2] ≈ exact[3] rtol = 1e-12
        @test solution.states[:, 3] ≈ exact[4] rtol = 1e-12
        @test solution.states[:, 4] ≈ exact[6] rtol = 1e-12
        # every saved state is a computed one: a save time between the time points is rejected
        @test_throws ArgumentError integrate(system, u0, times; save_times = [
            0.0, 0.03, 0.7])
        stats = solution.statistics
        @test stats.time == times[2:end]
        @test stats.time_step ≈ diff(times)
        @test all(stats.accepted)
        @test all(==(:converged), stats.status)
        @test all(==(2), stats.newton_iterations)
        @test all(stats.accepted_as_linear)
        @test all(==(0), stats.krylov_iterations)
        # the timed parts of a step are within its wall time, the factorization within the solve
        @test all(>=(0), stats.jacobian_time) && all(>=(0), stats.residual_time)
        @test all(stats.factorization_time .<= stats.linear_solve_time)
        @test all(stats.jacobian_time .+ stats.residual_time .+ stats.linear_solve_time .<=
                  stats.wall_time)
        @test all(>(0), stats.jacobian_time) && all(>(0), stats.factorization_time)
        @test length(stats.iteration_attempt) == sum(stats.newton_iterations)
        @test stats.iteration_attempt == repeat(1:(length(times) - 1), inner = 2)
        @test_throws ArgumentError integrate(system, u0, [0.0, 0.1, 0.1])
        @test_throws ArgumentError integrate(system, u0, times; save_times = [0.8])
        @test_throws ArgumentError integrate(system, u0, times; save_times = [0.1, 0.0])
    end

    @testset "a failed step is retried with shorter steps" begin
        # without a lower limit, Newton for -k √u overshoots below zero when k dt > 2 √u:
        # the first step (k dt = 0.96 > 2√0.2) fails, its two halves and the next step succeed
        k = 4.0
        system = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, k))
        times = [0.0, 0.24, 0.26]
        solution = integrate(system, [0.2], times)
        stats = solution.statistics
        @test stats.accepted == [false, true, true, true]
        @test stats.status[1] === :domain_error
        @test stats.time_step ≈ [0.24, 0.12, 0.12, 0.02]
        # the accepted steps end exactly on the requested time points
        @test stats.time[2:end] == [0.12, 0.24, 0.26]
        expected = foldl((u, h) -> square_root_step(u, k, h), stats.time_step[2:end]; init = 0.2)
        @test solution.states[1, end] ≈ expected rtol = 1e-12
        @test_throws TimeStepFailure integrate(system, [0.2], times; max_step_halvings = 0)
        @test_throws TimeStepFailure integrate(system, [0.2], times; stepper = FixedSteps(max_halvings = 0))
        @test_throws ArgumentError integrate(
            system, [0.2], times; max_step_halvings = 0, stepper = FixedSteps())
        @test_throws ArgumentError FixedSteps(max_halvings = -1)
    end

    @testset "run options" begin
        # the keyword form and the record give the same run and the same record
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k); nr = 2)
        times = [0.0, 0.01, 0.05, 0.2]
        options = RunOptions(save_times = [0.0, 0.05, 0.2], description = Dict("name" => "options"))
        by_record = integrate(system, [1.0, 2.0], times, options)
        by_keywords = integrate(system, [1.0, 2.0], times; save_times = [0.0, 0.05, 0.2],
            description = Dict("name" => "options"))
        @test by_record.states == by_keywords.states
        @test by_record.times == by_keywords.times == [0.0, 0.05, 0.2]
        @test by_record.settings["setup_hash"] == by_keywords.settings["setup_hash"]
        @test by_record.settings["time"]["stepper"] ==
              Dict{String, Any}("type" => "FixedSteps", "max_halvings" => 10)
        @test by_record.settings["segment"]["steps"] == 3
        # `environment` carries the start time, which differs when the two calls straddle a second
        without_environment(record) = filter(pair -> pair.first != "environment", record)
        record_of_options = run_settings(system, times, options)
        record_of_keywords = run_settings(system; times, save_times = [0.0, 0.05, 0.2],
            description = Dict("name" => "options"))
        @test without_environment(record_of_options) ==
              without_environment(record_of_keywords)
        @test haskey(record_of_options["environment"], "start_time")
        @test haskey(record_of_keywords["environment"], "start_time")
        @test RunOptions().stepper == FixedSteps()
        @test RunOptions().save_times === nothing
        @test describe(FixedSteps(max_halvings = 3)) ==
              Dict{String, Any}("type" => "FixedSteps", "max_halvings" => 3)
    end

    @testset "HDF5 output" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        mom = MomentumGrid(uniform_p_faces(5, 0.0, 2.0), uniform_xi_faces(3))
        g = cylindrical_grid(mom, rad)
        layout = UnknownLayout(rad,
            [ScalarUnknown(:plasma_current; scale = 2.0),
                RadialUnknown(:ion_densities; components = 2, nonnegative = true),
                KineticUnknown(:hot_tail_distribution, g)])
        decay(name) = Equation(name, TimeDerivative(), LinearTerm(name; coefficient = -1.0))
        system = EquationSystem(layout,
            [decay(:plasma_current), decay(:ion_densities),
                decay(:hot_tail_distribution)])
        u0 = collect(1.0:length(layout))
        solution = integrate(system, u0, [0.0, 0.1, 0.2])
        file = tempname() * ".h5"
        write_output(file, solution)
        h5open(file) do h5
            # the number itself is pinned in test_settings.jl
            @test read(attributes(h5)["format_version"]) == Runes2P.output_format_version()
            @test read(h5["time"]) == [0.0, 0.1, 0.2]
            f = read(h5["unknowns/hot_tail_distribution"])
            @test size(f) == (5, 3, 2, 3)
            @test f[:, :, 2, 3] == kinetic_view(solution.states[:, 3], layout, 3, 2)
            n = read(h5["unknowns/ion_densities"])
            @test size(n) == (2, 2, 3)
            @test n[:, 1, 2] ==
                  unknown_view(solution.states[:, 2], layout, :ion_densities, 1)
            @test read(h5["unknowns/plasma_current"]) == solution.states[end, :]
            @test read(h5["layout/order"]) ==
                  ["plasma_current", "ion_densities", "hot_tail_distribution"]
            @test read(h5["layout/layout_version"]) == 2
            grid_name = read(h5["layout/unknowns/hot_tail_distribution/grid"])
            @test read(h5["layout/grids/$grid_name/momentum_grid/p_faces"]) == mom.p_faces
            @test read(h5["layout/unknowns/ion_densities/lower_limit_tolerance"]) == 0.0
            @test read(h5["solver_statistics/time"]) == [0.1, 0.2]
            @test read(h5["solver_statistics/status"]) == ["converged", "converged"]
            @test read(h5["solver_statistics/newton_iterations"]) ==
                  solution.statistics.newton_iterations
        end
        rm(file)

        # per-radius grids: one dataset per radius
        other = MomentumGrid(uniform_p_faces(4, 0.0, 2.0), uniform_xi_faces(3))
        ragged = cylindrical_grid([mom, other], rad)
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, ragged)])
        system = EquationSystem(layout, [decay(:hot_tail_distribution)])
        solution = integrate(system, ones(length(layout)), [0.0, 0.1])
        file = tempname() * ".h5"
        write_output(file, solution)
        h5open(file) do h5
            @test size(read(h5["unknowns/hot_tail_distribution/radius_1"])) == (5, 3, 2)
            @test size(read(h5["unknowns/hot_tail_distribution/radius_2"])) == (4, 3, 2)
        end
        rm(file)
    end
end

@testset "the rows of a kinetic bulk take the size of their entry" begin
    # A Maxwellian of 1e20 m^-3 at 130 eV relaxing at 100 eV: f reaches 1.5e24 in the bulk,
    # against the scale 1e19 of the unknown. With a step of 1e-4 s, far above the collision
    # time, the collision fluxes of the bulk cancel in its rows, and the round-off of that
    # cancellation would hold the residual above the tolerance with the weight dt/scale. With
    # the weight dt/|u_old| the step converges to the tolerance; the tail keeps dt/scale.
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    theta = 100.0 / Constants.ELECTRON_REST_ENERGY
    grid = cylindrical_grid(
        MomentumGrid(uniform_p_faces(30, 0.0, 12 * sqrt(2theta)), uniform_xi_faces(2)), rad)
    settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :thermal)
    plasma(t) = PlasmaState(; cold_electron_temperature = [100.0],
        ions = [IonSpecies("D", 1, 1, [1e20])], electric_field = [0.0], magnetic_field = [5.0])
    operator = AdvectionDiffusion(grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero)
    layout = UnknownLayout(rad, [KineticUnknown(:f, grid; scale = 1e19)])
    system = EquationSystem(layout,
        [Equation(:f, TimeDerivative(), prescribed_fokker_planck_term(:f, operator, plasma; settings))])
    u_old = zeros(length(layout))
    f = kinetic_view(u_old, layout, 1, 1)
    p = momentum_grid(grid).p_centers
    for j in axes(f, 2), i in axes(f, 1)

        f[i, j] = exp(-p[i]^2 / (2 * 1.3 * theta))
    end
    f .*= 1e20 / density(grid, reshape(f, size(f)..., 1))[1]
    @test maximum(f) > 1e24
    dt = 1e-4
    u = copy(u_old)
    # without the acceptance of a linear step, so that the iteration goes through the test of
    # the residual: with the weight dt/scale it stalls at the floor and takes a third iteration
    ws = NewtonWorkspace(system, NewtonSolver(linear_acceptance = false))
    result = newton_solve!(u, u_old, ws, dt, dt)
    @test result.status === :converged
    @test !result.residual_floor
    @test result.iterations == 2
    @test result.residual_norm <= NewtonSolver().residual_tolerance
    # the bulk row is weighted by its entry, the tail row by the scale of the unknown
    @test ws.row_weights[1] == dt / abs(u_old[1])
    @test abs(u_old[end]) < 1e19
    @test ws.row_weights[end] == dt / 1e19
    # the residual of the accepted state with the weight dt/scale on every row: the round-off
    # floor of the bulk, above the tolerance
    F = similar(u)
    residual!(F, u, u_old, system, dt, dt)
    @test maximum(abs.(F)) * dt / 1e19 > NewtonSolver().residual_tolerance
end

@testset "the polish of negative entries" begin
    # the schemes decide which kinetic unknowns `:auto` polishes
    mom = MomentumGrid(uniform_p_faces(20, 0.0, 1.0), uniform_xi_faces(4))
    function polish_system(scheme)
        g, op = iter_plasma_operator(mom, 100.0, 1.0; scheme)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1e20)])
        return EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
    end
    tcdf, smart, central = polish_system(:tcdf), polish_system(:smart),
    polish_system(:central)
    polished(system, mode) = [group.unknown
                              for group in NewtonWorkspace(
        system, NewtonSolver(polish_negatives = mode)).polish]
    @test polished(tcdf, :auto) == [:f]
    @test isempty(polished(smart, :auto))
    @test isempty(polished(central, :auto))
    @test polished(smart, true) == [:f]
    @test isempty(polished(tcdf, false))
    @test_throws "polish_negatives must be" NewtonWorkspace(
        tcdf, NewtonSolver(polish_negatives = :always))
    # `:auto` stops after POLISH_MISSES polished solves in a row that leave negative entries,
    # with a warning; `true` never stops
    for (mode, stops) in ((:auto, true), (true, false))
        solver = NewtonSolver(polish_negatives = mode)
        ws = NewtonWorkspace(tcdf, solver)
        u = ones(length(tcdf.layout))
        u[3] = -1.0
        for step in 1:(POLISH_MISSES - 1)
            @test Runes2P._start_polish!(ws, u)
            @test_logs Runes2P._finish_polish!(ws, solver, u, 1e-3 * step)
        end
        @test Runes2P._start_polish!(ws, u)
        if stops
            @test_logs (:warn, r"polish_negatives = :auto stops polishing f") Runes2P._finish_polish!(
                ws, solver, u, 1.0)
        else
            @test_logs Runes2P._finish_polish!(ws, solver, u, 1.0)
        end
        @test ws.polish[1].enabled[] == !stops
        @test Runes2P._start_polish!(ws, u) == !stops
    end
    # an extra iteration that removes the negative entries resets the count
    solver = NewtonSolver()
    ws = NewtonWorkspace(tcdf, solver)
    negative, clean = ones(length(tcdf.layout)), ones(length(tcdf.layout))
    negative[3] = -1.0
    for _ in 1:(POLISH_MISSES - 1)
        Runes2P._start_polish!(ws, negative)
        Runes2P._finish_polish!(ws, solver, negative, 0.0)
    end
    Runes2P._start_polish!(ws, negative)
    Runes2P._finish_polish!(ws, solver, clean, 0.0)
    @test ws.polish[1].misses[] == 0
    @test ws.polish[1].enabled[]
end

@testset "a stiff step converges at the residual floor" begin
    # With rate·dt = 1e10 the transient and the right-hand side cancel to ten digits, so
    # round-off keeps the scaled residual near 1e-6, above the default tolerance. Newton
    # reaches the exact solution in one step; without the waiver of the residual floor (the
    # round-off bound or the stall) it would iterate to the limit and the step would fail.
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    layout = UnknownLayout(rad, [RadialUnknown(:x; components = 2, scale = 1.0)])
    rate = 1e10
    system = EquationSystem(layout, [Equation(:x, TimeDerivative(), ExchangeTerm(:x, rate))])
    u_old = [1.0, 0.0]
    u = copy(u_old)
    ws = NewtonWorkspace(system)
    result = newton_solve!(u, u_old, ws, 1.0, 1.0)
    @test result.status === :converged
    @test result.residual_norm > NewtonSolver().residual_tolerance
    # the waiver is flagged, so that a run can tell it from convergence to the tolerance
    @test result.residual_floor
    @test result.iterations <= 4
    soft = EquationSystem(layout, [Equation(:x, TimeDerivative(), ExchangeTerm(:x, 1.0))])
    solution = integrate(soft, [1.0, 0.0], [0.0, 0.1, 0.2])
    @test solution.statistics.accepted == [true, true]
    @test solution.statistics.accepted_at_residual_floor == [false, false]
    stiff = integrate(system, [1.0, 0.0], [0.0, 1.0])
    @test stiff.statistics.accepted_at_residual_floor == [true]
    # backward Euler on the exchange pair: the difference decays by 1/(1 + 2 rate dt)
    difference = (u_old[1] - u_old[2]) / (1 + 2 * rate)
    @test u[1] - u[2] ≈ difference rtol = 1e-6
    @test u[1] + u[2] ≈ sum(u_old) rtol = 1e-12
end

@testset "a stiff step is accepted within the round-off bound of its rows" begin
    # With rate·dt = 1e12 the two terms of each row cancel to twelve digits. The second
    # iterate is the solution to round-off, but its residual fell by chance (from 4.4e-5 to
    # 1.1e-5 in scaled units), so the stall test alone would wait for a third iteration. The
    # round-off bound eps Σ_j |(∂F/∂u)_ij u_j| of each row accepts the second.
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    layout = UnknownLayout(rad, [RadialUnknown(:x; components = 2, scale = 1.0)])
    rate = 1e12
    system = EquationSystem(layout, [Equation(:x, TimeDerivative(), ExchangeTerm(:x, rate))])
    u_old = [1.0, 0.0]
    u = copy(u_old)
    ws = NewtonWorkspace(system)
    result = newton_solve!(u, u_old, ws, 1.0, 1.0)
    @test result.status === :converged
    @test result.residual_floor
    @test result.iterations == 2
    @test result.residual_norm > NewtonSolver().residual_tolerance
    difference = (u_old[1] - u_old[2]) / (1 + 2 * rate)
    # the difference, 5e-13, is resolved to the round-off of the entries near 0.5
    @test u[1] - u[2] ≈ difference atol = 1e-15
    @test u[1] + u[2] ≈ sum(u_old) rtol = 1e-12
    F = similar(u)
    @test Runes2P._checked_residual!(F, u, u_old, system, 1.0, 1.0) === :ok
    @test Runes2P._residual_at_round_off(ws, F, u, 1.0)
    # an iterate off the solution by ±1e-6 has a residual of about 2e6 against a bound near
    # 2e-4:
    # it is not at the floor, and the iteration must go on
    off = u .+ [1e-6, -1e-6]
    @test Runes2P._checked_residual!(F, off, u_old, system, 1.0, 1.0) === :ok
    @test !Runes2P._residual_at_round_off(ws, F, off, 1.0)
end

# A repeated save time would be saved twice, which the reader of a segmented run refuses;
# the run refuses it, also when two save times snap to one time point.
@testset "repeated save times are refused" begin
    system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -1.0))
    u0 = [1.0]
    times = [0.0, 0.5, 1.0]
    @test length(integrate(system, u0, times; save_times = [0.0, 0.5, 1.0]).times) == 3
    @test_throws "strictly increasing" integrate(system, u0, times; save_times = [
        0.0, 0.5, 0.5, 1.0])
    @test_throws "strictly increasing" integrate(
        system, u0, times; save_times = [0.0, 0.5, 0.5 * (1 + 1e-13), 1.0])
end
