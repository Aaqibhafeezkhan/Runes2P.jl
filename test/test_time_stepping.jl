# Backward Euler with Newton and a sparse direct solver: convergence on update
# and residual, lower limits that shorten only steps towards the limit, the optional line
# search, failures handed to the time stepper, time points from any array, saved states
# independent of the steps, solver statistics and HDF5 output. Every test problem has a
# known backward-Euler solution.
using HDF5

# test terms on a radial unknown with one component: -rate √x (throws a DomainError for
# x < 0), rate atan(x - root) (for an algebraic equation) and a constant source
struct SquareRootDecayTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
end
struct ArctangentTerm <: EquationTerm
    unknown::Symbol
    root::Float64
end
struct ConstantSourceTerm <: EquationTerm
    value::Float64
end
Runes2P.dependencies(term::Union{SquareRootDecayTerm,ArctangentTerm}) = [term.unknown]
Runes2P.dependencies(::ConstantSourceTerm) = Symbol[]

function Runes2P.rhs!(out, term::SquareRootDecayTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        out[a] -= term.rate * sqrt(u[b])
    end
    return out
end
function Runes2P.rhs!(out, term::ArctangentTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        out[a] += atan(u[b] - term.root)
    end
    return out
end
function Runes2P.rhs!(out, term::ConstantSourceTerm, u, layout, row, t)
    for ir in 1:radial_cell_count(layout), a in unknown_range(layout, row, ir)
        out[a] += term.value
    end
    return out
end
function Runes2P.sparsity!(entries, term::Union{SquareRootDecayTerm,ArctangentTerm}, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end
Runes2P.sparsity!(entries, ::ConstantSourceTerm, layout, row) = entries

# a system of one radial unknown :x on `nr` radii
function single_unknown_system(kind, terms...; nr=1, scale=1.0, nonnegative=false, extra=())
    rad = RadialGrid(uniform_r_faces(nr, 0.5))
    unknowns = [RadialUnknown(:x; scale, nonnegative), (u for (u, _) in extra)...]
    equations = [Equation(:x, kind, terms...), (eq for (_, eq) in extra)...]
    return EquationSystem(UnknownLayout(rad, unknowns), equations)
end

# backward-Euler step of du/dt = -k √u: √u1 solves s² + k dt s - u0 = 0
square_root_step(u0, k, dt) = ((-k * dt + sqrt((k * dt)^2 + 4u0)) / 2)^2

@testset "Newton and backward-Euler time stepping" begin
    @testset "nonlinear step: quadratic convergence, lower limit" begin
        k, dt, u0 = 10.0, 1.0, [1.0]
        # the full Newton step from u0 lands at u0 - 5/3 < 0, where √ is undefined
        free = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, k))
        u = copy(u0)
        result = newton_solve!(u, u0, NewtonWorkspace(free), dt, dt)
        @test result.status === :domain_error
        @test result.iterations == 0

        limited = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, k); nonnegative=true)
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

    @testset "lower limits shorten only steps towards the limit (DREAM #91)" begin
        dt, u0 = 1.0, [2.0]
        # growth by a factor 100 in one step: never shortened
        growth = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=0.99); nonnegative=true)
        u = copy(u0)
        ws = NewtonWorkspace(growth)
        result = newton_solve!(u, u0, ws, dt, dt)
        @test result.status === :converged
        @test u[1] ≈ 200.0 rtol = 1e-12
        @test result.smallest_step_length == 1.0
        @test result.iterations == 2

        # decay by a factor 1000: the limit lets each iteration remove at most 90% of the
        # distance to zero, so a few more iterations, never below zero
        decay = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-999.0); nonnegative=true)
        u = copy(u0)
        ws = NewtonWorkspace(decay)
        result = newton_solve!(u, u0, ws, dt, dt)
        @test result.status === :converged
        @test u[1] ≈ 2e-3 rtol = 1e-10
        @test result.smallest_step_length < 1
        @test result.iterations > 2
        # a larger admissible reduction per iteration converges faster
        u = copy(u0)
        faster = newton_solve!(u, u0, NewtonWorkspace(decay, NewtonSolver(max_reduction=0.999)), dt, dt)
        @test faster.status === :converged
        @test faster.iterations < result.iterations
        # without the limit the linear step takes one iteration plus the check
        unlimited = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-999.0))
        u = copy(u0)
        @test newton_solve!(u, u0, NewtonWorkspace(unlimited), dt, dt).iterations == 2
    end

    @testset "lower limit relative to a reference unknown" begin
        # du/dt = -x - 3 with the limit x ≥ -0.1 n: the backward-Euler solution -1 (dt = 1)
        # lies below the limit; the iterates approach the limit and stop there, and the
        # step fails
        reference = RadialUnknown(:n; scale=1.0)
        n_equation = Equation(:n, TimeDerivative(), LinearTerm(:n; coefficient=0.0))
        limit = LowerLimit(tolerance=0.1, reference=:n)
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-1.0),
                                       ConstantSourceTerm(-3.0); nonnegative=limit,
                                       extra=((reference, n_equation),))
        layout = system.layout
        u_old = zeros(length(layout))
        unknown_view(u_old, layout, :x, 1)[1] = 1.0
        unknown_view(u_old, layout, :n, 1)[1] = 1.0
        u = copy(u_old)
        result = newton_solve!(u, u_old, NewtonWorkspace(system), 1.0, 1.0)
        @test result.status === :below_lower_limit
        @test -0.1 <= unknown_view(u, layout, :x, 1)[1] < -0.0999
        @test describe(layout)["unknowns"][1]["lower_limit_reference"] == "n"
        # checks on declaration
        @test_throws ArgumentError LowerLimit(tolerance=-1.0)
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        @test_throws ArgumentError UnknownLayout(rad, [RadialUnknown(:x; nonnegative=LowerLimit(reference=:y))])
        @test_throws ArgumentError UnknownLayout(rad, [RadialUnknown(:x; nonnegative=LowerLimit(reference=:y)),
                                                       RadialUnknown(:y; components=2)])
        @test_throws ArgumentError UnknownLayout(rad, [ScalarUnknown(:x; nonnegative=LowerLimit(reference=:y)),
                                                       RadialUnknown(:y)])
    end

    @testset "line search on the scaled residual (DREAM PR #460)" begin
        # 0 = atan(x - 1): plain Newton diverges from |x - 1| > 1.39
        system = single_unknown_system(Algebraic(), ArctangentTerm(:x, 1.0))
        u_old = [4.0]
        u = copy(u_old)
        plain = newton_solve!(u, u_old, NewtonWorkspace(system), 0.0, 1.0)
        @test plain.status !== :converged
        u = copy(u_old)
        ws = NewtonWorkspace(system, NewtonSolver(line_search=true))
        result = newton_solve!(u, u_old, ws, 0.0, 1.0)
        @test result.status === :converged
        @test u[1] ≈ 1.0 atol = 1e-10
        @test result.smallest_step_length < 1
        @test ws.step_lengths[end] == 1.0
        @test result.convergence_order > 2  # atan'' = 0 at the root: cubic
    end

    @testset "failures are reported, not thrown" begin
        singular = single_unknown_system(Algebraic(), LinearTerm(:x; coefficient=0.0))
        u = [1.0]
        @test newton_solve!(u, [1.0], NewtonWorkspace(singular), 0.0, 1.0).status === :singular_matrix
        nan_source = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-1.0),
                                           ConstantSourceTerm(NaN))
        u = [1.0]
        @test newton_solve!(u, [1.0], NewtonWorkspace(nan_source), 1.0, 1.0).status === :non_finite_residual
        u = [1.0]
        few = NewtonWorkspace(single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, 10.0);
                                                    nonnegative=true), NewtonSolver(max_iterations=2))
        @test newton_solve!(u, [1.0], few, 1.0, 1.0).status === :max_iterations
    end

    @testset "direct solver refuses implicit matrix-free terms" begin
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        g = cylindrical_grid(MomentumGrid(uniform_p_faces(6, 0.0, 3.0), uniform_xi_faces(4)), rad)
        op = KnockOnOperator(g, g, 0.2)
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g)])
        term(treatment) = KnockOnTerm(:hot_tail_distribution, op; target_density=[1e20], treatment)
        decay = LinearTerm(:hot_tail_distribution; coefficient=-1.0)
        implicit = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(), decay,
                                                    term(:implicit))])
        @test_throws ArgumentError NewtonWorkspace(implicit)
        explicit = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(), decay,
                                                    term(:explicit))])
        @test NewtonWorkspace(explicit) isa NewtonWorkspace
    end

    @testset "time points from any array, saved states independent of the steps" begin
        # du/dt = -k u on two radii with different initial values
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k); nr=2)
        u0 = [1.0, 2.0]
        times = [0.0, 0.01, 0.05, 0.06, 0.2, 0.7]
        save_times = [0.0, 0.03, 0.06, 0.7]
        solution = integrate(system, u0, times; save_times)
        exact = [u0 ./ prod(1 + k * (times[j+1] - times[j]) for j in 1:m-1; init=1.0) for m in eachindex(times)]
        @test solution.times == save_times
        @test solution.states[:, 1] == u0
        @test solution.states[:, 3] ≈ exact[4] rtol = 1e-12
        @test solution.states[:, 4] ≈ exact[6] rtol = 1e-12
        # between the ends of a step: linear interpolation of the end states
        θ = (0.03 - 0.01) / (0.05 - 0.01)
        @test solution.states[:, 2] ≈ (1 - θ) * exact[2] + θ * exact[3] rtol = 1e-12
        stats = solution.statistics
        @test stats.time == times[2:end]
        @test stats.time_step ≈ diff(times)
        @test all(stats.accepted)
        @test all(==(:converged), stats.status)
        @test all(==(2), stats.newton_iterations)
        @test all(==(0), stats.krylov_iterations)
        @test length(stats.iteration_attempt) == sum(stats.newton_iterations)
        @test stats.iteration_attempt == repeat(1:length(times)-1, inner=2)
        @test_throws ArgumentError integrate(system, u0, [0.0, 0.1, 0.1])
        @test_throws ArgumentError integrate(system, u0, times; save_times=[0.8])
        @test_throws ArgumentError integrate(system, u0, times; save_times=[0.1, 0.0])
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
        expected = foldl((u, h) -> square_root_step(u, k, h), stats.time_step[2:end]; init=0.2)
        @test solution.states[1, end] ≈ expected rtol = 1e-12
        @test_throws TimeStepFailure integrate(system, [0.2], times; max_step_halvings=0)
    end

    @testset "HDF5 output" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        mom = MomentumGrid(uniform_p_faces(5, 0.0, 2.0), uniform_xi_faces(3))
        g = cylindrical_grid(mom, rad)
        layout = UnknownLayout(rad, [ScalarUnknown(:plasma_current; scale=2.0),
                                     RadialUnknown(:ion_densities; components=2, nonnegative=true),
                                     KineticUnknown(:hot_tail_distribution, g)])
        decay(name) = Equation(name, TimeDerivative(), LinearTerm(name; coefficient=-1.0))
        system = EquationSystem(layout, [decay(:plasma_current), decay(:ion_densities),
                                         decay(:hot_tail_distribution)])
        u0 = collect(1.0:length(layout))
        solution = integrate(system, u0, [0.0, 0.1, 0.2])
        file = tempname() * ".h5"
        write_output(file, solution)
        h5open(file) do h5
            @test read(attributes(h5)["format_version"]) == 1
            @test read(h5["time"]) == [0.0, 0.1, 0.2]
            f = read(h5["unknowns/hot_tail_distribution"])
            @test size(f) == (5, 3, 2, 3)
            @test f[:, :, 2, 3] == kinetic_view(solution.states[:, 3], layout, 3, 2)
            n = read(h5["unknowns/ion_densities"])
            @test size(n) == (2, 2, 3)
            @test n[:, 1, 2] == unknown_view(solution.states[:, 2], layout, :ion_densities, 1)
            @test read(h5["unknowns/plasma_current"]) == solution.states[end, :]
            @test read(h5["layout/unknowns/order"]) == ["plasma_current", "ion_densities", "hot_tail_distribution"]
            @test read(h5["layout/unknowns/hot_tail_distribution/p_faces/2"]) == mom.p_faces
            @test read(h5["layout/unknowns/ion_densities/lower_limit_tolerance"]) == 0.0
            @test read(h5["solver_statistics/time"]) == [0.1, 0.2]
            @test read(h5["solver_statistics/status"]) == ["converged", "converged"]
            @test read(h5["solver_statistics/newton_iterations"]) == solution.statistics.newton_iterations
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
