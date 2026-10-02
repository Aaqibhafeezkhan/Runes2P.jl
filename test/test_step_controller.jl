# The step controller of Runes2P: the error estimate from the predictor
# against the exact local error of backward Euler on a linear decay, the doubling of the
# first step, the PI filter with its growth cap, rejections, the tolerances with their
# floor, a stiff relaxation whose step grows to the ceiling, the failure path, the record
# and the resume. The helpers are those of test_time_stepping.jl and test_step_doubling.jl.
using HDF5
using Random: Xoshiro

# the exact local error of one backward-Euler step of du/dt = -k u from u: the step
# against the exact solution over h
_local_error(u, k, h) = u / (1 + k * h) - u * exp(-k * h)

# a decay of a kinetic unknown cell by cell, -k f below the momentum `p_cut` and
# -k_tail f above it: the tail decays faster than the bulk, and its cells lie far below
# the absolute floor of the cell measure of the controller
struct TailDecayTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
    tail_rate::Float64
    p_cut::Float64
end
Runes2P.dependencies(term::TailDecayTerm) = [term.unknown]
_tail_decay_rate(term::TailDecayTerm, p) = p > term.p_cut ? term.tail_rate : term.rate
function Runes2P.rhs!(out, term::TailDecayTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    grid = layout.unknowns[id].grid
    for ir in 1:radial_cell_count(layout)
        p = momentum_grid(grid, ir).p_centers
        f, r = kinetic_view(u, layout, id, ir), kinetic_view(out, layout, row, ir)
        for j in axes(f, 2), i in axes(f, 1)
            r[i, j] -= _tail_decay_rate(term, p[i]) * f[i, j]
        end
    end
    return out
end
function Runes2P.sparsity!(entries, term::TailDecayTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end

@testset "step controller" begin
    @testset "settings" begin
        stepper = StepController()
        @test isnan(stepper.initial_step)
        @test (stepper.relative_tolerance, stepper.max_growth, stepper.min_shrink, stepper.safety) == (1e-3, 2.0, 0.2, 0.9)
        @test (stepper.gain_error, stepper.gain_memory) == (0.35, 0.2)
        @test (stepper.max_failures, stepper.failure_factor, stepper.max_rejections) == (5, 0.2, 20)
        @test_throws ArgumentError StepController(initial_step=0)
        @test_throws ArgumentError StepController(max_rejections=0)
        @test_throws ArgumentError StepController(relative_tolerance=0)
        @test_throws ArgumentError StepController(max_growth=1)
        @test_throws ArgumentError StepController(min_shrink=1)
        @test_throws ArgumentError StepController(safety=0)
        @test_throws ArgumentError StepController(gains=(0, 0.2))
        @test_throws ArgumentError StepController(gains=(0.5,))
        @test_throws ArgumentError StepController(max_failures=-1)
        @test_throws ArgumentError StepController(failure_factor=1)
        record = describe(StepController(initial_step=1e-3, gains=(0.5, 0.1)))
        @test record["type"] == "StepController"
        @test record["initial_step"] == 1e-3
        @test (record["gain_error"], record["gain_memory"]) == (0.5, 0.1)
        @test record["max_growth"] == 2.0
        @test (record["tail_cut"], record["tail_floor"]) == (1.0, 1e-12)
        @test isnan(describe(StepController(tail_cut=nothing))["tail_cut"])
        @test_throws ArgumentError StepController(tail_cut=-1.0)
        @test_throws ArgumentError StepController(tail_floor=-1e-3)
    end

    @testset "the estimate is the local error of backward Euler" begin
        # du/dt = -k u with u(0) = 1: the first step is checked by a doubling, the second
        # by the predictor. Both estimates converge to the exact local error of their
        # step as the step shrinks, with a misfit of first order in the step.
        k, rel = 3.0, 1e-3
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k))
        function estimates(h)
            # two steps of the length h, then the run ends: the second step is estimated
            # by the predictor through the first
            # the second step keeps the length of the first: the factor of the filter is
            # pinned to 1 (within the tolerance of the landing on the end of the run)
            stepper = StepController(initial_step=h, relative_tolerance=rel, max_growth=1 + 1e-12,
                                     min_shrink=1 - 1e-12)
            solution = integrate(system, [1.0], [0.0, 2h]; stepper)
            stats = solution.statistics
            steps = findall(==(:step), stats.kind)
            @test length(steps) == 2 && all(stats.accepted[steps])
            @test stats.kind[1:3] == [:half_step, :half_step, :step]
            u1 = 1 / (1 + k * h)
            # the error columns hold the estimate over the tolerance: absolute rel × scale
            # (the scale is 1) plus rel |u|
            first = stats.error[steps[1]] * (rel + rel * u1)
            second = stats.error[steps[2]] * (rel + rel * u1 / (1 + k * h))
            return first, second
        end
        exact_first(h) = abs(_local_error(1.0, k, h))
        exact_second(h) = abs(_local_error(1 / (1 + k * h), k, h))
        # steps whose errors lie under the tolerance, so no step is rejected (a rejected
        # step with the shrink pinned to 1 would repeat without end, which max_rejections
        # ends)
        ratios = map((0.01, 0.005, 0.0025)) do h
            first, second = estimates(h)
            (first / exact_first(h), second / exact_second(h))
        end
        for (first, second) in ratios
            @test 0.8 < first < 1.2
            @test 0.8 < second < 1.2
        end
        # the misfit of the predictor's estimate falls with h
        misfits = [abs(second - 1) for (_, second) in ratios]
        @test misfits[2] < 0.6 * misfits[1]
        @test misfits[3] < 0.6 * misfits[2]
    end

    @testset "a stiff relaxation: the step grows to the ceiling under the tolerance" begin
        # du/dt = -k (u - 1) from u = 0 with k = 1e4: the layer is resolved with short
        # steps, then the step grows by the cap of 2 per step to the end
        k = 1e4
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k), ConstantSourceTerm(k))
        stepper = StepController(initial_step=1e-5, relative_tolerance=1e-2)
        solution = integrate(system, [0.0], [0.0, 1.0]; stepper, save_times=[0.0, 1.0])
        stats = solution.statistics
        accepted = findall(stats.accepted)
        steps = stats.time_step[accepted]
        @test 10 <= length(accepted) <= 40
        @test all(stats.error[accepted] .<= 1)
        # the growth is capped at the factor 2 and reaches it in the plateau
        growth = steps[2:end] ./ steps[1:end-1]
        @test all(growth .<= 2 * (1 + 1e-12))
        @test count(g -> g > 1.9, growth) >= 5
        @test last(stats.time) == 1.0
        @test solution.states[1, end] ≈ 1.0 atol = 1e-6
        # against the fixed steps of the same count the controller keeps the error small
        exact(t) = 1 - exp(-k * t)
        for (m, t) in enumerate(stats.time[accepted])
            @test abs(solution.states[1, end] - exact(1.0)) < 1e-2 || m < length(accepted)
        end
    end

    @testset "rejections and the growth cap" begin
        # a large first step is rejected, the retry shrinks it by the elementary rule
        # with the floor min_shrink
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k))
        stepper = StepController(initial_step=2.0, relative_tolerance=1e-4)
        solution = integrate(system, [1.0], [0.0, 3.0]; stepper)
        stats = solution.statistics
        @test stats.kind[1:3] == [:half_step, :half_step, :step]
        @test !stats.accepted[3]
        @test stats.error[3] > 1
        @test all(==(:converged), stats.status)
        shrink = max(0.2, 0.9 * stats.error[3]^(-0.5))
        @test stats.kind[4:6] == [:half_step, :half_step, :step]  # the doubling again, from the same state
        @test stats.time_step[4:6] ≈ [shrink, shrink, 2 * shrink]
        accepted = findall(stats.accepted)
        steps = stats.time_step[accepted]
        growth = steps[2:end-1] ./ steps[1:end-2]                  # the last step is cut by the end of the run
        @test all(growth .<= 2 * (1 + 1e-12))
        @test all(growth .>= 0.2 * (1 - 1e-12))
        @test solution.states[1, end] ≈ foldl((u, h) -> u / (1 + k * h), steps; init=1.0) rtol = 1e-12
    end

    @testset "tail moments of a kinetic unknown" begin
        # f = 1 on a small grid, the bulk decays with k, the tail above p = 1 with 11 k: the
        # cells of the tail fall below the absolute floor of the cell measure (relative ×
        # scale) at once, so without the moments the step follows the bulk and grows past
        # 1 / k_tail, while with them the tail density holds the step. Measured values in
        # the comments.
        k, k_tail, t_end = 1.0, 11.0, 1.0
        mom = MomentumGrid(uniform_p_faces(8, 0.0, 2.0), uniform_xi_faces(2))
        grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=1.0)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), TailDecayTerm(:f, k, k_tail, 1.0))])
        u0 = ones(length(layout))
        region = MomentumRegion(1.0)
        tail(u) = density(grid, reshape(u, size(grid)); region)[1]
        exact_tail = tail(u0) * exp(-k_tail * t_end)
        runs = Dict{String,Any}()
        for (name, cut) in (("with" => 1.0), ("without" => nothing))
            stepper = StepController(initial_step=1e-3, relative_tolerance=1e-2, tail_cut=cut)
            solution = integrate(system, u0, [0.0, t_end]; stepper)
            stats = solution.statistics
            accepted = [k for k in eachindex(stats.kind) if stats.kind[k] === :step && stats.accepted[k]]
            runs[name] = (; h=stats.time_step[accepted], errors=stats.error[accepted],
                          ratio=tail(solution.states[:, end]) / exact_tail, record=solution.settings["time"]["stepper"])
        end
        @test runs["with"].record["tail_cut"] == 1.0 && runs["with"].record["tail_floor"] == 1e-12
        @test isnan(runs["without"].record["tail_cut"])
        @test all(runs["with"].errors .<= 1) && all(runs["without"].errors .<= 1)
        # without the moments the step grows past 1 / k_tail, with them it stays below
        @test maximum(runs["without"].h) > 1 / k_tail
        @test maximum(runs["with"].h) < 1 / k_tail
        @test length(runs["with"].h) > 2 * length(runs["without"].h)
        # the tail density at the end against the exact decay: backward Euler at this
        # tolerance overestimates it by a factor of order 2, the uncontrolled run by far more
        @test 1 / 3 < runs["with"].ratio < 3
        @test runs["without"].ratio > 10

        # pitch grids whose cell counts differ between the radii: the moments
        # are taken per radius, where they needed one shape of the grid before
        ragged = cylindrical_grid([mom, MomentumGrid(uniform_p_faces(8, 0.0, 2.0), uniform_xi_faces(3))],
                                  RadialGrid(uniform_r_faces(2, 1.0)))
        ragged_layout = UnknownLayout(ragged.radial_grid, [KineticUnknown(:f, ragged; scale=1.0)])
        ragged_system = EquationSystem(ragged_layout, [Equation(:f, TimeDerivative(), TailDecayTerm(:f, k, k_tail, 1.0))])
        solution = integrate(ragged_system, ones(length(ragged_layout)), [0.0, t_end];
                             stepper=StepController(initial_step=1e-3, relative_tolerance=1e-2))
        stats = solution.statistics
        @test maximum(stats.time_step[[k for k in eachindex(stats.kind) if stats.kind[k] === :step && stats.accepted[k]]]) <
              1 / k_tail
        # the weights give the moments of `density` and `current_density` at every radius
        same = cylindrical_grid(MomentumGrid(uniform_p_faces(8, 0.0, 2.0), uniform_xi_faces(3)),
                                RadialGrid(uniform_r_faces(2, 1.0)))
        f = rand(Xoshiro(146), size(same)...)
        weights = Runes2P._tail_moment_weights(same, region)
        @test [sum(weights[ir][1] .* f[:, :, ir]) for ir in 1:2] ≈ density(same, f; region) rtol = 1e-13
        @test [sum(weights[ir][2] .* f[:, :, ir]) for ir in 1:2] ≈ current_density(same, f; region) rtol = 1e-13
    end

    @testset "tolerances and the floor" begin
        k, h0 = 3.0, 0.02
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        layout = UnknownLayout(rad, [RadialUnknown(:x; scale=100.0), RadialUnknown(:y)])
        equations = [Equation(:x, TimeDerivative(), LinearTerm(:x; coefficient=-k)),
                     Equation(:y, TimeDerivative(), LinearTerm(:y; coefficient=-k))]
        # the default floor is relative × scale (the default relative tolerance 1e-3): x at 1e-3
        # of its scale lies under the floor and does not hold the step, y at its scale does
        system = EquationSystem(layout, equations)
        solution = integrate(system, [0.1, 1.0], [0.0, 0.1]; stepper=StepController(initial_step=h0))
        stats = solution.statistics
        first = findfirst(==(:step), stats.kind)
        u_full, u_half = 0.1 / (1 + k * h0), 0.1 / (1 + k * h0 / 2)^2
        rel = StepController().relative_tolerance
        error_x = 2 * abs(u_full - u_half) / (rel * 100 + rel * u_full)
        error_y = 2 * abs(10u_full - 10u_half) / (rel + rel * 10u_full)
        @test error_x < error_y
        @test stats.error[first] ≈ error_y
        # a tolerance of the system replaces the default, and zero tolerances leave an
        # unknown out
        system = EquationSystem(layout, equations; time_step_tolerances=Dict(:x => Tolerance(0.0, 1e-2),
                                                                             :y => Tolerance(0.0, 0.0)))
        solution = integrate(system, [0.1, 1.0], [0.0, 0.1]; stepper=StepController(initial_step=h0))
        stats = solution.statistics
        first = findfirst(==(:step), stats.kind)
        @test stats.error[first] ≈ 2 * abs(u_full - u_half) / (1e-2 * u_full)
    end

    @testset "a failed solve shrinks the step by the failure factor" begin
        # the bounded growth of test_step_doubling.jl: the half steps of 0.95 pass, the
        # full step overshoots the bound and fails, the retry has 0.2 times the length
        k = 1.0
        system = single_unknown_system(TimeDerivative(), BoundedGrowthTerm(:x, k, 10.0))
        solution = integrate(system, [1.0], [0.0, 1.0]; stepper=StepController(initial_step=0.95))
        stats = solution.statistics
        @test stats.kind[1:3] == [:half_step, :half_step, :step]
        @test stats.status[1:3] == [:converged, :converged, :domain_error]
        @test !any(stats.accepted[1:3])
        @test stats.time_step[1:3] ≈ [0.475, 0.475, 0.95]
        @test stats.kind[4:6] == [:half_step, :half_step, :step]
        @test stats.time_step[4:6] ≈ [0.095, 0.095, 0.19]
        @test all(==(:converged), stats.status[4:6])
        @test stats.accepted[6] == (stats.error[6] <= 1)
        @test solution.states[1, end] ≈ foldl((u, h) -> _growth_step(u, k, h), stats.time_step[stats.accepted];
                                              init=1.0) rtol = 1e-12
        @test_throws TimeStepFailure integrate(system, [1.0], [0.0, 1.0];
                                               stepper=StepController(initial_step=0.95, max_failures=0))
        # rejections in a row beyond the cap end the run: a shrink factor pinned to 1
        # repeats a rejected step
        k = 3.0
        decay = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k))
        pinned = StepController(initial_step=0.5, relative_tolerance=1e-4, min_shrink=1 - 1e-12, max_rejections=3)
        @test_throws TimeStepFailure integrate(decay, [1.0], [0.0, 1.0]; stepper=pinned)
        failure = try
            integrate(decay, [1.0], [0.0, 1.0]; stepper=pinned)
        catch err
            err
        end
        @test failure.status === :rejected
        @test length(failure.statistics.time) == 3 * 4      # four doublings, the last one over the cap
    end

    @testset "a failed step cut by a ceiling shrinks from its own length" begin
        # the planned length (100) lies far beyond the ceiling at t = 2, so
        # the step tried is 2. Growth k = 0.25 from 9 with the bound 15.5: the half steps of 1
        # give 12 and 16, the second fails. The retry takes 0.2 times the step tried, 0.4,
        # where 0.2 times the planned length (20) cut to the ceiling repeated the failed step.
        system = single_unknown_system(TimeDerivative(), BoundedGrowthTerm(:x, 0.25, 15.5))
        solution = integrate(system, [9.0], [0.0, 2.0]; stepper=StepController(initial_step=100.0, max_failures=2))
        stats = solution.statistics
        @test stats.kind[1:2] == [:half_step, :half_step]
        @test stats.status[1:2] == [:converged, :domain_error]
        @test stats.time_step[1:2] ≈ [1.0, 1.0]
        @test stats.time_step[3] ≈ 0.2                     # the half step of the retry of 0.4
        @test solution.times[end] == 2.0
    end

    @testset "save times within rounding of the time points" begin
        # save times built by another arithmetic than the time points lie an
        # ulp off them. They are taken as the time points, so no step of the rounding length
        # follows the time point.
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-3.0))
        times = collect(range(0.0, 1.0; length=11))
        save_times = [t * (1 + 1e-15) for t in times]
        solution = integrate(system, [1.0], times; save_times, stepper=StepController(initial_step=0.05))
        @test solution.times == times
        @test minimum(solution.statistics.time_step) > 1e-6
        # with the time points of FixedSteps as well
        fixed = integrate(system, [1.0], times; save_times)
        @test fixed.times == times
    end

    @testset "record and resume" begin
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k); nr=2)
        times = [0.0, 1.0]
        save_times = [0.0, 0.4, 1.0]
        stepper = StepController(initial_step=0.01, relative_tolerance=1e-3)
        whole = integrate(system, [1.0, 2.0], times; save_times, stepper)
        @test count(whole.statistics.accepted) > 8
        @test whole.settings["time"]["stepper"]["type"] == "StepController"
        # a segment stopped after its second accepted step (the first is the doubling),
        # resumed from the checkpoint: the predictor has its history back
        checkpoint = CheckpointSettings("controller"; directory=mktempdir(), interval=0)
        first_segment = integrate(system, [1.0, 2.0], times; save_times, stepper, checkpoint,
                                  stop=StopCondition(wall_budget=1e-9, slack=0))
        @test first_segment.settings["segment"]["steps"] == 1
        stored = read_checkpoint(checkpoint)
        @test stored.stepper["type"] == "StepController"
        @test stored.stepper["previous_step"] == whole.statistics.time_step[3]
        @test stored.stepper["previous_state"] == [1.0, 2.0]
        @test stored.stepper["failures"] == 0
        second_segment = @test_logs integrate(system, [1.0, 2.0], times; save_times, stepper, checkpoint, resume=true)
        @test run_status(second_segment) === :finished
        @test second_segment.states == whole.states[:, 2:end]
        rows = length(first_segment.statistics.time)
        @test second_segment.statistics.time == whole.statistics.time[rows+1:end]
        @test second_segment.statistics.time_step == whole.statistics.time_step[rows+1:end]
        @test isequal(second_segment.statistics.error, whole.statistics.error[rows+1:end])
        # no doubling in the resumed segment: the history came from the checkpoint
        @test all(==(:step), second_segment.statistics.kind)
        joined = read_output(checkpoint)
        @test joined.states == whole.states
        @test isequal(joined.statistics.error, whole.statistics.error)
        # a stored state of another layout is refused
        other = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k); nr=3)
        @test_throws DimensionMismatch integrate(other, [1.0, 2.0, 3.0], times; save_times, stepper, resume=stored)
    end

    @testset "the errors per unknown in the log" begin
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient=-k); nr=2)
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false)
        integrate(system, [1.0, 2.0], [0.0, 0.5]; stepper=StepController(initial_step=0.01), log)
        text = String(take!(io))
        @test occursin(r"x: .*error estimate [0-9.e+-]+ of the tolerance", text)
    end
end
