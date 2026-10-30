# DREAM's step doubling (`TimeStepperAdaptive.cpp`) on a linear decay whose
# backward-Euler steps are known in closed form: the error estimate, the step sequence by
# DREAM's rule with its rejections, the steps between checks, the failure path, the
# record of the stepper and the resume from a checkpoint, and the acceptance of an unknown
# at zero with no absolute tolerance, on synthetic states and in a run with its retry. The
# comparison with DREAM's own sequence is in test_time_stepping_dream.jl. The helpers are
# those of test_time_stepping.jl.
using HDF5

# du/dt = -k u: one backward-Euler step u/(1 + k h), two half steps u/(1 + k h/2)²; the
# error of the step relative to its tolerance, |u_full - u_half| / (rel |u_full|)
_decay_step(u, k, h) = u / (1 + k * h)

_doubling_error(k, h, rel) = abs(1 - (1 + k * h) / (1 + k * h / 2)^2) / rel

# the sequence DREAM's rule produces on the decay from t = 0 to t_end: per checked step
# its end time, length, error and acceptance, and the state at the end
function _dream_sequence(k, h0, rel, t_end; safety = 0.95)
    t, h, u = 0.0, h0, 1.0
    checks = NamedTuple{
        (:time, :step, :error, :accepted), Tuple{Float64, Float64, Float64, Bool}}[]
    while t < t_end
        t_next = t_end - t <= h * (1 + 1e-10) ? t_end : t + h
        h_step = t_next - t
        error = _doubling_error(k, h_step, rel)
        accepted = error <= 1
        push!(checks, (time = t_next, step = h_step, error = error, accepted = accepted))
        h = safety * h_step / error
        if accepted
            u = _decay_step(u, k, h_step)
            t = t_next
        end
    end
    return checks, u
end

# a source `rate` before the time `until` and none after it: from x = 0, a backward-Euler
# step that ends after `until` leaves x at zero, while its first half step adds to it
struct PulseSourceTerm <: EquationTerm
    rate::Float64
    until::Float64
end
Runes2P.dependencies(::PulseSourceTerm) = Symbol[]
function Runes2P.rhs!(out, term::PulseSourceTerm, u, layout, row, t)
    if t < term.until
        for ir in 1:radial_cell_count(layout), a in unknown_range(layout, row, ir)

            out[a] += term.rate
        end
    end
    return out
end
Runes2P.sparsity!(entries, ::PulseSourceTerm, layout, row) = entries

@testset "step doubling" begin
    @testset "settings" begin
        stepper = StepDoubling()
        @test isnan(stepper.initial_step)
        @test (stepper.relative_tolerance, stepper.check_every,
            stepper.safety, stepper.exponent) == (1e-2, 0, 0.95, 1.0)
        @test (stepper.max_failures, stepper.failure_factor) == (5, 0.2)
        @test StepDoubling(initial_step = 1e-3).initial_step == 1e-3
        @test_throws ArgumentError StepDoubling(initial_step = 0)
        @test_throws ArgumentError StepDoubling(relative_tolerance = 0)
        @test_throws ArgumentError StepDoubling(check_every = -1)
        @test_throws ArgumentError StepDoubling(safety = 1.5)
        @test_throws ArgumentError StepDoubling(exponent = 0)
        @test_throws ArgumentError StepDoubling(max_failures = -1)
        @test_throws ArgumentError StepDoubling(failure_factor = 1)
        @test describe(StepDoubling(initial_step = 1e-3, check_every = 2)) ==
              Dict{String, Any}(
            "type" => "StepDoubling", "initial_step" => 1e-3, "relative_tolerance" => 1e-2,
            "check_every" => 2, "safety" => 0.95, "exponent" => 1.0, "max_failures" => 5,
            "failure_factor" => 0.2)
    end

    @testset "DREAM's rule on a linear decay, with rejections" begin
        # k h0 = 1.5 is rejected (error 18 times the tolerance); the rule then alternates
        # between accepted short steps and rejected long ones, since with error ∝ h² and
        # the exponent 1 the map h → 0.95 h / error has a marginal fixed point
        k, h0, rel, t_end = 3.0, 0.5, 1e-2, 1.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k))
        solution = integrate(system, [1.0], [0.0, t_end]; save_times = [0.0, t_end],
            stepper = StepDoubling(initial_step = h0, relative_tolerance = rel))
        stats = solution.statistics
        checks, u_end = _dream_sequence(k, h0, rel, t_end)
        checked = findall(==(:step), stats.kind)
        @test length(checked) == length(checks) > 20
        @test stats.time[checked] ≈ [c.time for c in checks]
        @test stats.time_step[checked] ≈ [c.step for c in checks]
        @test stats.error[checked] ≈ [c.error for c in checks] rtol = 1e-10
        @test stats.accepted[checked] == [c.accepted for c in checks]
        @test count(c -> !c.accepted, checks) >= 5
        @test all(==(:converged), stats.status)          # a rejected step converged all the same
        # two half steps before every checked step, never accepted, without an error
        @test count(==(:half_step), stats.kind) == 2 * length(checked)
        half = findall(==(:half_step), stats.kind)
        @test !any(stats.accepted[half])
        @test all(isnan, stats.error[half])
        @test stats.time[checked .- 1] ≈ stats.time[checked]            # the second half step ends with the step
        @test stats.time[checked .- 2] ≈
              stats.time[checked] .- stats.time_step[checked] ./ 2
        @test solution.states[1, end] ≈ u_end rtol = 1e-12
        @test solution.settings["time"]["stepper"]["type"] == "StepDoubling"
        @test solution.settings["segment"]["steps"] == count(c -> c.accepted, checks)
        # the last accepted step lands on the end of the run
        @test last(stats.time) == t_end
    end

    @testset "steps between checks" begin
        k, h0, rel = 3.0, 0.02, 1e-2
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k))
        solution = integrate(system, [1.0], [0.0, 2.0]; save_times = [0.0, 2.0],
            stepper = StepDoubling(initial_step = h0, relative_tolerance = rel, check_every = 2))
        stats = solution.statistics
        # the first step is checked, then two unchecked steps at the new length, and so on
        @test stats.kind[1:3] == [:half_step, :half_step, :step]
        @test stats.error[3] ≈ _doubling_error(k, h0, rel)
        @test stats.accepted[3]
        h1 = 0.95 * h0 / stats.error[3]
        @test stats.kind[4:5] == [:step, :step]
        @test stats.time_step[4:5] ≈ [h1, h1]
        @test all(isnan, stats.error[4:5])
        @test all(stats.accepted[4:5])
        @test stats.kind[6:8] == [:half_step, :half_step, :step]
        @test stats.time_step[8] ≈ h1
        # the check at row 8 rejects its step, which is redone as a new doubling, as in DREAM
        @test !stats.accepted[8]
        @test stats.error[8] ≈ _doubling_error(k, h1, rel)
        @test stats.kind[9:11] == [:half_step, :half_step, :step]
        @test stats.time_step[11] ≈ 0.95 * h1 / stats.error[8]
        @test solution.states[1, end] ≈
              foldl((u, h) -> _decay_step(u, k, h), stats.time_step[stats.accepted]; init = 1.0)
    end

    @testset "save times between the steps, the last step lands on the end" begin
        # the time points are the ceilings of the steps and the save times must be among
        # them: every saved state is a computed one, as in DREAM, and a save time between
        # the time points is rejected
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k))
        times = [0.0, 0.3, 0.55, 1.0]
        stepper = StepDoubling(initial_step = 0.05, relative_tolerance = 1e-2)
        solution = integrate(system, [1.0], times; save_times = [0.0, 0.55, 1.0], stepper)
        stats = solution.statistics
        accepted_times = stats.time[stats.accepted]
        @test all(t -> t in accepted_times, times[2:end])
        @test solution.times == [0.0, 0.55, 1.0]
        for (k_save, s) in enumerate([0.55, 1.0])
            upto = findfirst(==(s), accepted_times)
            states = foldl(
                (u, h) -> _decay_step(u, k, h), diff([0.0;
                                                      accepted_times[1:upto]]); init = 1.0)
            @test solution.states[1, k_save + 1] ≈ states rtol = 1e-12
        end
        @test_throws ArgumentError integrate(
            system, [1.0], times; save_times = [0.0, 0.4, 1.0], stepper)
    end

    @testset "tolerances per unknown" begin
        k, h0 = 3.0, 0.1
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        layout = UnknownLayout(rad, [RadialUnknown(:x), RadialUnknown(:y)])
        equations = [Equation(:x, TimeDerivative(), LinearTerm(:x; coefficient = -k)),
            Equation(:y, TimeDerivative(), LinearTerm(:y; coefficient = -2k))]
        # a looser tolerance on y than the stepper's default on x: x sets the error
        loose = EquationSystem(layout, equations; time_step_tolerances = Dict(:y =>
            Tolerance(0.0, 1.0)))
        solution = integrate(loose, [1.0, 1.0], [0.0, 0.2]; stepper = StepDoubling(initial_step = h0))
        first_check = findfirst(==(:step), solution.statistics.kind)
        @test solution.statistics.error[first_check] ≈ _doubling_error(k, h0, 1e-2)
        # an absolute tolerance enters the scale
        absolute = EquationSystem(layout,
            equations;
            time_step_tolerances = Dict(:x => Tolerance(0.5, 1e-2),
                :y => Tolerance(0.5, 1e-2)))
        solution = integrate(absolute, [1.0, 1.0], [0.0, 0.2]; stepper = StepDoubling(initial_step = h0))
        first_check = findfirst(==(:step), solution.statistics.kind)
        u_full, u_half = _decay_step(1.0, 2k, h0),
        _decay_step(_decay_step(1.0, 2k, h0 / 2), 2k, h0 / 2)
        @test solution.statistics.error[first_check] ≈
              abs(u_full - u_half) / (0.5 + 1e-2 * abs(u_full))
        # both tolerances zero: the unknown is left out, and with every unknown left out
        # the error is zero and the next step unbounded (one step to the end)
        excluded = EquationSystem(layout,
            equations;
            time_step_tolerances = Dict(:x => Tolerance(0.0, 0.0),
                :y => Tolerance(0.0, 0.0)))
        solution = integrate(excluded, [1.0, 1.0], [0.0, 0.2]; stepper = StepDoubling(initial_step = h0))
        stats = solution.statistics
        @test stats.kind == [:half_step, :half_step, :step, :half_step, :half_step, :step]
        @test stats.error[[3, 6]] == [0.0, 0.0]
        @test stats.time_step[6] ≈ 0.2 - h0
    end

    @testset "an unknown at zero with no absolute tolerance" begin
        # DREAM accepts a step only with dx ≤ epsa + epsr x for every unknown
        # (ConvergenceChecker::IsConverged), so with epsa = 0 and x = 0 any difference fails
        # it; the zero scale stays out of the largest error, which sets the next step
        rad = RadialGrid(uniform_r_faces(1, 0.5))
        layout = UnknownLayout(rad, [RadialUnknown(:x), RadialUnknown(:y)])
        state = Runes2P.StepDoublingState{Float64}(
            0.1, 0, 0, [0.0, 0.0], [1e-2, 1e-2], zeros(2), zeros(2), fill(NaN, 2))
        # x at zero after the full step and not after the halves; y within its tolerance
        largest, converged = Runes2P._doubling_errors!(state, layout, [0.0, 1.0], [
            1e-9, 1.0005])
        @test !converged
        @test largest ≈ 0.05
        @test state.errors == [Inf, largest]
        # no difference at the zero scale: no error there, and the step converges
        largest, converged = Runes2P._doubling_errors!(state, layout, [0.0, 1.0], [
            0.0, 1.0005])
        @test converged
        @test state.errors == [0.0, largest]
    end

    @testset "a run rejects the step that an unknown at zero fails" begin
        # x = 0 with the pulse until 0.75: the full step of 1 leaves x at zero, the half steps
        # give 0.5, so the step converges with no error and is rejected all the same. With no
        # other error to scale by, it is retried with half its length (a choice of this
        # package), which ends inside the pulse; then the error is zero, and the next step
        # runs to the end.
        system = single_unknown_system(TimeDerivative(), PulseSourceTerm(1.0, 0.75))
        solution = integrate(system, [0.0], [0.0, 2.0]; stepper = StepDoubling(initial_step = 1.0))
        stats = solution.statistics
        checked = findall(==(:step), stats.kind)
        @test stats.time_step[checked] ≈ [1.0, 0.5, 1.5]
        @test stats.accepted[checked] == [false, true, true]
        @test stats.error[checked] == [0.0, 0.0, 0.0]
        @test all(==(:converged), stats.status)
        @test solution.states[1, end] ≈ 0.5
    end

    @testset "the step after such a rejection follows from the other errors" begin
        # a decaying y whose error is within its tolerance: the step is rejected with an
        # error below 1, and the next step follows from that error by DREAM's rule, longer
        # than the step rejected
        k, h0 = 0.1, 1.0
        decay = (RadialUnknown(:y),
            Equation(:y, TimeDerivative(), LinearTerm(:y; coefficient = -k)))
        system = single_unknown_system(TimeDerivative(), PulseSourceTerm(1.0, 0.75); extra = (decay,))
        solution = integrate(system, [0.0, 1.0], [0.0, 10.0]; stepper = StepDoubling(initial_step = h0))
        stats = solution.statistics
        checked = findall(==(:step), stats.kind)
        step_error = _doubling_error(k, h0, 1e-2)
        @test step_error < 1
        @test !stats.accepted[checked[1]]
        @test stats.error[checked[1]] ≈ step_error
        @test stats.time_step[checked[2]] ≈ 0.95 * h0 / step_error
    end

    @testset "a failed solve shrinks the step by the failure factor" begin
        # growth from 1 with the bound 10 and k = 1: the half steps of the initial length
        # 0.95 pass (1.9, then 3.6), the full step overshoots the bound (20) and fails,
        # and the stepper retries with 0.2 times the length
        k = 1.0
        system = single_unknown_system(TimeDerivative(), BoundedGrowthTerm(:x, k, 10.0))
        solution = integrate(system, [1.0], [0.0, 1.0]; stepper = StepDoubling(initial_step = 0.95))
        stats = solution.statistics
        @test stats.kind[1:3] == [:half_step, :half_step, :step]
        @test stats.status[1:3] == [:converged, :converged, :domain_error]
        @test !any(stats.accepted[1:3])
        @test stats.time_step[1:3] ≈ [0.475, 0.475, 0.95]
        @test stats.kind[4:6] == [:half_step, :half_step, :step]
        @test stats.time_step[4:6] ≈ [0.095, 0.095, 0.19]
        @test all(==(:converged), stats.status[4:6])
        # the retry is checked as any step: accepted when its error is within the tolerance
        @test stats.accepted[6] == (stats.error[6] <= 1)
        @test solution.states[1, end] ≈
              foldl((u, h) -> _growth_step(u, k, h), stats.time_step[stats.accepted];
            init = 1.0) rtol = 1e-12
        @test solution.states[1, end] < 10
        @test_throws TimeStepFailure integrate(system, [1.0], [0.0, 1.0];
            stepper = StepDoubling(initial_step = 0.95, max_failures = 0))
    end

    @testset "a failed step cut by a ceiling shrinks from its own length" begin
        # the planned length (100) lies far beyond the ceiling at t = 2, so the step tried is
        # 2. Growth k = 0.25 from 9 with the bound 15.5: the half steps of 1 give 12 and 16,
        # the second fails. The retry takes 0.2 times the step tried, 0.4; 0.2 times the
        # planned length (20), cut to the ceiling, would repeat the failed step.
        system = single_unknown_system(TimeDerivative(), BoundedGrowthTerm(:x, 0.25, 15.5))
        solution = integrate(system, [9.0], [0.0, 2.0];
            stepper = StepDoubling(initial_step = 100.0, max_failures = 2))
        stats = solution.statistics
        @test stats.kind[1:2] == [:half_step, :half_step]
        @test stats.status[1:2] == [:converged, :domain_error]
        @test stats.time_step[1:2] ≈ [1.0, 1.0]
        @test stats.time_step[3] ≈ 0.2                     # the half step of the retry of 0.4
        @test solution.times[end] == 2.0
    end

    @testset "record and resume" begin
        k, h0 = 3.0, 0.05
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k))
        times = [0.0, 0.5, 1.0]            # the save times are time points
        save_times = [0.0, 0.5, 1.0]
        stepper = StepDoubling(initial_step = h0, relative_tolerance = 1e-2, check_every = 1)
        whole = integrate(system, [1.0], times; save_times, stepper)
        @test count(whole.statistics.accepted) > 10
        # a segment stopped after its first accepted step, resumed from the checkpoint
        checkpoint = CheckpointSettings("doubling"; directory = mktempdir(), interval = 0)
        first_segment = integrate(system, [1.0], times; save_times, stepper, checkpoint,
            stop = StopCondition(wall_budget = 1e-9, slack = 0))
        @test run_status(first_segment) === :stopped
        @test first_segment.settings["segment"]["steps"] == 1
        stored = read_checkpoint(checkpoint)
        @test stored.time == whole.statistics.time[findfirst(whole.statistics.accepted)]
        @test stored.stepper["type"] == "StepDoubling"
        @test stored.stepper["dt"] ≈ 0.95 * h0 / whole.statistics.error[3]
        @test stored.stepper["steps_since_check"] == 0
        @test stored.stepper["failures"] == 0
        second_segment = @test_logs integrate(
            system, [1.0], times; save_times, stepper, checkpoint, resume = true)
        @test run_status(second_segment) === :finished
        @test second_segment.times == save_times[2:end]
        @test second_segment.states == whole.states[:, 2:end]
        done = count(first_segment.statistics.accepted)
        rows = length(first_segment.statistics.time)
        @test second_segment.statistics.time == whole.statistics.time[(rows + 1):end]
        @test second_segment.statistics.time_step ==
              whole.statistics.time_step[(rows + 1):end]
        @test second_segment.statistics.kind == whole.statistics.kind[(rows + 1):end]
        @test isequal(second_segment.statistics.error, whole.statistics.error[(rows + 1):end])
        joined = read_output(checkpoint)
        @test joined.states == whole.states
        @test isequal(joined.statistics.kind, whole.statistics.kind)
        @test isequal(joined.statistics.error, whole.statistics.error)
        # the output file holds the new columns
        file = joinpath(checkpoint.directory, "whole.h5")
        write_output(file, whole)
        h5open(file) do h5
            @test read(h5["solver_statistics/kind"]) == String.(whole.statistics.kind)
            @test isequal(read(h5["solver_statistics/error"]), whole.statistics.error)
        end
        back = read_output(file)
        @test isequal(back.statistics.kind, whole.statistics.kind)
        @test isequal(back.statistics.error, whole.statistics.error)
    end
end
