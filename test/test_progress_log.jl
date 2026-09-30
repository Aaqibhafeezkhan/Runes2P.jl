# Progress log of a run: the header, the entries at milestones, after a
# silence and on demand, the events with their limit per kind, and the bounded size of
# the log. The kinetic case is the one of test_checkpoint.jl, the stop request the one of
# test_stop.jl, the failing step the one of test_time_stepping.jl.
using Logging
using Profile

# an interval update that asks for an entry at one time point, as the signal handler does
struct AskForEntry <: IntervalUpdate
    time::Float64
end
Runes2P.update_name(::AskForEntry) = "ask_for_entry"
Runes2P.apply_update!(update::AskForEntry, u, layout, t) = (t == update.time && Profile.peek_report[](); false)
Runes2P.update_state(::AskForEntry) = [0.0]
Runes2P.restore_update!(update::AskForEntry, state) = update

# an interval update that sends SIGUSR1 to the process at one time point and waits until
# the report of Julia has reached the log (at most `timeout` seconds), so that the entry
# comes at the next time point also on a loaded node
struct SendSignal <: IntervalUpdate
    time::Float64
    log::ProgressLog
    timeout::Float64
end
Runes2P.update_name(::SendSignal) = "send_signal"
function Runes2P.apply_update!(update::SendSignal, u, layout, t)
    t == update.time || return false
    ccall(:kill, Cint, (Cint, Cint), getpid(), 10)
    deadline = Base.time() + update.timeout
    while !update.log.requested[] && Base.time() < deadline
        sleep(0.01)
    end
    return false
end
Runes2P.update_state(::SendSignal) = [0.0]
Runes2P.restore_update!(update::SendSignal, state) = update

# exponential decay of a radial unknown, or of a kinetic one, over `intervals` intervals
function _decay_case(intervals; kinetic=false)
    radial = RadialGrid(uniform_r_faces(2, 1.0))
    unknown = kinetic ? KineticUnknown(:f, cylindrical_grid(MomentumGrid(uniform_p_faces(3, 0.0, 1.0),
                                                                          uniform_xi_faces(2)), radial)) :
              RadialUnknown(:n)
    name = kinetic ? :f : :n
    layout = UnknownLayout(radial, [unknown])
    system = EquationSystem(layout, [Equation(name, TimeDerivative(), LinearTerm(name; coefficient=-1.0))])
    return (; system, layout, u0=ones(length(layout)), times=collect(range(0.0, 1.0; length=intervals + 1)))
end

_progress_lines(text) = filter(line -> occursin(r"^\[[0-9T:-]+\] .*progress", line), split(text, '\n'))
_count(pattern, text) = length(collect(eachmatch(pattern, text)))

@testset "progress log" begin
    @testset "settings and formats" begin
        log = ProgressLog(io=IOBuffer())
        @test log.milestones[1:4] == [1e-4, 1e-3, 0.01, 0.02]
        @test length(log.milestones) == 102 && last(log.milestones) == 1.0
        @test log.guard_seconds == 60.0 .* [1, 2, 5, 10, 20, 30, 60, 90, 120]
        @test log.events_shown == 3
        @test log.color === false
        @test describe(log) == Dict{String,Any}("type" => "ProgressLog", "milestones" => log.milestones,
                                                "guard_minutes" => [1.0, 2, 5, 10, 20, 30, 60, 90, 120],
                                                "events_shown" => 3, "state" => true, "on_signal" => true,
                                                "request_file" => "")
        @test_throws ArgumentError ProgressLog(milestones=[0.0, 0.5])
        @test_throws ArgumentError ProgressLog(milestones=[0.5, 1.5])
        @test_throws ArgumentError ProgressLog(guard_minutes=())
        @test_throws ArgumentError ProgressLog(guard_minutes=(1, 0))
        @test_throws ArgumentError ProgressLog(events_shown=-1)

        @test Runes2P._duration(12.34) == "12.3 s"
        @test Runes2P._duration(0.2694) == "269.4 ms"
        @test Runes2P._duration(125) == "2 min 05 s"
        @test Runes2P._duration(3725) == "1 h 02 min"
        @test Runes2P._duration(90000) == "1 d 01 h"
        @test Runes2P._duration(NaN) == "unknown"
        @test Runes2P._bytes(3 * 1024^2) == "3 MB"
        @test Runes2P._bytes(1.5 * 1024^3) == "1.50 GB"
        @test Runes2P._values([1.0, 2.5]) == "1 2.5"
        @test Runes2P._values(collect(1.0:20.0)) == "20 values from 1 to 20, first 1, last 20"
        @test Runes2P._values([1.0 2.0; 3.0 4.0]) == "2 × 2 values from 1 to 4, first 1, last 4"
        @test Runes2P._values(["a", "b"]) == "a, b"
    end

    @testset "header and entries of a run" begin
        case = _checkpoint_case()
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false)
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),),
                             log, description=Dict("name" => "log test", "delta_b_over_b" => 1e-3))
        text = String(take!(io))
        reference = _checkpoint_case()
        plain = integrate(reference.system, reference.u0, reference.times; save_times=reference.save_times,
                          updates=(reference.update(),),
                          description=Dict("name" => "log test", "delta_b_over_b" => 1e-3))
        @test solution.states == plain.states           # the log does not change the run
        @test solution.settings["log"]["type"] == "ProgressLog"
        @test solution.settings["setup_hash"] == plain.settings["setup_hash"]

        # the header: what identifies the run, then the solver, then the physics and grids
        code = code_version()
        @test occursin("Runes2P run \"log test\", segment 1", text)
        @test occursin("commit " * code["commit"] * (code["tree_clean"] ? " (clean tree)" : " (changed tree)"), text)
        @test occursin("julia       " * string(VERSION) * " on " * gethostname(), text)
        @test occursin("setup hash  " * solution.settings["setup_hash"], text)
        position(title) = first(findfirst("- " * title * " -", text))
        @test first(findfirst("Runes2P run", text)) < position("solver") < position("time") <
              position("layout and grids") < position("system") < position("prescribed") < position("run")
        @test occursin("max_iterations = 30", text)
        @test occursin("  type = NewtonSolver\n  floor_linear_tolerance = 1e-08", text)     # the type first
        @test occursin("type = SparseLU", text)
        @test occursin("log: events_shown = 3, guard_minutes = 9 values from 1 to 120, first 1, last 120, " *
                       "milestones = 102 values from 0.0001 to 1, first 0.0001, last 1, on_signal = false, " *
                       "state = true", text)
        @test occursin("9 time points from t = 0 to 0.0008 s, intervals from 0.0001 to 0.0001 s", text)
        @test occursin("7 save times, stepper FixedSteps with max_halvings = 10", text)
        @test occursin("f: kinetic on grid_1, scale 1e+20, 128 entries", text)
        @test occursin("hot_density: radial, scale 1e+20, 2 entries", text)
        @test occursin("grid_1: cylindrical geometry, 2 radii from r = 0 to 1 m, 16 momentum cells from p = 0 to 1.2 " *
                       "(first cell 0.025), 4 pitch cells", text)
        @test occursin("type = PrescribedFokkerPlanckTerm", text)
        @test occursin("p_scheme = exponential_fitting", text)
        @test occursin("type = KnockOnTerm", text)
        @test occursin("type = KnockOnCutoffUpdate", text)
        @test occursin("diffusion_r_r = 16 × 4 × 3 values from 50 to 50", text)   # arrays are summarized
        @test occursin("placement = energy_midpoint", text)
        @test occursin("table_1:", text)
        @test occursin("name = log test", text)
        @test occursin("delta_b_over_b = 0.001", text)
        header_lines = count(==('\n'), first(split(text, "] progress"; limit=2)))
        @test header_lines < 250

        # the entries: the start, a milestone at each of the seven time points between, the end
        entries = _progress_lines(text)
        @test length(entries) == 9 == log.entries
        @test occursin("  0.000 %  t = 0.000000e+00 s of 8.000000e-04 s, time point 1 of 9  (segment start)", entries[1])
        @test occursin(" 12.500 %  t = 1.000000e-04 s of 8.000000e-04 s, time point 2 of 9  (milestone)", entries[2])
        @test occursin("100.000 %  t = 8.000000e-04 s of 8.000000e-04 s, time point 9 of 9  (segment end, finished)",
                       entries[9])
        @test count(entry -> occursin("(milestone)", entry), entries) == 7
        @test occursin("steps   1 accepted of 1 attempted since the last entry (0 failed, 0 rejected, 0 half steps", text)
        @test occursin("steps   none since the last entry", text)       # the entry of the start
        @test occursin(r"solver  Newton iterations \d+\.\d\d per step, Krylov iterations 0\.00 per Newton iteration", text)
        @test occursin(r"cost    \d+\.\d (ms|s) per step: Jacobian \d+ %, residual \d+ %, factorization \d+ %", text)
        @test occursin(r"wall    .* since the start of the segment, .* since the last entry, estimate for the 7 time points left", text)
        @test occursin("estimate for the 0 time points left: none", text)
        @test occursin(r"memory  .* at most resident, .* live in Julia", text)

        # the state, set off from the solver block
        @test _count(r"-{20} state at t = ", text) == 9
        final = solution.states[:, end]
        distribution = stack(collect(unknown_view(final, case.layout, :f, ir)) for ir in 1:2)
        @test occursin("f: density [m^-3] " * Runes2P._values(density(case.grid, distribution)), text)
        currents = current_density(case.grid, distribution)
        @test occursin("current density [A/m^2] " * Runes2P._values(currents), text)
        radial = case.grid.radial_grid
        total = 2pi * sum(radial.r_centers .* radial.r_widths .* currents)
        @test occursin("total current [A] " * Runes2P._number(total), text)
        @test occursin("negative cells 0", text)
        @test occursin("hot_density: " * Runes2P._values([only(unknown_view(final, case.layout, :hot_density, ir))
                                                       for ir in 1:2]), text)

        # without the state
        case = _checkpoint_case()
        log = ProgressLog(io=io, on_signal=false, state=false)
        integrate(case.system, case.u0, case.times; updates=(case.update(),), log)
        text = String(take!(io))
        @test !occursin("state at t", text)
        @test !occursin("density [m^-3]", text)
        @test length(_progress_lines(text)) == 9
    end

    @testset "the size of the log is bounded" begin
        case = _decay_case(2000)
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false)
        integrate(case.system, case.u0, case.times; save_times=[0.0, 1.0], log)
        text = String(take!(io))
        entries = _progress_lines(text)
        # the start, 0.01 %, 0.1 %, 1 % to 99 %, and the end at 100 %
        @test length(entries) == 103 == log.entries
        @test occursin("  0.050 %", entries[2]) && occursin("(milestone)", entries[2])
        @test all(entry -> occursin("(milestone)", entry), entries[2:102])
        @test occursin("100.000 %", entries[103]) && occursin("(segment end, finished)", entries[103])
        lines = count(==('\n'), text)
        @test lines < 1500
        # the same run with ten times the steps has the same number of entries
        case = _decay_case(20000)
        integrate(case.system, case.u0, case.times; save_times=[0.0, 1.0], log)
        longer = String(take!(io))
        @test length(_progress_lines(longer)) == 103
        @test count(==('\n'), longer) == lines
    end

    @testset "silence guard" begin
        # the first limit is passed at once, the second never: one entry of the guard
        case = _decay_case(50)
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], guard_minutes=(1e-12, 1e9))
        integrate(case.system, case.u0, case.times; log)
        entries = _progress_lines(String(take!(io)))
        @test length(entries) == 3
        @test occursin("time point 2 of 51  (silence)", entries[2])
        @test log.guard_index == 2
        # the last limit stays: an entry at every time point
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], guard_minutes=(1e-12,))
        integrate(case.system, case.u0, case.times; log)
        entries = _progress_lines(String(take!(io)))
        @test count(entry -> occursin("(silence)", entry), entries) == 49
        # an entry at a milestone does not set the sequence back
        case = _decay_case(10)
        log = ProgressLog(io=io, on_signal=false, milestones=[0.5, 1.0], guard_minutes=(1e-12, 1e9))
        integrate(case.system, case.u0, case.times; log)
        entries = _progress_lines(String(take!(io)))
        @test length(entries) == 4
        @test occursin("(silence)", entries[2]) && occursin(" 10.000 %", entries[2])
        @test occursin("(milestone)", entries[3]) && occursin(" 50.000 %", entries[3])
        @test log.guard_index == 2
    end

    @testset "events" begin
        # a step that fails and is halved
        system = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, 4.0))
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0])
        solution = integrate(system, [0.2], [0.0, 0.24, 0.26]; log)
        text = String(take!(io))
        @test solution.statistics.accepted == [false, true, true, true]
        @test occursin("] WARNING: step failed: the step to t = 0.24 s with the time step 0.24 s ended with the " *
                       "status domain_error and is retried with half the time step", text)
        @test occursin("steps   3 accepted of 4 attempted since the last entry (1 failed, 0 rejected, 0 half steps", text)
        @test occursin("events  since the last entry: 1 × step failed", text)
        # a run that fails
        log = ProgressLog(io=io, on_signal=false)
        @test_throws TimeStepFailure integrate(system, [0.2], [0.0, 0.24, 0.26]; max_step_halvings=0, log)
        text = String(take!(io))
        @test occursin("] ERROR: run failed: the step to t = 0.24 s with the time step 0.24 s failed with the status " *
                       "domain_error after 0 halvings", text)
        @test occursin("(segment end, failed)", text)

        # of a kind of event a few are written, the rest is counted
        case = _decay_case(10; kinetic=true)
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], events_shown=2)
        integrate(case.system, -case.u0, case.times; log)
        text = String(take!(io))
        @test _count(r"WARNING: negative values: 12 negative cells of the kinetic unknowns after the step", text) == 2
        @test _count(r"further events of this kind are counted in the next entry", text) == 1
        @test occursin("events  since the last entry: 10 × negative values (8 not shown)", text)
        @test occursin("negative cells 12 (smallest value", text)
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], events_shown=0)
        integrate(case.system, -case.u0, case.times; log)
        text = String(take!(io))
        @test !occursin("WARNING", text)
        @test occursin("10 × negative values (10 not shown)", text)

        # colour for a terminal
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], color=true)
        integrate(case.system, -case.u0, case.times; log)
        text = String(take!(io))
        @test occursin("\e[1;33mWARNING\e[0m: negative values", text)
        @test occursin("\e[1;36mprogress\e[0m", text)

        # messages of the logging macros inside a run go to the log
        with_logger(log) do
            @warn "a warning" value = 3 values = [1.0, 2.0]
            @error "an error"
            @debug "not written"
        end
        text = String(take!(io))
        @test occursin("\e[1;33mWARNING\e[0m: a warning (value = 3, values = 1 2)", text)
        @test occursin("\e[1;31mERROR\e[0m: an error", text)
        @test !occursin("not written", text)
    end

    @testset "entry on demand" begin
        case = _checkpoint_case()
        io = IOBuffer()
        # by the request file, which the log removes
        request = tempname()
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0], request_file=request)
        integrate(case.system, case.u0, case.times; updates=(case.update(), RequestStop(request, case.times[4])), log)
        entries = _progress_lines(String(take!(io)))
        # asked at the time points 4 to 8 by the update, which runs after the accepted step
        # and before the report of that step, so each request is served at its own time point
        @test count(entry -> occursin("(on demand)", entry), entries) == 5
        @test occursin("time point 4 of 9  (on demand)", entries[2])
        rm(request; force=true)

        # by the report of Julia on SIGUSR1, which the log replaces during the run only
        before = Profile.peek_report[]
        case = _checkpoint_case()
        log = ProgressLog(io=io, milestones=[1.0])
        integrate(case.system, case.u0, case.times; updates=(case.update(), AskForEntry(case.times[3])), log)
        entries = _progress_lines(String(take!(io)))
        @test length(entries) == 3
        @test occursin("time point 3 of 9  (on demand)", entries[2])
        @test Profile.peek_report[] === before
        # also after a run that failed
        system = single_unknown_system(TimeDerivative(), SquareRootDecayTerm(:x, 4.0))
        @test_throws TimeStepFailure integrate(system, [0.2], [0.0, 0.24, 0.26]; max_step_halvings=0,
                                               log=ProgressLog(io=io))
        @test Profile.peek_report[] === before
        take!(io)
        # without on_signal the report of Julia stays
        case = _checkpoint_case()
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0])
        integrate(case.system, case.u0, case.times; updates=(case.update(), AskForEntry(-1.0)), log)
        @test length(_progress_lines(String(take!(io)))) == 2

        # the signal itself: Julia profiles for the duration set, then calls the report
        duration = Profile.get_peek_duration()
        Profile.set_peek_duration(0.1)
        try
            case = _decay_case(8)
            log = ProgressLog(io=io, milestones=[1.0])
            integrate(case.system, case.u0, case.times; updates=(SendSignal(case.times[2], log, 30.0),), log)
            entries = _progress_lines(String(take!(io)))
            @test count(entry -> occursin("(on demand)", entry), entries) == 1
        finally
            Profile.set_peek_duration(duration)
        end
        @test Profile.peek_report[] === before
    end

    @testset "segments, checkpoints and stop" begin
        directory = _checkpoint_directory()
        file = joinpath(directory, "log", "run.log")
        checkpoint = CheckpointSettings("logged"; directory, interval=0)
        case = _checkpoint_case()
        log = ProgressLog(file=file, on_signal=false)
        @test log.color === false
        integrate(case.system, case.u0, case.times[1:5]; save_times=filter(<=(case.times[5]), case.save_times),
                  updates=(case.update(),), checkpoint, log)
        close(log)
        first_text = read(file, String)
        @test _count(r"info: checkpoint: written at t = ", first_text) == 4
        @test occursin("events  since the last entry: 1 × checkpoint", first_text)
        @test occursin("checkpoint: directory = $directory, interval = 0, name = logged, overwrite = false", first_text)

        # the second segment appends to the file; its warnings go to the log, not to the terminal
        case = _checkpoint_case()
        log = ProgressLog(file=file, on_signal=false)
        second = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                      updates=(case.update(),), checkpoint, resume=true, log)
        close(log)
        text = read(file, String)
        @test startswith(text, first_text)
        rest = text[ncodeunits(first_text)+1:end]
        @test occursin("Runes2P run, segment 2", rest)
        @test occursin("resumed     from " * Runes2P.checkpoint_file(checkpoint) * " at t = 0.0004 s, changed sections: " *
                       "prescribed, time", rest)
        @test occursin("WARNING: resuming " * Runes2P.checkpoint_file(checkpoint) * " with changed settings", rest)
        # the milestones behind the segment are not written again
        entries = _progress_lines(rest)
        @test length(entries) == 5 == log.entries
        @test occursin(" 50.000 %", entries[1]) && occursin("time point 5 of 9  (segment start)", entries[1])
        @test occursin(" 62.500 %", entries[2]) && occursin("(milestone)", entries[2])
        @test second.settings["segment"]["index"] == 2

        # a stop
        case = _checkpoint_case()
        io = IOBuffer()
        log = ProgressLog(io=io, on_signal=false, milestones=[1.0])
        solution = integrate(case.system, case.u0, case.times; updates=(case.update(),),
                             stop=StopCondition(wall_budget=1e-9, slack=0), log)
        text = String(take!(io))
        @test run_status(solution) === :stopped
        @test occursin("WARNING: stop: the segment stops at t = 0.0001 s: wall time", text)
        @test occursin("time point 2 of 9  (segment end, stopped: wall time)", text)
        @test occursin("stop: end_time = ", text)
    end
end
